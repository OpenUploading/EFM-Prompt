"""FineMI CBraMod EEG-only and frozen-head prompt transfer experiments."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_EEG_CACHE = Path(r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_rawuv_binary_1v6")
sys.path.insert(0, str(ROOT))

from finemi_prompt_data import SPLITS, load_prompt_trials
from cbramod_mope_boundary import MoPEBoundaryPrompt
from run_shin2017_cbramod_fnirs_feature_stage1 import FnirsTemporalEncoder
from foundation_tmpa_token_alignment import FoundationTMPAFinalAdapter
from foundation_deep_prompt import DeepConditionalPrompt, SharedDeepThreeComponentPrompt
from foundation_hierarchical_cross_attention import FoundationHierarchicalCrossAttentionAdapter
from foundation_hierarchical_cross_attention_bidirectional_contrast import (
    FoundationHierarchicalBidirectionalContrastAdapter,
)
from run_shin2017_cbramod_fnirs_feature_stage1 import metrics
from run_shin2017_foundation_boundary_prompt import load_compatible
from run_shin2017_foundation_tmpa_token_alignment import class_aware_contrastive_loss


class Trials(Dataset):
    def __init__(self, eeg, fnirs, labels, indices):
        self.eeg, self.fnirs, self.labels, self.indices = eeg, fnirs, labels, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        return self.eeg[index], self.fnirs[index], self.labels[index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "eegonly",
            "mope",
            "deep_conditional",
            "deep_three_component_shared",
            "tmpa_final",
            "hierarchical_cross_attention",
            "bidirectional_contrast",
            # Backward-compatible names used by earlier FineMI runs.
            "three_component",
            "tmpa",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, default=ROOT.parent)
    parser.add_argument("--eeg-cache-root", type=Path, default=DEFAULT_EEG_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eegonly-head-checkpoint", type=Path, default=None,
                        help="EEG-only best_prompt_and_head.pth; required with --freeze-classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--prompt-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-pair", type=float, default=0.1)
    parser.add_argument("--lambda-class", type=float, default=0.02)
    parser.add_argument("--importance-weight", type=float, default=0.01)
    parser.add_argument("--freeze-classifier", action="store_true")
    parser.add_argument("--prompt-boundary", choices=("pre", "pre_post"), default="pre")
    parser.add_argument("--fnirs-window", type=float, nargs=2, metavar=("START", "END"), default=(3.0, 7.0))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FineMICBraModPrompt(nn.Module):
    def __init__(self, args):
        super().__init__()
        sys.path.insert(0, str(args.portable_root / "CBraMod"))
        from models.cbramod import CBraMod

        aliases = {"three_component": "mope", "tmpa": "tmpa_final"}
        self.method = aliases.get(args.method, args.method)
        self.backbone = CBraMod(200, 200, 200, 800, 62, 12, 8)
        self.pretrained_report = load_compatible(self.backbone, args.checkpoint)
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(62 * 4 * 200, 4 * 200), nn.ELU(), nn.Dropout(args.dropout),
            nn.Linear(4 * 200, 200), nn.ELU(), nn.Dropout(args.dropout), nn.Linear(200, 2),
        )
        if args.eegonly_head_checkpoint is not None:
            state = torch.load(args.eegonly_head_checkpoint, map_location="cpu", weights_only=True)
            if any(key.startswith("classifier.") for key in state):
                state = {key.removeprefix("classifier."): value for key, value in state.items()
                         if key.startswith("classifier.")}
            else:
                # The paper-protocol EEG-only runner serializes a Sequential:
                # backbone, Flatten, Linear, ELU, Dropout, Linear, ELU,
                # Dropout, Linear.  Map its three linear layers to this head.
                mapping = {"2.": "0.", "5.": "3.", "8.": "6."}
                state = {
                    new_prefix + key[len(old_prefix):]: value
                    for key, value in state.items()
                    for old_prefix, new_prefix in mapping.items()
                    if key.startswith(old_prefix)
                }
            missing, unexpected = self.classifier.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Invalid EEG-only head checkpoint: missing={missing}, unexpected={unexpected}")
        self.prompt = None
        if self.method == "mope":
            # Ordinary three-component MoPE: static + routed dynamic + mapped;
            # injected only at the pre/post boundaries, as in SHIN pre_post.
            prompt_args = dict(
                condition_dim=256, d_model=200, prompt_count=6, rank=8,
                token_count=62 * 4, hidden_dim=256, dropout=args.dropout,
                expert_count=16, temperature=0.1, router_noise_std=0.00390625,
                importance_threshold=0.05,
            )
            self.prompt = nn.ModuleDict({
                "fnirs_encoder": FnirsTemporalEncoder(24 * 2, 256, args.dropout),
                "pre": MoPEBoundaryPrompt(**prompt_args),
            })
            if args.prompt_boundary == "pre_post":
                self.prompt["post"] = MoPEBoundaryPrompt(**prompt_args)
        elif self.method == "deep_conditional":
            self.prompt = DeepConditionalPrompt(
                eeg_dim=200, prompt_dim=128, prompt_tokens=4, stages=3,
                fnirs_temporal_tokens=10, attention_heads=8,
                prompt_scale=0.05, dropout=args.dropout,
            )
            self.deep_stages = {5: 0, 8: 1, 11: 2}
        elif self.method == "deep_three_component_shared":
            self.prompt = SharedDeepThreeComponentPrompt(
                eeg_dim=200, prompt_dim=128, prompt_tokens=6, stages=4,
                fnirs_temporal_tokens=10, attention_heads=8,
                prompt_scale=0.05, dropout=args.dropout,
                expert_count=16, router_temperature=0.1,
                router_noise_std=0.00390625, importance_threshold=0.05,
                prompt_rank=8, prompt_hidden=256,
            )
            self.deep_stages = {2: 0, 5: 1, 8: 2, 11: 3}
        elif self.method == "tmpa_final":
            self.prompt = FoundationTMPAFinalAdapter(
                eeg_dim=200, alignment_dim=128, fnirs_temporal_tokens=10,
                mode_count=4, prompt_tokens_per_mode=2, attention_heads=8,
                token_cost_weight=1.0, prompt_scale=0.05,
                sinkhorn_epsilon=0.1, sinkhorn_iterations=100, dropout=args.dropout,
            )
        elif self.method == "hierarchical_cross_attention":
            self.prompt = FoundationHierarchicalCrossAttentionAdapter(
                eeg_dim=200, alignment_dim=128, fnirs_temporal_tokens=10,
                mode_count=4, prompt_tokens_per_mode=2, attention_heads=8,
                token_cost_weight=1.0, prompt_scale=0.05,
                sinkhorn_epsilon=0.1, sinkhorn_iterations=100, dropout=args.dropout,
            )
        elif self.method == "bidirectional_contrast":
            self.prompt = FoundationHierarchicalBidirectionalContrastAdapter(
                eeg_dim=200, alignment_dim=128, fnirs_temporal_tokens=10,
                mode_count=4, prompt_tokens_per_mode=2, attention_heads=8,
                token_cost_weight=1.0, prompt_scale=0.05,
                sinkhorn_epsilon=0.1, sinkhorn_iterations=100, dropout=args.dropout,
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if args.freeze_classifier:
            for parameter in self.classifier.parameters():
                parameter.requires_grad = False

    def forward(self, eeg, fnirs, pair_matrix=True):
        tokens = self.backbone.patch_embedding(eeg)
        auxiliary = None
        if self.method == "mope":
            condition = self.prompt["fnirs_encoder"](fnirs.permute(0, 3, 1, 2).reshape(fnirs.shape[0], fnirs.shape[-1], -1))
            tokens = tokens + self.prompt["pre"](condition).view_as(tokens)
            tokens = self.backbone.encoder(tokens)
            if "post" in self.prompt:
                tokens = tokens + self.prompt["post"](condition).view_as(tokens)
        elif self.method in {"deep_conditional", "deep_three_component_shared"}:
            context = (
                self.prompt.encode_fnirs(fnirs)
                if self.method == "deep_three_component_shared" else None
            )
            for layer_index, layer in enumerate(self.backbone.encoder.layers):
                if layer_index in self.deep_stages:
                    stage = self.deep_stages[layer_index]
                    tokens = (
                        self.prompt.inject(tokens, context, stage)
                        if context is not None else self.prompt(tokens, fnirs, stage)
                    )
                tokens = layer(tokens)
        elif self.method in {
            "tmpa_final", "hierarchical_cross_attention", "bidirectional_contrast"
        }:
            tokens, auxiliary = self.prompt(tokens, fnirs, compute_pair_matrix=pair_matrix)
            tokens = self.backbone.encoder(tokens)
        else:
            tokens = self.backbone.encoder(tokens)
        return self.classifier(tokens.flatten(1)), auxiliary

    def importance_loss(self):
        if self.method == "deep_three_component_shared":
            return self.prompt.importance_loss()
        if self.method != "mope":
            return next(self.parameters()).new_zeros(())
        losses = [self.prompt["pre"].importance_loss()]
        if "post" in self.prompt:
            losses.append(self.prompt["post"].importance_loss())
        return sum(losses) / len(losses)


def run_epoch(model, loader, optimizer, device, args, training):
    model.train(training)
    model.backbone.eval()
    if args.freeze_classifier:
        model.classifier.eval()
    criterion = nn.CrossEntropyLoss()
    total = seen = 0
    predictions, targets = [], []
    for eeg, fnirs, target in loader:
        eeg, fnirs = eeg.to(device).float(), fnirs.to(device).float()
        target = target.to(device).long()
        with torch.set_grad_enabled(training):
            logits, aux = model(eeg, fnirs, pair_matrix=training)
            loss = criterion(logits, target)
            if training and model.method in {"mope", "deep_three_component_shared"}:
                loss = loss + args.importance_weight * model.importance_loss()
            elif training and model.method in {
                "tmpa_final", "hierarchical_cross_attention", "bidirectional_contrast"
            }:
                pair, same_class = class_aware_contrastive_loss(aux["sample_distance"], target, 0.1)
                loss = loss + args.lambda_pair * pair + args.lambda_class * same_class
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += float(loss.detach()) * len(target); seen += len(target)
        predictions.append(logits.detach().argmax(1).cpu().numpy())
        targets.append(target.cpu().numpy())
    return metrics(np.concatenate(targets), np.concatenate(predictions), total / seen)


def main():
    args = parse_args()
    args.portable_root = args.portable_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.eeg_cache_root = args.eeg_cache_root.resolve()
    args.checkpoint = (args.checkpoint or args.portable_root / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth").resolve()
    if args.freeze_classifier and args.eegonly_head_checkpoint is None:
        raise ValueError("--freeze-classifier requires --eegonly-head-checkpoint")
    if args.eegonly_head_checkpoint is not None:
        args.eegonly_head_checkpoint = args.eegonly_head_checkpoint.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    eeg, fnirs, labels, meta = load_prompt_trials(
        list(range(1, 19)), SPLITS["train"], tuple(args.fnirs_window), args.eeg_cache_root)
    subject_ids = meta["subject_ids"]
    split_indices = {name: np.flatnonzero(np.isin(subject_ids, subjects)) for name, subjects in SPLITS.items()}
    loaders = {
        name: DataLoader(Trials(eeg, fnirs, labels, indices), batch_size=args.batch_size,
                         shuffle=name == "train", num_workers=args.num_workers,
                         pin_memory=torch.cuda.is_available())
        for name, indices in split_indices.items()
    }
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    model = FineMICBraModPrompt(args).to(device)
    groups = []
    if not args.freeze_classifier:
        groups.append({"params": model.classifier.parameters(), "lr": args.head_lr})
    if model.prompt is not None:
        groups.append({"params": model.prompt.parameters(), "lr": args.prompt_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    diagnostic = {
        "method": model.method, "requested_method": args.method,
        "prompt_boundary": args.prompt_boundary, "split": SPLITS, "eeg_shape": list(eeg.shape),
        "fnirs_shape": list(fnirs.shape), "data": {k: v for k, v in meta.items() if k != "subject_ids"},
        "backbone_frozen": True, "classifier_frozen": bool(args.freeze_classifier),
        "pretrained_load": model.pretrained_report, "args": vars(args),
    }
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    history, best_acc, best_epoch, best_state = [], -1.0, 0, None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], optimizer, device, args, True)
        val = run_epoch(model, loaders["val"], None, device, args, False)
        history.append({"epoch": epoch, "train": train, "val": val})
        print(f"epoch {epoch:03d}/{args.epochs} train={train['acc']:.4f} val={val['acc']:.4f}", flush=True)
        if val["acc"] > best_acc:
            best_acc, best_epoch = val["acc"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if not key.startswith("backbone.")}
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current)
    test = run_epoch(model, loaders["test"], None, device, args, False)
    torch.save(best_state, args.output_dir / "best_prompt_and_head.pth")
    summary = {"method": model.method, "seed": args.seed, "best_epoch": best_epoch,
               "best_val_acc": best_acc, "test": test, "elapsed_seconds": time.time() - started}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
