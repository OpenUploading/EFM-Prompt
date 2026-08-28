"""Strict prompt-only deep fNIRS prompting for CBraMod, CodeBrain and CSBrain.

The pretrained EFM and a matching EEG-only classifier are both frozen.  Only
the fNIRS conditioner and layer-wise residual prompt parameters are optimized.
"""

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
from torch.utils.data import DataLoader

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from foundation_deep_prompt import DeepConditionalPrompt, SharedDeepThreeComponentPrompt
from run_shin2017_cbramod_fnirs_feature_stage1 import SHIN_TASKS, load_paired_bids_trial_cache, metrics
from run_shin2017_foundation_boundary_prompt import (
    CodeBrainBoundaryEncoder,
    CSBrainBoundaryEncoder,
    FoundationTrialDataset,
    load_compatible,
)
from sgformer_mapped_prompt import load_sgformer_graph_trials, normalize_graph_from_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict frozen-EFM deep conditional prompt on SHIN")
    parser.add_argument("--portable-root", type=Path, default=SCRIPT_ROOT.parent)
    parser.add_argument("--eeg-bids-root", type=Path, default=Path(r"D:\0senior student creation\datasets\shin2017_eeg_bids_bdf"))
    parser.add_argument("--shin-root", type=Path, default=Path(r"D:\0senior student creation\datasets\shin2017_eeg_nirs_left_right_hand_mi"))
    parser.add_argument("--backbone", choices=("cbramod", "codebrain", "csbrain"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--fnirs-cache-path", type=Path, default=None)
    parser.add_argument("--task", choices=tuple(SHIN_TASKS), default="mi")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 30)))
    parser.add_argument("--train-subjects", nargs="+", type=int, default=list(range(1, 20)))
    parser.add_argument("--val-subjects", nargs="+", type=int, default=list(range(20, 25)))
    parser.add_argument("--test-subjects", nargs="+", type=int, default=list(range(25, 30)))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--eeg-scale", type=float, default=100.0)
    parser.add_argument("--fnirs-window", type=float, default=10.0)
    parser.add_argument("--fnirs-offset", type=float, default=0.0)
    parser.add_argument(
        "--deep-prompt-mode",
        choices=("conditional", "three_component_shared"),
        default="conditional",
    )
    parser.add_argument("--prompt-dim", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=4)
    parser.add_argument("--fnirs-temporal-tokens", type=int, default=10)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--prompt-scale", type=float, default=0.05)
    parser.add_argument("--expert-count", type=int, default=16)
    parser.add_argument("--router-temperature", type=float, default=0.1)
    parser.add_argument("--router-noise-std", type=float, default=0.00390625)
    parser.add_argument("--importance-threshold", type=float, default=0.05)
    parser.add_argument("--importance-weight", type=float, default=0.01)
    parser.add_argument("--prompt-rank", type=int, default=8)
    parser.add_argument("--prompt-hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_head(classifier: nn.Module, checkpoint: Path, task: str, seed: int) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EEG-only head checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    head_state = {key.removeprefix("classifier."): value for key, value in state.items() if key.startswith("classifier.")}
    if not head_state:
        raise ValueError(f"{checkpoint} does not contain classifier.* weights")
    saved = payload.get("args", {}) if isinstance(payload, dict) else {}
    if saved.get("task") is not None and saved["task"] != task:
        raise ValueError(f"Head checkpoint task is {saved['task']}, requested {task}")
    if saved.get("seed") is not None and int(saved["seed"]) != int(seed):
        raise ValueError(f"Head checkpoint seed is {saved['seed']}, requested {seed}")
    result = classifier.load_state_dict(head_state, strict=True)
    return {"checkpoint": str(checkpoint.resolve()), "epoch": payload.get("epoch"), "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


def _prompt_context(prompt: nn.Module, fnirs: torch.Tensor):
    return prompt.encode_fnirs(fnirs) if hasattr(prompt, "encode_fnirs") else fnirs


def _inject_prompt(prompt: nn.Module, tokens: torch.Tensor, context, stage: int) -> torch.Tensor:
    return prompt.inject(tokens, context, stage) if hasattr(prompt, "inject") else prompt(tokens, context, stage)


class CBraModDeepEncoder(nn.Module):
    stages = {5: 0, 8: 1, 11: 2}
    def __init__(self, root: Path, checkpoint: Path, dropout: float):
        super().__init__()
        sys.path.insert(0, str(root / "CBraMod"))
        from models.cbramod import CBraMod
        self.backbone = CBraMod(200, 200, 200, 800, 30, 12, 8)
        self.pretrained_report = load_compatible(self.backbone, checkpoint)
        self.backbone.proj_out = nn.Identity()
    def forward(self, eeg, fnirs, prompt):
        tokens = self.backbone.patch_embedding(eeg)
        context = _prompt_context(prompt, fnirs)
        for index, layer in enumerate(self.backbone.encoder.layers):
            if index in self.stages: tokens = _inject_prompt(prompt, tokens, context, self.stages[index])
            tokens = layer(tokens)
        return tokens


class CSBrainDeepEncoder(CSBrainBoundaryEncoder):
    stages = {5: 0, 8: 1, 11: 2}
    def forward(self, eeg, fnirs, prompt):
        tokens = self.backbone.patch_embedding(eeg[:, self.sorted_indices, :, :])
        context = _prompt_context(prompt, fnirs)
        for index in range(self.backbone.encoder.num_layers):
            tokens = self.backbone.TemEmbedEEGLayer(tokens) + tokens
            tokens = self.backbone.BrainEmbedEEGLayer(tokens, self.backbone.area_config) + tokens
            if index in self.stages: tokens = _inject_prompt(prompt, tokens, context, self.stages[index])
            tokens = self.backbone.encoder.layers[index](tokens, self.backbone.area_config)
        return self.backbone.proj_out(tokens)


class CodeBrainDeepEncoder(CodeBrainBoundaryEncoder):
    stages = {3: 0, 5: 1, 7: 2}
    def forward(self, eeg, fnirs, prompt):
        batch, channels, patches, _ = eeg.shape
        tokens = self.backbone.patch_embedding(eeg)
        hidden = self.backbone.init_conv(tokens.permute(0, 3, 1, 2).reshape(batch, 200, channels * patches))
        context = _prompt_context(prompt, fnirs)
        original = hidden
        skip = 0
        for index, block in enumerate(self.backbone.residual_layer.residual_blocks):
            if index in self.stages:
                grid = hidden.reshape(batch, 200, channels, patches).permute(0, 2, 3, 1)
                hidden = _inject_prompt(prompt, grid, context, self.stages[index]).permute(0, 3, 1, 2).reshape(batch, 200, channels * patches)
            hidden, skip_one = block((hidden, original))
            skip = skip + skip_one
        skip = skip * (1.0 / len(self.backbone.residual_layer.residual_blocks)) ** 0.5
        output = self.backbone.final_conv(skip)
        return self.backbone.norm(output.reshape(batch, 200, channels, patches).permute(0, 2, 3, 1))


class FrozenDeepPromptModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        defaults = {"cbramod": args.portable_root / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth", "codebrain": args.portable_root / "CodeBrain" / "pretrained_weights" / "CodeBrain.pth", "csbrain": args.portable_root / "CSBrain" / "pretrained_weights" / "CSBrain.pth"}
        checkpoint = args.checkpoint or defaults[args.backbone]
        if args.backbone == "cbramod": self.encoder = CBraModDeepEncoder(args.portable_root, checkpoint, args.dropout)
        elif args.backbone == "codebrain": self.encoder = CodeBrainDeepEncoder(args.portable_root, checkpoint, args.dropout)
        else: self.encoder = CSBrainDeepEncoder(args.portable_root, checkpoint, args.dropout)
        if args.deep_prompt_mode == "three_component_shared":
            stage_maps = {
                "cbramod": {2: 0, 5: 1, 8: 2, 11: 3},
                "csbrain": {2: 0, 5: 1, 8: 2, 11: 3},
                "codebrain": {1: 0, 3: 1, 5: 2, 7: 3},
            }
            self.encoder.stages = stage_maps[args.backbone]
            self.prompt = SharedDeepThreeComponentPrompt(
                eeg_dim=200,
                prompt_dim=args.prompt_dim,
                prompt_tokens=args.prompt_tokens,
                stages=4,
                fnirs_temporal_tokens=args.fnirs_temporal_tokens,
                attention_heads=args.attention_heads,
                prompt_scale=args.prompt_scale,
                dropout=args.dropout,
                expert_count=args.expert_count,
                router_temperature=args.router_temperature,
                router_noise_std=args.router_noise_std,
                importance_threshold=args.importance_threshold,
                prompt_rank=args.prompt_rank,
                prompt_hidden=args.prompt_hidden,
            )
        else:
            self.prompt = DeepConditionalPrompt(
                200, args.prompt_dim, args.prompt_tokens, 3,
                args.fnirs_temporal_tokens, args.attention_heads,
                args.prompt_scale, args.dropout,
            )
        if args.backbone == "codebrain":
            sys.path.insert(0, str(args.portable_root / "CodeBrain" / "scripts"))
            from shin_linear_head import OfficialClassificationHead
            self.classifier = OfficialClassificationHead(30, 10, 200, 2, args.dropout)
        else:
            self.classifier = nn.Sequential(nn.Linear(60000, 2000), nn.ELU(), nn.Dropout(args.dropout), nn.Linear(2000, 200), nn.ELU(), nn.Dropout(args.dropout), nn.Linear(200, 2))
        self.head_report = _load_head(self.classifier, args.head_checkpoint, args.task, args.seed)
        for parameter in self.encoder.parameters(): parameter.requires_grad = False
        for parameter in self.classifier.parameters(): parameter.requires_grad = False
    def forward(self, eeg, fnirs):
        features = self.encoder(eeg, fnirs, self.prompt)
        if hasattr(self.classifier, "flatten"):
            return self.classifier(self.classifier.flatten(features))
        return self.classifier(features.flatten(1))


def run_epoch(model, loader, optimizer, device, training: bool, args: argparse.Namespace):
    model.train(training)
    model.encoder.eval(); model.classifier.eval()
    criterion = nn.CrossEntropyLoss(); total = total_classification = total_importance = seen = 0; predictions = []; labels = []
    for eeg, _, fnirs, target in loader:
        eeg, fnirs, target = eeg.to(device).float(), fnirs.to(device).float(), target.to(device)
        with torch.set_grad_enabled(training):
            logits = model(eeg, fnirs)
            classification_loss = criterion(logits, target)
            if training and args.deep_prompt_mode == "three_component_shared":
                importance_loss = model.prompt.importance_loss()
                loss = classification_loss + args.importance_weight * importance_loss
            else:
                importance_loss = classification_loss.detach() * 0.0
                loss = classification_loss
            if training: optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += float(loss.detach()) * len(target)
        total_classification += float(classification_loss.detach()) * len(target)
        total_importance += float(importance_loss.detach()) * len(target)
        seen += len(target)
        predictions.append(logits.detach().argmax(1).cpu().numpy()); labels.append(target.cpu().numpy())
    result = metrics(np.concatenate(labels), np.concatenate(predictions), total / seen)
    result["classification_loss"] = total_classification / seen
    result["importance_loss"] = total_importance / seen
    return result


def prompt_parameter_counts(prompt: nn.Module, mode: str) -> dict[str, int]:
    if mode != "three_component_shared":
        return {
            "shared_parameters": sum(parameter.numel() for parameter in prompt.parameters()),
            "stage_specific_parameters": 0,
            "total_trainable_prompt_parameters": sum(parameter.numel() for parameter in prompt.parameters()),
        }
    shared_modules = (
        prompt.fnirs_encoder, prompt.router, prompt.mapper, prompt.eeg_in,
        prompt.tokens_from_prompt, prompt.token_norm,
    )
    shared = sum(parameter.numel() for module in shared_modules for parameter in module.parameters())
    shared += prompt.prompt_experts.numel()
    stage_specific = (
        prompt.static_prompts.numel()
        + sum(parameter.numel() for module in prompt.stage_down for parameter in module.parameters())
        + sum(parameter.numel() for module in prompt.stage_up for parameter in module.parameters())
        + prompt.gates.numel()
    )
    return {
        "shared_parameters": shared,
        "stage_specific_parameters": stage_specific,
        "total_trainable_prompt_parameters": shared + stage_specific,
    }


def main() -> None:
    args = parse_args(); args.portable_root = args.portable_root.resolve(); args.eeg_bids_root = args.eeg_bids_root.resolve(); args.shin_root = args.shin_root.resolve(); args.head_checkpoint = args.head_checkpoint.resolve(); args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists(): raise FileExistsError(f"Output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True); seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    if args.cache_path is None: args.cache_path = SCRIPT_ROOT / "cache" / f"shin2017_{args.task}_paired_10patch.npz"
    if args.fnirs_cache_path is None: args.fnirs_cache_path = SCRIPT_ROOT / "cache" / f"shin2017_{args.task}_hbo_hbr_10s.npz"
    data_args = argparse.Namespace(cache_path=args.cache_path, rebuild_cache=args.rebuild_cache, task=args.task, eeg_scale=args.eeg_scale, eeg_bids_root=args.eeg_bids_root, shin_root=args.shin_root, subjects=args.subjects, fnirs_offset=args.fnirs_offset, fnirs_window=args.fnirs_window)
    eeg, _, _, labels, meta = load_paired_bids_trial_cache(data_args)
    fnirs, fnirs_labels, fnirs_subjects, fnirs_meta = load_sgformer_graph_trials(args.shin_root, args.subjects, SHIN_TASKS[args.task]["sessions"], args.fnirs_window, args.fnirs_offset, cache_path=args.fnirs_cache_path, rebuild_cache=args.rebuild_cache)
    subject_ids = np.asarray(meta["subject_ids"])
    if not np.array_equal(labels, fnirs_labels) or not np.array_equal(subject_ids, fnirs_subjects): raise RuntimeError("EEG/fNIRS trial ordering differs")
    splits = {name: np.flatnonzero(np.isin(subject_ids, values)) for name, values in {"train": args.train_subjects, "val": args.val_subjects, "test": args.test_subjects}.items()}
    fnirs = normalize_graph_from_train(fnirs, splits["train"])
    loaders = {name: DataLoader(FoundationTrialDataset(eeg, fnirs, fnirs, labels, indices), batch_size=args.batch_size, shuffle=name == "train", num_workers=args.num_workers, pin_memory=torch.cuda.is_available()) for name, indices in splits.items()}
    device = torch.device(args.device); model = FrozenDeepPromptModel(args).to(device)
    optimizer = torch.optim.AdamW(model.prompt.parameters(), lr=args.prompt_lr, weight_decay=args.weight_decay)
    parameter_counts = prompt_parameter_counts(model.prompt, args.deep_prompt_mode)
    backbone_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    parameter_counts["backbone_parameters"] = backbone_parameters
    parameter_counts["trainable_backbone_ratio"] = parameter_counts["total_trainable_prompt_parameters"] / max(backbone_parameters, 1)
    if args.deep_prompt_mode == "three_component_shared":
        prompt_locations = {
            "cbramod_csbrain": "layers 3, 6, 9, 12 before native encoder block",
            "codebrain": "residual SSSM blocks 2, 4, 6, 8 before native block",
        }
    else:
        prompt_locations = {
            "cbramod_csbrain": "layers 6, 9, 12 before native encoder block",
            "codebrain": "residual SSSM blocks 4, 6, 8 before native block",
        }
    diagnostic = {"method": model.prompt.method_name, "deep_prompt_mode": args.deep_prompt_mode, "backbone": args.backbone, "task": args.task, "prompt_locations": prompt_locations, "stage_map": model.encoder.stages, "backbone_frozen": True, "classifier_frozen": True, "head_load": model.head_report, "parameter_counts": parameter_counts, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "all_parameters": sum(p.numel() for p in model.parameters()), "fnirs_preprocessing": fnirs_meta.get("preprocessing"), "args": vars(args)}
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.diagnose_only: print(json.dumps(diagnostic, ensure_ascii=False, indent=2, default=str)); return
    best = None; best_state = None; best_acc = -1.0; history = []; started = time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], optimizer, device, True, args); val = run_epoch(model, loaders["val"], None, device, False, args)
        record = {"epoch": epoch, "train": train, "val": val, "elapsed_seconds": time.time() - started}; history.append(record)
        if args.deep_prompt_mode == "three_component_shared":
            record["routing"] = model.prompt.routing_statistics()
        message = f"epoch {epoch:03d}/{args.epochs} train_loss={train['loss']:.4f} val_acc={val['acc']:.4f} val_kappa={val['kappa']:.4f}"
        if args.deep_prompt_mode == "three_component_shared":
            message += f" imp={train['importance_loss']:.4f}"
        print(message, flush=True)
        if val["acc"] > best_acc: best_acc = val["acc"]; best = copy.deepcopy(record); best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); test = run_epoch(model, loaders["test"], None, device, False, args)
    summary = {"method": model.prompt.method_name, "deep_prompt_mode": args.deep_prompt_mode, "backbone": args.backbone, "stage_map": model.encoder.stages, "best_epoch": best["epoch"], "best_val": best["val"], "test_at_best_epoch": test, "parameter_counts": parameter_counts, "args": vars(args)}
    (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__": main()
