"""Model-agnostic final TMPA runner for CBraMod, CodeBrain, and CSBrain."""

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

from foundation_tmpa_token_alignment import FoundationTMPAFinalAdapter  # noqa: E402
from run_shin2017_cbramod_fnirs_feature_stage1 import (  # noqa: E402
    SHIN_TASKS,
    load_paired_bids_trial_cache,
    metrics,
)
from run_shin2017_foundation_boundary_prompt import (  # noqa: E402
    CodeBrainBoundaryEncoder,
    CSBrainBoundaryEncoder,
    FoundationTrialDataset,
    load_classifier_checkpoint,
    load_compatible,
)
from sgformer_mapped_prompt import (  # noqa: E402
    load_sgformer_graph_trials,
    normalize_graph_from_train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Foundation-model-agnostic token OT on SHIN")
    parser.add_argument("--portable-root", type=Path, default=SCRIPT_ROOT.parent)
    parser.add_argument("--eeg-bids-root", type=Path, default=Path(r"D:\0senior student creation\datasets\shin2017_eeg_bids_bdf"))
    parser.add_argument("--shin-root", type=Path, default=Path(r"D:\0senior student creation\datasets\shin2017_eeg_nirs_left_right_hand_mi"))
    parser.add_argument("--backbone", choices=("cbramod", "codebrain", "csbrain"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--fnirs-cache-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(SHIN_TASKS), default="mi")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 30)))
    parser.add_argument("--train-subjects", nargs="+", type=int, default=list(range(1, 20)))
    parser.add_argument("--val-subjects", nargs="+", type=int, default=list(range(20, 25)))
    parser.add_argument("--test-subjects", nargs="+", type=int, default=list(range(25, 30)))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--eeg-scale", type=float, default=100.0)
    parser.add_argument("--fnirs-window", type=float, default=10.0)
    parser.add_argument("--fnirs-offset", type=float, default=0.0)
    parser.add_argument("--alignment-dim", type=int, default=128)
    parser.add_argument("--fnirs-temporal-tokens", type=int, default=10)
    parser.add_argument("--token-cost-weight", type=float, default=1.0)
    parser.add_argument("--contrast-temperature", type=float, default=0.1)
    parser.add_argument("--lambda-pair", type=float, default=0.1)
    parser.add_argument("--lambda-class", type=float, default=0.02)
    parser.add_argument("--mode-count", type=int, default=4)
    parser.add_argument("--prompt-tokens-per-mode", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iterations", type=int, default=100)
    parser.add_argument("--prompt-scale", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument(
        "--freeze-classifier",
        action="store_true",
        help="Load a matching EEG-only classifier and train only the TMPA adapter.",
    )
    parser.add_argument(
        "--head-checkpoint",
        type=Path,
        default=None,
        help="Matching EEG-only best_model.pth; required with --freeze-classifier.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CodeBrainTokenInterface(CodeBrainBoundaryEncoder):
    def tokenize(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.backbone.patch_embedding(eeg)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, channels, patches, _ = tokens.shape
        x = tokens.permute(0, 3, 1, 2).reshape(batch, 200, channels * patches)
        x = self.backbone.init_conv(x)
        x = self.backbone.residual_layer(x)
        x = self.backbone.final_conv(x)
        x = x.reshape(batch, 200, channels, patches).permute(0, 2, 3, 1)
        return self.backbone.norm(x)


class CSBrainTokenInterface(CSBrainBoundaryEncoder):
    def tokenize(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.backbone.patch_embedding(eeg[:, self.sorted_indices, :, :])

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer_idx in range(self.backbone.encoder.num_layers):
            tokens = self.backbone.TemEmbedEEGLayer(tokens) + tokens
            tokens = self.backbone.BrainEmbedEEGLayer(tokens, self.backbone.area_config) + tokens
            tokens = self.backbone.encoder.layers[layer_idx](tokens, self.backbone.area_config)
        return self.backbone.proj_out(tokens)


class CBraModTokenInterface(nn.Module):
    def __init__(self, portable_root: Path, checkpoint: Path, dropout: float) -> None:
        super().__init__()
        root = portable_root / "CBraMod"
        sys.path.insert(0, str(root))
        from models.cbramod import CBraMod

        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )
        self.pretrained_report = load_compatible(self.backbone, checkpoint)
        self.backbone.proj_out = nn.Identity()

    def tokenize(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.backbone.patch_embedding(eeg)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone.proj_out(self.backbone.encoder(tokens))


class FoundationTMPATokenModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        if args.backbone == "cbramod":
            self.encoder = CBraModTokenInterface(args.portable_root, args.checkpoint, args.dropout)
        elif args.backbone == "codebrain":
            self.encoder = CodeBrainTokenInterface(args.portable_root, args.checkpoint, args.dropout)
        else:
            self.encoder = CSBrainTokenInterface(args.portable_root, args.checkpoint, args.dropout)
        self.adapter = FoundationTMPAFinalAdapter(
            eeg_dim=200,
            alignment_dim=args.alignment_dim,
            fnirs_temporal_tokens=args.fnirs_temporal_tokens,
            mode_count=args.mode_count,
            prompt_tokens_per_mode=args.prompt_tokens_per_mode,
            attention_heads=args.attention_heads,
            token_cost_weight=args.token_cost_weight,
            prompt_scale=args.prompt_scale,
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_iterations=args.sinkhorn_iterations,
            dropout=args.dropout,
        )
        if args.backbone == "codebrain":
            sys.path.insert(0, str(args.portable_root / "CodeBrain" / "scripts"))
            from shin_linear_head import OfficialClassificationHead

            self.classifier = OfficialClassificationHead(30, 10, 200, 2, args.dropout)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(30 * 10 * 200, 10 * 200),
                nn.ELU(), nn.Dropout(args.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(), nn.Dropout(args.dropout),
                nn.Linear(200, 2),
            )

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs_graph: torch.Tensor,
        compute_pair_matrix: bool = True,
    ):
        tokens = self.encoder.tokenize(eeg)
        tokens, ot = self.adapter(
            tokens, fnirs_graph, compute_pair_matrix=compute_pair_matrix
        )
        features = self.encoder.encode_tokens(tokens)
        if hasattr(self.classifier, "flatten"):
            features = self.classifier.flatten(features)
            logits = self.classifier(features)
        else:
            logits = self.classifier(features.flatten(1))
        return logits, ot


def set_trainable(
    model: FoundationTMPATokenModel,
    train_backbone: bool,
    freeze_classifier: bool,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.adapter.parameters():
        parameter.requires_grad = True
    if not freeze_classifier:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    if train_backbone:
        for parameter in model.encoder.backbone.parameters():
            parameter.requires_grad = True


def make_optimizer(model, args):
    groups = [{"params": list(model.adapter.parameters()), "lr": args.feature_lr}]
    if not args.freeze_classifier:
        groups.append({"params": list(model.classifier.parameters()), "lr": args.head_lr})
    if args.train_backbone:
        groups.append({"params": list(model.encoder.backbone.parameters()), "lr": args.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _directional_class_aware_contrast(
    distance: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strong paired contrast plus weak same-class supervised contrast."""
    scores = -distance / temperature
    batch = target.numel()
    identity = torch.eye(batch, dtype=torch.bool, device=target.device)
    same_class = target[:, None].eq(target[None, :])
    different_class = ~same_class

    # Same-class non-paired trials are neutral here, never false negatives.
    pair_candidates = identity | different_class
    pair_denominator = torch.logsumexp(
        scores.masked_fill(~pair_candidates, -torch.inf), dim=1
    )
    pair_loss = -(scores.diagonal() - pair_denominator).mean()

    weak_positive = same_class & ~identity
    candidates = ~identity
    class_denominator = torch.logsumexp(
        scores.masked_fill(~candidates, -torch.inf), dim=1
    )
    per_pair = -(scores - class_denominator[:, None])
    positive_count = weak_positive.sum(dim=1)
    valid = positive_count > 0
    if valid.any():
        class_loss = (
            (per_pair * weak_positive).sum(dim=1)[valid]
            / positive_count[valid]
        ).mean()
    else:
        class_loss = scores.sum() * 0.0
    return pair_loss, class_loss


def class_aware_contrastive_loss(
    distance: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("contrast_temperature must be positive")
    eeg_pair, eeg_class = _directional_class_aware_contrast(
        distance, target, temperature
    )
    fnirs_pair, fnirs_class = _directional_class_aware_contrast(
        distance.transpose(0, 1), target, temperature
    )
    return 0.5 * (eeg_pair + fnirs_pair), 0.5 * (eeg_class + fnirs_class)


def run_epoch(model, loader, optimizer, device, args, training: bool):
    model.train(training)
    if not args.train_backbone:
        model.encoder.backbone.eval()
    if args.freeze_classifier:
        model.classifier.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_prompt_ot, total_token_ot, total_mode_similarity = 0.0, 0.0, 0.0, 0.0
    total_pair, total_class, seen = 0.0, 0.0, 0
    predictions, labels = [], []
    for eeg, _, fnirs_graph, target in loader:
        eeg = eeg.to(device).float()
        fnirs_graph = fnirs_graph.to(device).float()
        target = target.to(device)
        with torch.set_grad_enabled(training):
            logits, ot = model(eeg, fnirs_graph, compute_pair_matrix=training)
            classification = criterion(logits, target)
            if training:
                pair_loss, class_loss = class_aware_contrastive_loss(
                    ot["sample_distance"], target, args.contrast_temperature
                )
                loss = (
                    classification
                    + args.lambda_pair * pair_loss
                    + args.lambda_class * class_loss
                )
            else:
                pair_loss = classification.detach() * 0.0
                class_loss = classification.detach() * 0.0
                loss = classification
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        count = len(target)
        seen += count
        total_loss += float(loss.detach().item()) * count
        total_prompt_ot += float(ot["paired_prompt_ot"].detach().item()) * count
        total_token_ot += float(ot["paired_token_ot"].detach().item()) * count
        total_mode_similarity += float(ot["mode_similarity"].detach().item()) * count
        total_pair += float(pair_loss.detach().item()) * count
        total_class += float(class_loss.detach().item()) * count
        predictions.append(logits.detach().argmax(1).cpu().numpy())
        labels.append(target.detach().cpu().numpy())
    result = metrics(np.concatenate(labels), np.concatenate(predictions), total_loss / seen)
    result["prompt_ot"] = total_prompt_ot / seen
    result["token_ot"] = total_token_ot / seen
    result["mode_similarity"] = total_mode_similarity / seen
    result["pair_contrast"] = total_pair / seen
    result["class_contrast"] = total_class / seen
    return result


def main() -> None:
    args = parse_args()
    args.portable_root = args.portable_root.resolve()
    args.eeg_bids_root = args.eeg_bids_root.resolve()
    args.shin_root = args.shin_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if args.freeze_classifier:
        if args.train_backbone:
            raise ValueError("--freeze-classifier requires a frozen EFM; remove --train-backbone.")
        if args.head_checkpoint is None:
            raise ValueError("--freeze-classifier requires --head-checkpoint from a matching EEG-only run.")
        args.head_checkpoint = args.head_checkpoint.resolve()
    if args.checkpoint is None:
        defaults = {
            "cbramod": args.portable_root / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth",
            "codebrain": args.portable_root / "CodeBrain" / "pretrained_weights" / "CodeBrain.pth",
            "csbrain": args.portable_root / "CSBrain" / "pretrained_weights" / "CSBrain.pth",
        }
        args.checkpoint = defaults[args.backbone]
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"{args.backbone} checkpoint not found: {args.checkpoint}. "
            "Provide the original pretrained weight with --checkpoint."
        )
    if args.cache_path is None:
        args.cache_path = SCRIPT_ROOT / "cache" / f"shin2017_{args.task}_paired_10patch.npz"
    if args.fnirs_cache_path is None:
        args.fnirs_cache_path = SCRIPT_ROOT / "cache" / f"shin2017_{args.task}_hbo_hbr_10s.npz"
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    loader_args = argparse.Namespace(
        cache_path=args.cache_path, rebuild_cache=args.rebuild_cache,
        task=args.task, eeg_scale=args.eeg_scale,
        eeg_bids_root=args.eeg_bids_root, shin_root=args.shin_root,
        subjects=args.subjects, fnirs_offset=args.fnirs_offset,
        fnirs_window=args.fnirs_window,
    )
    eeg, _, fnirs_sequence, labels, meta = load_paired_bids_trial_cache(loader_args)
    fnirs_graph, graph_labels, graph_subjects, graph_meta = load_sgformer_graph_trials(
        args.shin_root, args.subjects, SHIN_TASKS[args.task]["sessions"],
        args.fnirs_window, args.fnirs_offset,
        cache_path=args.fnirs_cache_path, rebuild_cache=args.rebuild_cache,
    )
    subject_ids = np.asarray(meta["subject_ids"])
    if not np.array_equal(labels, graph_labels) or not np.array_equal(subject_ids, graph_subjects):
        raise RuntimeError("EEG and fNIRS trial ordering differs")
    split_indices = {
        "train": np.flatnonzero(np.isin(subject_ids, args.train_subjects)),
        "val": np.flatnonzero(np.isin(subject_ids, args.val_subjects)),
        "test": np.flatnonzero(np.isin(subject_ids, args.test_subjects)),
    }
    fnirs_graph = normalize_graph_from_train(fnirs_graph, split_indices["train"])
    datasets = {
        name: FoundationTrialDataset(eeg, fnirs_sequence, fnirs_graph, labels, indices)
        for name, indices in split_indices.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }

    model = FoundationTMPATokenModel(args)
    head_report = None
    if args.freeze_classifier:
        head_report = load_classifier_checkpoint(
            model, args.head_checkpoint, args.task, args.seed
        )
    set_trainable(model, args.train_backbone, args.freeze_classifier)
    model.to(device)
    optimizer = make_optimizer(model, args)
    method_name = getattr(model.adapter, "method_name", "foundation_tmpa_hierarchical_contrastive")
    alignment_description = getattr(
        model.adapter,
        "alignment_description",
        "multi-mode prompts + token-level OT + prompt-level OT",
    )
    diagnostics = {
        "method": method_name,
        "backbone": args.backbone,
        "token_partition": "none; flatten native EFM tokens",
        "alignment": alignment_description,
        "alignment_metric_note": getattr(model.adapter, "metric_note", None),
        "contrast": "paired strong positive + same-class weak positive + different-class negative",
        "eeg_shape": list(eeg.shape),
        "fnirs_shape": list(fnirs_graph.shape),
        "fnirs_preprocessing": graph_meta.get("preprocessing"),
        "backbone_frozen": not args.train_backbone,
        "classifier_frozen": args.freeze_classifier,
        "classifier_load": head_report,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "pretrained_load": model.encoder.pretrained_report,
        "args": vars(args),
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if args.diagnose_only:
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), flush=True)
        return

    history, best, best_state, best_accuracy = [], None, None, -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], optimizer, device, args, True)
        val = run_epoch(model, loaders["val"], None, device, args, False)
        record = {"epoch": epoch, "train": train, "val": val, "elapsed_seconds": time.time() - started}
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs} train_loss={train['loss']:.4f} "
            f"val_acc={val['acc']:.4f} val_kappa={val['kappa']:.4f} "
            f"prompt_ot={train['prompt_ot']:.4f} pair={train['pair_contrast']:.4f} "
            f"class={train['class_contrast']:.4f}",
            flush=True,
        )
        if val["acc"] > best_accuracy:
            best_accuracy = val["acc"]
            best = copy.deepcopy(record)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    model.load_state_dict(best_state)
    test = run_epoch(model, loaders["test"], None, device, args, False)
    summary = {
        "method": method_name,
        "backbone": args.backbone,
        "best_epoch": best["epoch"],
        "best_val": best["val"],
        "test_at_best_epoch": test,
        "args": vars(args),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
