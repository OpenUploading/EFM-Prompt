"""Spatiotemporal-partition TMPA for CBraMod/CodeBrain and SHIN."""

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
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from run_shin2017_cbramod_fnirs_feature_stage1 import (  # noqa: E402
    SHIN_TASKS,
    load_paired_bids_trial_cache,
)
from sgformer_mapped_prompt import (  # noqa: E402
    load_sgformer_graph_trials,
    normalize_graph_from_train,
)
from tmpa_lite_cbramod import TMPALiteAdapter  # noqa: E402
from run_shin2017_foundation_boundary_prompt import (  # noqa: E402
    CodeBrainBoundaryEncoder,
    CSBrainBoundaryEncoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatiotemporal TMPA on paired SHIN EEG-fNIRS")
    parser.add_argument("--portable-root", type=Path, default=SCRIPT_ROOT.parent)
    parser.add_argument(
        "--backbone", choices=("cbramod", "codebrain", "csbrain"), default="cbramod"
    )
    parser.add_argument("--eeg-bids-root", type=Path, default=Path(r"D:\0senior student creation\datasets\shin2017_eeg_bids_bdf"))
    parser.add_argument("--shin-root", type=Path, default=Path(r"D:\0senior student creation\datasets\shin2017_eeg_nirs_left_right_hand_mi"))
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--sgformer-cache-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(SHIN_TASKS), default="mi")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 30)))
    parser.add_argument("--train-subjects", nargs="+", type=int, default=list(range(1, 20)))
    parser.add_argument("--val-subjects", nargs="+", type=int, default=list(range(20, 25)))
    parser.add_argument("--test-subjects", nargs="+", type=int, default=list(range(25, 30)))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--eeg-scale", type=float, default=1.0)
    parser.add_argument("--fnirs-window", type=float, default=10.0)
    parser.add_argument("--fnirs-offset", type=float, default=0.0)
    parser.add_argument("--alignment-dim", type=int, default=128)
    parser.add_argument("--lambda-spatial", type=float, default=0.05)
    parser.add_argument("--lambda-temporal", type=float, default=0.05)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--prompt-scale", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--selection-metric", choices=("accuracy", "cohen_kappa"), default="accuracy")
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


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, loss: float) -> dict:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


class CBraModTMPALite(nn.Module):
    def __init__(self, cbramod_cls, args: argparse.Namespace) -> None:
        super().__init__()
        self.backbone = cbramod_cls(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )
        self.adapter = TMPALiteAdapter(
            d_model=200,
            alignment_dim=args.alignment_dim,
            prompt_scale=args.prompt_scale,
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_iterations=args.sinkhorn_iterations,
            dropout=args.dropout,
        )
        # Keep the current CBraMod SHIN three-layer all-patch head unchanged.
        self.classifier = nn.Sequential(
            nn.Linear(30 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(args.dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(args.dropout),
            nn.Linear(200, 2),
        )

    def features(self, eeg: torch.Tensor, fnirs: torch.Tensor):
        patch_tokens = self.backbone.patch_embedding(eeg)
        patch_tokens, losses = self.adapter(patch_tokens, fnirs)
        features = self.backbone.encoder(patch_tokens)
        features = self.backbone.proj_out(features)
        return features.flatten(start_dim=1), losses

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor):
        features, losses = self.features(eeg, fnirs)
        return self.classifier(features), losses


class CodeBrainTMPALite(nn.Module):
    """Apply the unchanged spatiotemporal adapter to CodeBrain's native grid."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.encoder = CodeBrainBoundaryEncoder(
            args.portable_root, args.checkpoint, args.dropout
        )
        self.backbone = self.encoder.backbone
        self.pretrained_report = self.encoder.pretrained_report
        self.adapter = TMPALiteAdapter(
            d_model=200,
            alignment_dim=args.alignment_dim,
            prompt_scale=args.prompt_scale,
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_iterations=args.sinkhorn_iterations,
            dropout=args.dropout,
        )
        sys.path.insert(0, str(args.portable_root / "CodeBrain" / "scripts"))
        from shin_linear_head import OfficialClassificationHead

        self.classifier = OfficialClassificationHead(30, 10, 200, 2, args.dropout)

    def features(self, eeg: torch.Tensor, fnirs: torch.Tensor):
        batch, channels, patches, _ = eeg.shape
        patch_tokens = self.backbone.patch_embedding(eeg)
        patch_tokens, losses = self.adapter(patch_tokens, fnirs)
        x = patch_tokens.permute(0, 3, 1, 2).reshape(batch, 200, channels * patches)
        x = self.backbone.init_conv(x)
        x = self.backbone.residual_layer(x)
        x = self.backbone.final_conv(x)
        features = x.reshape(batch, 200, channels, patches).permute(0, 2, 3, 1)
        return self.backbone.norm(features), losses

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor):
        features, losses = self.features(eeg, fnirs)
        return self.classifier(features), losses


class CSBrainTMPALite(nn.Module):
    """Apply the unchanged spatiotemporal adapter to CSBrain's native grid."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.encoder = CSBrainBoundaryEncoder(
            args.portable_root, args.checkpoint, args.dropout
        )
        self.backbone = self.encoder.backbone
        self.sorted_indices = self.encoder.sorted_indices
        self.pretrained_report = self.encoder.pretrained_report
        self.adapter = TMPALiteAdapter(
            d_model=200,
            alignment_dim=args.alignment_dim,
            prompt_scale=args.prompt_scale,
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_iterations=args.sinkhorn_iterations,
            dropout=args.dropout,
        )
        # CSBrain's official SHIN adaptation keeps the all-patch three-layer head.
        self.classifier = nn.Sequential(
            nn.Linear(30 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(args.dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(args.dropout),
            nn.Linear(200, 2),
        )

    def features(self, eeg: torch.Tensor, fnirs: torch.Tensor):
        tokens = self.backbone.patch_embedding(eeg[:, self.sorted_indices, :, :])
        tokens, losses = self.adapter(tokens, fnirs)
        for layer_idx in range(self.backbone.encoder.num_layers):
            tokens = self.backbone.TemEmbedEEGLayer(tokens) + tokens
            tokens = self.backbone.BrainEmbedEEGLayer(
                tokens, self.backbone.area_config
            ) + tokens
            tokens = self.backbone.encoder.layers[layer_idx](
                tokens, self.backbone.area_config
            )
        return self.backbone.proj_out(tokens).flatten(1), losses

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor):
        features, losses = self.features(eeg, fnirs)
        return self.classifier(features), losses


def load_pretrained(model: CBraModTMPALite, checkpoint: Path) -> dict:
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(source, dict) and "model" in source:
        source = source["model"]
    result = model.backbone.load_state_dict(source, strict=True)
    model.backbone.proj_out = nn.Identity()
    return {
        "checkpoint": str(checkpoint.resolve()),
        "loaded_keys": len(source),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def set_trainable(model: nn.Module, train_backbone: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.adapter, model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True
    if train_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True


def make_optimizer(model: nn.Module, args: argparse.Namespace):
    groups = [
        {"params": list(model.adapter.parameters()), "lr": args.feature_lr, "name": "fnirs_adapter"},
        {"params": list(model.classifier.parameters()), "lr": args.head_lr, "name": "head"},
    ]
    if args.train_backbone:
        groups.append({"params": list(model.backbone.parameters()), "lr": args.backbone_lr, "name": "backbone"})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def make_loader(eeg, fnirs, labels, indices, args, shuffle: bool):
    dataset = TensorDataset(
        torch.from_numpy(eeg[indices]),
        torch.from_numpy(fnirs[indices]),
        torch.from_numpy(labels[indices]),
    )
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, optimizer, device, args, train: bool):
    model.train(train)
    if not args.train_backbone:
        model.backbone.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_spatial, total_temporal, seen = 0.0, 0.0, 0.0, 0
    labels, predictions = [], []
    for eeg, fnirs, target in loader:
        eeg = eeg.to(device, non_blocking=True).float()
        fnirs = fnirs.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits, ot = model(eeg, fnirs)
            cls_loss = criterion(logits, target)
            loss = cls_loss + args.lambda_spatial * ot["spatial_ot"] + args.lambda_temporal * ot["temporal_ot"]
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        count = len(target)
        total_loss += float(loss.detach().item()) * count
        total_spatial += float(ot["spatial_ot"].detach().item()) * count
        total_temporal += float(ot["temporal_ot"].detach().item()) * count
        seen += count
        labels.append(target.detach().cpu().numpy())
        predictions.append(logits.detach().argmax(1).cpu().numpy())
    result = classification_metrics(np.concatenate(labels), np.concatenate(predictions), total_loss / seen)
    result.update({
        "spatial_ot": total_spatial / seen,
        "temporal_ot": total_temporal / seen,
    })
    return result


def main() -> None:
    args = parse_args()
    args.portable_root = args.portable_root.resolve()
    args.eeg_bids_root = args.eeg_bids_root.resolve()
    args.shin_root = args.shin_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.cache_path is None:
        args.cache_path = args.output_dir / "paired_eeg_fnirs_cache.npz"
    else:
        args.cache_path = args.cache_path.resolve()
    if args.sgformer_cache_path is None:
        args.sgformer_cache_path = args.output_dir / "hbo_hbr_graph_cache.npz"
    else:
        args.sgformer_cache_path = args.sgformer_cache_path.resolve()
    if args.checkpoint is None:
        defaults = {
            "cbramod": args.portable_root / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth",
            "codebrain": args.portable_root / "CodeBrain" / "pretrained_weights" / "CodeBrain.pth",
            "csbrain": args.portable_root / "CSBrain" / "pretrained_weights" / "CSBrain.pth",
        }
        args.checkpoint = defaults[args.backbone]
    args.checkpoint = args.checkpoint.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)

    paired_args = argparse.Namespace(
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
        task=args.task,
        eeg_bids_root=args.eeg_bids_root,
        shin_root=args.shin_root,
        subjects=args.subjects,
        fnirs_offset=args.fnirs_offset,
        fnirs_window=args.fnirs_window,
        eeg_scale=args.eeg_scale,
    )
    eeg, _, _, labels, paired_meta = load_paired_bids_trial_cache(paired_args)
    graph, graph_labels, graph_subjects, graph_meta = load_sgformer_graph_trials(
        args.shin_root,
        args.subjects,
        SHIN_TASKS[args.task]["sessions"],
        args.fnirs_window,
        args.fnirs_offset,
        cache_path=args.sgformer_cache_path,
        rebuild_cache=args.rebuild_cache,
    )
    if not np.array_equal(labels, graph_labels) or not np.array_equal(
        np.asarray(paired_meta["subject_ids"]), graph_subjects
    ):
        raise RuntimeError("Paired EEG/fNIRS ordering does not match")
    if len(eeg) != len(graph):
        raise RuntimeError(f"EEG/fNIRS trial count mismatch: {len(eeg)} vs {len(graph)}")

    subject_ids = np.asarray(paired_meta["subject_ids"])
    train_idx = np.flatnonzero(np.isin(subject_ids, args.train_subjects))
    val_idx = np.flatnonzero(np.isin(subject_ids, args.val_subjects))
    test_idx = np.flatnonzero(np.isin(subject_ids, args.test_subjects))
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise RuntimeError("One of train/val/test splits is empty")
    graph = normalize_graph_from_train(graph, train_idx)

    if args.backbone == "cbramod":
        cbramod_root = args.portable_root / "CBraMod"
        sys.path.insert(0, str(cbramod_root))
        from models.cbramod import CBraMod

        model = CBraModTMPALite(CBraMod, args)
        load_report = load_pretrained(model, args.checkpoint)
    elif args.backbone == "codebrain":
        model = CodeBrainTMPALite(args)
        load_report = model.pretrained_report
    else:
        model = CSBrainTMPALite(args)
        load_report = model.pretrained_report
    set_trainable(model, args.train_backbone)
    model.to(device)
    optimizer = make_optimizer(model, args)
    loaders = {
        "train": make_loader(eeg, graph, labels, train_idx, args, True),
        "val": make_loader(eeg, graph, labels, val_idx, args, False),
        "test": make_loader(eeg, graph, labels, test_idx, args, False),
    }
    diagnostics = {
        "method": "TMPA-spatiotemporal-partition",
        "backbone": args.backbone,
        "backbone_frozen": not args.train_backbone,
        "data": {"eeg_shape": list(eeg.shape), "fnirs_shape": list(graph.shape)},
        "preprocessing": graph_meta.get("preprocessing"),
        "splits": {"train": args.train_subjects, "val": args.val_subjects, "test": args.test_subjects},
        "pretrained_load": load_report,
        "args": vars(args),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    write_json(args.output_dir / "diagnostics.json", diagnostics)
    if args.diagnose_only:
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), flush=True)
        return

    history, best_record, best_state, best_value = [], None, None, -float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], optimizer, device, args, True)
        val = run_epoch(model, loaders["val"], None, device, args, False)
        value = val[args.selection_metric]
        record = {"epoch": epoch, "train": train, "val": val, "elapsed_seconds": time.time() - started}
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs} train_loss={train['loss']:.4f} "
            f"val_acc={val['accuracy']:.4f} val_kappa={val['cohen_kappa']:.4f} "
            f"spatial_ot={train['spatial_ot']:.4f} temporal_ot={train['temporal_ot']:.4f}",
            flush=True,
        )
        if value > best_value:
            best_value = value
            best_record = copy.deepcopy(record)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        write_json(args.output_dir / "history.json", history)

    if best_state is None or best_record is None:
        raise RuntimeError("No best validation record was produced")
    model.load_state_dict(best_state)
    best_test = run_epoch(model, loaders["test"], None, device, args, False)
    summary = {
        "method": "TMPA-spatiotemporal-partition",
        "backbone": args.backbone,
        "best_epoch": best_record["epoch"],
        "selection_metric": args.selection_metric,
        "best_val": best_record["val"],
        "test_at_best_epoch": best_test,
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "EXPERIMENT_RECORD.md").write_text(
        f"# {args.backbone} TMPA spatiotemporal partition\n\n"
        + json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
