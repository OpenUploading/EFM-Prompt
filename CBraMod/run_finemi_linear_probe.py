"""CBraMod training on FineMI EEG (subjects 12/3/3)."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.cbramod import CBraMod
from run_shin_finetune import (
    cache_features,
    evaluate,
    load_pretrained,
    loader,
    seed_everything,
    train_epoch,
    write_json,
)


class FineMICBraMod(nn.Module):
    """CBraMod backbone plus its official all_patch_reps three-layer head."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=62, n_layer=12, nhead=8,
        )
        # Official all_patch_reps pattern: C*P*D -> P*D -> D -> classes.
        self.classifier = nn.Sequential(
            nn.Linear(62 * 4 * 200, 4 * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, 2),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CBraMod + FineMI EEG binary task")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_rawuv_binary_1v6"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained_weights/pretrained_weights.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--full-finetune", action="store_true",
                        help="Update backbone from epoch 1; otherwise train only the classifier.")
    return parser.parse_args()


def load_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        eeg = data["eeg"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.int64, copy=False)
    if eeg.ndim != 3 or eeg.shape[1:] != (62, 800):
        raise ValueError(f"{path}: expected EEG [trials,62,800], got {eeg.shape}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{path}: expected binary labels 0/1, got {np.unique(labels)}")
    # Physical microvolts, matching the SHIN raw-uV CBraMod path: no /100.
    return eeg.reshape(-1, 62, 4, 200), labels


def load_split(root: Path, subjects: list[int]) -> tuple[np.ndarray, np.ndarray]:
    arrays, targets = [], []
    for subject in subjects:
        x, y = load_subject(root / f"subject{subject:02d}_paired.npz")
        arrays.append(x)
        targets.append(y)
    return np.concatenate(arrays), np.concatenate(targets)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "mplconfig"))
    seed_everything(args.seed)

    split_subjects = {
        "train": list(range(1, 13)),
        "val": list(range(13, 16)),
        "test": list(range(16, 19)),
    }
    arrays, targets = {}, {}
    for name, subjects in split_subjects.items():
        arrays[name], targets[name] = load_split(args.data_root, subjects)
        print(f"[{name}] subjects={subjects}, X={arrays[name].shape}, labels={dict(Counter(targets[name]))}", flush=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    raw_loaders = {
        name: loader(arrays[name], targets[name], args.batch_size,
                     name == "train" and args.full_finetune,
                     args.num_workers, args.seed)
        for name in arrays
    }

    model = FineMICBraMod(dropout=args.dropout)
    pretrained = load_pretrained(model, args.checkpoint)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = args.full_finetune
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    model.to(device)

    if args.full_finetune:
        active_loaders = raw_loaders
        feature_mode = False
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ], weight_decay=args.weight_decay)
    else:
        feature_arrays, feature_targets = {}, {}
        for name in ("train", "val", "test"):
            feature_arrays[name], feature_targets[name] = cache_features(
                model, raw_loaders[name], device, name
            )
        active_loaders = {
            name: loader(feature_arrays[name], feature_targets[name], args.batch_size,
                         name == "train", args.num_workers, args.seed)
            for name in feature_arrays
        }
        feature_mode = True
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
        )
    history, best_epoch, best_accuracy = [], 0, -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, active_loaders["train"], optimizer, device, feature_mode)
        val = evaluate(model, active_loaders["val"], device, feature_mode)
        record = {"epoch": epoch, "train_loss": loss, "val": val}
        history.append(record)
        print(f"epoch {epoch:03d}/{args.epochs} train_loss={loss:.4f} "
              f"val_acc={val['accuracy']:.4f} val_f1={val['f1_macro']:.4f}", flush=True)
        if val["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = val["accuracy"], epoch
            state = model.state_dict() if args.full_finetune else model.classifier.state_dict()
            torch.save(state, args.output_dir / ("best_model.pth" if args.full_finetune else "best_head.pth"))
        write_json(args.output_dir / "history.json", history)

    best_path = args.output_dir / ("best_model.pth" if args.full_finetune else "best_head.pth")
    best_state = torch.load(best_path, map_location=device, weights_only=True)
    if args.full_finetune:
        model.load_state_dict(best_state)
    else:
        model.classifier.load_state_dict(best_state)
    test = evaluate(model, active_loaders["test"], device, feature_mode)
    summary = {
        "experiment": ("FineMI EEG-only full CBraMod fine-tuning" if args.full_finetune
                       else "FineMI EEG-only frozen CBraMod backbone"),
        "preprocessing": "physical-uV EEG; no CAR/z-score/additional scaling and no /100; reshape 62x800 to 62x4x200",
        "split_subjects": split_subjects,
        "classifier": "official all_patch_reps (49600->800->200->2)",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_test": test,
        "elapsed_seconds": time.time() - started,
        "pretrained_load": pretrained,
        "args": vars(args),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
