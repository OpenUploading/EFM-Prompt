"""Paper-aligned FineMI EEG-only subject-independent baselines.

The default is one fixed 12/3/3 subject split.  Five-fold evaluation remains
available for a later robustness analysis.  Test subjects are never used for
model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = Path(r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_paper_car_uv100_binary_1v6")
CBRAMOD_19_CHANNELS = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)


class Trials(Dataset):
    def __init__(self, eeg: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.eeg, self.labels, self.indices = eeg, labels, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row = int(self.indices[index])
        return self.eeg[row], self.labels[row]


class TemporalMeanFlatten(nn.Module):
    """Average the four 1-second patches while retaining channel-specific features."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=2).flatten(start_dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("eegnet", "cbramod"), required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, default=ROOT.parent)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1, help="model/data-order seed")
    parser.add_argument("--protocol", choices=("single", "fivefold"), default="single")
    parser.add_argument("--fold-seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--selection-metric", choices=("accuracy", "macro_f1"), default="macro_f1")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--finetune-mode", choices=("frozen", "full"), default="frozen")
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="CBraMod backbone LR for full fine-tuning; defaults to --lr.")
    parser.add_argument("--classifier", choices=("official_all_patch", "temporal_mean"),
                        default="official_all_patch")
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="CosineAnnealingLR lower bound.")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Global gradient-norm limit; omit to disable clipping.")
    parser.add_argument("--binary-loss", choices=("ce", "bce"), default="ce",
                        help="Use a one-logit BCE head for the binary CBraMod task, or two-logit CE.")
    parser.add_argument("--channel-set", choices=("all62", "cbramod19"), default="all62",
                        help="Use all FineMI scalp channels or the 19 channels used in CBraMod pre-training.")
    parser.add_argument("--white-noise-augment-times", type=int, default=0,
                        help="Number of one-time FineMI-style noisy copies added to the training set only.")
    parser.add_argument("--white-noise-std", type=float, default=0.02)
    parser.add_argument("--white-noise-snr-db", type=float, default=1.0)
    parser.add_argument("--eegnet-f1", type=int, default=16)
    parser.add_argument("--eegnet-depth-multiplier", type=int, default=2)
    parser.add_argument("--eegnet-f2", type=int, default=32)
    parser.add_argument("--eegnet-kernel-length", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def augment_with_white_noise(
    eeg: np.ndarray, labels: np.ndarray, times: int, std: float, snr_db: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match FineMI's released one-time white-noise augmentation formula."""
    if times <= 0:
        return eeg, labels
    flat = eeg.reshape(eeg.shape[0], eeg.shape[1], -1)
    signal_power = np.mean(flat ** 2, axis=-1, keepdims=True)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    copies = [eeg]
    for _ in range(times):
        noise = rng.normal(0.0, std, size=flat.shape) * np.sqrt(noise_power)
        copies.append((flat + noise).astype(np.float32).reshape(eeg.shape))
    return np.concatenate(copies), np.tile(labels, times + 1)


def subject_folds(seed: int) -> list[tuple[list[int], list[int], list[int]]]:
    subjects = np.arange(1, 19)
    permutation = np.random.default_rng(seed).permutation(subjects).tolist()
    test_groups = [permutation[:4], permutation[4:8], permutation[8:12], permutation[12:15], permutation[15:18]]
    folds = []
    for group in test_groups:
        remaining = [subject for subject in permutation if subject not in group]
        val = remaining[:3]
        train = remaining[3:]
        folds.append((train, val, group))
    return folds


def load_cache(root: Path, channel_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    pieces, labels, subjects = [], [], []
    metadata = []
    for subject in range(1, 19):
        path = root / f"subject{subject:02d}_paired.npz"
        with np.load(path, allow_pickle=False) as item:
            eeg = item["eeg"].astype(np.float32, copy=False)
            y = item["labels"].astype(np.int64, copy=False)
            if eeg.shape != (80, 62, 800) or np.bincount(y, minlength=2).tolist() != [40, 40]:
                raise RuntimeError(f"{path}: invalid EEG/labels")
            available = [str(name).upper() for name in item["eeg_channels"].tolist()]
            selected = list(CBRAMOD_19_CHANNELS) if channel_set == "cbramod19" else available
            missing = [name for name in selected if name not in available]
            if missing:
                raise RuntimeError(f"{path}: missing requested EEG channels {missing}")
            channel_indices = [available.index(name) for name in selected]
            pieces.append(eeg[:, channel_indices].reshape(-1, len(selected), 4, 200))
            labels.append(y)
            subjects.append(np.full(len(y), subject, dtype=np.int16))
            metadata.append(str(item["preprocessing"]) if "preprocessing" in item else "missing")
    return np.concatenate(pieces), np.concatenate(labels), np.concatenate(subjects), {
        "preprocessing": sorted(set(metadata)), "channel_set": channel_set,
        "eeg_channels": selected,
    }


def make_model(args: argparse.Namespace, num_channels: int) -> nn.Module:
    if args.model == "eegnet":
        prep = Path(r"D:\0senior student creation\braindecode_codebrain_prep\scripts")
        sys.path.insert(0, str(prep))
        from foundation_prompt_models import EEGNetBaseline
        return EEGNetBaseline(sample_shape=(num_channels, 4, 200), dropout=args.dropout, n_outputs=2,
                               f1=args.eegnet_f1, depth_multiplier=args.eegnet_depth_multiplier,
                               f2=args.eegnet_f2, kernel_length=args.eegnet_kernel_length)
    sys.path.insert(0, str(args.portable_root / "CBraMod"))
    from models.cbramod import CBraMod
    from run_shin2017_foundation_boundary_prompt import load_compatible
    model = CBraMod(200, 200, 200, 800, num_channels, 12, 8)
    checkpoint = args.checkpoint or args.portable_root / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth"
    load_compatible(model, checkpoint)
    model.proj_out = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = args.finetune_mode == "full"
    output_dim = 1 if args.binary_loss == "bce" else 2
    if args.classifier == "temporal_mean":
        classifier = nn.Sequential(
            TemporalMeanFlatten(), nn.LayerNorm(num_channels * 200),
            nn.Linear(num_channels * 200, 256), nn.ELU(), nn.Dropout(args.dropout),
            nn.Linear(256, output_dim),
        )
    else:
        classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(num_channels * 4 * 200, 800), nn.ELU(), nn.Dropout(args.dropout),
            nn.Linear(800, 200), nn.ELU(), nn.Dropout(args.dropout), nn.Linear(200, output_dim),
        )
    return nn.Sequential(model, classifier)


def run_epoch(model, loader, optimizer, device, frozen_backbone: bool, binary_loss: str, grad_clip: float | None = None):
    training = optimizer is not None
    model.train(training)
    if frozen_backbone:
        model[0].eval()
    loss_fn = nn.BCEWithLogitsLoss() if binary_loss == "bce" else nn.CrossEntropyLoss()
    losses, predicted, target = [], [], []
    for eeg, y in loader:
        eeg, y = eeg.to(device).float(), y.to(device).long()
        with torch.set_grad_enabled(training):
            logits = model(eeg)
            if binary_loss == "bce":
                logits = logits.squeeze(1)
                loss = loss_fn(logits, y.float())
            else:
                loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        losses.append(float(loss.detach()) * len(y))
        if binary_loss == "bce":
            predicted.append((logits.detach() >= 0).long().cpu().numpy())
        else:
            predicted.append(logits.detach().argmax(1).cpu().numpy())
        target.append(y.cpu().numpy())
    y_true, y_pred = np.concatenate(target), np.concatenate(predicted)
    return {"loss": sum(losses) / len(y_true), **scores(y_true, y_pred)}, y_true, y_pred


def main() -> None:
    started_at = time.time()
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.cache_root, args.output_dir, args.portable_root = args.cache_root.resolve(), args.output_dir.resolve(), args.portable_root.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    if args.batch_size is None:
        args.batch_size = 64 if args.model == "eegnet" else 16
    eeg, labels, subject_ids, meta = load_cache(args.cache_root, args.channel_set)
    device = torch.device(args.device)
    fold_rows, subject_rows = [], []
    if args.protocol == "single":
        evaluation_splits = [(list(range(1, 13)), list(range(13, 16)), list(range(16, 19)))]
        protocol_description = "single fixed subject-independent split; train=1-12, validation=13-15, test=16-18"
    else:
        evaluation_splits = subject_folds(args.fold_seed)
        protocol_description = "subject_grouped_5fold; test groups 4/4/4/3/3; validation subjects selected only from outer-train"
    for fold, (train_subjects, val_subjects, test_subjects) in enumerate(evaluation_splits, start=1):
        seed_all(args.seed + fold * 1000)
        indices = {name: np.flatnonzero(np.isin(subject_ids, values)) for name, values in {
            "train": train_subjects, "val": val_subjects, "test": test_subjects}.items()}
        train_eeg, train_labels = augment_with_white_noise(
            eeg[indices["train"]], labels[indices["train"]], args.white_noise_augment_times,
            args.white_noise_std, args.white_noise_snr_db, args.seed + fold * 1000,
        )
        loaders = {
            "train": DataLoader(
                TensorDataset(torch.from_numpy(train_eeg), torch.from_numpy(train_labels)),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            ),
            **{name: DataLoader(Trials(eeg, labels, indices[name]), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=torch.cuda.is_available()) for name in ("val", "test")},
        }
        model = make_model(args, eeg.shape[1]).to(device)
        frozen_backbone = args.model == "cbramod" and args.finetune_mode == "frozen"
        optimizer_type = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
        if args.model == "cbramod" and args.finetune_mode == "full":
            optimizer = optimizer_type([
                {"params": model[0].parameters(), "lr": args.backbone_lr or args.lr},
                {"params": model[1].parameters(), "lr": args.lr},
            ], weight_decay=args.weight_decay)
        else:
            optimizer = optimizer_type((p for p in model.parameters() if p.requires_grad),
                                       lr=args.lr, weight_decay=args.weight_decay)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
                     if args.scheduler == "cosine" else None)
        best_state, best_val, best_epoch, stale = None, -np.inf, 0, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            train, _, _ = run_epoch(model, loaders["train"], optimizer, device, frozen_backbone, args.binary_loss, args.grad_clip)
            val, _, _ = run_epoch(model, loaders["val"], None, device, frozen_backbone, args.binary_loss)
            history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train, "val": val})
            if val[args.selection_metric] > best_val:
                best_val, best_epoch, stale = val[args.selection_metric], epoch, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break
            if scheduler is not None:
                scheduler.step()
        model.load_state_dict(best_state)
        test, y_true, y_pred = run_epoch(model, loaders["test"], None, device, frozen_backbone, args.binary_loss)
        torch.save(best_state, args.output_dir / f"fold{fold}_best.pth")
        (args.output_dir / f"fold{fold}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        fold_rows.append({"fold": fold, "train_subjects": train_subjects, "val_subjects": val_subjects,
                          "test_subjects": test_subjects, "best_epoch": best_epoch,
                          "selection_metric": args.selection_metric,
                          "best_val_selection_score": best_val, "test": test})
        test_subject_rows = subject_ids[indices["test"]]
        for subject in test_subjects:
            mask = test_subject_rows == subject
            subject_rows.append({"fold": fold, "subject": subject, **scores(y_true[mask], y_pred[mask])})
        print(f"fold {fold}: best_val_{args.selection_metric}={best_val:.4f}, test_acc={test['accuracy']:.4f}, test_f1={test['macro_f1']:.4f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregate = {metric: float(np.mean([row[metric] for row in subject_rows])) for metric in ("accuracy", "balanced_accuracy", "macro_f1", "kappa")}
    summary = {"model": args.model, "seed": args.seed, "fold_seed": args.fold_seed, "outer_protocol": protocol_description,
               "cache": str(args.cache_root), "data": {"eeg_shape": list(eeg.shape), **meta}, "args": vars(args),
               "folds": fold_rows, "subject_oof_mean": aggregate, "subject_rows": subject_rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with (args.output_dir / "subject_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0]))
        writer.writeheader(); writer.writerows(subject_rows)
    print(json.dumps({"subject_oof_mean": aggregate, "seconds": time.time() - started_at}, indent=2), flush=True)


if __name__ == "__main__":
    main()
