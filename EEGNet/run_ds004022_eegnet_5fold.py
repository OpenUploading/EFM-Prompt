"""Compare subject-grouped and trial-mixed 5-fold EEGNet on ds004022."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

from eegnet_pytorch import EEGNet
import run_shin_eegnet as portable


CACHE = Path(r"D:\0senior student creation\braindecode_codebrain_prep\cache\ds004022_eeg_mi5s_4to40hz_200hz.npz")
DIRECT_RESULT = Path(r"D:\data\EEGNet-SHIN\ds004022_eegnet_tuning_20260821-121049\tuning_summary.json")
OUTPUT_ROOT = Path(r"D:\0senior student creation\results")
CLASSES = 4


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray, loss: float) -> dict:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(CLASSES))).tolist(),
    }


@torch.no_grad()
def evaluate(model, data_loader, device, criterion):
    model.eval()
    loss_sum = seen = 0
    ys, preds = [], []
    for x, y in data_loader:
        x, y = x.to(device).float(), y.to(device)
        logits = model(x)
        loss_sum += float(criterion(logits, y).item()) * len(y)
        seen += len(y)
        ys.append(y.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(preds)
    return metrics(y_true, y_pred, loss_sum / seen), y_true, y_pred


def normalize_from_train(x: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = x[train_idx].mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = x[train_idx].std(axis=(0, 2), keepdims=True, dtype=np.float64)
    return ((x - mean) / (std + 1e-6)).astype(np.float32)


def train_fold(x, y, train_idx, val_idx, test_idx, args, fold_seed):
    portable.seed_all(fold_seed)
    x_norm = normalize_from_train(x, train_idx)
    loaders = {
        "train": portable.loader(x_norm[train_idx], y[train_idx], args.batch_size, True, 0, fold_seed),
        "val": portable.loader(x_norm[val_idx], y[val_idx], args.batch_size, False, 0, fold_seed),
        "test": portable.loader(x_norm[test_idx], y[test_idx], args.batch_size, False, 0, fold_seed),
    }
    device = torch.device(args.device)
    model = EEGNet(
        channels=x.shape[1], samples=x.shape[2], classes=CLASSES,
        dropout=0.5, kernel_length=100,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best_state = best_val = None
    best_epoch = stalled = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, by in loaders["train"]:
            bx, by = bx.to(device).float(), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            model.constrain_weights()
        val, _, _ = evaluate(model, loaders["val"], device, criterion)
        improved = best_val is None or (
            val["f1_macro"] > best_val["f1_macro"] + 1e-6
            or (abs(val["f1_macro"] - best_val["f1_macro"]) <= 1e-6 and val["loss"] < best_val["loss"])
        )
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val, best_epoch, stalled = val, epoch, 0
        else:
            stalled += 1
        if stalled >= args.patience:
            break
    model.load_state_dict(best_state)
    test, y_true, y_pred = evaluate(model, loaders["test"], device, criterion)
    return {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "validation": best_val,
        "test": test,
    }, y_true, y_pred


def grouped_splits(x, y, groups, seed, n_splits):
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (trainval, test) in enumerate(outer.split(x, y, groups), 1):
        inner = GroupShuffleSplit(n_splits=1, test_size=1, random_state=seed + fold)
        inner_train, inner_val = next(inner.split(trainval, y[trainval], groups[trainval]))
        yield fold, trainval[inner_train], trainval[inner_val], test


def mixed_splits(x, y, groups, seed, n_splits):
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (trainval, test) in enumerate(outer.split(x, y), 1):
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
        inner_train, inner_val = next(inner.split(trainval, y[trainval]))
        yield fold, trainval[inner_train], trainval[inner_val], test


def run_protocol(name, splitter, x, y, groups, args):
    folds, all_true, all_pred = [], [], []
    for fold, train_idx, val_idx, test_idx in splitter(x, y, groups, args.seed, args.n_splits):
        result, y_true, y_pred = train_fold(x, y, train_idx, val_idx, test_idx, args, args.seed)
        result.update({
            "fold": fold,
            "train_subjects": sorted(np.unique(groups[train_idx]).astype(int).tolist()),
            "val_subjects": sorted(np.unique(groups[val_idx]).astype(int).tolist()),
            "test_subjects": sorted(np.unique(groups[test_idx]).astype(int).tolist()),
        })
        folds.append(result)
        all_true.append(y_true)
        all_pred.append(y_pred)
        print(f"{name} fold={fold} acc={result['test']['accuracy']:.4f} kappa={result['test']['cohen_kappa']:.4f}", flush=True)
    keys = ("accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa")
    fold_summary = {
        key: {
            "mean": float(np.mean([item["test"][key] for item in folds])),
            "std_sample": float(np.std([item["test"][key] for item in folds], ddof=1)),
        }
        for key in keys
    }
    pooled = metrics(np.concatenate(all_true), np.concatenate(all_pred), 0.0)
    pooled["loss"] = None
    return {"protocol": name, "folds": folds, "fold_mean_std": fold_summary, "pooled": pooled}


def main():
    args = arguments()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    item = np.load(CACHE, allow_pickle=False)
    x = item["eeg"].astype(np.float32)
    y = item["labels"].astype(np.int64)
    groups = item["subject_ids"].astype(np.int64)
    output = args.output_dir or OUTPUT_ROOT / f"ds004022_eegnet_5fold_{datetime.now():%Y%m%d-%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    grouped = run_protocol(f"subject_grouped_{args.n_splits}fold", grouped_splits, x, y, groups, args)
    mixed = run_protocol(f"trial_mixed_{args.n_splits}fold", mixed_splits, x, y, groups, args)
    direct = None
    if DIRECT_RESULT.is_file():
        direct = json.loads(DIRECT_RESULT.read_text(encoding="utf-8"))["final_test_once"]
    report = {
        "dataset": "OpenNeuro ds004022",
        "task": "4-class reach/grasp/lift/twist",
        "input": {"shape": list(x.shape), "sampling_hz": 200, "window": "complete 5-s MI"},
        "preprocessing": "continuous CAR, 4-40 Hz, 500->200 Hz; normalization fitted on each fold's training subset only",
        "hyperparameters": {
            "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
            "lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.5,
            "kernel_length": 100, "label_smoothing": 0.05, "seed": args.seed,
        },
        "direct_subject_split_1to5_6_7": direct,
        "subject_grouped_5fold": grouped,
        "trial_mixed_5fold": mixed,
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    if direct:
        rows.append({"protocol": "direct_subject_split_1to5_6_7", "aggregation": "single_test", **{k: direct[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa")}})
    for result in (grouped, mixed):
        rows.append({"protocol": result["protocol"], "aggregation": "pooled_oof", **{k: result["pooled"][k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa")}})
        rows.append({"protocol": result["protocol"], "aggregation": "fold_mean", **{k: result["fold_mean_std"][k]["mean"] for k in ("accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa")}})
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "comparison": rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
