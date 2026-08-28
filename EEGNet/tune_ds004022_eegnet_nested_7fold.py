"""Nested subject-grouped seven-fold tuning for ds004022 EEGNet.

Each outer fold holds one participant out for testing.  Hyperparameters are
ranked only on one participant drawn from the remaining six, exactly mirroring
the earlier fixed 1--5/6/7 tuning procedure while preventing outer-test use.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from tune_ds004022_eegnet import CONFIGS, parse_seeds, train_one


CACHE = Path(r"D:\0senior student creation\braindecode_codebrain_prep\cache\ds004022_eeg_mi5s_4to40hz_200hz.npz")
OUTPUT_ROOT = Path(r"D:\0senior student creation\results")
CLASSES = 4


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested subject-grouped 7-fold tuning for ds004022 EEGNet")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", default="1", help="Comma-separated model seeds used during each inner search")
    parser.add_argument("--fold-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-configs", type=int, default=None, help="For a smoke test only")
    return parser.parse_args()


def normalize_from_inner_train(x: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = x[train_idx].mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = x[train_idx].std(axis=(0, 2), keepdims=True, dtype=np.float64)
    return ((x - mean) / (std + 1e-6)).astype(np.float32)


def outer_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    for fold, (trainval, test) in enumerate(outer.split(x, y, groups), start=1):
        inner = GroupShuffleSplit(n_splits=1, test_size=1, random_state=seed + fold)
        train_relative, val_relative = next(inner.split(trainval, y[trainval], groups[trainval]))
        yield fold, trainval[train_relative], trainval[val_relative], test


def metric_summary(rows: list[dict]) -> dict:
    keys = ("accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa")
    return {
        key: {
            "mean": float(np.mean([row["test"][key] for row in rows])),
            "std_sample": float(np.std([row["test"][key] for row in rows], ddof=1)),
        }
        for key in keys
    }


def pooled_metrics(rows: list[dict]) -> dict:
    matrix = np.sum([np.asarray(row["test"]["confusion_matrix"], dtype=np.int64) for row in rows], axis=0)
    y_true = np.repeat(np.arange(CLASSES), matrix.sum(axis=1))
    y_pred = np.concatenate([np.repeat(np.arange(CLASSES), matrix[row]) for row in range(CLASSES)])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seeds = parse_seeds(args.seeds)
    configs = CONFIGS[:args.max_configs] if args.max_configs else CONFIGS
    item = np.load(CACHE, allow_pickle=False)
    x, y, groups = item["eeg"].astype(np.float32), item["labels"].astype(np.int64), item["subject_ids"].astype(np.int64)
    output = args.output_dir or OUTPUT_ROOT / f"ds004022_eegnet_nested_7fold_{datetime.now():%Y%m%d-%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    folds = []
    for fold, train_idx, val_idx, test_idx in outer_splits(x, y, groups, args.fold_seed):
        x_norm = normalize_from_inner_train(x, train_idx)
        arrays = {name: x_norm[index] for name, index in {"train": train_idx, "val": val_idx, "test": test_idx}.items()}
        labels = {name: y[index] for name, index in {"train": train_idx, "val": val_idx, "test": test_idx}.items()}
        trials = []
        for config_id, config in enumerate(configs, start=1):
            per_seed = []
            for seed in seeds:
                result, _ = train_one(arrays, labels, config, seed, args.epochs, args.patience, args.batch_size, device)
                per_seed.append(result)
            trials.append({
                "config_id": config_id, "config": config, "runs": per_seed,
                "mean_val_f1": float(np.mean([run["best_val"]["f1_macro"] for run in per_seed])),
                "mean_val_loss": float(np.mean([run["best_val"]["loss"] for run in per_seed])),
            })
        trials.sort(key=lambda row: (-row["mean_val_f1"], row["mean_val_loss"]))
        winner = trials[0]
        final, test = train_one(arrays, labels, winner["config"], seeds[0], args.epochs, args.patience, args.batch_size, device, evaluate_test=True)
        record = {
            "fold": fold,
            "train_subjects": sorted(np.unique(groups[train_idx]).astype(int).tolist()),
            "val_subjects": sorted(np.unique(groups[val_idx]).astype(int).tolist()),
            "test_subjects": sorted(np.unique(groups[test_idx]).astype(int).tolist()),
            "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "selected": winner,
            "final_selected_run": final,
            "test": test,
        }
        folds.append(record)
        (output / f"fold{fold}_tuning.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"fold={fold} test_subject={record['test_subjects']} config={winner['config_id']} test_acc={test['accuracy']:.4f} test_f1={test['f1_macro']:.4f}", flush=True)
    report = {
        "dataset": "OpenNeuro ds004022",
        "task": "4-class reach/grasp/lift/twist",
        "protocol": "nested subject-grouped 7-fold; outer test subject never enters the inner hyperparameter search",
        "input": {"shape": list(x.shape), "sampling_hz": 200, "window": "complete 5-s MI"},
        "preprocessing": "continuous CAR, 4-40 Hz, 500->200 Hz; per-fold normalization fitted on inner-train subjects only",
        "inner_tuning": {"candidate_configs": configs, "seeds": seeds, "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size},
        "folds": folds,
        "fold_mean_std": metric_summary(folds),
        "pooled_oof": pooled_metrics(folds),
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "pooled_oof": report["pooled_oof"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
