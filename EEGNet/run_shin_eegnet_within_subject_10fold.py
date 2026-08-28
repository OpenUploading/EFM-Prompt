"""Within-subject stratified 10-fold EEGNet evaluation on SHIN.

This is an evaluation protocol parallel to the subject-independent runner.
Every original trial belongs to exactly one test fold; no sliding windows are
created. A validation split is drawn only from each fold's training trials.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import run_shin_eegnet as base
from eegnet_pytorch import EEGNet


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Within-subject 10-fold EEGNet on SHIN")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(base.TASKS), required=True)
    parser.add_argument("--subjects", default="1-29")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernel-length", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def load_subject_cached(
    data_root: Path,
    cache_dir: Path,
    subject: int,
    task_key: str,
    task: dict,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{task_key}_within_sub-{subject:02d}_30ch_2000_global-zscore.npz"
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["X"], cached["y"]
    x, y, _ = base.load_subject(data_root, subject, task)
    np.savez(cache_path, X=x, y=y)
    return x, y


@torch.no_grad()
def evaluate_predictions(
    model: EEGNet,
    data_loader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int | None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum, seen, predictions, targets = 0.0, 0, [], []
    for step, (x, y) in enumerate(data_loader, 1):
        if max_batches and step > max_batches:
            break
        x, y = x.to(device).float(), y.to(device)
        logits = model(x)
        loss_sum += float(criterion(logits, y).item()) * len(y)
        seen += len(y)
        predictions.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    truth = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    return base.metric_values(truth, prediction, loss_sum / seen), truth, prediction


def mean_std(items: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([item[key] for item in items], dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
    fold_seed: int,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_fraction, random_state=fold_seed
    )
    fold_train_indices = train_indices
    train_relative, val_relative = next(
        splitter.split(fold_train_indices, y[fold_train_indices])
    )
    train_indices = fold_train_indices[train_relative]
    val_indices = fold_train_indices[val_relative]

    base.seed_all(fold_seed)
    model = EEGNet(
        channels=30,
        samples=2000,
        classes=2,
        dropout=args.dropout,
        kernel_length=args.kernel_length,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loaders = {
        "train": base.loader(x[train_indices], y[train_indices], args.batch_size, True, args.num_workers, fold_seed),
        "val": base.loader(x[val_indices], y[val_indices], args.batch_size, False, args.num_workers, fold_seed),
        "test": base.loader(x[test_indices], y[test_indices], args.batch_size, False, args.num_workers, fold_seed),
    }

    best_state, best_epoch, best_val, best_val_accuracy = None, 0, None, -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, (batch_x, batch_y) in enumerate(loaders["train"], 1):
            if args.max_batches and step > args.max_batches:
                break
            batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            model.constrain_weights()
        val = base.evaluate(model, loaders["val"], device, criterion, args.max_batches)
        if val["accuracy"] > best_val_accuracy:
            best_val_accuracy = val["accuracy"]
            best_epoch = epoch
            best_val = copy.deepcopy(val)
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    test, truth, prediction = evaluate_predictions(
        model, loaders["test"], device, criterion, args.max_batches
    )
    return {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "test": test,
        "counts": {"train": len(train_indices), "val": len(val_indices), "test": len(test_indices)},
    }, truth, prediction


def main() -> None:
    args = arguments()
    if not (args.data_root / "dataset_description.json").is_file():
        raise FileNotFoundError(f"Not a SHIN BIDS root: {args.data_root}")
    if args.folds != 10:
        raise ValueError("This runner is intentionally fixed to the DAMFNet-style 10-fold protocol")
    if not 0 < args.val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    subjects = base.parse_subjects(args.subjects)
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    task = base.TASKS[args.task]
    all_subjects, started = [], time.time()
    for subject in subjects:
        x, y = load_subject_cached(args.data_root, args.cache_dir, subject, args.task, task)
        if len(np.unique(y)) != 2 or min(np.bincount(y)) < args.folds:
            raise RuntimeError(f"Subject {subject} cannot support stratified {args.folds}-fold CV")
        folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + subject)
        fold_records, pooled_truth, pooled_prediction = [], [], []
        for fold, (train_indices, test_indices) in enumerate(folds.split(x, y), 1):
            fold_seed = args.seed * 100000 + subject * 100 + fold
            record, truth, prediction = run_fold(
                x, y, train_indices, test_indices, args, fold_seed, device
            )
            record["fold"] = fold
            fold_records.append(record)
            pooled_truth.append(truth)
            pooled_prediction.append(prediction)
            print(
                f"subject {subject:02d} fold {fold:02d}/10 "
                f"epoch={record['best_epoch']} test_acc={record['test']['accuracy']:.4f}",
                flush=True,
            )
        pooled_truth_array = np.concatenate(pooled_truth)
        pooled_prediction_array = np.concatenate(pooled_prediction)
        pooled = base.metric_values(pooled_truth_array, pooled_prediction_array, 0.0)
        fold_tests = [item["test"] for item in fold_records]
        all_subjects.append({
            "subject": subject,
            "trial_count": len(y),
            "folds": fold_records,
            "pooled_test": pooled,
            "fold_test_mean_std": {
                key: mean_std(fold_tests, key)
                for key in ("accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa")
            },
        })

    pooled_subject_metrics = [item["pooled_test"] for item in all_subjects]
    summary = {
        "protocol": "within-subject stratified 10-fold; validation split from each fold training set",
        "task": {"key": args.task, **task},
        "subjects": subjects,
        "seed": args.seed,
        "schedule": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "dropout": args.dropout,
            "kernel_length": args.kernel_length, "label_smoothing": args.label_smoothing,
        },
        "input": "30 channels x 2000 samples (10 seconds at 200 Hz), per-trial global z-score",
        "subject_results": all_subjects,
        "subjects_mean_std_pooled_test": {
            key: mean_std(pooled_subject_metrics, key)
            for key in ("accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa")
        },
        "elapsed_seconds": time.time() - started,
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    base.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["subjects_mean_std_pooled_test"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
