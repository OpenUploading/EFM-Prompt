"""Train the official-architecture EEGNet-8,2 baseline on SHIN EEG."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
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
from torch.utils.data import DataLoader, TensorDataset

from eegnet_pytorch import EEGNet


OFFICIAL_COMMIT = "4a512e503198db2010848813ead9afbf8cd54c97"
OUTPUT_ROOT = Path(r"D:\data\EEGNet-SHIN")
SHIN_CHANNELS = [
    "F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8",
    "AFF1h", "AFF2h", "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h",
    "CCP3h", "T7", "P7", "P3", "PPO1h", "POO1", "POO2", "PPO2h",
    "P4", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "P8", "T8",
]
TASKS = {
    "mi": {
        "name": "EEG-MI",
        "description": "left_hand (0) vs right_hand (1)",
        "sessions": ("ses-0imagery", "ses-2imagery", "ses-4imagery"),
        "labels": {"left_hand": 0, "right_hand": 1},
    },
    "ma": {
        "name": "EEG-MA",
        "description": "subtraction (0) vs rest (1)",
        "sessions": ("ses-1arithmetic", "ses-3arithmetic", "ses-5arithmetic"),
        "labels": {"subtraction": 0, "rest": 1},
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEGNet-8,2 on SHIN EEG")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\DataSets\SHIN\v1.0.1"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--train-subjects", default="1-19")
    parser.add_argument("--val-subjects", default="20-24")
    parser.add_argument("--test-subjects", default="25-29")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernel-length", type=int, default=100)
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-subjects-per-split", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def parse_subjects(text: str) -> list[int]:
    result: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            result.extend(range(start, stop + 1))
        elif part:
            result.append(int(part))
    if not result or len(result) != len(set(result)):
        raise ValueError(f"Invalid subjects: {text}")
    return result


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def one(folder: Path, pattern: str) -> Path:
    items = list(folder.glob(pattern))
    if len(items) != 1:
        raise RuntimeError(f"{folder}: expected one {pattern}, found {len(items)}")
    return items[0]


def load_subject(root: Path, subject: int, task: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    import mne

    trials: list[np.ndarray] = []
    labels: list[int] = []
    session_info: list[dict] = []
    for session in task["sessions"]:
        eeg_dir = root / f"sub-{subject:02d}" / session / "eeg"
        bdf = one(eeg_dir, "*_eeg.bdf")
        events = one(eeg_dir, "*_events.tsv")
        channels = one(eeg_dir, "*_channels.tsv")
        eeg_names = [
            row["name"] for row in read_tsv(channels)
            if row.get("type", "").upper() == "EEG"
        ]
        if eeg_names != SHIN_CHANNELS:
            raise RuntimeError(f"Unexpected channel order in {channels}: {eeg_names}")
        raw = mne.io.read_raw_bdf(bdf, preload=True, verbose="ERROR")
        if abs(float(raw.info["sfreq"]) - 200.0) > 1e-6:
            raise RuntimeError(f"Expected 200 Hz in {bdf}, got {raw.info['sfreq']}")
        continuous = np.nan_to_num(
            raw.get_data(picks=SHIN_CHANNELS, units="uV"),
            copy=False,
        )
        raw.close()

        counts: Counter[str] = Counter()
        for event in read_tsv(events):
            label_name = event.get("trial_type", "")
            if label_name not in task["labels"]:
                continue
            start = int(event["sample"])
            trial = continuous[:, start:start + 2000].astype(np.float32, copy=True)
            if trial.shape != (30, 2000):
                raise RuntimeError(f"Bad trial shape {trial.shape} in {events}")
            mean = float(trial.mean(dtype=np.float64))
            std = float(trial.std(dtype=np.float64))
            if not np.isfinite(std) or std < 1e-6:
                raise RuntimeError(f"Invalid std={std} in {bdf}")
            trials.append(((trial - mean) / std).astype(np.float32))
            labels.append(task["labels"][label_name])
            counts[label_name] += 1
        expected = Counter({label: 10 for label in task["labels"]})
        if counts != expected:
            raise RuntimeError(f"Bad event counts in {events}: {dict(counts)}")
        session_info.append({"session": session, "label_counts": dict(counts)})
    return (
        np.stack(trials),
        np.asarray(labels, dtype=np.int64),
        {"subject": subject, "trials": len(labels), "sessions": session_info},
    )


def load_split(
    root: Path,
    name: str,
    subjects: list[int],
    cache_dir: Path,
    task_key: str,
    task: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(map(str, subjects))
    cache = cache_dir / f"{task_key}_{name}_sub-{tag}_30ch_2000_global-zscore.npz"
    if cache.is_file():
        item = np.load(cache, allow_pickle=False)
        x, y = item["X"], item["y"]
        print(f"[{name}] cache {cache}: X={x.shape}, y={Counter(y.tolist())}", flush=True)
        return x, y, [{"subject": subject, "source": "cache"} for subject in subjects]
    arrays, targets, details = [], [], []
    for subject in subjects:
        x, y, detail = load_subject(root, subject, task)
        arrays.append(x)
        targets.append(y)
        details.append(detail)
        print(f"[{name}] sub-{subject}: X={x.shape}, y={Counter(y.tolist())}", flush=True)
    x_all, y_all = np.concatenate(arrays), np.concatenate(targets)
    np.savez(cache, X=x_all, y=y_all)
    print(f"[{name}] wrote cache {cache}", flush=True)
    return x_all, y_all, details


def loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, loss: float) -> dict:
    classes = np.unique(np.concatenate([y_true, y_pred])).tolist()
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


@torch.no_grad()
def evaluate(
    model: EEGNet,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int | None,
) -> dict:
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
    return metric_values(
        np.concatenate(targets),
        np.concatenate(predictions),
        loss_sum / seen,
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_accuracy",
            "val_balanced_accuracy", "val_macro_f1", "val_kappa", "elapsed_seconds",
        ])
        for row in history:
            val = row["val"]
            writer.writerow([
                row["epoch"], row["train_loss"], val["loss"], val["accuracy"],
                val["balanced_accuracy"], val["f1_macro"], val["cohen_kappa"],
                row["elapsed_seconds"],
            ])


def record(summary: dict) -> str:
    best, test, final = summary["best"], summary["best_test"], summary["final"]
    return f"""# EEGNet-8,2 × SHIN EEG 实验记录

## 身份

架构逐层对应作者官方 `vlawhern/arl-eegmodels` 的新版 EEGNet-8,2。官方代码为旧版 TensorFlow/Keras，本实验在现有 PyTorch CUDA 环境中运行等价移植，不使用预训练权重。

## 参数

| 参数 | 值 |
|---|---|
| 任务 | {summary['task']['name']}：{summary['task']['description']} |
| 划分 | 19/5/5：train 1–19 / val 20–24 / test 25–29 |
| Seed | {summary['seed']} |
| 输入 | 30×2000，200 Hz，逐trial全局z-score |
| 模型 | EEGNet-8,2，F1=8，D=2，F2=16 |
| 首层时间核 | {summary['model']['kernel_length']}（200 Hz下约0.5秒） |
| 参数量 | {summary['model']['parameters']} |
| 初始化 | 随机初始化，无预训练权重 |
| Epoch | {summary['schedule']['epochs']} |
| Batch size | {summary['schedule']['batch_size']} |
| 学习率 | {summary['schedule']['lr']} |
| Weight decay | {summary['schedule']['weight_decay']} |
| Dropout | {summary['schedule']['dropout']} |

## 结果

| 检查点 | Epoch | Val Acc | Test Acc | Test Macro-F1 | Test Kappa |
|---|---:|---:|---:|---:|---:|
| 最佳验证模型 | {best['epoch']} | {best['val']['accuracy']:.4f} | {test['accuracy']:.4f} | {test['f1_macro']:.4f} | {test['cohen_kappa']:.4f} |
| 最终模型 | {final['epoch']} | {final['val']['accuracy']:.4f} | {final['test']['accuracy']:.4f} | {final['test']['f1_macro']:.4f} | {final['test']['cohen_kappa']:.4f} |
"""


def main() -> None:
    args = arguments()
    if not (args.data_root / "dataset_description.json").is_file():
        raise FileNotFoundError(f"Not a SHIN BIDS root: {args.data_root}")
    seed_all(args.seed)
    task = TASKS[args.task]
    subjects = {
        "train": parse_subjects(args.train_subjects),
        "val": parse_subjects(args.val_subjects),
        "test": parse_subjects(args.test_subjects),
    }
    flat = [subject for values in subjects.values() for subject in values]
    if len(flat) != len(set(flat)):
        raise ValueError("Train/val/test subjects overlap")
    if args.max_subjects_per_split:
        subjects = {
            name: values[:args.max_subjects_per_split]
            for name, values in subjects.items()
        }

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = args.output_root / (
            f"{stamp}_{args.task}_eegnet82_ep{args.epochs}_lr{args.lr:g}_seed{args.seed}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    cache_dir = args.cache_dir or (args.output_root / "cache")

    arrays, labels, details = {}, {}, {}
    for name in ("train", "val", "test"):
        arrays[name], labels[name], details[name] = load_split(
            args.data_root, name, subjects[name], cache_dir, args.task, task
        )

    input_channels, input_samples = arrays["train"].shape[1:]
    output_classes = int(max(labels[name].max() for name in labels)) + 1
    model = EEGNet(
        channels=input_channels,
        samples=input_samples,
        classes=output_classes,
        dropout=args.dropout,
        kernel_length=args.kernel_length,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model.to(device)
    with torch.no_grad():
        probe = torch.from_numpy(arrays["train"][:2]).to(device).float()
        logits = model(probe)
    diagnostics = {
        "model_identity": "EEGNet-8,2 PyTorch port of official ARL architecture",
        "official_repository": "https://github.com/vlawhern/arl-eegmodels",
        "official_commit": OFFICIAL_COMMIT,
        "official_runtime": "TensorFlow/Keras 2.0-2.3",
        "adapted_runtime": {
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "pretrained_weights": None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "feature_dim": model.feature_dim,
        "input_shape": [input_channels, input_samples],
        "sampling_rate_hz": args.sampling_rate,
        "normalization": "physical microvolts, then per-trial global z-score",
        "kernel_length": args.kernel_length,
        "task": {"key": args.task, **task},
        "seed": args.seed,
        "splits": {
            name: {
                "subjects": subjects[name],
                "shape": list(arrays[name].shape),
                "label_counts": dict(Counter(map(int, labels[name]))),
                "details": details[name],
            }
            for name in arrays
        },
        "forward_check": {
            "input": list(probe.shape),
            "logits": list(logits.shape),
            "finite": bool(torch.isfinite(logits).all()),
        },
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    print(json.dumps(diagnostics["forward_check"], ensure_ascii=False), flush=True)
    print(f"parameters={diagnostics['parameter_count']}, feature_dim={model.feature_dim}", flush=True)
    if args.diagnose_only:
        print(f"Diagnostics written to {output_dir}", flush=True)
        return

    loaders = {
        name: loader(
            arrays[name], labels[name], args.batch_size, name == "train",
            args.num_workers, args.seed,
        )
        for name in arrays
    }
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history, best, best_accuracy = [], None, -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for step, (x, y) in enumerate(loaders["train"], 1):
            if args.max_batches and step > args.max_batches:
                break
            x, y = x.to(device).float(), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            model.constrain_weights()
            loss_sum += float(loss.item()) * len(y)
            seen += len(y)
        val = evaluate(model, loaders["val"], device, criterion, args.max_batches)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "val": val,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={row['train_loss']:.5f} "
            f"val_acc={val['accuracy']:.4f} val_f1={val['f1_macro']:.4f}",
            flush=True,
        )
        if val["accuracy"] > best_accuracy:
            best_accuracy, best = val["accuracy"], row
            torch.save({"model": model.state_dict(), "record": row}, output_dir / "best.pt")
        write_history(output_dir / "history.csv", history)

    final_test = evaluate(model, loaders["test"], device, criterion, args.max_batches)
    final = {"epoch": args.epochs, "val": history[-1]["val"], "test": final_test}
    torch.save({"model": model.state_dict(), "record": final}, output_dir / "last.pt")
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    best_test = evaluate(model, loaders["test"], device, criterion, args.max_batches)
    assert best is not None
    summary = {
        "model_identity": "EEGNet-8,2 official-architecture PyTorch port",
        "run_finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "task": {"key": args.task, "name": task["name"], "description": task["description"]},
        "seed": args.seed,
        "split": subjects,
        "model": {
            "parameters": diagnostics["parameter_count"],
            "kernel_length": args.kernel_length,
            "f1": 8,
            "depth_multiplier": 2,
            "f2": 16,
            "pretrained": False,
        },
        "schedule": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
        },
        "best": best,
        "best_test": best_test,
        "final": final,
        "history": history,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "EXPERIMENT_RECORD.md").write_text(record(summary), encoding="utf-8")
    print(json.dumps({
        "output_dir": summary["output_dir"],
        "best_epoch": best["epoch"],
        "best_val_accuracy": best["val"]["accuracy"],
        "best_test": best_test,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
