"""Train the repository DAMFNet on aligned SHIN EEG/HbR windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
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
from torch.utils.data import DataLoader

from models.shin_damfnet import SHINDAMFNet
from shin_data import (
    DAMF_FIXED_EEG_INDICES,
    SENSOR_LAYOUTS,
    SHIN_EEG_CHANNELS,
    TASKS,
    WindowDataset,
    load_split,
)


OFFICIAL_COMMIT = "86ce33d4925d5e5603ccb2d7f4833d430f37ae2e"
DEFAULT_OUTPUT_ROOT = Path(r"D:\data\DAMFNet-SHIN")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAMFNet on aligned SHIN EEG/fNIRS")
    parser.add_argument("--eeg-root", type=Path, default=Path(r"D:\DataSets\SHIN\v1.0.1"))
    parser.add_argument("--fnirs-root", type=Path, default=Path(r"D:\DataSets\SHIN\NIRS_01-29"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--train-subjects", default="1-19")
    parser.add_argument("--val-subjects", default="20-24")
    parser.add_argument("--test-subjects", default="25-29")
    parser.add_argument(
        "--sensor-layout",
        choices=tuple(SENSOR_LAYOUTS),
        default="project_all",
    )
    parser.add_argument("--epoch-start-s", type=float, default=0.0)
    parser.add_argument("--epoch-stop-s", type=float, default=10.0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--window-stride-seconds", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation improvement; 0 disables.",
    )
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--loss-w-eeg", type=float, default=1.0)
    parser.add_argument("--loss-w-hbr", type=float, default=1.0)
    parser.add_argument("--loss-w-fuse", type=float, default=1.0)
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


def loader(
    dataset: WindowDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def aggregate_trials(
    logits: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    expected_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    grouped_logits: dict[int, list[np.ndarray]] = defaultdict(list)
    grouped_labels: dict[int, set[int]] = defaultdict(set)
    for logit, label, trial_id in zip(logits, labels, trial_ids):
        grouped_logits[int(trial_id)].append(logit)
        grouped_labels[int(trial_id)].add(int(label))
    true, prediction = [], []
    for trial_id in sorted(grouped_logits):
        if len(grouped_logits[trial_id]) != expected_windows:
            raise RuntimeError(
                f"Trial {trial_id} has {len(grouped_logits[trial_id])} windows, "
                f"expected {expected_windows}"
            )
        if len(grouped_labels[trial_id]) != 1:
            raise RuntimeError(f"Trial {trial_id} has inconsistent labels")
        true.append(next(iter(grouped_labels[trial_id])))
        prediction.append(np.mean(grouped_logits[trial_id], axis=0).argmax())
    return np.asarray(true), np.asarray(prediction)


def weighted_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    weights: tuple[float, float, float],
) -> torch.Tensor:
    eeg_logits, hbr_logits, fusion_logits = outputs
    return (
        weights[0] * criterion(eeg_logits, labels)
        + weights[1] * criterion(hbr_logits, labels)
        + weights[2] * criterion(fusion_logits, labels)
    )


@torch.no_grad()
def evaluate(
    model: SHINDAMFNet,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    weights: tuple[float, float, float],
    max_batches: int | None,
) -> dict:
    model.eval()
    expected_windows = data_loader.dataset.windows_per_trial
    loss_sum, seen = 0.0, 0
    logits_all, labels_all, trial_ids_all = [], [], []
    for step, (eeg, hbr, labels, trial_ids) in enumerate(data_loader, 1):
        if max_batches and step > max_batches:
            break
        eeg = eeg.to(device, non_blocking=True).float()
        hbr = hbr.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)
        outputs = model(eeg, hbr)
        loss = weighted_loss(outputs, labels, criterion, weights)
        loss_sum += float(loss.item()) * len(labels)
        seen += len(labels)
        logits_all.append(outputs[2].cpu().numpy())
        labels_all.append(labels.cpu().numpy())
        trial_ids_all.append(trial_ids.numpy())
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    trial_ids = np.concatenate(trial_ids_all)
    window_metrics = metric_values(labels, logits.argmax(1))
    if max_batches:
        # Smoke tests may contain incomplete trials. Trial metrics are only
        # valid when every configured window is present.
        counts = Counter(map(int, trial_ids))
        complete_ids = {
            trial_id
            for trial_id, count in counts.items()
            if count == expected_windows
        }
        mask = np.asarray([int(item) in complete_ids for item in trial_ids])
        if complete_ids:
            trial_true, trial_pred = aggregate_trials(
                logits[mask],
                labels[mask],
                trial_ids[mask],
                expected_windows,
            )
            trial_metrics = metric_values(trial_true, trial_pred)
        else:
            trial_metrics = None
    else:
        trial_true, trial_pred = aggregate_trials(
            logits,
            labels,
            trial_ids,
            expected_windows,
        )
        trial_metrics = metric_values(trial_true, trial_pred)
    return {
        "loss": loss_sum / seen,
        "window": window_metrics,
        "trial": trial_metrics,
        "windows": int(seen),
        "trials": int(len(set(map(int, trial_ids)))),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_window_accuracy",
            "val_window_macro_f1", "val_trial_accuracy", "val_trial_macro_f1",
            "elapsed_seconds",
        ])
        for row in history:
            val = row["val"]
            writer.writerow([
                row["epoch"],
                row["train_loss"],
                val["loss"],
                val["window"]["accuracy"],
                val["window"]["f1_macro"],
                val["trial"]["accuracy"] if val["trial"] else "",
                val["trial"]["f1_macro"] if val["trial"] else "",
                row["elapsed_seconds"],
            ])


def experiment_record(summary: dict) -> str:
    best, test, final = summary["best"], summary["best_test"], summary["final"]
    schedule = summary["schedule"]
    data = summary["data"]
    return f"""# DAMFNet × SHIN EEG–fNIRS 实验记录

## 实验身份

使用 `useflf/DAMFNet` 仓库的完整 EEG–HbR 双分支、空间/时间双节点融合和残差CTAM。{summary['model']['sensor_description']}其余融合主干保持官方张量结构。无预训练权重。

## 参数

| 参数 | 值 |
|---|---|
| 任务 | {summary['task']['name']}：{summary['task']['description']} |
| 划分 | 19/5/5：train 1–19 / val 20–24 / test 25–29 |
| Seed | {summary['seed']} |
| Trial | EEG/HbR严格按session及事件序列对齐；每trial产生{data['windows_per_trial']}个{data['window_seconds']:g}秒窗 |
| 窗口范围 | {data['window_intervals']} |
| EEG窗口 | {data['eeg_window_shape']}，200 Hz |
| HbR窗口 | {data['hbr_window_shape']}，10 Hz |
| fNIRS处理 | OD→MBLL→HbR；0.01–0.1 Hz；-5..-2秒基线；epoch {data['epoch_seconds']}秒 |
| 参数量 | {summary['model']['parameters']} |
| Epoch | {schedule['epochs']} |
| Early stopping | patience={schedule['patience']}；实际训练{schedule['trained_epochs']}轮 |
| Batch size | {schedule['batch_size']} |
| 学习率 | {schedule['lr']} |
| Weight decay | {schedule['weight_decay']} |
| Dropout | {schedule['dropout']} |
| 辅助损失权重 | EEG/HbR/Fusion = {schedule['loss_weights']} |

## Trial级结果

| 检查点 | Epoch | Val Acc | Test Acc | Test Macro-F1 | Test Kappa |
|---|---:|---:|---:|---:|---:|
| 最佳验证模型 | {best['epoch']} | {best['val']['trial']['accuracy']:.4f} | {test['trial']['accuracy']:.4f} | {test['trial']['f1_macro']:.4f} | {test['trial']['cohen_kappa']:.4f} |
| 最终模型 | {final['epoch']} | {final['val']['trial']['accuracy']:.4f} | {final['test']['trial']['accuracy']:.4f} | {final['test']['trial']['f1_macro']:.4f} | {final['test']['trial']['cohen_kappa']:.4f} |

最佳模型的window级测试准确率：{test['window']['accuracy']:.4f}。
"""


def main() -> None:
    args = arguments()
    # Dataset-specific loaders validate their own cache/raw layout.  The
    # shared DAMFNet runner only needs both supplied roots to exist; requiring
    # a BIDS dataset_description.json here wrongly rejects FineMI caches.
    if not args.eeg_root.is_dir():
        raise FileNotFoundError(f"Missing EEG data root: {args.eeg_root}")
    if not args.fnirs_root.is_dir():
        raise FileNotFoundError(f"Missing SHIN fNIRS root: {args.fnirs_root}")
    if abs(args.window_seconds - 3.0) > 1e-9:
        raise ValueError("The official DAMFNet core requires 3-second windows")
    if args.epoch_stop_s <= args.epoch_start_s:
        raise ValueError("epoch-stop-s must be greater than epoch-start-s")
    span_after_window = (
        args.epoch_stop_s - args.epoch_start_s - args.window_seconds
    )
    if span_after_window < 0 or abs(
        span_after_window / args.window_stride_seconds
        - round(span_after_window / args.window_stride_seconds)
    ) > 1e-9:
        raise ValueError(
            "Epoch, window and stride do not produce windows ending exactly at epoch-stop-s"
        )
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
            f"{stamp}_{args.task}_{args.sensor_layout}_damfnet_"
            f"ep{args.epochs}_lr{args.lr:g}_seed{args.seed}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    cache_dir = args.cache_dir or (args.output_root / "cache")

    arrays, details, datasets = {}, {}, {}
    for name in ("train", "val", "test"):
        eeg, hbr, labels, details[name] = load_split(
            args.eeg_root,
            args.fnirs_root,
            subjects[name],
            name,
            args.task,
            cache_dir,
            args.sensor_layout,
            args.epoch_start_s,
            args.epoch_stop_s,
        )
        arrays[name] = {"eeg": eeg, "hbr": hbr, "labels": labels}
        datasets[name] = WindowDataset(
            eeg,
            hbr,
            labels,
            epoch_start_s=args.epoch_start_s,
            window_seconds=args.window_seconds,
            stride_seconds=args.window_stride_seconds,
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = SHINDAMFNet(
        dropout=args.dropout,
        sensor_layout=args.sensor_layout,
    ).to(device)
    with torch.no_grad():
        sample_eeg, sample_hbr, _, _ = datasets["train"][0]
        outputs = model(
            sample_eeg.unsqueeze(0).to(device),
            sample_hbr.unsqueeze(0).to(device),
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    diagnostics = {
        "model_identity": (
            "useflf/DAMFNet with fixed SHIN DAMF 8/24 sensors"
            if args.sensor_layout == "damf_fixed"
            else "useflf/DAMFNet with SHIN learnable node projections"
        ),
        "repository": "https://github.com/useflf/DAMFNet",
        "official_commit": OFFICIAL_COMMIT,
        "pretrained_weights": None,
        "parameter_count": parameter_count,
        "environment": {
            "conda_env": "csbrain-bcic2a",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "task": {"key": args.task, **task},
        "seed": args.seed,
        "adaptation": {
            "sensor_layout": args.sensor_layout,
            "eeg_nodes": (
                "fixed 8 DAMF bilateral FCC/CCP sensors; no projection"
                if args.sensor_layout == "damf_fixed"
                else "30 -> learnable Linear -> 8 official virtual nodes"
            ),
            "hbr_nodes": (
                "fixed final 24 DAMF fNIRS nodes; no projection"
                if args.sensor_layout == "damf_fixed"
                else "36 -> learnable Linear -> 24 official virtual nodes"
            ),
            "epoch_seconds": [args.epoch_start_s, args.epoch_stop_s],
            "window_seconds": args.window_seconds,
            "stride_seconds": args.window_stride_seconds,
            "windows_per_trial": datasets["train"].windows_per_trial,
            "window_intervals": datasets["train"].window_intervals,
            "model_selection": (
                "trial-level validation accuracy from mean of all window logits"
            ),
        },
        "splits": {
            name: {
                "subjects": subjects[name],
                "trials": int(len(arrays[name]["labels"])),
                "windows": int(len(datasets[name])),
                "eeg_shape": list(arrays[name]["eeg"].shape),
                "hbr_shape": list(arrays[name]["hbr"].shape),
                "label_counts": dict(Counter(map(int, arrays[name]["labels"]))),
                "details": details[name],
            }
            for name in arrays
        },
        "forward_check": {
            "eeg": [1, *sample_eeg.shape],
            "hbr": [1, *sample_hbr.shape],
            "outputs": [list(output.shape) for output in outputs],
            "finite": all(bool(torch.isfinite(output).all()) for output in outputs),
        },
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    print(json.dumps(diagnostics["forward_check"], ensure_ascii=False), flush=True)
    print(f"parameters={parameter_count}", flush=True)
    if args.diagnose_only:
        print(f"Diagnostics written to {output_dir}", flush=True)
        return

    loaders = {
        name: loader(
            datasets[name],
            args.batch_size,
            name == "train",
            args.num_workers,
            args.seed,
        )
        for name in datasets
    }
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    weights = (args.loss_w_eeg, args.loss_w_hbr, args.loss_w_fuse)
    history, best, best_accuracy, epochs_without_improvement = [], None, -1.0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for step, (eeg, hbr, labels, _) in enumerate(loaders["train"], 1):
            if args.max_batches and step > args.max_batches:
                break
            eeg = eeg.to(device, non_blocking=True).float()
            hbr = hbr.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(eeg, hbr)
            loss = weighted_loss(outputs, labels, criterion, weights)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            seen += len(labels)
        val = evaluate(
            model, loaders["val"], device, criterion, weights, args.max_batches
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "val": val,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        selection_accuracy = (
            val["trial"]["accuracy"]
            if val["trial"] is not None
            else val["window"]["accuracy"]
        )
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={row['train_loss']:.5f} "
            f"val_window_acc={val['window']['accuracy']:.4f} "
            f"val_trial_acc={selection_accuracy:.4f}",
            flush=True,
        )
        if selection_accuracy > best_accuracy:
            best_accuracy, best = selection_accuracy, row
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "record": row}, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        write_history(output_dir / "history.csv", history)
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"early_stop epoch={epoch} patience={args.patience} "
                f"best_epoch={best['epoch']}",
                flush=True,
            )
            break

    final_test = evaluate(model, loaders["test"], device, criterion, weights, None)
    final = {"epoch": history[-1]["epoch"], "val": history[-1]["val"], "test": final_test}
    torch.save({"model": model.state_dict(), "record": final}, output_dir / "last.pt")
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    best_test = evaluate(model, loaders["test"], device, criterion, weights, None)
    assert best is not None
    summary = {
        "model_identity": "useflf/DAMFNet SHIN EEG-HbR fusion",
        "run_finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "task": {"key": args.task, "name": task["name"], "description": task["description"]},
        "seed": args.seed,
        "split": subjects,
        "model": {
            "parameters": parameter_count,
            "pretrained": False,
            "sensor_layout": args.sensor_layout,
            "eeg_node_projection": (
                None if args.sensor_layout == "damf_fixed" else "30->8"
            ),
            "hbr_node_projection": (
                None if args.sensor_layout == "damf_fixed" else "36->24"
            ),
            "sensor_description": (
                "SHIN直接选取与合作版DAMFNet一致的8个EEG通道和24个HbR节点，"
                "不使用节点投影并放弃其余通道。"
                if args.sensor_layout == "damf_fixed"
                else (
                    "SHIN的30个EEG电极和36个fNIRS节点通过可学习线性层"
                    "投影到官方公开数据配置的8/24节点。"
                )
            ),
        },
        "data": {
            "sensor_layout": args.sensor_layout,
            "eeg_channels": (
                [SHIN_EEG_CHANNELS[index] for index in DAMF_FIXED_EEG_INDICES]
                if args.sensor_layout == "damf_fixed"
                else SHIN_EEG_CHANNELS
            ),
            "hbr_channels": (
                "fixed source-detector indices 12..35"
                if args.sensor_layout == "damf_fixed"
                else "all 36 source-detector nodes"
            ),
            "epoch_seconds": [args.epoch_start_s, args.epoch_stop_s],
            "window_seconds": args.window_seconds,
            "stride_seconds": args.window_stride_seconds,
            "windows_per_trial": datasets["train"].windows_per_trial,
            "window_intervals": datasets["train"].window_intervals,
            "eeg_window_shape": (
                f"{sample_eeg.shape[0]}×{sample_eeg.shape[1]}"
            ),
            "hbr_window_shape": (
                f"{sample_hbr.shape[0]}×{sample_hbr.shape[1]}"
            ),
        },
        "schedule": {
            "epochs": args.epochs,
            "trained_epochs": history[-1]["epoch"],
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "loss_weights": list(weights),
        },
        "best": best,
        "best_test": best_test,
        "final": final,
        "history": history,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "EXPERIMENT_RECORD.md").write_text(
        experiment_record(summary),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": summary["output_dir"],
        "best_epoch": best["epoch"],
        "best_val_trial_accuracy": best["val"]["trial"]["accuracy"],
        "best_test_trial": best_test["trial"],
        "best_test_window": best_test["window"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
