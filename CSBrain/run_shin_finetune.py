"""Fine-tune the official CSBrain foundation model on SHIN EEG MI or MA."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from models.model_for_shin import Model, SHIN_BRAIN_REGIONS, SHIN_ELECTRODES


OFFICIAL_REPO = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = OFFICIAL_REPO / "pth" / "CSBrain.pth"
DEFAULT_OUTPUT_ROOT = Path(r"D:\data\CSBrain-Official-SHIN")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official CSBrain fine-tuning on SHIN EEG")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\DataSets\SHIN\v1.0.1"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--foundation-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--train-subjects", default="1-19")
    parser.add_argument("--val-subjects", default="20-24")
    parser.add_argument("--test-subjects", default="25-29")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--unfreeze-epoch", type=int, default=91)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-subjects-per-split", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def parse_subjects(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            result.extend(range(start, stop + 1))
        elif part:
            result.append(int(part))
    if not result or len(result) != len(set(result)):
        raise ValueError(f"Invalid or duplicate subject list: {value}")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def exactly_one(folder: Path, pattern: str) -> Path:
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"{folder}: expected one {pattern}, found {len(paths)}")
    return paths[0]


def load_subject(root: Path, subject: int, task: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    import mne

    trials: list[np.ndarray] = []
    labels: list[int] = []
    session_details: list[dict] = []
    for session in task["sessions"]:
        eeg_dir = root / f"sub-{subject}" / session / "eeg"
        bdf_path = exactly_one(eeg_dir, "*_eeg.bdf")
        events_path = exactly_one(eeg_dir, "*_events.tsv")
        channels_path = exactly_one(eeg_dir, "*_channels.tsv")
        eeg_names = [
            row["name"] for row in read_tsv(channels_path)
            if row.get("type", "").upper() == "EEG"
        ]
        if eeg_names != SHIN_ELECTRODES:
            raise RuntimeError(
                f"Unexpected EEG channel order in {channels_path}: {eeg_names}"
            )

        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
        if abs(float(raw.info["sfreq"]) - 200.0) > 1e-6:
            raise RuntimeError(f"Expected 200 Hz, got {raw.info['sfreq']} in {bdf_path}")
        continuous = np.nan_to_num(
            raw.get_data(picks=SHIN_ELECTRODES, units="uV"),
            copy=False,
        )
        raw.close()

        counts: Counter[str] = Counter()
        for event in read_tsv(events_path):
            label_name = event.get("trial_type", "")
            if label_name not in task["labels"]:
                continue
            start = int(event["sample"])
            trial = continuous[:, start:start + 2000].astype(np.float32, copy=True)
            if trial.shape != (30, 2000):
                raise RuntimeError(f"Bad trial shape {trial.shape} in {events_path}")
            mean = float(trial.mean(dtype=np.float64))
            std = float(trial.std(dtype=np.float64))
            if not np.isfinite(std) or std < 1e-6:
                raise RuntimeError(f"Invalid trial std={std} in {bdf_path}")
            normalized = (trial - mean) / std
            trials.append(normalized.reshape(30, 10, 200).astype(np.float32))
            labels.append(task["labels"][label_name])
            counts[label_name] += 1

        expected = Counter({name: 10 for name in task["labels"]})
        if counts != expected:
            raise RuntimeError(
                f"Unexpected event counts in {events_path}: {dict(counts)}, expected {dict(expected)}"
            )
        session_details.append({
            "session": session,
            "trials": sum(counts.values()),
            "label_counts": dict(counts),
        })

    return (
        np.stack(trials),
        np.asarray(labels, dtype=np.int64),
        {"subject": subject, "trials": len(labels), "sessions": session_details},
    )


def load_split(
    root: Path,
    split_name: str,
    subjects: list[int],
    cache_dir: Path,
    task_key: str,
    task: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    subject_tag = "-".join(map(str, subjects))
    cache_path = cache_dir / (
        f"{task_key}_{split_name}_sub-{subject_tag}_30ch_10x200_global-zscore.npz"
    )
    if cache_path.is_file():
        item = np.load(cache_path, allow_pickle=False)
        x, y = item["X"], item["y"]
        print(f"[{split_name}] cache {cache_path}: X={x.shape}, y={Counter(y.tolist())}", flush=True)
        return x, y, [{"subject": item, "source": "cache"} for item in subjects]

    arrays: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    details: list[dict] = []
    for subject in subjects:
        x, y, detail = load_subject(root, subject, task)
        arrays.append(x)
        targets.append(y)
        details.append(detail)
        print(f"[{split_name}] sub-{subject}: X={x.shape}, y={Counter(y.tolist())}", flush=True)
    x_all = np.concatenate(arrays)
    y_all = np.concatenate(targets)
    np.savez(cache_path, X=x_all, y=y_all)
    print(f"[{split_name}] wrote cache {cache_path}", flush=True)
    return x_all, y_all, details


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def cache_backbone_features(
    model: Model,
    loaders: dict[str, DataLoader],
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> dict[str, DataLoader]:
    model.backbone.eval()
    result: dict[str, DataLoader] = {}
    for split_name, loader in loaders.items():
        features: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for step, (x, y) in enumerate(loader, 1):
            encoded = model.encode(x.to(device, non_blocking=True).float())
            features.append(encoded.flatten(1).cpu())
            targets.append(y)
            if step % 20 == 0 or step == len(loader):
                print(f"[feature cache] {split_name}: {step}/{len(loader)}", flush=True)
        dataset = TensorDataset(torch.cat(features), torch.cat(targets))
        result[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(seed) if split_name == "train" else None,
        )
    return result


def metrics(y_true: np.ndarray, y_pred: np.ndarray, loss: float) -> dict:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def evaluate(
    model: Model,
    loader: DataLoader,
    device: torch.device,
    feature_mode: bool,
    criterion: nn.Module,
    max_batches: int | None,
) -> dict:
    model.eval()
    total_loss = 0.0
    seen = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for step, (x, y) in enumerate(loader, 1):
        if max_batches and step > max_batches:
            break
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        logits = model.classifier(x) if feature_mode else model(x)
        total_loss += float(criterion(logits, y).item()) * len(y)
        seen += len(y)
        predictions.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    return metrics(
        np.concatenate(targets),
        np.concatenate(predictions),
        total_loss / seen,
    )


def train_epoch(
    model: Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    feature_mode: bool,
    criterion: nn.Module,
    max_batches: int | None,
) -> float:
    model.classifier.train() if feature_mode else model.train()
    if feature_mode:
        model.backbone.eval()
    total_loss = 0.0
    seen = 0
    for step, (x, y) in enumerate(loader, 1):
        if max_batches and step > max_batches:
            break
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model.classifier(x) if feature_mode else model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for group in optimizer.param_groups for parameter in group["params"]),
            max_norm=1.0,
        )
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        seen += len(y)
    return total_loss / seen


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "stage", "train_loss", "val_loss", "val_accuracy",
            "val_balanced_accuracy", "val_macro_f1", "val_kappa", "elapsed_seconds",
        ])
        for row in history:
            val = row["val"]
            writer.writerow([
                row["epoch"], row["stage"], row["train_loss"], val["loss"],
                val["accuracy"], val["balanced_accuracy"], val["f1_macro"],
                val["cohen_kappa"], row["elapsed_seconds"],
            ])


def experiment_record(summary: dict) -> str:
    best = summary["best"]
    test = summary["best_test"]
    final = summary["final"]
    schedule = summary["schedule"]
    return f"""# 官方 CSBrain × SHIN EEG 实验记录

## 实验身份

本实验使用 yuchen2199/CSBrain 官方代码和官方基础预训练权重，并新增 SHIN 30通道脑区拓扑。模型不是此前的 CSBrain-compatible CNN fallback。

## 参数

| 参数 | 值 |
|---|---|
| 任务 | {summary['task']['name']}：{summary['task']['description']} |
| 随机种子 | {summary['seed']} |
| 受试者划分 | train 1–19 / val 20–24 / test 25–29 |
| 输入 | 30×10×200，200 Hz，逐 trial 全局 z-score |
| 主干 | 官方 CSBrain，12层，d_model=200，8头 |
| 主干初始化 | 官方 CSBrain.pth；匹配 {summary['pretrained']['matched_tensor_count']}/{summary['pretrained']['checkpoint_tensor_count']} tensors |
| 分类头 | 官方30通道10-patch三层MLP：60000→2000→200→2 |
| Epoch | {schedule['epochs']} |
| Batch size | {schedule['batch_size']} |
| 分类头学习率 | {schedule['head_lr']} |
| 主干学习率 | {schedule['backbone_lr']} |
| Weight decay | {schedule['weight_decay']} |
| 解冻 | 第 {schedule['unfreeze_epoch']} epoch，仅最后 {schedule['epochs'] - schedule['unfreeze_epoch'] + 1} epoch |
| 提前解冻实验 | 未进行 |

## 结果

| 检查点 | Epoch | 验证准确率 | 验证 Macro-F1 | 测试准确率 | 测试 Macro-F1 | 测试 Kappa |
|---|---:|---:|---:|---:|---:|---:|
| 最佳验证模型 | {best['epoch']} | {best['val']['accuracy']:.4f} | {best['val']['f1_macro']:.4f} | {test['accuracy']:.4f} | {test['f1_macro']:.4f} | {test['cohen_kappa']:.4f} |
| 最后一轮模型 | {final['epoch']} | {final['val']['accuracy']:.4f} | {final['val']['f1_macro']:.4f} | {final['test']['accuracy']:.4f} | {final['test']['f1_macro']:.4f} | {final['test']['cohen_kappa']:.4f} |

最佳模型测试混淆矩阵：`{test['confusion_matrix']}`
"""


def update_results_page(output_root: Path) -> None:
    summaries: list[dict] = []
    for path in output_root.glob("*/summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("model_identity") == "official CSBrain":
                summary["_directory"] = str(path.parent)
                summaries.append(summary)
        except (OSError, json.JSONDecodeError):
            continue
    summaries.sort(key=lambda item: item.get("run_started", ""))
    lines = [
        "# 官方 CSBrain × SHIN 独立结果页",
        "",
        "本页只收录官方 CSBrain 主干和官方预训练权重实验；旧的 CNN fallback 不在此表中。",
        "固定协议：seed=1，19/5/5受试者划分，100 epoch，head lr=1e-4，backbone lr=1e-5，第91轮解冻。",
        "",
        "| 日期 | 任务 | Seed | 划分 | 最佳Epoch | 最佳Val Acc | Test Acc | Test Macro-F1 | Test Kappa | 结果目录 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        test = item["best_test"]
        lines.append(
            f"| {item['run_started'][:10]} | {item['task']['name']} | {item['seed']} | "
            f"19/5/5 | {item['best']['epoch']} | {item['best']['val']['accuracy']:.4f} | "
            f"{test['accuracy']:.4f} | {test['f1_macro']:.4f} | "
            f"{test['cohen_kappa']:.4f} | `{item['_directory']}` |"
        )
    (output_root / "官方CSBrain_SHIN_结果表.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 1 <= args.unfreeze_epoch <= args.epochs:
        raise ValueError("--unfreeze-epoch must be within 1..epochs")
    if not args.data_root.joinpath("dataset_description.json").is_file():
        raise FileNotFoundError(f"Not a SHIN BIDS root: {args.data_root}")
    if not args.foundation_dir.is_file():
        raise FileNotFoundError(f"Missing foundation checkpoint: {args.foundation_dir}")

    seed_everything(args.seed)
    split_subjects = {
        "train": parse_subjects(args.train_subjects),
        "val": parse_subjects(args.val_subjects),
        "test": parse_subjects(args.test_subjects),
    }
    flat = [subject for values in split_subjects.values() for subject in values]
    if len(flat) != len(set(flat)):
        raise ValueError("Train/val/test subjects must be disjoint")
    if args.max_subjects_per_split:
        split_subjects = {
            name: values[:args.max_subjects_per_split]
            for name, values in split_subjects.items()
        }

    run_started = datetime.now().astimezone().isoformat(timespec="seconds")
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = args.output_root / (
            f"{stamp}_{args.task}_official_ep{args.epochs}_headlr1e-4_"
            f"backbonelr1e-5_unfreeze{args.unfreeze_epoch}_seed{args.seed}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    cache_dir = args.cache_dir or (args.output_root / "cache")

    task = TASKS[args.task]
    arrays: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    split_details: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        arrays[split_name], targets[split_name], split_details[split_name] = load_split(
            args.data_root,
            split_name,
            split_subjects[split_name],
            cache_dir,
            args.task,
            task,
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in csbrain-bcic2a")
    model = Model(
        args.foundation_dir,
        num_classes=2,
        n_layer=12,
        dropout=args.dropout,
    )
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "backbone": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "classifier": sum(parameter.numel() for parameter in model.classifier.parameters()),
    }
    diagnostics = {
        "model_identity": "official CSBrain",
        "official_repository": "https://github.com/yuchen2199/CSBrain",
        "official_commit": "185aee55b24d0410a830df8dd08d03f675616998",
        "foundation_checkpoint": str(args.foundation_dir.resolve()),
        "foundation_sha256": file_sha256(args.foundation_dir),
        "pretrained": model.pretrained_report,
        "parameter_counts": parameter_counts,
        "environment": {
            "conda_env": "csbrain-bcic2a",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
        },
        "task": {"key": args.task, **task},
        "seed": args.seed,
        "input_shape": [30, 10, 200],
        "sampling_rate_hz": 200,
        "normalization": "physical microvolts, then per-trial global z-score",
        "electrodes": SHIN_ELECTRODES,
        "brain_regions": SHIN_BRAIN_REGIONS,
        "splits": {
            name: {
                "subjects": split_subjects[name],
                "shape": list(arrays[name].shape),
                "label_counts": dict(Counter(map(int, targets[name]))),
                "details": split_details[name],
            }
            for name in arrays
        },
    }
    write_json(output_dir / "diagnostics.json", diagnostics)

    model.to(device)
    with torch.no_grad():
        probe = torch.from_numpy(arrays["train"][:1]).to(device).float()
        probe_logits = model(probe)
    diagnostics["forward_check"] = {
        "input": list(probe.shape),
        "logits": list(probe_logits.shape),
        "finite": bool(torch.isfinite(probe_logits).all()),
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    print(json.dumps(diagnostics["forward_check"], ensure_ascii=False), flush=True)
    print(json.dumps(model.pretrained_report, ensure_ascii=False, indent=2), flush=True)
    if args.diagnose_only:
        print(f"Diagnostics written to {output_dir}", flush=True)
        return

    raw_loaders = {
        name: make_loader(
            arrays[name], targets[name], args.batch_size, name == "train",
            args.seed, args.num_workers,
        )
        for name in arrays
    }
    model.backbone.requires_grad_(False)
    feature_loaders = cache_backbone_features(
        model, raw_loaders, device, args.batch_size, args.num_workers, args.seed
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr, "name": "backbone"},
            {"params": model.classifier.parameters(), "lr": args.head_lr, "name": "classifier"},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history: list[dict] = []
    best_record: dict | None = None
    best_accuracy = -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            model.backbone.requires_grad_(True)
            print(f"[epoch {epoch}] unfroze official CSBrain backbone", flush=True)
        feature_mode = epoch < args.unfreeze_epoch
        active_loaders = feature_loaders if feature_mode else raw_loaders
        train_loss = train_epoch(
            model, active_loaders["train"], optimizer, device,
            feature_mode, criterion, args.max_batches,
        )
        val = evaluate(
            model, active_loaders["val"], device,
            feature_mode, criterion, args.max_batches,
        )
        record = {
            "epoch": epoch,
            "stage": "frozen_backbone" if feature_mode else "fine_tune",
            "train_loss": train_loss,
            "val": val,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs} stage={record['stage']} "
            f"loss={train_loss:.5f} val_acc={val['accuracy']:.4f} "
            f"val_f1={val['f1_macro']:.4f}",
            flush=True,
        )
        if val["accuracy"] > best_accuracy:
            best_accuracy = val["accuracy"]
            best_record = record
            torch.save(
                {"model": model.state_dict(), "record": record},
                output_dir / "best.pt",
            )
        write_history(output_dir / "history.csv", history)

    final_test = evaluate(
        model, raw_loaders["test"], device, False, criterion, args.max_batches
    )
    final = {"epoch": args.epochs, "val": history[-1]["val"], "test": final_test}
    torch.save(
        {"model": model.state_dict(), "record": final},
        output_dir / "last.pt",
    )

    best_checkpoint = torch.load(
        output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    best_test = evaluate(
        model, raw_loaders["test"], device, False, criterion, args.max_batches
    )
    assert best_record is not None
    summary = {
        "model_identity": "official CSBrain",
        "run_started": run_started,
        "run_finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "task": {"key": args.task, "name": task["name"], "description": task["description"]},
        "seed": args.seed,
        "split": split_subjects,
        "pretrained": model.pretrained_report,
        "parameters": parameter_counts,
        "head": "Linear(60000,2000)-ELU-Dropout-Linear(2000,200)-ELU-Dropout-Linear(200,2)",
        "schedule": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "unfreeze_epoch": args.unfreeze_epoch,
            "early_unfreeze_experiment": False,
        },
        "best": best_record,
        "best_test": best_test,
        "final": final,
        "history": history,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "EXPERIMENT_RECORD.md").write_text(
        experiment_record(summary),
        encoding="utf-8",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    update_results_page(args.output_root)
    print(json.dumps({
        "output_dir": summary["output_dir"],
        "best_epoch": best_record["epoch"],
        "best_val_accuracy": best_record["val"]["accuracy"],
        "best_test": best_test,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
