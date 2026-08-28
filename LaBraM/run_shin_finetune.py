"""Fine-tune LaBraM on the EEG part of the SHIN dataset.

The protocol mirrors the local CodeBrain experiment: subject-independent split,
100 epochs, head/backbone learning rates of 1e-4/1e-5, and backbone unfreezing
for the final ten epochs.  All run artifacts include a Chinese experiment log.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from timm.models import create_model
from torch.utils.data import DataLoader, TensorDataset

import modeling_finetune  # noqa: F401 - register LaBraM models in timm
import utils


TASKS = {
    "mi": {
        "name": "EEG-MI", "description": "left_hand (0) vs right_hand (1)",
        "sessions": ("ses-0imagery", "ses-2imagery", "ses-4imagery"),
        "labels": {"left_hand": 0, "right_hand": 1},
    },
    "ma": {
        "name": "EEG-MA", "description": "subtraction (0) vs rest (1)",
        "sessions": ("ses-1arithmetic", "ses-3arithmetic", "ses-5arithmetic"),
        "labels": {"subtraction": 0, "rest": 1},
    },
}
SHIN_CHANNELS = [
    "F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8",
    "AFF1h", "AFF2h", "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h",
    "CCP3h", "T7", "P7", "P3", "PPO1h", "POO1", "POO2", "PPO2h",
    "P4", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "P8", "T8",
]
LABRAM_CHANNELS = [
    "F7", "AF5", "F3", "FP1", "FP2", "AF6", "F4", "F8", "AF1",
    "AF2", "CZ", "PZ", "FC5", "FC3", "CP5", "CP3", "T7", "P7",
    "P3", "PO1", "O1", "O2", "PO2", "P4", "FC4", "FC6", "CP4",
    "CP6", "P8", "T8",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaBraM + SHIN EEG 二分类微调")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\DataSets\SHIN\v1.0.1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/labram-base.pth"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:\data\LaBraM-SHIN\cache"))
    parser.add_argument("--task", choices=tuple(TASKS), default="mi")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=91)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--experiment-note", default=None)
    parser.add_argument("--max-subjects-per-split", type=int, default=None,
                        help="仅用于快速冒烟测试；正式实验保持为空。")
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


def subject_ids(start: int, stop: int, limit: int | None) -> list[int]:
    ids = list(range(start, stop + 1))
    return ids[:limit] if limit else ids


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _single(folder: Path, pattern: str) -> Path:
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"期望 {folder} 中有且仅有一个 {pattern}，实际为 {len(paths)}")
    return paths[0]


def _load_subject(subject: int, task: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    # Lazy import keeps syntax/argument checks usable even when MNE is unavailable.
    import mne

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    run_info = []
    for session in task["sessions"]:
        eeg_dir = ACTIVE_DATA_ROOT / f"sub-{subject:02d}" / session / "eeg"
        bdf_path = _single(eeg_dir, "*_eeg.bdf")
        events_path = _single(eeg_dir, "*_events.tsv")
        channels_path = _single(eeg_dir, "*_channels.tsv")

        channel_rows = _read_tsv(channels_path)
        eeg_names = [row["name"] for row in channel_rows if row.get("type", "").upper() == "EEG"]
        if eeg_names != SHIN_CHANNELS:
            raise RuntimeError(f"{channels_path} 的 EEG 通道顺序与预期不一致：{eeg_names}")

        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        if abs(sfreq - 200.0) > 1e-6:
            raise RuntimeError(f"{bdf_path} 采样率为 {sfreq} Hz，当前适配器要求 200 Hz")
        data_uv = raw.get_data(picks=SHIN_CHANNELS, units="uV")
        raw.close()
        data_uv = np.nan_to_num(data_uv, copy=False)

        counts = Counter()
        for event in _read_tsv(events_path):
            trial_type = event.get("trial_type", "")
            if trial_type not in task["labels"]:
                continue
            start = int(event["sample"])
            stop = start + 2000
            if stop > data_uv.shape[1]:
                raise RuntimeError(f"{events_path} 中 trial 越界：sample={start}")
            all_x.append(data_uv[:, start:stop].astype(np.float32, copy=True))
            all_y.append(task["labels"][trial_type])
            counts[trial_type] += 1
        expected = Counter({label: 10 for label in task["labels"]})
        if counts != expected:
            raise RuntimeError(f"{events_path} 标签数量异常：{dict(counts)}")
        run_info.append({"session": session, "trials": sum(counts.values()), "labels": dict(counts)})

    return np.stack(all_x), np.asarray(all_y, dtype=np.int64), {
        "subject": subject, "trials": len(all_y), "runs": run_info
    }


def load_split(name: str, subjects: list[int], cache_dir: Path, task_key: str,
               task: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(map(str, subjects))
    cache_path = cache_dir / f"{task_key}_{name}_sub-{tag}_30ch_10s_200hz.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        info = [{"subject": s, "trials": 60, "source": "cache"} for s in subjects]
        print(f"[{name}] 使用缓存 {cache_path}，shape={cached['X'].shape}", flush=True)
        return cached["X"], cached["y"], info

    arrays, labels, info = [], [], []
    for subject in subjects:
        x, y, subject_info = _load_subject(subject, task)
        arrays.append(x)
        labels.append(y)
        info.append(subject_info)
        print(f"[{name}] sub-{subject}: {len(y)} trials", flush=True)
    X = np.concatenate(arrays)
    y = np.concatenate(labels)
    np.savez(cache_path, X=X, y=y)
    print(f"[{name}] 缓存写入 {cache_path}，shape={X.shape}", flush=True)
    return X, y, info


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool,
                num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def load_pretrained(model: torch.nn.Module, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint.get("module", checkpoint))
    source = {key[8:] if key.startswith("student.") else key: value for key, value in source.items()}
    target = model.state_dict()
    compatible = {}
    skipped = {}
    for key, value in source.items():
        if "relative_position_index" in key:
            skipped[key] = "relative_position_index"
        elif key not in target:
            skipped[key] = "目标模型无此参数"
        elif target[key].shape != value.shape:
            skipped[key] = f"形状 {tuple(value.shape)} -> {tuple(target[key].shape)}"
        else:
            compatible[key] = value
    result = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "loaded_keys": len(compatible),
        "source_keys": len(source),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "skipped_keys": skipped,
    }


def prepare_x(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    # LaBraM's official downstream adapters expect physical microvolts / 100.
    x = x.to(device, non_blocking=True).float().div_(100.0)
    return x.reshape(x.shape[0], x.shape[1], 10, 200)


@torch.no_grad()
def cache_features(model: torch.nn.Module, loader: DataLoader, input_chans: list[int],
                   device: torch.device, label: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features, targets = [], []
    for step, (x, y) in enumerate(loader, start=1):
        feat = model.forward_features(prepare_x(x, device), input_chans=input_chans)
        features.append(feat.cpu().numpy().astype(np.float32))
        targets.append(y.numpy())
        if step % 25 == 0 or step == len(loader):
            print(f"[特征缓存] {label}: {step}/{len(loader)}", flush=True)
    return np.concatenate(features), np.concatenate(targets)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, loss: float) -> dict:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, input_chans: list[int],
             device: torch.device, feature_mode: bool) -> dict:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, seen, predictions, labels = 0.0, 0, [], []
    for x, y in loader:
        y = y.to(device, non_blocking=True)
        logits = model.head(x.to(device, non_blocking=True).float()) if feature_mode else model(
            prepare_x(x, device), input_chans=input_chans
        )
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * len(y)
        seen += len(y)
        predictions.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return metric_dict(np.concatenate(labels), np.concatenate(predictions), total_loss / seen)


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                input_chans: list[int], device: torch.device, feature_mode: bool) -> float:
    model.head.train() if feature_mode else model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, seen = 0.0, 0
    for x, y in loader:
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model.head(x.to(device, non_blocking=True).float()) if feature_mode else model(
            prepare_x(x, device), input_chans=input_chans
        )
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        seen += len(y)
    return total_loss / seen


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def report_markdown(args: argparse.Namespace, diagnostics: dict, summary: dict) -> str:
    best = summary["best"]
    final = summary["final"]
    test = summary["best_test"]
    task = TASKS[args.task]
    return f"""# LaBraM × SHIN EEG 实验记录

## 实验思路

{args.experiment_note}

SHIN 的完整 10 秒 EEG trial 保留为 30×2000；因采样率正好为 200 Hz，直接重排为 30×10×200 输入 LaBraM。模型专用幅值归一化采用物理微伏除以 100。前 {args.unfreeze_epoch - 1} epoch 冻结骨干并缓存特征，只训练线性分类头；从第 {args.unfreeze_epoch} epoch 起解冻全模型，共微调 {args.epochs - args.unfreeze_epoch + 1} epoch。

## 参数

| 参数 | 值 |
|---|---:|
| 模型 | LaBraM Base（patch 200） |
| 数据任务 | SHIN {task['name']}；{task['description']} |
| 分类头 | Linear(200, 2) |
| 预训练权重 | `{args.checkpoint.resolve()}` |
| 随机种子 | {args.seed} |
| Epoch | {args.epochs} |
| Batch size | {args.batch_size} |
| 分类头学习率 | {args.head_lr:g} |
| 骨干学习率 | {args.backbone_lr:g} |
| Weight decay | {args.weight_decay:g} |
| 骨干解冻 | 第 {args.unfreeze_epoch} epoch（最后 {args.epochs - args.unfreeze_epoch + 1} epoch） |
| 数据划分 | 训练 sub-1~19；验证 sub-20~24；测试 sub-25~29 |
| 输入 | 30 通道，10 秒，200 Hz，10 个 patch |
| 标签 | {task['description']} |

## 数据适配结论

| 检查项 | 结论 |
|---|---|
| BDF 与采样率 | 可读取；200 Hz，无需重采样 |
| trial 长度 | 2000 点，可整除 patch size 200 |
| EEG 通道 | 30 个全部保留 |
| 通道命名 | 18 个 10-5 半步名称需映射到 LaBraM 10-20 位置表，映射唯一且无冲突 |
| 标签 | 每个 session 两类各 10 trial，二分类无需改输出维度 |
| EOG | 2 个 EOG 通道不输入模型 |
| 幅值处理 | BDF 转物理微伏，再除以 100（LaBraM 下游代码约定；不同于 CodeBrain 的逐 trial 全局 z-score） |

完整映射及逐 split 数量见 `diagnostics.json`。

## 结果

| 检查点 | Epoch | 验证准确率 | 验证 Macro-F1 | 测试准确率 | 测试 Macro-F1 | 测试 Kappa |
|---|---:|---:|---:|---:|---:|---:|
| 最优验证模型 | {best['epoch']} | {best['val']['accuracy']:.4f} | {best['val']['f1_macro']:.4f} | {test['accuracy']:.4f} | {test['f1_macro']:.4f} | {test['cohen_kappa']:.4f} |
| 最后模型 | {final['epoch']} | {final['val']['accuracy']:.4f} | {final['val']['f1_macro']:.4f} | {final['test']['accuracy']:.4f} | {final['test']['f1_macro']:.4f} | {final['test']['cohen_kappa']:.4f} |

最优模型测试集混淆矩阵（行是真实标签，列是预测标签）：`{test['confusion_matrix']}`。
"""


def main() -> None:
    global ACTIVE_DATA_ROOT
    args = parse_args()
    task = TASKS[args.task]
    if args.experiment_note is None:
        args.experiment_note = (
            f"复用统一的 SHIN 被试独立划分和微调参数，评估 LaBraM 在 "
            f"{task['name']}（{task['description']}）上的迁移效果。"
        )
    ACTIVE_DATA_ROOT = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖已停止：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "mplconfig"))
    seed_everything(args.seed)

    if SHIN_CHANNELS.__len__() != LABRAM_CHANNELS.__len__() or len(set(LABRAM_CHANNELS)) != 30:
        raise RuntimeError("通道映射不是 30 个唯一位置")
    absent = sorted(set(LABRAM_CHANNELS) - set(utils.standard_1020))
    if absent:
        raise RuntimeError(f"LaBraM 位置表缺少映射后的通道：{absent}")

    split_subjects = {
        "train": subject_ids(1, 19, args.max_subjects_per_split),
        "val": subject_ids(20, 24, args.max_subjects_per_split),
        "test": subject_ids(25, 29, args.max_subjects_per_split),
    }
    arrays, labels, details = {}, {}, {}
    for name, ids in split_subjects.items():
        arrays[name], labels[name], details[name] = load_split(
            name, ids, args.cache_dir, args.task, task
        )

    diagnostics = {
        "data_root": str(ACTIVE_DATA_ROOT),
        "sampling_rate_hz": 200,
        "trial_samples": 2000,
        "patch_size": 200,
        "patch_count": 10,
        "normalization": "physical_uV / 100",
        "task": {"key": args.task, "name": task["name"],
                 "description": task["description"],
                 "sessions": list(task["sessions"]), "labels": task["labels"]},
        "channel_mapping": dict(zip(SHIN_CHANNELS, LABRAM_CHANNELS)),
        "mapping_is_unique": len(set(LABRAM_CHANNELS)) == len(LABRAM_CHANNELS),
        "splits": {
            name: {
                "subjects": split_subjects[name],
                "shape": list(arrays[name].shape),
                "label_counts": {str(k): int(v) for k, v in Counter(labels[name].tolist()).items()},
                "details": details[name],
            } for name in split_subjects
        },
    }
    write_json(args.output_dir / "diagnostics.json", diagnostics)
    if args.diagnose_only:
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("要求 CUDA 训练，但当前 LaBraM 环境的 torch.cuda.is_available() 为 False")
    device = torch.device(args.device)
    raw_loaders = {
        name: make_loader(arrays[name], labels[name], args.batch_size, name == "train", args.num_workers, args.seed)
        for name in arrays
    }
    model = create_model(
        "labram_base_patch200_200", pretrained=False, num_classes=2,
        use_mean_pooling=True, init_scale=0.001, use_rel_pos_bias=False,
        use_abs_pos_emb=True, init_values=0.1, qkv_bias=False,
    )
    pretrained = load_pretrained(model, args.checkpoint)
    diagnostics["pretrained_load"] = pretrained
    diagnostics["model_parameters"] = {
        "total": sum(p.numel() for p in model.parameters()),
        "linear_head": sum(p.numel() for p in model.head.parameters()),
    }
    write_json(args.output_dir / "diagnostics.json", diagnostics)
    model.to(device)
    input_chans = utils.get_input_chans(LABRAM_CHANNELS)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("head.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "name": "backbone"},
        {"params": list(model.head.parameters()), "lr": args.head_lr, "name": "head"},
    ], weight_decay=args.weight_decay)

    # Reusing frozen features makes the first 90 epochs exactly equivalent while
    # avoiding 90 redundant backbone passes.
    feature_arrays, feature_labels = {}, {}
    for name in ("train", "val", "test"):
        feature_arrays[name], feature_labels[name] = cache_features(
            model, raw_loaders[name], input_chans, device, name
        )
    feature_loaders = {
        name: make_loader(feature_arrays[name], feature_labels[name], args.batch_size,
                          name == "train", args.num_workers, args.seed)
        for name in feature_arrays
    }

    history = []
    best_accuracy = -1.0
    best_record = None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            for parameter in model.parameters():
                parameter.requires_grad = True
            print(f"[训练] epoch {epoch}: 解冻 LaBraM 骨干", flush=True)
        feature_mode = epoch < args.unfreeze_epoch
        loaders = feature_loaders if feature_mode else raw_loaders
        train_loss = train_epoch(model, loaders["train"], optimizer, input_chans, device, feature_mode)
        val = evaluate(model, loaders["val"], input_chans, device, feature_mode)
        record = {
            "epoch": epoch,
            "stage": "linear_probe" if feature_mode else "full_finetune",
            "train_loss": train_loss,
            "val": val,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs} stage={record['stage']} "
            f"train_loss={train_loss:.4f} val_acc={val['accuracy']:.4f} "
            f"val_f1={val['f1_macro']:.4f}", flush=True
        )
        if val["accuracy"] > best_accuracy:
            best_accuracy = val["accuracy"]
            best_record = record
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       args.output_dir / "best_model.pth")
        write_json(args.output_dir / "history.json", history)

    final_test = evaluate(model, raw_loaders["test"], input_chans, device, feature_mode=False)
    final_record = {"epoch": args.epochs, "val": history[-1]["val"], "test": final_test}
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args)},
               args.output_dir / "last_model.pth")

    best_checkpoint = torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    best_test = evaluate(model, raw_loaders["test"], input_chans, device, feature_mode=False)
    summary = {
        "best": best_record,
        "best_test": best_test,
        "final": final_record,
        "elapsed_seconds": time.time() - started,
        "seed": args.seed,
        "task": diagnostics["task"],
        "experiment_note": args.experiment_note,
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "EXPERIMENT_RECORD.md").write_text(
        report_markdown(args, diagnostics, summary), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


ACTIVE_DATA_ROOT = Path(".")


if __name__ == "__main__":
    main()
