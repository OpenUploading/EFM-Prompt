"""CBraMod fine-tuning on SHIN EEG with reproducible reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
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

from models.cbramod import CBraMod


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


class SHINCBraMod(nn.Module):
    """Pretrained backbone plus the repository's default downstream head."""

    def __init__(self, head_type: str = "official_all_patch", dropout: float = 0.1) -> None:
        super().__init__()
        if head_type != "official_all_patch":
            raise ValueError(f"仅支持官方默认分类头 official_all_patch，收到：{head_type}")
        self.head_type = head_type
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )
        # Official all_patch_reps pattern:
        # C*P*D -> P*D -> D -> classes.
        self.classifier = nn.Sequential(
            nn.Linear(30 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, 2),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return features.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CBraMod + SHIN EEG 二分类")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\DataSets\SHIN\v1.0.1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained_weights/pretrained_weights.pth"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:\data\CBraMod-SHIN\cache"))
    parser.add_argument("--task", choices=tuple(TASKS), default="mi")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--head-type",
        choices=("official_all_patch",),
        default="official_all_patch",
        help="CBraMod 官方默认 all_patch_reps 三层分类头",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-epoch", type=int, default=91)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--experiment-note", default=None)
    parser.add_argument("--max-subjects-per-split", type=int, default=None,
                        help="只用于冒烟测试；正式实验不要设置。")
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


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _single(folder: Path, pattern: str) -> Path:
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"{folder} 中 {pattern} 的文件数应为 1，实际为 {len(paths)}")
    return paths[0]


def preprocess(data_uv: np.ndarray) -> np.ndarray:
    return data_uv.astype(np.float32, copy=False)


def load_subject(data_root: Path, subject: int, task: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    import pyedflib

    trials, labels, run_info = [], [], []
    for session in task["sessions"]:
        eeg_dir = data_root / f"sub-{subject:02d}" / session / "eeg"
        bdf_path = _single(eeg_dir, "*_eeg.bdf")
        events_path = _single(eeg_dir, "*_events.tsv")
        channels_path = _single(eeg_dir, "*_channels.tsv")
        rows = _read_tsv(channels_path)
        names = [row["name"] for row in rows if row.get("type", "").upper() == "EEG"]
        if names != SHIN_CHANNELS:
            raise RuntimeError(f"{channels_path} 的 EEG 通道或顺序异常：{names}")

        reader = pyedflib.EdfReader(str(bdf_path))
        try:
            signal_labels = reader.getSignalLabels()
            picks = [signal_labels.index(name) for name in SHIN_CHANNELS]
            sfreqs = [float(reader.getSampleFrequency(index)) for index in picks]
            if any(abs(sfreq - 200.0) > 1e-6 for sfreq in sfreqs):
                raise RuntimeError(f"{bdf_path} sampling rates are {sfreqs}; expected 200 Hz")
            units = [reader.getPhysicalDimension(index) for index in picks]
            if any(unit != "uV" for unit in units):
                raise RuntimeError(f"{bdf_path} physical units are {units}; expected uV")
            data_uv = np.stack([reader.readSignal(index) for index in picks])
        finally:
            reader.close()
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
            trial = preprocess(data_uv[:, start:stop].copy())
            trials.append(trial.reshape(30, 10, 200))
            labels.append(task["labels"][trial_type])
            counts[trial_type] += 1
        expected = Counter({label: 10 for label in task["labels"]})
        if counts != expected:
            raise RuntimeError(f"{events_path} 标签数量异常：{dict(counts)}")
        run_info.append({"session": session, "trials": 20, "labels": dict(counts)})
    return np.stack(trials), np.asarray(labels, dtype=np.int64), {
        "subject": subject, "trials": len(labels), "runs": run_info,
    }


def load_split(data_root: Path, name: str, subjects: list[int], cache_dir: Path,
               task_key: str, task: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(map(str, subjects))
    cache_path = cache_dir / f"{task_key}_{name}_sub-{tag}_30ch_10patch_raw_uv.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        print(f"[{name}] 使用缓存 {cache_path}，shape={cached['X'].shape}", flush=True)
        info = [{"subject": subject, "trials": 60, "source": "cache"} for subject in subjects]
        return cached["X"], cached["y"], info
    arrays, targets, info = [], [], []
    for subject in subjects:
        x, y, detail = load_subject(data_root, subject, task)
        arrays.append(x)
        targets.append(y)
        info.append(detail)
        print(f"[{name}] sub-{subject}: {len(y)} trials", flush=True)
    X, y = np.concatenate(arrays), np.concatenate(targets)
    np.savez(cache_path, X=X, y=y)
    print(f"[{name}] 缓存写入 {cache_path}，shape={X.shape}", flush=True)
    return X, y, info


def loader(X, y, batch_size, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def load_pretrained(model: SHINCBraMod, checkpoint: Path) -> dict:
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(source, dict) and "model" in source:
        source = source["model"]
    result = model.backbone.load_state_dict(source, strict=True)
    # Official downstream models remove the pretraining output projection.
    model.backbone.proj_out = nn.Identity()
    return {
        "checkpoint": str(checkpoint.resolve()),
        "loaded_keys": len(source),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "proj_out_after_loading": "Identity",
    }


@torch.no_grad()
def cache_features(model, data_loader, device, name):
    model.eval()
    features, labels = [], []
    for step, (x, y) in enumerate(data_loader, start=1):
        feat = model.features(x.to(device, non_blocking=True).float())
        features.append(feat.cpu().numpy().astype(np.float32))
        labels.append(y.numpy())
        if step % 25 == 0 or step == len(data_loader):
            print(f"[特征缓存] {name}: {step}/{len(data_loader)}", flush=True)
    return np.concatenate(features), np.concatenate(labels)


def metrics(y_true, y_pred, loss):
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def evaluate(model, data_loader, device, feature_mode):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, seen, predictions, labels = 0.0, 0, [], []
    for x, y in data_loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        logits = model.classifier(x) if feature_mode else model(x)
        loss = criterion(logits, y)
        total += float(loss.item()) * len(y)
        seen += len(y)
        predictions.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return metrics(np.concatenate(labels), np.concatenate(predictions), total / seen)


def train_epoch(model, data_loader, optimizer, device, feature_mode):
    model.classifier.train() if feature_mode else model.train()
    criterion = nn.CrossEntropyLoss()
    total, seen = 0.0, 0
    for x, y in data_loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model.classifier(x) if feature_mode else model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(y)
        seen += len(y)
    return total / seen


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def make_report(args, summary):
    best, test, final = summary["best"], summary["best_test"], summary["final"]
    head_text = (
        "官方 all_patch_reps：Flatten(60000) → Linear(60000,2000) → ELU "
        f"→ Dropout({args.dropout:g}) → Linear(2000,200) → ELU "
        f"→ Dropout({args.dropout:g}) → Linear(200,2)"
    )
    task = TASKS[args.task]
    return f"""# CBraMod × SHIN EEG 实验记录

## 实验思路

{args.experiment_note}

保留 SHIN 完整 10 秒 trial。BDF 中的物理微伏数据不执行公共平均参考或带通滤波，直接重排为 30×10×200。前 {args.unfreeze_epoch - 1} epoch 缓存冻结骨干特征，只训练所选线性分类头；从第 {args.unfreeze_epoch} epoch 起解冻全模型。

## 参数

| 参数 | 值 |
|---|---:|
| 模型 | CBraMod Base |
| 数据任务 | SHIN {task['name']}；{task['description']} |
| 分类头 | {head_text} |
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

## 数据适配结论

| 检查项 | 结论 |
|---|---|
| BDF/采样率 | 可读取；200 Hz，无需重采样 |
| trial/patch | 2000 点，直接分成 10×200 |
| EEG/EOG | 保留 30 个 EEG，排除 2 个 EOG |
| 电极名称 | CBraMod 无固定电极词表，无需名称映射 |
| 标签 | {task['description']}；每 session 两类各 10 个 |
| 模型专用预处理 | 无 CAR、无额外滤波；BDF 物理微伏直接转为 float32 |

## 结果

| 检查点 | Epoch | 验证准确率 | 验证 Macro-F1 | 测试准确率 | 测试 Macro-F1 | 测试 Kappa |
|---|---:|---:|---:|---:|---:|---:|
| 最优验证模型 | {best['epoch']} | {best['val']['accuracy']:.4f} | {best['val']['f1_macro']:.4f} | {test['accuracy']:.4f} | {test['f1_macro']:.4f} | {test['cohen_kappa']:.4f} |
| 最后模型 | {final['epoch']} | {final['val']['accuracy']:.4f} | {final['val']['f1_macro']:.4f} | {final['test']['accuracy']:.4f} | {final['test']['f1_macro']:.4f} | {final['test']['cohen_kappa']:.4f} |

最优模型测试混淆矩阵（行是真实标签，列是预测标签）：`{test['confusion_matrix']}`。

## 结果分析

正式结果始终采用验证准确率最高的检查点，而非强制采用最后一轮。不同分类头的结果必须作为独立实验记录，并在相同数据划分和超参数下比较。
"""


def main() -> None:
    args = parse_args()
    task = TASKS[args.task]
    if args.experiment_note is None:
        args.experiment_note = (
            f"复用统一的 SHIN 被试独立划分和分阶段微调参数，评估 CBraMod 在 "
            f"{task['name']}（{task['description']}）上的迁移效果。"
        )
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，为避免覆盖已停止：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "mplconfig"))
    seed_everything(args.seed)

    ranges = {"train": (1, 19), "val": (20, 24), "test": (25, 29)}
    split_subjects = {}
    for name, (start, stop) in ranges.items():
        ids = list(range(start, stop + 1))
        split_subjects[name] = ids[:args.max_subjects_per_split] if args.max_subjects_per_split else ids
    arrays, targets, details = {}, {}, {}
    for name, subjects in split_subjects.items():
        arrays[name], targets[name], details[name] = load_split(
            args.data_root, name, subjects, args.cache_dir, args.task, task
        )

    diagnostics = {
        "data_root": str(args.data_root), "sampling_rate_hz": 200,
        "trial_samples": 2000, "patch_size": 200, "patch_count": 10,
        "channels": SHIN_CHANNELS, "eeg_channels": 30, "excluded_eog_channels": 2,
        "channel_mapping_required": False,
        "task": {"key": args.task, "name": task["name"],
                 "description": task["description"],
                 "sessions": list(task["sessions"]), "labels": task["labels"]},
        "preprocessing": "physical_uV -> float32; no CAR; no additional filtering",
        "splits": {
            name: {
                "subjects": split_subjects[name], "shape": list(arrays[name].shape),
                "label_counts": {str(k): int(v) for k, v in Counter(targets[name].tolist()).items()},
                "details": details[name],
            } for name in split_subjects
        },
    }
    write_json(args.output_dir / "diagnostics.json", diagnostics)
    if args.diagnose_only:
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("要求 CUDA，但 cbramod_env 中 CUDA 不可用")
    device = torch.device(args.device)
    raw_loaders = {
        name: loader(arrays[name], targets[name], args.batch_size, name == "train",
                     args.num_workers, args.seed) for name in arrays
    }

    model = SHINCBraMod(args.head_type, dropout=args.dropout)
    diagnostics["pretrained_load"] = load_pretrained(model, args.checkpoint)
    diagnostics["model_parameters"] = {
        "total": sum(p.numel() for p in model.parameters()),
        "linear_head": sum(p.numel() for p in model.classifier.parameters()),
    }
    diagnostics["head_type"] = args.head_type
    write_json(args.output_dir / "diagnostics.json", diagnostics)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    backbone_params = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "name": "backbone"},
        {"params": list(model.classifier.parameters()), "lr": args.head_lr, "name": "head"},
    ], weight_decay=args.weight_decay)

    feature_arrays, feature_targets = {}, {}
    for name in ("train", "val", "test"):
        feature_arrays[name], feature_targets[name] = cache_features(
            model, raw_loaders[name], device, name
        )
    feature_loaders = {
        name: loader(feature_arrays[name], feature_targets[name], args.batch_size,
                     name == "train", args.num_workers, args.seed)
        for name in feature_arrays
    }

    history, best_record, best_accuracy = [], None, -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            for parameter in model.parameters():
                parameter.requires_grad = True
            print(f"[训练] epoch {epoch}: 解冻 CBraMod 骨干", flush=True)
        feature_mode = epoch < args.unfreeze_epoch
        active = feature_loaders if feature_mode else raw_loaders
        train_loss = train_epoch(model, active["train"], optimizer, device, feature_mode)
        val = evaluate(model, active["val"], device, feature_mode)
        record = {
            "epoch": epoch, "stage": "linear_probe" if feature_mode else "full_finetune",
            "train_loss": train_loss, "val": val, "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(f"epoch {epoch:03d}/{args.epochs} stage={record['stage']} "
              f"train_loss={train_loss:.4f} val_acc={val['accuracy']:.4f} "
              f"val_f1={val['f1_macro']:.4f}", flush=True)
        if val["accuracy"] > best_accuracy:
            best_accuracy, best_record = val["accuracy"], record
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       args.output_dir / "best_model.pth")
        write_json(args.output_dir / "history.json", history)

    final_test = evaluate(model, raw_loaders["test"], device, False)
    final = {"epoch": args.epochs, "val": history[-1]["val"], "test": final_test}
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args)},
               args.output_dir / "last_model.pth")
    best_checkpoint = torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    best_test = evaluate(model, raw_loaders["test"], device, False)
    summary = {
        "best": best_record, "best_test": best_test, "final": final,
        "elapsed_seconds": time.time() - started, "seed": args.seed,
        "task": diagnostics["task"],
        "head_type": args.head_type,
        "experiment_note": args.experiment_note,
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "EXPERIMENT_RECORD.md").write_text(make_report(args, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
