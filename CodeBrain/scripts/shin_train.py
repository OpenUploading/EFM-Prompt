"""Linear-probe/fine-tune CodeBrain on the SHIN BIDS EEG release.

The loader targets D:/DataSets/SHIN/v1.0.1, reads BDF with pyEDFlib, and uses
the BIDS sidecars as the source of channel types and trial labels.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyedflib
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODEBRAIN_ROOT = PROJECT_ROOT / "external" / "CodeBrain-source"
sys.path.insert(0, str(CODEBRAIN_ROOT))

from Models.SSSM import SSSM  # noqa: E402
from shin_linear_head import OfficialClassificationHead  # noqa: E402


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


def parse_subjects(text):
    result = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"invalid subject list: {text!r}")
    if any(item < 1 or item > 29 for item in result):
        raise ValueError("SHIN subject IDs must be in 1..29")
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_one(folder, pattern):
    matches = list(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern!r} under {folder}, found {len(matches)}")
    return matches[0]


def read_eeg_channel_names(channels_tsv):
    with channels_tsv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    names = [row["name"] for row in rows if row["type"].strip().upper() == "EEG"]
    if len(names) != 30:
        raise ValueError(f"{channels_tsv}: expected 30 EEG channels, got {len(names)}")
    return names


def read_events(events_tsv, sfreq, label_to_id):
    events = []
    with events_tsv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            label = row["trial_type"].strip()
            if label not in label_to_id:
                continue
            sample_text = row.get("sample", "").strip()
            sample = int(float(sample_text)) if sample_text else int(round(float(row["onset"]) * sfreq))
            events.append((sample, label_to_id[label], label))
    expected = Counter({class_id: 10 for class_id in label_to_id.values()})
    if len(events) != 20 or Counter(label for _, label, _ in events) != expected:
        raise ValueError(
            f"{events_tsv}: expected 20 balanced task trials, "
            f"got {Counter(label for _, label, _ in events)}"
        )
    return events


def unit_scale_to_microvolts(unit, path, channel):
    normalized = unit.strip().replace("μ", "u").replace("µ", "u").lower()
    if normalized in {"uv", "microvolt", "microvolts"}:
        return 1.0
    if normalized in {"v", "volt", "volts"}:
        return 1e6
    raise ValueError(f"{path}: unsupported physical unit {unit!r} for {channel}")


def load_run(eeg_dir, tmin, tmax, label_to_id, expected_channels=None):
    bdf_path = find_one(eeg_dir, "*_eeg.bdf")
    events_path = find_one(eeg_dir, "*_events.tsv")
    channels_path = find_one(eeg_dir, "*_channels.tsv")
    eeg_names = read_eeg_channel_names(channels_path)
    if expected_channels is not None and eeg_names != expected_channels:
        raise ValueError(f"{channels_path}: EEG channel order differs from the first run")

    reader = pyedflib.EdfReader(str(bdf_path))
    try:
        labels = reader.getSignalLabels()
        label_to_index = {name: index for index, name in enumerate(labels)}
        missing = [name for name in eeg_names if name not in label_to_index]
        if missing:
            raise ValueError(f"{bdf_path}: BDF is missing EEG channels {missing}")
        picks = [label_to_index[name] for name in eeg_names]
        sfreqs = [float(reader.getSampleFrequency(index)) for index in picks]
        if any(abs(value - 200.0) > 1e-6 for value in sfreqs):
            raise ValueError(f"{bdf_path}: expected every EEG channel at 200 Hz, got {sfreqs}")
        sfreq = sfreqs[0]
        n_times = int(reader.getNSamples()[picks[0]])
        continuous = []
        source_units = []
        for name, index in zip(eeg_names, picks):
            unit = reader.getPhysicalDimension(index)
            source_units.append(unit)
            scale = unit_scale_to_microvolts(unit, bdf_path, name)
            continuous.append(reader.readSignal(index).astype(np.float32, copy=False) * scale)
        continuous = np.stack(continuous)
    finally:
        reader.close()

    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))
    n_points = stop_offset - start_offset
    if n_points <= 0 or n_points % 200:
        raise ValueError("the selected window must contain a positive whole number of 200-sample patches")

    trials, labels_out = [], []
    for event_sample, label_id, _ in read_events(events_path, sfreq, label_to_id):
        start = event_sample + start_offset
        stop = event_sample + stop_offset
        if start < 0 or stop > n_times:
            raise ValueError(f"{events_path}: window [{start}, {stop}) is outside BDF length {n_times}")
        trial = np.asarray(continuous[:, start:stop], dtype=np.float32)
        mean = trial.mean(dtype=np.float64)
        std = trial.std(dtype=np.float64)
        if not np.isfinite(std) or std < 1e-6:
            raise ValueError(f"{bdf_path}: invalid trial standard deviation {std}")
        trials.append((trial - mean) / std)
        labels_out.append(label_id)

    x = np.ascontiguousarray(np.stack(trials), dtype=np.float32)
    y = np.asarray(labels_out, dtype=np.int64)
    info = {
        "bdf": str(bdf_path),
        "events": str(events_path),
        "source_units": sorted(set(source_units)),
        "sfreq": sfreq,
        "continuous_samples": n_times,
        "trials": len(y),
        "class_counts": dict(Counter(int(value) for value in y)),
        "output_shape": list(x.shape),
    }
    return x, y, eeg_names, info


def load_subjects(data_root, subjects, tmin, tmax, task):
    xs, ys, infos = [], [], []
    channel_names = None
    for subject in subjects:
        subject_x, subject_y = [], []
        for session in task["sessions"]:
            eeg_dir = data_root / f"sub-{subject}" / session / "eeg"
            if not eeg_dir.is_dir():
                raise FileNotFoundError(eeg_dir)
            x, y, names, info = load_run(
                eeg_dir, tmin, tmax, task["labels"], channel_names
            )
            channel_names = names
            info.update({"subject": subject, "session": session})
            infos.append(info)
            subject_x.append(x)
            subject_y.append(y)
        xs.append(np.concatenate(subject_x))
        ys.append(np.concatenate(subject_y))
    return np.concatenate(xs), np.concatenate(ys), channel_names, infos


class SHINCodeBrainClassifier(nn.Module):
    def __init__(self, num_channels, num_patches, num_classes=2, dropout=0.1, n_layer=8):
        super().__init__()
        self.feature_dim = 200
        self.backbone = SSSM(
            in_channels=200,
            res_channels=200,
            skip_channels=200,
            out_channels=200,
            num_res_layers=n_layer,
            diffusion_step_embed_dim_in=200,
            diffusion_step_embed_dim_mid=200,
            diffusion_step_embed_dim_out=200,
            s4_lmax=570,
            s4_d_state=64,
            s4_dropout=dropout,
            s4_bidirectional=True,
            s4_layernorm=True,
            codebook_size_t=4096,
            codebook_size_f=4096,
            if_codebook=False,
        )
        self.backbone.proj_out = nn.Identity()
        self.classifier = OfficialClassificationHead(
            num_channels=num_channels,
            num_patches=num_patches,
            feature_dim=self.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def encode(self, x):
        batch, channels, patches, _ = x.shape
        features = self.backbone(x)
        # SSSM ends with squeeze(); reshape restores dimensions when batch == 1.
        features = features.reshape(batch, channels, patches, self.feature_dim)
        return self.classifier.flatten(features)

    def forward(self, x):
        return self.classifier(self.encode(x))


def load_pretrained_backbone(model, checkpoint):
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw.get("model", raw)) if isinstance(raw, dict) else raw
    clean = {key.removeprefix("module."): value for key, value in state.items()}
    current = model.backbone.state_dict()
    compatible = {key: value for key, value in clean.items() if key in current and value.shape == current[key].shape}
    if not compatible:
        raise RuntimeError(f"no compatible CodeBrain keys found in {path}")
    missing, unexpected = model.backbone.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(path),
        "checkpoint_keys": len(clean),
        "matched_keys": len(compatible),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def make_loader(x, y, batch_size, shuffle):
    # CodeBrain consumes one-second, 200-sample patches.
    tensor_x = torch.from_numpy(x).reshape(x.shape[0], x.shape[1], -1, 200)
    tensor_y = torch.from_numpy(y)
    return DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def cache_backbone_features(model, loaders, device, batch_size):
    """Run a frozen backbone once; subsequent epochs train only the official head."""
    model.eval()
    feature_loaders = {}
    with torch.no_grad():
        for name, loader in loaders.items():
            features, targets = [], []
            for x, y in loader:
                features.append(model.encode(x.to(device, non_blocking=True)).cpu())
                targets.append(y)
            feature_dataset = TensorDataset(torch.cat(features), torch.cat(targets))
            feature_loaders[name] = DataLoader(
                feature_dataset,
                batch_size=batch_size,
                shuffle=name == "train",
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            print(f"cached {name} CodeBrain features: {tuple(feature_dataset.tensors[0].shape)}")
    return feature_loaders


def score(model, loader, device, criterion):
    model.eval()
    losses, logits_all, labels_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            losses.append(float(criterion(logits, y).item()) * len(y))
            logits_all.append(logits.cpu())
            labels_all.append(y.cpu())
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    predictions = logits.argmax(axis=1)
    return {
        "loss": float(sum(losses) / len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).astype(int).tolist(),
    }


def write_experiment_record(out_dir, summary, splits):
    note = str(summary["experiment_note"]).replace("|", "\\|")
    schedule = summary["schedule"]
    best = summary["test"]
    final = summary["final_epoch_test"]
    mode_name = {
        "head-only": "仅训练官方分类头",
        "fine-tune": "全程微调",
        "staged-fine-tune": "分阶段微调",
    }.get(summary["mode"], summary["mode"])
    lines = [
        "# 实验记录",
        "",
        "## 实验思路",
        "",
        note,
        "",
        "## 实验参数",
        "",
        "| 参数 | 取值 |",
        "|---|---|",
        f"| 数据集 | SHIN {summary['task']['name']} |",
        f"| 分类任务 | {summary['task']['description']} |",
        "| 模型 | CodeBrain + 官方三层下游分类头 |",
        f"| 训练模式 | {mode_name} |",
        f"| 随机种子 | {summary['seed']} |",
        f"| 训练轮数 | {schedule['epochs']} |",
        f"| Batch size | {schedule['batch_size']} |",
        f"| 分类头学习率 | {schedule['head_lr']} |",
        f"| 骨干网络学习率 | {schedule['backbone_lr']} |",
        f"| Weight decay | {schedule['weight_decay']} |",
        f"| 骨干网络解冻轮次 | {schedule['unfreeze_backbone_epoch']} |",
        f"| 训练集受试者 | {','.join(map(str, splits['train']))} |",
        f"| 验证集受试者 | {','.join(map(str, splits['val']))} |",
        f"| 测试集受试者 | {','.join(map(str, splits['test']))} |",
        "",
        "## 实验结果",
        "",
        "| 检查点 | 轮次 | 验证准确率 | 测试准确率 | Macro F1 | Kappa |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 验证集最佳 | {summary['best_epoch']} | {summary['best_val_accuracy']:.4f} | {best['accuracy']:.4f} | {best['f1_macro']:.4f} | {best['kappa']:.4f} |",
        f"| 最后一轮 | {schedule['epochs']} | {summary['history'][-1]['val']['accuracy']:.4f} | {final['accuracy']:.4f} | {final['f1_macro']:.4f} | {final['kappa']:.4f} |",
        "",
        f"最佳模型测试集混淆矩阵：`{best['confusion_matrix']}`",
        "",
    ]
    (out_dir / "EXPERIMENT_RECORD.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="CodeBrain official-head fine-tuning on SHIN BIDS EEG")
    parser.add_argument("--data-root", default=r"D:\DataSets\SHIN\v1.0.1")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "shin_codebrain"))
    parser.add_argument("--pretrained-backbone", default=str(PROJECT_ROOT / "external" / "CodeBrain" / "Checkpoints" / "CodeBrain.pth"))
    parser.add_argument("--train-subjects", default=",".join(str(i) for i in range(1, 20)))
    parser.add_argument("--val-subjects", default="20,21,22,23,24")
    parser.add_argument("--test-subjects", default="25,26,27,28,29")
    parser.add_argument("--task", choices=tuple(TASKS), default="mi",
                        help="mi=左右手运动想象；ma=心算减法与静息")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--experiment-note",
        default=None,
    )
    parser.add_argument(
        "--finetune-backbone",
        action="store_true",
        help="Default keeps the backbone frozen and trains the official downstream head.",
    )
    parser.add_argument(
        "--unfreeze-backbone-epoch",
        type=int,
        default=0,
        help="Start backbone fine-tuning at this 1-based epoch; 0 keeps it frozen.",
    )
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    task = TASKS[args.task]
    if args.experiment_note is None:
        args.experiment_note = (
            f"使用预训练 CodeBrain 骨干和官方三层下游分类头，评估 SHIN "
            f"{task['name']}（{task['description']}）跨受试者分类性能。"
        )

    seed_everything(args.seed)
    if args.finetune_backbone and args.unfreeze_backbone_epoch:
        raise ValueError("use either --finetune-backbone or --unfreeze-backbone-epoch, not both")
    if args.unfreeze_backbone_epoch < 0 or args.unfreeze_backbone_epoch > args.epochs:
        raise ValueError("--unfreeze-backbone-epoch must be 0 or within 1..epochs")
    data_root = Path(args.data_root)
    if not (data_root / "dataset_description.json").exists():
        raise FileNotFoundError(f"not a SHIN BIDS root: {data_root}")
    splits = {
        "train": parse_subjects(args.train_subjects),
        "val": parse_subjects(args.val_subjects),
        "test": parse_subjects(args.test_subjects),
    }
    flat_subjects = [item for values in splits.values() for item in values]
    if len(flat_subjects) != len(set(flat_subjects)):
        raise ValueError("train/val/test subject lists must be disjoint")

    arrays, labels, all_info, channel_names = {}, {}, [], None
    for split_name, subjects in splits.items():
        x, y, names, info = load_subjects(
            data_root, subjects, args.tmin, args.tmax, task
        )
        if channel_names is not None and names != channel_names:
            raise ValueError("EEG channel order changed between splits")
        channel_names = names
        arrays[split_name], labels[split_name] = x, y
        all_info.extend(info)
        print(f"{split_name}: subjects={subjects} x={x.shape} labels={dict(Counter(y.tolist()))}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "data_root": str(data_root),
        "format": "BIDS sidecars + BDF read with pyEDFlib",
        "task_key": args.task,
        "task": task["description"],
        "subjects": splits,
        "sessions": list(task["sessions"]),
        "sfreq": 200,
        "window_seconds": [args.tmin, args.tmax],
        "channels": channel_names,
        "num_channels": len(channel_names),
        "normalization": "per-trial global z-score after conversion to microvolts",
        "seed": args.seed,
        "experiment_note": args.experiment_note,
        "input_shapes_before_patching": {name: list(value.shape) for name, value in arrays.items()},
        "input_shapes_to_codebrain": {
            name: [value.shape[0], value.shape[1], value.shape[2] // 200, 200]
            for name, value in arrays.items()
        },
        "runs": all_info,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    if args.diagnose_only:
        print(f"diagnostics written to {out_dir / 'diagnostics.json'}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_patches = arrays["train"].shape[2] // 200
    model = SHINCodeBrainClassifier(
        len(channel_names),
        num_patches=num_patches,
        dropout=args.dropout,
    )
    pretrained = load_pretrained_backbone(model, args.pretrained_backbone)
    unfreeze_epoch = 1 if args.finetune_backbone else args.unfreeze_backbone_epoch
    initially_frozen = unfreeze_epoch != 1
    if initially_frozen:
        model.backbone.requires_grad_(False)
    model.to(device)

    raw_loaders = {
        name: make_loader(arrays[name], labels[name], args.batch_size, name == "train")
        for name in arrays
    }
    feature_loaders = None
    if initially_frozen:
        feature_loaders = cache_backbone_features(model, raw_loaders, device, args.batch_size)
    training_model = model if unfreeze_epoch == 1 else model.classifier
    loaders = raw_loaders if unfreeze_epoch == 1 else feature_loaders
    parameter_groups = [{"params": model.classifier.parameters(), "lr": args.head_lr}]
    if unfreeze_epoch:
        parameter_groups.append({"params": model.backbone.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        if unfreeze_epoch > 1 and epoch == unfreeze_epoch:
            model.backbone.requires_grad_(True)
            training_model = model
            loaders = raw_loaders
            print(f"epoch {epoch:03d}: backbone unfrozen at lr={args.backbone_lr:g}")
        training_model.train()
        if epoch < unfreeze_epoch or unfreeze_epoch == 0:
            model.backbone.eval()
        total_loss, count = 0.0, 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = training_model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            count += len(y)
        val_metrics = score(training_model, loaders["val"], device, criterion)
        row = {"epoch": epoch, "train_loss": total_loss / count, "val": val_metrics}
        history.append(row)
        print(f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} val_acc={val_metrics['accuracy']:.4f}")
        if val_metrics["accuracy"] > best_val:
            best_val, best_epoch = val_metrics["accuracy"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics}, out_dir / "best.pt")

    final_test_metrics = score(training_model, loaders["test"], device, criterion)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs,
            "val": history[-1]["val"],
            "test": final_test_metrics,
        },
        out_dir / "last.pt",
    )
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = score(training_model, loaders["test"], device, criterion)
    summary = {
        "device": str(device),
        "seed": args.seed,
        "task": {
            "key": args.task,
            "name": task["name"],
            "description": task["description"],
        },
        "experiment_note": args.experiment_note,
        "mode": (
            "fine-tune"
            if unfreeze_epoch == 1
            else "staged-fine-tune"
            if unfreeze_epoch > 1
            else "head-only"
        ),
        "schedule": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr if unfreeze_epoch else None,
            "weight_decay": args.weight_decay,
            "unfreeze_backbone_epoch": unfreeze_epoch or None,
        },
        "head": (
            f"Official CodeBrain: Flatten({len(channel_names) * num_patches * 200})"
            f"-Linear({len(channel_names) * num_patches * 200},{num_patches * 200})"
            f"-ELU-Dropout-Linear({num_patches * 200},200)"
            "-ELU-Dropout-Linear(200,2)"
        ),
        "head_trainable_parameters": sum(p.numel() for p in model.classifier.parameters() if p.requires_grad),
        "pretrained": pretrained,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "test": test_metrics,
        "final_epoch_test": final_test_metrics,
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_experiment_record(out_dir, summary, splits)
    print(json.dumps({key: summary[key] for key in ("device", "mode", "head", "best_epoch", "test")}, indent=2))


if __name__ == "__main__":
    main()
