"""CBraMod transfer-learning baseline for HYGRIP left/right grip decoding."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "HYGRIP"))
from hefmi_within_subject_folds import (  # noqa: E402
    SPLIT_TO_FOLDS,
    make_within_subject_five_fold_indices,
)


HERE = Path(__file__).resolve().parent
CBRAMOD_ROOT = HERE
CHECKPOINT = CBRAMOD_ROOT / "pretrained_weights" / "pretrained_weights.pth"
SUBJECTS = list("ABCDEFGHIJKLMN")
CHANNELS = 24
SAMPLE_RATE = 200
WINDOW_SECONDS = 10
PATCH_SECONDS = 1
PATCH_SAMPLES = SAMPLE_RATE * PATCH_SECONDS
PATCHES = WINDOW_SECONDS // PATCH_SECONDS
SCALE_DIVISOR = 100.0


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, default=Path(r"D:\data\HYGRIP-Baselines\prepared"))
    parser.add_argument("--eeg-preprocessing", choices=("legacy_causal_div100", "v2_channel_zscore"), default="legacy_causal_div100")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cbramod-root", type=Path, default=CBRAMOD_ROOT,
                        help="CBraMod repository root containing models/cbramod.py")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:\data\HYGRIP-Baselines\cache\eeg_cbramod"))
    parser.add_argument("--train-subjects", default="A-J")
    parser.add_argument("--val-subjects", default="K-L")
    parser.add_argument("--test-subjects", default="M-N")
    parser.add_argument(
        "--split-protocol",
        choices=("subject_holdout", "within_subject_5fold"),
        default="subject_holdout",
    )
    parser.add_argument("--all-subjects", default="A-N")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--fine-tune-backbone", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def parse_subjects(text):
    result = []
    for item in text.upper().split(","):
        item = item.strip()
        if "-" in item:
            start, stop = item.split("-", 1)
            result.extend(chr(value) for value in range(ord(start), ord(stop) + 1))
        elif item:
            result.append(item)
    if not result or any(subject not in SUBJECTS for subject in result):
        raise ValueError(text)
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def preprocess_trial(eeg_uv, mode):
    eeg = np.asarray(eeg_uv[:, : WINDOW_SECONDS * SAMPLE_RATE], dtype=np.float64)
    if mode == "v2_channel_zscore":
        eeg -= eeg.mean(axis=-1, keepdims=True)
        scale = eeg.std(axis=-1, keepdims=True)
        eeg = eeg / np.maximum(scale, 1e-12)
        return eeg.astype(np.float32).reshape(CHANNELS, PATCHES, PATCH_SAMPLES)
    eeg -= eeg.mean(axis=0, keepdims=True)
    b, a = butter(5, [0.3 / (SAMPLE_RATE / 2), 50.0 / (SAMPLE_RATE / 2)], btype="band")
    eeg = lfilter(b, a, eeg, axis=-1) / SCALE_DIVISOR
    return eeg.astype(np.float32).reshape(CHANNELS, PATCHES, PATCH_SAMPLES)


def load_subject(root, subject, preprocessing):
    path = root / f"subject_{subject}_trials.mat"
    data = loadmat(path, variable_names=["eeg_uv", "labels"])
    eeg = np.asarray(data["eeg_uv"], dtype=np.float32)
    target = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
    if eeg.shape[1:] != (CHANNELS, 4000) or len(eeg) != len(target):
        raise RuntimeError(f"{path}: unexpected shapes {eeg.shape}, {target.shape}")
    if Counter(target.tolist()) not in (Counter({0: 10, 1: 10}), Counter({0: 13, 1: 13})):
        raise RuntimeError(f"{path}: unexpected labels {Counter(target.tolist())}")
    x = np.stack([preprocess_trial(trial, preprocessing) for trial in eeg])
    if not np.isfinite(x).all():
        raise RuntimeError(f"{path}: non-finite preprocessed EEG")
    return x, target


def load_split(root, cache_dir, name, subjects, preprocessing):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"hygrip_{name}_{''.join(subjects)}_24ch_10patch_{preprocessing}.npz"
    if cache_path.exists():
        saved = np.load(cache_path, allow_pickle=False)
        print(f"[{name}] cache {cache_path} X={saved['X'].shape}", flush=True)
        return saved["X"], saved["y"], saved["subjects"]
    arrays, labels, subject_ids = [], [], []
    for subject in subjects:
        x, target = load_subject(root, subject, preprocessing)
        arrays.append(x)
        labels.append(target)
        subject_ids.append(np.full(len(target), subject))
        print(f"[{name}] subject {subject}: {len(target)} trials", flush=True)
    X, y, subject_array = np.concatenate(arrays), np.concatenate(labels), np.concatenate(subject_ids)
    np.savez(cache_path, X=X, y=y, subjects=subject_array)
    print(f"[{name}] cache written X={X.shape}", flush=True)
    return X, y, subject_array


def load_within_subject_splits(root, cache_dir, subjects, seed, preprocessing):
    cache_dir.mkdir(parents=True, exist_ok=True)
    subject_tag = "".join(subjects)
    cache_paths = {
        name: cache_dir / (
            f"hygrip_within5_seed-{seed}_{name}_{subject_tag}_"
            f"24ch_10patch_{preprocessing}.npz"
        )
        for name in SPLIT_TO_FOLDS
    }
    metadata_path = cache_dir / f"hygrip_within5_seed-{seed}_{subject_tag}_folds.json"
    if metadata_path.is_file() and all(path.is_file() for path in cache_paths.values()):
        arrays = {}
        for name, path in cache_paths.items():
            saved = np.load(path, allow_pickle=False)
            arrays[name] = saved["X"], saved["y"], saved["subjects"]
            print(f"[{name}] cache {path} X={saved['X'].shape}", flush=True)
        return arrays, json.loads(metadata_path.read_text(encoding="utf-8"))

    x_parts = {name: [] for name in SPLIT_TO_FOLDS}
    y_parts = {name: [] for name in SPLIT_TO_FOLDS}
    subject_parts = {name: [] for name in SPLIT_TO_FOLDS}
    subject_fold_rows = []
    for subject in subjects:
        x, target = load_subject(root, subject, preprocessing)
        numeric_subject = ord(subject) - ord("A") + 1
        numeric_subjects = np.full(len(target), numeric_subject, dtype=np.int16)
        indices, subject_metadata = make_within_subject_five_fold_indices(
            target, numeric_subjects, seed
        )
        row = subject_metadata["subject_folds"][0]
        row["subject"] = subject
        subject_fold_rows.append(row)
        for name in SPLIT_TO_FOLDS:
            chosen = indices[name]
            x_parts[name].append(x[chosen])
            y_parts[name].append(target[chosen])
            subject_parts[name].append(np.full(len(chosen), subject))
        print(
            f"[within5] subject {subject}: total={len(target)} "
            + " ".join(f"{name}={len(indices[name])}" for name in SPLIT_TO_FOLDS),
            flush=True,
        )

    arrays = {}
    for name in SPLIT_TO_FOLDS:
        X = np.concatenate(x_parts[name])
        y = np.concatenate(y_parts[name])
        subject_array = np.concatenate(subject_parts[name])
        arrays[name] = X, y, subject_array
        np.savez(cache_paths[name], X=X, y=y, subjects=subject_array)
    metadata = {
        "protocol": "within_subject_stratified_5fold_fixed_3_1_1",
        "description": (
            "Each subject is independently stratified into five folds; "
            "folds 1-3 train, fold 4 validation, fold 5 test."
        ),
        "seed": int(seed),
        "subjects": subjects,
        "folds": {name: list(values) for name, values in SPLIT_TO_FOLDS.items()},
        "splits": {
            name: {
                "trials": int(len(arrays[name][1])),
                "label_counts": {
                    str(label): int((arrays[name][1] == label).sum())
                    for label in sorted(np.unique(arrays[name][1]).tolist())
                },
            }
            for name in SPLIT_TO_FOLDS
        },
        "subject_folds": subject_fold_rows,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return arrays, metadata


class HYGRIPCBraMod(nn.Module):
    def __init__(self, cbramod_root: Path, dropout=0.1):
        super().__init__()
        if not (cbramod_root / "models" / "cbramod.py").is_file():
            raise FileNotFoundError(f"Invalid CBraMod repository root: {cbramod_root}")
        sys.path.insert(0, str(cbramod_root))
        from models.cbramod import CBraMod

        self.backbone = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )
        self.classifier = nn.Sequential(
            nn.Linear(CHANNELS * PATCHES * 200, PATCHES * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(PATCHES * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, 2),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x).flatten(start_dim=1))


def load_pretrained(model, checkpoint, fine_tune_backbone=False):
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(source, dict) and "model" in source:
        source = source["model"]
    result = model.backbone.load_state_dict(source, strict=True)
    model.backbone.proj_out = nn.Identity()
    for parameter in model.backbone.parameters():
        parameter.requires_grad = fine_tune_backbone
    return {
        "checkpoint": str(checkpoint.resolve()),
        "loaded_keys": len(source),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "proj_out_after_loading": "Identity",
        "backbone_frozen_all_epochs": not fine_tune_backbone,
    }


def metrics(y_true, prediction, loss):
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, prediction)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def make_loader(X, y, batch_size, shuffle, workers, seed):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def run_epoch(model, data_loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum, seen, predictions, labels = 0.0, 0, [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for data, target in data_loader:
            data, target = data.to(device).float(), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.item()) * len(target)
            seen += len(target)
            predictions.append(logits.argmax(1).detach().cpu().numpy())
            labels.append(target.cpu().numpy())
    labels, predictions = np.concatenate(labels), np.concatenate(predictions)
    return metrics(labels, predictions, loss_sum / seen), labels, predictions


def main():
    args = arguments()
    seed_all(args.seed)
    fold_metadata = None
    if args.split_protocol == "within_subject_5fold":
        all_subjects = parse_subjects(args.all_subjects)
        splits = {name: list(all_subjects) for name in SPLIT_TO_FOLDS}
    else:
        splits = {
            "train": parse_subjects(args.train_subjects),
            "val": parse_subjects(args.val_subjects),
            "test": parse_subjects(args.test_subjects),
        }
        flattened = sum(splits.values(), [])
        if len(flattened) != len(set(flattened)):
            raise ValueError("subject splits overlap")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "args.json", vars(args))

    if args.split_protocol == "within_subject_5fold":
        arrays, fold_metadata = load_within_subject_splits(
            args.prepared_root.resolve(),
            args.cache_dir.resolve(),
            splits["train"],
            args.seed,
            args.eeg_preprocessing,
        )
    else:
        arrays = {
            name: load_split(
                args.prepared_root.resolve(),
                args.cache_dir.resolve(),
                name,
                subjects,
                args.eeg_preprocessing,
            )
            for name, subjects in splits.items()
        }
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model = HYGRIPCBraMod(cbramod_root=args.cbramod_root, dropout=args.dropout)
    checkpoint_record = load_pretrained(
        model, args.checkpoint, fine_tune_backbone=args.fine_tune_backbone
    )
    model.to(device)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]
    if not trainable_names:
        raise RuntimeError("No trainable parameters")
    backbone_trainable = [name for name in trainable_names if name.startswith("backbone.")]
    if args.fine_tune_backbone:
        if not backbone_trainable:
            raise RuntimeError("Fine-tuning requested but no backbone parameter is trainable")
    elif backbone_trainable or any(not name.startswith("classifier.") for name in trainable_names):
        raise RuntimeError(f"Unexpected trainable parameters: {trainable_names[:10]}")

    probe = torch.from_numpy(arrays["train"][0][:2]).to(device)
    with torch.no_grad():
        output = model(probe)
    diagnostics = {
        "dataset": "HYGRIP",
        "task": "left hand (0) vs right hand (1) dynamic grip",
        "splits": splits,
        "split_protocol": args.split_protocol,
        "fold_metadata": fold_metadata,
        "shapes": {
            name: {
                "X": list(value[0].shape),
                "y": list(value[1].shape),
                "subjects": list(value[2].shape),
                "labels": dict(Counter(value[1].tolist())),
            }
            for name, value in arrays.items()
        },
        "preprocessing": {
            "source": "prepared HYGRIP eeg_uv, task onset 0-20 s, 200 Hz, microvolts",
            "selected_window_seconds": [0, WINDOW_SECONDS],
            "CAR": True,
            "filter": "already zero-phase 1-45 Hz in v2" if args.eeg_preprocessing == "v2_channel_zscore" else "5th-order causal Butterworth 0.3-50 Hz at 200 Hz",
            "normalization": args.eeg_preprocessing,
            "scale_divisor": None if args.eeg_preprocessing == "v2_channel_zscore" else SCALE_DIVISOR,
            "model_input": [CHANNELS, PATCHES, PATCH_SAMPLES],
        },
        "protocol": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr if args.fine_tune_backbone else None,
            "training_mode": "full fine-tuning" if args.fine_tune_backbone else "classification head only",
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seed": args.seed,
            "early_stopping": None,
            "checkpoint_selection": "validation accuracy",
        },
        "checkpoint_load": checkpoint_record,
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "trainable_tensors": trainable_names,
            "frozen_tensor_count": len(frozen_names),
        },
        "forward": {
            "input": list(probe.shape),
            "output": list(output.shape),
            "finite": bool(torch.isfinite(output).all()),
        },
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "checkpoint_load.json", checkpoint_record)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
    if args.diagnose_only:
        return

    loaders = {
        name: make_loader(
            arrays[name][0],
            arrays[name][1],
            args.batch_size,
            name == "train",
            args.num_workers,
            args.seed,
        )
        for name in arrays
    }
    parameter_groups = [
        {"params": list(model.classifier.parameters()), "lr": args.head_lr}
    ]
    if args.fine_tune_backbone:
        parameter_groups.append(
            {"params": list(model.backbone.parameters()), "lr": args.backbone_lr}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    history, best_epoch, best_accuracy, best_val, best_state = [], 0, -1.0, None, None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train, _, _ = run_epoch(model, loaders["train"], device, optimizer)
        validation, _, _ = run_epoch(model, loaders["val"], device)
        row = {"epoch": epoch, "train": train, "val": validation}
        history.append(row)
        if validation["accuracy"] > best_accuracy + 1e-6:
            best_epoch = epoch
            best_accuracy = validation["accuracy"]
            best_val = validation
            selected_state = model.state_dict() if args.fine_tune_backbone else model.classifier.state_dict()
            best_state = {
                name: value.detach().cpu().clone() for name, value in selected_state.items()
            }
            torch.save(
                best_state,
                output_dir / ("best_model.pt" if args.fine_tune_backbone else "best_head.pt"),
            )
        print(
            f"epoch {epoch:03d}/{args.epochs} train_acc={train['accuracy']:.4f} "
            f"val_acc={validation['accuracy']:.4f} "
            f"best_val_acc={best_accuracy:.4f} (ep {best_epoch}) "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )
        write_json(output_dir / "history.json", history)

    if args.fine_tune_backbone:
        model.load_state_dict(best_state)
    else:
        model.classifier.load_state_dict(best_state)
    test, test_labels, test_predictions = run_epoch(model, loaders["test"], device)
    per_subject = []
    test_subjects = arrays["test"][2]
    for subject in sorted(set(test_subjects.tolist())):
        mask = test_subjects == subject
        per_subject.append(
            {
                "subject": subject,
                "trials": int(mask.sum()),
                **metrics(test_labels[mask], test_predictions[mask], 0.0),
            }
        )
    summary = {
        "model": (
            "CBraMod official pretrained initialization; full model fine-tuning"
            if args.fine_tune_backbone
            else "CBraMod official pretrained backbone, frozen; all-patch classification head only"
        ),
        "dataset": "HYGRIP EEG",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_val": best_val,
        "test_metrics": test,
        "per_subject": per_subject,
        "checkpoint_load": checkpoint_record,
        "diagnostics": diagnostics,
        "elapsed_seconds": time.time() - started,
        "split_protocol": args.split_protocol,
        "split_protocol_description": (
            "逐受试者分层五折固定 3/1/1：fold 1–3 train、fold 4 val、fold 5 test"
            if args.split_protocol == "within_subject_5fold"
            else "受试者互斥的固定 Train/Validation/Test 划分"
        ),
        "fold_metadata": fold_metadata,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
