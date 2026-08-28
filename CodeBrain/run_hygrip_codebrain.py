"""Frozen-backbone CodeBrain/CSBrain baselines for HYGRIP EEG."""

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
PORTABLE_ROOT = HERE.parent
CODEBRAIN_ADAPTER = PORTABLE_ROOT / "CodeBrain"
CODEBRAIN_ROOT = CODEBRAIN_ADAPTER / "external" / "CodeBrain-source"
CSBRAIN_ROOT = PORTABLE_ROOT / "CSBrain"
DEFAULT_CODEBRAIN_WEIGHT = (
    CODEBRAIN_ADAPTER / "pretrained_weights" / "CodeBrain.pth"
)
DEFAULT_CSBRAIN_WEIGHT = CSBRAIN_ROOT / "pretrained_weights" / "CSBrain.pth"

SUBJECTS = list("ABCDEFGHIJKLMN")
CHANNELS = 24
PATCHES = 10
PATCH_SIZE = 200
FEATURE_DIM = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("codebrain", "csbrain"), required=True)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(r"D:\data\HYGRIP-Baselines\prepared_eeg_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--codebrain-root", type=Path, default=CODEBRAIN_ROOT,
                        help="CodeBrain source root containing Models/SSSM.py")
    parser.add_argument("--csbrain-root", type=Path, default=CSBRAIN_ROOT,
                        help="CSBrain source root containing models/CSBrain.py")
    parser.add_argument("--train-subjects", default="A-J")
    parser.add_argument("--val-subjects", default="K-L")
    parser.add_argument("--test-subjects", default="M-N")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--fine-tune-backbone", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def parse_subjects(text: str) -> list[str]:
    result: list[str] = []
    for part in text.upper().split(","):
        part = part.strip()
        if "-" in part:
            start, stop = part.split("-", 1)
            result.extend(chr(value) for value in range(ord(start), ord(stop) + 1))
        elif part:
            result.append(part)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"invalid or duplicate subjects: {text}")
    if any(subject not in SUBJECTS for subject in result):
        raise ValueError(f"HYGRIP subjects must be A-N: {text}")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def load_split(root: Path, subjects: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays, labels, subject_ids = [], [], []
    for subject in subjects:
        path = root / f"subject_{subject}_trials.mat"
        item = loadmat(path, variable_names=["eeg_uv", "labels"])
        eeg = np.asarray(item["eeg_uv"], dtype=np.float32)
        target = np.asarray(item["labels"], dtype=np.int64).reshape(-1)
        if eeg.shape[1:] != (CHANNELS, 4000) or len(eeg) != len(target):
            raise RuntimeError(f"{path}: unexpected shapes {eeg.shape}, {target.shape}")
        # Match the corrected HYGRIP v2 CBraMod protocol: task-onset 0-10 s,
        # followed by per-trial, per-channel z-score at model load time.
        eeg = eeg[:, :, : PATCHES * PATCH_SIZE].astype(np.float64)
        eeg -= eeg.mean(axis=-1, keepdims=True)
        scale = eeg.std(axis=-1, keepdims=True)
        eeg = (eeg / np.maximum(scale, 1e-12)).astype(np.float32)
        eeg = np.ascontiguousarray(eeg.reshape(-1, CHANNELS, PATCHES, PATCH_SIZE))
        if not np.isfinite(eeg).all():
            raise RuntimeError(f"{path}: non-finite normalized EEG")
        arrays.append(eeg)
        labels.append(target)
        subject_ids.append(np.full(len(target), subject))
        print(f"subject {subject}: X={eeg.shape}, y={Counter(target.tolist())}", flush=True)
    return np.concatenate(arrays), np.concatenate(labels), np.concatenate(subject_ids)


class OfficialHead(nn.Module):
    """Official downstream MLP shape used by both foundation-model repositories."""

    def __init__(self, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(CHANNELS * PATCHES * FEATURE_DIM, PATCHES * FEATURE_DIM),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(PATCHES * FEATURE_DIM, FEATURE_DIM),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(FEATURE_DIM, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features.flatten(start_dim=1))


class FoundationClassifier(nn.Module):
    def __init__(self, model_name: str, checkpoint: Path, dropout: float,
                 fine_tune_backbone: bool, codebrain_root: Path, csbrain_root: Path):
        super().__init__()
        self.model_name = model_name
        if model_name == "codebrain":
            if not (codebrain_root / "Models" / "SSSM.py").is_file():
                raise FileNotFoundError(f"Invalid CodeBrain source root: {codebrain_root}")
            sys.path.insert(0, str(codebrain_root))
            from Models.SSSM import SSSM

            self.backbone = SSSM(
                in_channels=200,
                res_channels=200,
                skip_channels=200,
                out_channels=200,
                num_res_layers=8,
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
        else:
            if not (csbrain_root / "models" / "CSBrain.py").is_file():
                raise FileNotFoundError(f"Invalid CSBrain source root: {csbrain_root}")
            sys.path.insert(0, str(csbrain_root))
            from models.CSBrain import CSBrain

            # HYGRIP's 24 scalp channels form two dense grids around C3/C4;
            # under the official convention they all belong to central region 4.
            brain_regions = [4] * CHANNELS
            self.backbone = CSBrain(
                in_dim=200,
                out_dim=200,
                d_model=200,
                dim_feedforward=800,
                seq_len=CHANNELS,
                n_layer=12,
                nhead=8,
                brain_regions=brain_regions,
                sorted_indices=list(range(CHANNELS)),
            )
        self.pretrained_report = self._load_checkpoint(checkpoint)
        self.backbone.proj_out = nn.Identity()
        self.backbone.requires_grad_(fine_tune_backbone)
        self.classifier = OfficialHead(dropout)
        if model_name == "codebrain":
            # Match CodeBrain's published downstream-head initialization.
            for layer in self.classifier.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _load_checkpoint(self, checkpoint: Path) -> dict:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            state = raw.get("state_dict", raw.get("model", raw))
        else:
            state = raw
        cleaned = {key.removeprefix("module."): value for key, value in state.items()}
        current = self.backbone.state_dict()
        matched = {
            key: value
            for key, value in cleaned.items()
            if key in current and current[key].shape == value.shape
        }
        if not matched:
            raise RuntimeError(f"no compatible {self.model_name} checkpoint tensors")
        missing, unexpected = self.backbone.load_state_dict(matched, strict=False)
        return {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_tensors": len(cleaned),
            "matched_tensors": len(matched),
            "matched_parameters": int(sum(value.numel() for value in matched.values())),
            "backbone_parameters": int(sum(value.numel() for value in current.values())),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return features.reshape(x.shape[0], CHANNELS, PATCHES, FEATURE_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


def metrics(labels: np.ndarray, predictions: np.ndarray, loss: float) -> dict:
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(labels, predictions)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def cache_features(
    model: FoundationClassifier,
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
    batch_size: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
    model.eval()
    result = {}
    with torch.no_grad():
        for name, (x, y, subjects) in arrays.items():
            loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size)
            parts = []
            for (batch,) in loader:
                parts.append(model.encode(batch.to(device, non_blocking=True)).cpu())
            features = torch.cat(parts)
            result[name] = features, torch.from_numpy(y), subjects
            print(f"cached {name} features: {tuple(features.shape)}", flush=True)
    return result


def make_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    losses, labels, predictions = 0.0, [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses += float(loss.item()) * len(y)
            labels.append(y.cpu().numpy())
            predictions.append(logits.argmax(1).detach().cpu().numpy())
    labels_array = np.concatenate(labels)
    prediction_array = np.concatenate(predictions)
    return metrics(labels_array, prediction_array, losses / len(labels_array)), labels_array, prediction_array


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = args.checkpoint or (
        DEFAULT_CODEBRAIN_WEIGHT if args.model == "codebrain" else DEFAULT_CSBRAIN_WEIGHT
    )
    splits = {
        "train": parse_subjects(args.train_subjects),
        "val": parse_subjects(args.val_subjects),
        "test": parse_subjects(args.test_subjects),
    }
    flat = sum(splits.values(), [])
    if len(flat) != len(set(flat)):
        raise ValueError("train/val/test subjects overlap")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "args.json", vars(args) | {"resolved_checkpoint": checkpoint})

    arrays = {
        name: load_split(args.prepared_root.resolve(), subjects)
        for name, subjects in splits.items()
    }
    device = torch.device(args.device)
    model = FoundationClassifier(
        args.model,
        checkpoint.resolve(),
        args.dropout,
        fine_tune_backbone=args.fine_tune_backbone,
        codebrain_root=args.codebrain_root.resolve(),
        csbrain_root=args.csbrain_root.resolve(),
    ).to(device)
    probe = torch.from_numpy(arrays["train"][0][:1]).to(device)
    with torch.no_grad():
        probe_features = model.encode(probe)
        probe_logits = model.classifier(probe_features)
    diagnostics = {
        "dataset": "HYGRIP EEG v2",
        "task": "left hand (0) vs right hand (1) dynamic grip",
        "model": args.model,
        "training_mode": (
            "official pretrained initialization; full backbone fine-tuning"
            if args.fine_tune_backbone
            else "official pretrained backbone frozen; classification head only"
        ),
        "split_protocol": "subject-disjoint fixed split (normal split)",
        "splits": splits,
        "shapes": {
            name: {"X": list(values[0].shape), "y": list(values[1].shape), "labels": dict(Counter(values[1].tolist()))}
            for name, values in arrays.items()
        },
        "preprocessing": "HYGRIP v2 continuous correction; onset 0-10 s; per-trial per-channel z-score; [24,10,200]",
        "channel_regions": "24 channels around C3/C4; CSBrain region IDs are all central=4",
        "checkpoint_load": model.pretrained_report,
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "backbone_trainable": sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
        },
        "probe": {"input": list(probe.shape), "features": list(probe_features.shape), "logits": list(probe_logits.shape), "finite": bool(torch.isfinite(probe_logits).all())},
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
    if args.diagnose_only:
        return

    if args.fine_tune_backbone:
        cached = None
        training_model = model
        loaders = {
            name: make_loader(
                torch.from_numpy(values[0]),
                torch.from_numpy(values[1]),
                args.batch_size,
                name == "train",
                args.seed,
            )
            for name, values in arrays.items()
        }
        optimizer = torch.optim.AdamW(
            [
                {"params": model.classifier.parameters(), "lr": args.head_lr},
                {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        cached = cache_features(model, arrays, device, args.feature_batch_size)
        training_model = model.classifier.to(device)
        loaders = {
            name: make_loader(values[0], values[1], args.batch_size, name == "train", args.seed)
            for name, values in cached.items()
        }
        optimizer = torch.optim.AdamW(
            training_model.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
        )
    history, best_epoch, best_accuracy, best_val, best_state = [], 0, -1.0, None, None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train, _, _ = run_epoch(training_model, loaders["train"], device, optimizer)
        val, _, _ = run_epoch(training_model, loaders["val"], device)
        history.append({"epoch": epoch, "train": train, "val": val})
        if val["accuracy"] > best_accuracy + 1e-12:
            best_epoch, best_accuracy, best_val = epoch, val["accuracy"], val
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in training_model.state_dict().items()
            }
            torch.save(
                best_state,
                output_dir / ("best_model.pt" if args.fine_tune_backbone else "best_head.pt"),
            )
        print(
            f"epoch {epoch:03d}/{args.epochs} train_acc={train['accuracy']:.4f} "
            f"val_acc={val['accuracy']:.4f} best={best_accuracy:.4f}@{best_epoch} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )
        write_json(output_dir / "history.json", history)

    training_model.load_state_dict(best_state)
    test, test_labels, test_predictions = run_epoch(training_model, loaders["test"], device)
    test_subjects = arrays["test"][2]
    per_subject = []
    for subject in sorted(set(test_subjects.tolist())):
        chosen = test_subjects == subject
        per_subject.append({"subject": subject, "trials": int(chosen.sum()), **metrics(test_labels[chosen], test_predictions[chosen], 0.0)})
    summary = {
        "model": args.model,
        "dataset": "HYGRIP EEG v2",
        "training_mode": (
            "full backbone fine-tuning"
            if args.fine_tune_backbone
            else "classification head only"
        ),
        "learning_rates": {
            "head": args.head_lr,
            "backbone": args.backbone_lr if args.fine_tune_backbone else None,
        },
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_val": best_val,
        "test_metrics": test,
        "per_subject": per_subject,
        "elapsed_seconds": time.time() - started,
        "diagnostics": diagnostics,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
