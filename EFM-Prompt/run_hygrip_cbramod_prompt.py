"""Frozen-CBraMod, frozen-head cross-modal prompt experiments on HYGRIP."""

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
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
PORTABLE_ROOT = ROOT.parent
SUBJECTS = list("ABCDEFGHIJKLMN")
CHANNELS = 24
SAMPLE_RATE = 200
EEG_SECONDS = 10
PATCH_SAMPLES = 200
PATCHES = EEG_SECONDS

sys.path.insert(0, str(ROOT))
from cbramod_mope_boundary import MoPEBoundaryPrompt
from foundation_deep_prompt import DeepConditionalPrompt, SharedDeepThreeComponentPrompt
from foundation_hierarchical_cross_attention import FoundationHierarchicalCrossAttentionAdapter
from foundation_hierarchical_cross_attention_bidirectional_contrast import (
    FoundationHierarchicalBidirectionalContrastAdapter,
)
from foundation_tmpa_token_alignment import FoundationTMPAFinalAdapter
from run_shin2017_cbramod_fnirs_feature_stage1 import FnirsTemporalEncoder
from run_shin2017_foundation_boundary_prompt import load_compatible
from run_shin2017_foundation_tmpa_token_alignment import class_aware_contrastive_loss


class Trials(Dataset):
    def __init__(self, eeg, fnirs, labels, subjects, indices):
        self.eeg = eeg
        self.fnirs = fnirs
        self.labels = labels
        self.subjects = subjects
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        return self.eeg[index], self.fnirs[index], self.labels[index], index


def parse_subjects(text: str) -> list[str]:
    result = []
    for part in text.upper().split(","):
        part = part.strip()
        if "-" in part:
            start, stop = part.split("-", 1)
            result.extend(chr(value) for value in range(ord(start), ord(stop) + 1))
        elif part:
            result.append(part)
    if not result or len(result) != len(set(result)) or any(x not in SUBJECTS for x in result):
        raise ValueError(f"Invalid HYGRIP subjects: {text}")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "mope",
            "deep_conditional",
            "deep_three_component_shared",
            "tmpa_final",
            "hierarchical_cross_attention",
            "bidirectional_contrast",
        ),
        required=True,
    )
    parser.add_argument(
        "--prepared-root", type=Path,
        default=Path(r"D:\data\HYGRIP-Baselines\prepared_eeg_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, default=PORTABLE_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eegonly-head-checkpoint", type=Path, required=True)
    parser.add_argument("--train-subjects", default="A-J")
    parser.add_argument("--val-subjects", default="K-L")
    parser.add_argument("--test-subjects", default="M-N")
    parser.add_argument("--fnirs-window", type=float, nargs=2, default=(3.0, 13.0), metavar=("START", "END"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-pair", type=float, default=0.1)
    parser.add_argument("--lambda-class", type=float, default=0.02)
    parser.add_argument("--importance-weight", type=float, default=0.01)
    parser.add_argument("--prompt-boundary", choices=("pre", "pre_post"), default="pre")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_eeg(eeg_uv: np.ndarray) -> np.ndarray:
    eeg = np.asarray(eeg_uv[..., : EEG_SECONDS * SAMPLE_RATE], dtype=np.float64)
    eeg -= eeg.mean(axis=-1, keepdims=True)
    eeg /= np.maximum(eeg.std(axis=-1, keepdims=True), 1e-12)
    return eeg.astype(np.float32).reshape(-1, CHANNELS, PATCHES, PATCH_SAMPLES)


def load_trials(root: Path, train_subjects: list[str], fnirs_window: tuple[float, float]):
    eeg_parts, fnirs_parts, label_parts, subject_parts = [], [], [], []
    for subject in SUBJECTS:
        path = root / f"subject_{subject}_trials.mat"
        data = loadmat(path, variable_names=["eeg_uv", "fnirs_um", "labels"])
        eeg = np.asarray(data["eeg_uv"], dtype=np.float32)
        fnirs = np.asarray(data["fnirs_um"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
        if eeg.shape != (len(labels), CHANNELS, 4000):
            raise RuntimeError(f"{path}: invalid EEG shape {eeg.shape}")
        if fnirs.shape != (len(labels), 2, CHANNELS, 250):
            raise RuntimeError(f"{path}: invalid fNIRS shape {fnirs.shape}")
        if Counter(labels.tolist()) not in (Counter({0: 10, 1: 10}), Counter({0: 13, 1: 13})):
            raise RuntimeError(f"{path}: invalid labels {Counter(labels.tolist())}")
        times = np.arange(fnirs.shape[-1], dtype=np.float64) / (fnirs.shape[-1] / 20.0)
        selected = (times >= fnirs_window[0] - 1e-9) & (times < fnirs_window[1] - 1e-9)
        if not selected.any():
            raise RuntimeError(f"{path}: empty fNIRS window {fnirs_window}")
        eeg_parts.append(preprocess_eeg(eeg))
        fnirs_parts.append(fnirs[..., selected].transpose(0, 2, 1, 3))
        label_parts.append(labels)
        subject_parts.append(np.full(len(labels), subject))
    eeg = np.concatenate(eeg_parts)
    fnirs = np.concatenate(fnirs_parts).astype(np.float32, copy=False)
    labels = np.concatenate(label_parts)
    subject_ids = np.concatenate(subject_parts)
    train = fnirs[np.isin(subject_ids, train_subjects)]
    mean = train.mean(axis=(0, 3), keepdims=True, dtype=np.float64)
    std = train.std(axis=(0, 3), keepdims=True, dtype=np.float64)
    fnirs = ((fnirs - mean) / np.maximum(std, 1e-12)).astype(np.float32)
    if not np.isfinite(eeg).all() or not np.isfinite(fnirs).all():
        raise RuntimeError("HYGRIP preprocessing produced non-finite values")
    return eeg, fnirs, labels, subject_ids


def load_head(classifier: nn.Module, checkpoint: Path) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HYGRIP EEG-only head not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(key.startswith("classifier.") for key in state):
        state = {
            key.removeprefix("classifier."): value
            for key, value in state.items() if key.startswith("classifier.")
        }
    result = classifier.load_state_dict(state, strict=True)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "loaded_keys": len(state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


class HYGRIPCBraModPrompt(nn.Module):
    def __init__(self, args):
        super().__init__()
        sys.path.insert(0, str(args.portable_root / "CBraMod"))
        from models.cbramod import CBraMod

        self.method = args.method
        # Keep the official 30-channel CBraMod positional configuration used by
        # the existing HYGRIP EEG-only runner; the actual token grid has 24 channels.
        self.backbone = CBraMod(200, 200, 200, 800, 30, 12, 8)
        self.pretrained_report = load_compatible(self.backbone, args.checkpoint)
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(CHANNELS * PATCHES * 200, PATCHES * 200),
            nn.ELU(), nn.Dropout(args.dropout),
            nn.Linear(PATCHES * 200, 200),
            nn.ELU(), nn.Dropout(args.dropout), nn.Linear(200, 2),
        )
        self.head_report = load_head(self.classifier, args.eegonly_head_checkpoint)
        self.prompt = None
        if self.method == "mope":
            prompt_args = dict(
                condition_dim=256, d_model=200, prompt_count=6, rank=8,
                token_count=CHANNELS * PATCHES, hidden_dim=256, dropout=args.dropout,
                expert_count=16, temperature=0.1, router_noise_std=0.00390625,
                importance_threshold=0.05,
            )
            self.prompt = nn.ModuleDict({
                "fnirs_encoder": FnirsTemporalEncoder(CHANNELS * 2, 256, args.dropout),
                "pre": MoPEBoundaryPrompt(**prompt_args),
            })
            if args.prompt_boundary == "pre_post":
                self.prompt["post"] = MoPEBoundaryPrompt(**prompt_args)
        elif self.method == "deep_conditional":
            self.prompt = DeepConditionalPrompt(
                200, 128, 4, 3, 10, 8, 0.05, args.dropout,
            )
            self.deep_stages = {5: 0, 8: 1, 11: 2}
        elif self.method == "deep_three_component_shared":
            self.prompt = SharedDeepThreeComponentPrompt(
                eeg_dim=200, prompt_dim=128, prompt_tokens=6, stages=4,
                fnirs_temporal_tokens=10, attention_heads=8,
                prompt_scale=0.05, dropout=args.dropout, expert_count=16,
                router_temperature=0.1, router_noise_std=0.00390625,
                importance_threshold=0.05, prompt_rank=8, prompt_hidden=256,
            )
            self.deep_stages = {2: 0, 5: 1, 8: 2, 11: 3}
        else:
            adapter_type = {
                "tmpa_final": FoundationTMPAFinalAdapter,
                "hierarchical_cross_attention": FoundationHierarchicalCrossAttentionAdapter,
                "bidirectional_contrast": FoundationHierarchicalBidirectionalContrastAdapter,
            }[self.method]
            self.prompt = adapter_type(
                eeg_dim=200, alignment_dim=128, fnirs_temporal_tokens=10,
                mode_count=4, prompt_tokens_per_mode=2, attention_heads=8,
                token_cost_weight=1.0, prompt_scale=0.05,
                sinkhorn_epsilon=0.1, sinkhorn_iterations=100,
                dropout=args.dropout,
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.classifier.parameters():
            parameter.requires_grad = False

    def forward(self, eeg, fnirs, pair_matrix=True):
        tokens = self.backbone.patch_embedding(eeg)
        auxiliary = None
        if self.method == "mope":
            sequence = fnirs.permute(0, 3, 1, 2).reshape(fnirs.shape[0], fnirs.shape[-1], -1)
            condition = self.prompt["fnirs_encoder"](sequence)
            tokens = tokens + self.prompt["pre"](condition).view_as(tokens)
            tokens = self.backbone.encoder(tokens)
            if "post" in self.prompt:
                tokens = tokens + self.prompt["post"](condition).view_as(tokens)
        elif self.method in {"deep_conditional", "deep_three_component_shared"}:
            context = (
                self.prompt.encode_fnirs(fnirs)
                if self.method == "deep_three_component_shared" else None
            )
            for layer_index, layer in enumerate(self.backbone.encoder.layers):
                if layer_index in self.deep_stages:
                    stage = self.deep_stages[layer_index]
                    tokens = (
                        self.prompt.inject(tokens, context, stage)
                        if context is not None else self.prompt(tokens, fnirs, stage)
                    )
                tokens = layer(tokens)
        else:
            tokens, auxiliary = self.prompt(tokens, fnirs, compute_pair_matrix=pair_matrix)
            tokens = self.backbone.encoder(tokens)
        return self.classifier(tokens.flatten(1)), auxiliary

    def importance_loss(self):
        if self.method == "deep_three_component_shared":
            return self.prompt.importance_loss()
        if self.method != "mope":
            return next(self.parameters()).new_zeros(())
        losses = [self.prompt["pre"].importance_loss()]
        if "post" in self.prompt:
            losses.append(self.prompt["post"].importance_loss())
        return sum(losses) / len(losses)


def scores(y_true, y_pred, loss):
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def run_epoch(model, loader, optimizer, device, args, training):
    model.train(training)
    model.backbone.eval()
    model.classifier.eval()
    criterion = nn.CrossEntropyLoss()
    total = seen = 0
    predicted, targets, indices = [], [], []
    for eeg, fnirs, target, index in loader:
        eeg, fnirs = eeg.to(device).float(), fnirs.to(device).float()
        target = target.to(device).long()
        with torch.set_grad_enabled(training):
            logits, auxiliary = model(eeg, fnirs, pair_matrix=training)
            loss = criterion(logits, target)
            if training and model.method in {"mope", "deep_three_component_shared"}:
                loss = loss + args.importance_weight * model.importance_loss()
            elif training and model.method in {
                "tmpa_final", "hierarchical_cross_attention", "bidirectional_contrast"
            }:
                pair, same_class = class_aware_contrastive_loss(
                    auxiliary["sample_distance"], target, 0.1
                )
                loss = loss + args.lambda_pair * pair + args.lambda_class * same_class
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss.detach()) * len(target)
        seen += len(target)
        predicted.append(logits.detach().argmax(1).cpu().numpy())
        targets.append(target.cpu().numpy())
        indices.append(index.numpy())
    y_true, y_pred = np.concatenate(targets), np.concatenate(predicted)
    return scores(y_true, y_pred, total / seen), y_true, y_pred, np.concatenate(indices)


def main():
    args = parse_args()
    seed_all(args.seed)
    args.prepared_root = args.prepared_root.resolve()
    args.portable_root = args.portable_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = (
        args.checkpoint
        or args.portable_root / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth"
    ).resolve()
    args.eegonly_head_checkpoint = args.eegonly_head_checkpoint.resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    splits = {
        "train": parse_subjects(args.train_subjects),
        "val": parse_subjects(args.val_subjects),
        "test": parse_subjects(args.test_subjects),
    }
    flattened = sum(splits.values(), [])
    if len(flattened) != len(set(flattened)):
        raise ValueError("HYGRIP subject splits overlap")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eeg, fnirs, labels, subject_ids = load_trials(
        args.prepared_root, splits["train"], tuple(args.fnirs_window)
    )
    split_indices = {
        name: np.flatnonzero(np.isin(subject_ids, subjects))
        for name, subjects in splits.items()
    }
    loaders = {
        name: DataLoader(
            Trials(eeg, fnirs, labels, subject_ids, indices),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(args.seed) if name == "train" else None,
        )
        for name, indices in split_indices.items()
    }
    device = torch.device(args.device)
    model = HYGRIPCBraModPrompt(args).to(device)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(name.startswith(("backbone.", "classifier.")) for name in trainable):
        raise RuntimeError(f"Invalid trainable parameter set: {trainable[:10]}")
    optimizer = torch.optim.AdamW(
        model.prompt.parameters(), lr=args.prompt_lr, weight_decay=args.weight_decay
    )
    probe_eeg = torch.from_numpy(eeg[split_indices["train"][:2]]).to(device)
    probe_fnirs = torch.from_numpy(fnirs[split_indices["train"][:2]]).to(device)
    with torch.no_grad():
        probe_logits, _ = model(probe_eeg, probe_fnirs, pair_matrix=False)
    diagnostic = {
        "method": args.method,
        "dataset": "HYGRIP left/right dynamic grip",
        "splits": splits,
        "eeg_shape": list(eeg.shape),
        "fnirs_shape": list(fnirs.shape),
        "fnirs_window_seconds": list(args.fnirs_window),
        "fnirs_normalization": "per node/chromophore; train subjects only; trials/time",
        "eeg_preprocessing": "prepared_eeg_v2; first 10 s; per-trial per-channel z-score",
        "probe_logits_shape": list(probe_logits.shape),
        "backbone_frozen": True,
        "classifier_frozen": True,
        "pretrained_load": model.pretrained_report,
        "head_load": model.head_report,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "args": vars(args),
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if args.diagnose_only:
        print(json.dumps(diagnostic, ensure_ascii=False, indent=2, default=str))
        return
    history, best_acc, best_epoch, best_state = [], -1.0, 0, None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train, _, _, _ = run_epoch(model, loaders["train"], optimizer, device, args, True)
        val, _, _, _ = run_epoch(model, loaders["val"], None, device, args, False)
        history.append({"epoch": epoch, "train": train, "val": val})
        print(
            f"epoch {epoch:03d}/{args.epochs} train={train['accuracy']:.4f} "
            f"val={val['accuracy']:.4f}", flush=True,
        )
        if val["accuracy"] > best_acc:
            best_acc, best_epoch = val["accuracy"], epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.prompt.state_dict().items()
            }
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    model.prompt.load_state_dict(best_state)
    test, y_true, y_pred, test_indices = run_epoch(
        model, loaders["test"], None, device, args, False
    )
    per_subject = []
    test_subject_ids = subject_ids[test_indices]
    for subject in splits["test"]:
        mask = test_subject_ids == subject
        per_subject.append({
            "subject": subject,
            "trials": int(mask.sum()),
            **scores(y_true[mask], y_pred[mask], 0.0),
        })
    torch.save(best_state, args.output_dir / "best_prompt.pth")
    summary = {
        "method": args.method,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_acc,
        "test": test,
        "per_subject": per_subject,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
