"""Prompt tuning for CBraMod on SHIN EEG.

The experiment code lives outside the CBraMod source tree so the prompt adapter
can later be ported to other EEG foundation models with minimal changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CBraMod prompt tuning on SHIN EEG")
    parser.add_argument("--portable-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\DataSets\SHIN\v1.0.1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:\data\EFM-Prompt-SHIN\cache"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--task", choices=("mi", "ma"), default="mi")
    parser.add_argument("--prompt-mode", choices=("none", "static", "context"), default="static")
    parser.add_argument("--prompt-scale", type=float, default=0.05)
    parser.add_argument("--prompt-hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-epoch", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-subjects-per-split", type=int, default=None)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--experiment-note", default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def metrics(y_true, y_pred, loss):
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


class CBraModPromptModel(nn.Module):
    def __init__(
        self,
        cbramod_cls,
        prompt_mode: str,
        prompt_scale: float,
        prompt_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.prompt_mode = prompt_mode
        self.prompt_scale = prompt_scale
        self.channels = 30
        self.patch_count = 10
        self.d_model = 200
        self.backbone = cbramod_cls(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )
        if prompt_mode == "static":
            self.static_prompt = nn.Parameter(torch.zeros(1, self.channels, self.patch_count, self.d_model))
            self.context_prompt = None
        elif prompt_mode == "context":
            self.static_prompt = None
            self.context_prompt = nn.Sequential(
                nn.Linear(self.channels * 2, prompt_hidden),
                nn.GELU(),
                nn.Linear(prompt_hidden, self.channels * self.d_model + self.patch_count * self.d_model),
            )
        else:
            self.static_prompt = None
            self.context_prompt = None

        self.classifier = nn.Sequential(
            nn.Linear(self.channels * self.patch_count * self.d_model, self.patch_count * self.d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.patch_count * self.d_model, self.d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 2),
        )

    def prompt_parameters(self):
        if self.prompt_mode == "static":
            return [self.static_prompt]
        if self.prompt_mode == "context":
            return list(self.context_prompt.parameters())
        return []

    def _prompt(self, x: torch.Tensor, patch_emb: torch.Tensor) -> torch.Tensor:
        if self.prompt_mode == "none":
            return torch.zeros_like(patch_emb)
        if self.prompt_mode == "static":
            return self.static_prompt.expand(x.shape[0], -1, -1, -1)

        channel_mean = x.mean(dim=(2, 3))
        channel_std = x.std(dim=(2, 3), unbiased=False)
        stats = torch.cat([channel_mean, channel_std], dim=1)
        factors = self.context_prompt(stats)
        ch_end = self.channels * self.d_model
        channel_factor = factors[:, :ch_end].view(-1, self.channels, 1, self.d_model)
        patch_factor = factors[:, ch_end:].view(-1, 1, self.patch_count, self.d_model)
        return channel_factor + patch_factor

    def features(self, x: torch.Tensor) -> torch.Tensor:
        patch_emb = self.backbone.patch_embedding(x)
        patch_emb = patch_emb + self.prompt_scale * self._prompt(x, patch_emb)
        feats = self.backbone.encoder(patch_emb)
        feats = self.backbone.proj_out(feats)
        return feats.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_pretrained(model: CBraModPromptModel, checkpoint: Path) -> dict:
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(source, dict) and "model" in source:
        source = source["model"]
    result = model.backbone.load_state_dict(source, strict=True)
    model.backbone.proj_out = nn.Identity()
    return {
        "checkpoint": str(checkpoint.resolve()),
        "loaded_keys": len(source),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "proj_out_after_loading": "Identity",
    }


def set_trainable(model: CBraModPromptModel, train_backbone: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    for parameter in model.prompt_parameters():
        parameter.requires_grad = True
    if train_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True


def make_optimizer(model: CBraModPromptModel, args: argparse.Namespace, train_backbone: bool):
    groups = [
        {"params": list(model.classifier.parameters()), "lr": args.head_lr, "name": "head"},
    ]
    prompt_params = model.prompt_parameters()
    if prompt_params:
        groups.insert(0, {"params": prompt_params, "lr": args.prompt_lr, "name": "prompt"})
    if train_backbone:
        groups.insert(0, {"params": list(model.backbone.parameters()), "lr": args.backbone_lr, "name": "backbone"})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, seen, predictions, labels = 0.0, 0, [], []
    for x, y in data_loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total += float(loss.item()) * len(y)
        seen += len(y)
        predictions.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return metrics(np.concatenate(labels), np.concatenate(predictions), total / seen)


def train_epoch(model, data_loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, seen = 0.0, 0
    for x, y in data_loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(y)
        seen += len(y)
    return total / seen


def make_report(args, summary, trainable_counts) -> str:
    task = summary["task"]
    best, test, final = summary["best"], summary["best_test"], summary["final"]
    return f"""# EFM Prompt × CBraMod × SHIN EEG

## Experiment

{args.experiment_note}

The prompt is inserted after CBraMod patch embedding and before the Transformer
encoder. This changes the token representation seen by the backbone without
modifying the original SHIN loader or the pretrained CBraMod implementation.

## Parameters

| Item | Value |
|---|---:|
| Backbone | CBraMod Base |
| Task | SHIN {task['name']}; {task['description']} |
| Prompt mode | {args.prompt_mode} |
| Prompt scale | {args.prompt_scale:g} |
| Prompt lr | {args.prompt_lr:g} |
| Head lr | {args.head_lr:g} |
| Backbone lr | {args.backbone_lr:g} |
| Backbone unfreeze epoch | {args.unfreeze_epoch} |
| Epochs | {args.epochs} |
| Batch size | {args.batch_size} |
| Seed | {args.seed} |
| Checkpoint | `{args.checkpoint.resolve()}` |
| Trainable prompt params | {trainable_counts['prompt']} |
| Trainable head params | {trainable_counts['head']} |
| Backbone params | {trainable_counts['backbone']} |

## Results

| Checkpoint | Epoch | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Test Kappa |
|---|---:|---:|---:|---:|---:|---:|
| Best validation | {best['epoch']} | {best['val']['accuracy']:.4f} | {best['val']['f1_macro']:.4f} | {test['accuracy']:.4f} | {test['f1_macro']:.4f} | {test['cohen_kappa']:.4f} |
| Last | {final['epoch']} | {final['val']['accuracy']:.4f} | {final['val']['f1_macro']:.4f} | {final['test']['accuracy']:.4f} | {final['test']['f1_macro']:.4f} | {final['test']['cohen_kappa']:.4f} |

Best-test confusion matrix: `{test['confusion_matrix']}`.
"""


def main() -> None:
    args = parse_args()
    args.portable_root = args.portable_root.resolve()
    cbramod_root = args.portable_root / "CBraMod"
    sys.path.insert(0, str(cbramod_root))
    from models.cbramod import CBraMod
    from run_shin_finetune import TASKS, load_split, loader

    if args.checkpoint is None:
        args.checkpoint = cbramod_root / "pretrained_weights" / "pretrained_weights.pth"
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.experiment_note is None:
        args.experiment_note = (
            "Evaluate prompt tuning on CBraMod under the fixed SHIN subject-independent split."
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "mplconfig"))
    seed_everything(args.seed)

    task = TASKS[args.task]
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
        "backbone": "CBraMod",
        "prompt_position": "after patch_embedding, before encoder",
        "prompt_mode": args.prompt_mode,
        "prompt_scale": args.prompt_scale,
        "task": {"key": args.task, "name": task["name"], "description": task["description"]},
        "splits": {
            name: {
                "subjects": split_subjects[name],
                "shape": list(arrays[name].shape),
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
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    loaders = {
        name: loader(arrays[name], targets[name], args.batch_size, name == "train",
                     args.num_workers, args.seed)
        for name in arrays
    }

    model = CBraModPromptModel(
        CBraMod,
        prompt_mode=args.prompt_mode,
        prompt_scale=args.prompt_scale,
        prompt_hidden=args.prompt_hidden,
        dropout=args.dropout,
    )
    diagnostics["pretrained_load"] = load_pretrained(model, args.checkpoint)
    trainable_counts = {
        "prompt": sum(p.numel() for p in model.prompt_parameters()),
        "head": sum(p.numel() for p in model.classifier.parameters()),
        "backbone": sum(p.numel() for p in model.backbone.parameters()),
    }
    diagnostics["parameters"] = trainable_counts
    write_json(args.output_dir / "diagnostics.json", diagnostics)
    model.to(device)

    train_backbone = args.unfreeze_epoch <= 1
    set_trainable(model, train_backbone=train_backbone)
    optimizer = make_optimizer(model, args, train_backbone=train_backbone)

    history, best_record, best_accuracy = [], None, -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch and not train_backbone:
            train_backbone = True
            set_trainable(model, train_backbone=True)
            optimizer = make_optimizer(model, args, train_backbone=True)
            print(f"[train] epoch {epoch}: unfreezing CBraMod backbone", flush=True)

        train_loss = train_epoch(model, loaders["train"], optimizer, device)
        val = evaluate(model, loaders["val"], device)
        record = {
            "epoch": epoch,
            "stage": "prompt_tuning" if not train_backbone else "prompt_plus_finetune",
            "train_loss": train_loss,
            "val": val,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs} stage={record['stage']} "
            f"train_loss={train_loss:.4f} val_acc={val['accuracy']:.4f} val_f1={val['f1_macro']:.4f}",
            flush=True,
        )
        if val["accuracy"] > best_accuracy:
            best_accuracy, best_record = val["accuracy"], record
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       args.output_dir / "best_model.pth")
        write_json(args.output_dir / "history.json", history)

    final_test = evaluate(model, loaders["test"], device)
    final = {"epoch": args.epochs, "val": history[-1]["val"], "test": final_test}
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args)},
               args.output_dir / "last_model.pth")
    best_checkpoint = torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    best_test = evaluate(model, loaders["test"], device)
    summary = {
        "best": best_record,
        "best_test": best_test,
        "final": final,
        "elapsed_seconds": time.time() - started,
        "seed": args.seed,
        "task": diagnostics["task"],
        "prompt_mode": args.prompt_mode,
        "prompt_scale": args.prompt_scale,
        "experiment_note": args.experiment_note,
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "EXPERIMENT_RECORD.md").write_text(
        make_report(args, summary, trainable_counts), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

