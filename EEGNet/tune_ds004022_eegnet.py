"""Validation-only hyperparameter tuning for ds004022 four-class EEGNet.

The test split is deliberately held out during tuning.  Only the selected
configuration is evaluated on it once, after all validation comparisons finish.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eegnet_pytorch import EEGNet
import run_ds004022_eegnet as dataset
import run_shin_eegnet as portable


OUTPUT_ROOT = Path(r"D:\data\EEGNet-SHIN")

# Curated around the observed failure mode: training fit improves while the
# validation loss rises.  This is intentionally a small search, not a large
# random sweep on a 7-subject dataset.
CONFIGS = [
    {"lr": 3e-4, "weight_decay": 1e-4, "dropout": 0.25, "kernel_length": 64, "label_smoothing": 0.00},
    {"lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.25, "kernel_length": 64, "label_smoothing": 0.05},
    {"lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.50, "kernel_length": 64, "label_smoothing": 0.00},
    {"lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.50, "kernel_length": 64, "label_smoothing": 0.05},
    {"lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.50, "kernel_length": 100, "label_smoothing": 0.05},
    {"lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.50, "kernel_length": 64, "label_smoothing": 0.05},
    {"lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.50, "kernel_length": 64, "label_smoothing": 0.05},
    {"lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.50, "kernel_length": 100, "label_smoothing": 0.05},
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune EEGNet on ds004022 using validation Macro-F1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds; use 1,2,3 for a robust rerun")
    parser.add_argument("--train-subjects", default="1-5")
    parser.add_argument("--val-subjects", default="6")
    parser.add_argument("--test-subjects", default="7")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-configs", type=int, default=None, help="For a quick smoke test")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must be non-empty and unique")
    return seeds


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_arrays(subjects: dict[str, list[int]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays, labels = {}, {}
    for name in ("train", "val", "test"):
        arrays[name], labels[name], _ = dataset.load_split(None, name, subjects[name], Path("."), "mi", {})
    return arrays, labels


def build_model(x: np.ndarray, classes: int, config: dict) -> EEGNet:
    return EEGNet(
        channels=x.shape[1], samples=x.shape[2], classes=classes,
        dropout=config["dropout"], kernel_length=config["kernel_length"],
    )


def train_one(
    arrays: dict[str, np.ndarray], labels: dict[str, np.ndarray], config: dict,
    seed: int, epochs: int, patience: int, batch_size: int, device: torch.device,
    evaluate_test: bool = False,
) -> tuple[dict, dict | None]:
    """Train with validation early stopping; test is optional and final-only."""
    portable.seed_all(seed)
    classes = int(max(values.max() for values in labels.values())) + 1
    model = build_model(arrays["train"], classes, config).to(device)
    loaders = {
        name: portable.loader(arrays[name], labels[name], batch_size, name == "train", 0, seed)
        for name in ("train", "val", "test")
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    best_state, best_val, best_epoch, stalled = None, None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = seen = 0
        for x, y in loaders["train"]:
            x, y = x.to(device).float(), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            model.constrain_weights()
            loss_sum += float(loss.item()) * len(y)
            seen += len(y)
        val = portable.evaluate(model, loaders["val"], device, criterion, None)
        train_loss = loss_sum / seen
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val})
        improved = best_val is None or (
            val["f1_macro"] > best_val["f1_macro"] + 1e-6
            or (abs(val["f1_macro"] - best_val["f1_macro"]) <= 1e-6 and val["loss"] < best_val["loss"])
        )
        if improved:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_val, best_epoch, stalled = val, epoch, 0
        else:
            stalled += 1
        if stalled >= patience:
            break
    assert best_state is not None and best_val is not None
    model.load_state_dict(best_state)
    result = {
        "seed": seed, "config": config, "best_epoch": best_epoch,
        "epochs_ran": len(history), "best_val": best_val,
        "best_train_loss": history[best_epoch - 1]["train_loss"], "history": history,
    }
    test = portable.evaluate(model, loaders["test"], device, criterion, None) if evaluate_test else None
    return result, test


def main() -> None:
    args = arguments()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seeds = parse_seeds(args.seeds)
    subjects = {
        "train": portable.parse_subjects(args.train_subjects),
        "val": portable.parse_subjects(args.val_subjects),
        "test": portable.parse_subjects(args.test_subjects),
    }
    flat = [subject for split in subjects.values() for subject in split]
    if len(flat) != len(set(flat)):
        raise ValueError("Train/val/test subjects must not overlap")
    configs = CONFIGS[:args.max_configs] if args.max_configs else CONFIGS
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output_dir or OUTPUT_ROOT / f"ds004022_eegnet_tuning_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    arrays, labels = load_arrays(subjects)
    print(f"Tuning {len(configs)} configurations x {len(seeds)} seed(s); held-out test subjects={subjects['test']}.", flush=True)
    trials = []
    for config_id, config in enumerate(configs, 1):
        per_seed = []
        for seed in seeds:
            result, _ = train_one(arrays, labels, config, seed, args.epochs, args.patience, args.batch_size, device)
            per_seed.append(result)
            val = result["best_val"]
            print(f"config={config_id} seed={seed} epoch={result['best_epoch']} val_f1={val['f1_macro']:.4f} val_acc={val['accuracy']:.4f}", flush=True)
        mean_f1 = float(np.mean([item["best_val"]["f1_macro"] for item in per_seed]))
        mean_loss = float(np.mean([item["best_val"]["loss"] for item in per_seed]))
        trials.append({"config_id": config_id, "config": config, "runs": per_seed, "mean_val_f1": mean_f1, "mean_val_loss": mean_loss})
    trials.sort(key=lambda item: (-item["mean_val_f1"], item["mean_val_loss"]))
    winner = trials[0]
    # One final test evaluation only, using the first requested seed.
    final_run, final_test = train_one(arrays, labels, winner["config"], seeds[0], args.epochs, args.patience, args.batch_size, device, evaluate_test=True)
    report = {
        "protocol": "Strict subject-independent split. Configurations ranked by validation Macro-F1 only; test participants evaluated once after selection.",
        "subjects": subjects,
        "search": {"configs": len(configs), "seeds": seeds, "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size},
        "ranking": trials,
        "selected": winner,
        "final_selected_run": final_run,
        "final_test_once": final_test,
    }
    write_json(output / "tuning_summary.json", report)
    with (output / "ranking.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "config_id", "lr", "weight_decay", "dropout", "kernel_length", "label_smoothing", "mean_val_f1", "mean_val_loss"])
        writer.writeheader()
        for rank, item in enumerate(trials, 1):
            writer.writerow({"rank": rank, "config_id": item["config_id"], **item["config"], "mean_val_f1": item["mean_val_f1"], "mean_val_loss": item["mean_val_loss"]})
    print(json.dumps({"output": str(output), "selected": winner["config"], "final_test_once": final_test}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
