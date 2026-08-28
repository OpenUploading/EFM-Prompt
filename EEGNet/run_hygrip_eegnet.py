"""Run EEGNet, fNIRS-T, or DAMFNet on prepared HYGRIP trials."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.signal import resample
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
PORTABLE_ROOT = HERE.parent
sys.path.insert(0, str(PORTABLE_ROOT / "EEGNet"))
from eegnet_pytorch import EEGNet
sys.path.insert(0, str(PORTABLE_ROOT / "fNIRS-Transformer"))
from model import fNIRS_T
sys.path.insert(0, str(PORTABLE_ROOT / "DAMFNet"))
from models.shin_damfnet import SHINDAMFNet
from shin_data import WindowDataset


SUBJECTS = list("ABCDEFGHIJKLMN")
TASK = {"name": "HYGRIP hand classification", "labels": {"left": 0, "right": 1}}


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["eegnet", "fnirst", "damfnet"], required=True)
    p.add_argument("--prepared-root", type=Path, default=Path(r"D:\data\HYGRIP-Baselines\prepared"))
    p.add_argument("--eeg-normalization", choices=("global_zscore", "channel_zscore"), default="global_zscore")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-subjects", default="A-J")
    p.add_argument("--val-subjects", default="K-L")
    p.add_argument("--test-subjects", default="M-N")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int)
    p.add_argument("--diagnose-only", action="store_true")
    return p.parse_args()


def parse_subjects(value: str) -> list[str]:
    result = []
    for part in value.upper().split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(chr(i) for i in range(ord(a), ord(b) + 1))
        elif part:
            result.append(part)
    if not result or any(x not in SUBJECTS for x in result) or len(result) != len(set(result)):
        raise ValueError(f"Invalid subjects: {value}")
    return result


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_split(root: Path, subjects: list[str]):
    eeg, fnirs, labels, details = [], [], [], []
    for subject in subjects:
        path = root / f"subject_{subject}_trials.mat"
        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        x_eeg = np.asarray(d["eeg_uv"], dtype=np.float32)
        x_fnirs = np.asarray(d["fnirs_um"], dtype=np.float32)
        y = np.asarray(d["labels"], dtype=np.int64).reshape(-1)
        if x_eeg.shape != (len(y), 24, 4000) or x_fnirs.shape != (len(y), 2, 24, 250):
            raise RuntimeError(f"{path}: unexpected shapes {x_eeg.shape}, {x_fnirs.shape}, {y.shape}")
        if Counter(y.tolist()) not in (Counter({0: 10, 1: 10}), Counter({0: 13, 1: 13})):
            raise RuntimeError(f"{path}: unexpected labels {Counter(y.tolist())}")
        if not np.isfinite(x_eeg).all() or not np.isfinite(x_fnirs).all():
            raise RuntimeError(f"{path}: non-finite input")
        eeg.append(x_eeg); fnirs.append(x_fnirs); labels.append(y)
        details.append({"subject": subject, "trials": len(y), "labels": dict(Counter(y.tolist())), "file": str(path)})
    return np.concatenate(eeg), np.concatenate(fnirs), np.concatenate(labels), details


def zscore_trial(x):
    mean = x.mean(axis=tuple(range(1, x.ndim)), keepdims=True, dtype=np.float64)
    std = x.std(axis=tuple(range(1, x.ndim)), keepdims=True, dtype=np.float64)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def normalize_eeg(x, mode):
    if mode == "global_zscore":
        return zscore_trial(x)
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float64)
    std = x.std(axis=-1, keepdims=True, dtype=np.float64)
    return ((x - mean) / np.maximum(std, 1e-12)).astype(np.float32)


def metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p)), "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y, p)),
            "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist()}


def make_loader(dataset, batch, shuffle, workers, seed):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)


@torch.no_grad()
def evaluate_single(model, loader, device, criterion):
    model.eval(); loss_sum = seen = 0; pred = []; true = []
    for x, y in loader:
        x, y = x.to(device).float(), y.to(device)
        logits = model(x); loss = criterion(logits, y)
        loss_sum += float(loss.item()) * len(y); seen += len(y)
        pred.append(logits.argmax(1).cpu().numpy()); true.append(y.cpu().numpy())
    y, p = np.concatenate(true), np.concatenate(pred)
    return {"loss": loss_sum / seen, **metrics(y, p)}


def damf_loss(outputs, y, criterion):
    return sum(criterion(output, y) for output in outputs)


@torch.no_grad()
def evaluate_damf(model, loader, device, criterion):
    model.eval(); loss_sum = seen = 0; logits_all = []; y_all = []; ids_all = []
    for eeg, hbr, y, ids in loader:
        eeg, hbr, y = eeg.to(device).float(), hbr.to(device).float(), y.to(device)
        outputs = model(eeg, hbr); loss = damf_loss(outputs, y, criterion)
        loss_sum += float(loss.item()) * len(y); seen += len(y)
        logits_all.append(outputs[2].cpu().numpy()); y_all.append(y.cpu().numpy()); ids_all.append(ids.numpy())
    logits, y, ids = np.concatenate(logits_all), np.concatenate(y_all), np.concatenate(ids_all)
    grouped, grouped_y = defaultdict(list), {}
    for logit, label, trial in zip(logits, y, ids, strict=True):
        grouped[int(trial)].append(logit); grouped_y[int(trial)] = int(label)
    true = np.asarray([grouped_y[k] for k in sorted(grouped)])
    pred = np.asarray([np.mean(grouped[k], axis=0).argmax() for k in sorted(grouped)])
    return {"loss": loss_sum / seen, "window": metrics(y, logits.argmax(1)), "trial": metrics(true, pred)}


def main():
    args = arguments(); seed_all(args.seed)
    splits = {"train": parse_subjects(args.train_subjects), "val": parse_subjects(args.val_subjects), "test": parse_subjects(args.test_subjects)}
    if len(sum(splits.values(), [])) != len(set(sum(splits.values(), []))): raise ValueError("Subject splits overlap")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, details = {}, {}
    for name in ("train", "val", "test"):
        eeg, fnirs, y, detail = load_split(args.prepared_root, splits[name])
        raw[name] = {"eeg": eeg, "fnirs": fnirs, "y": y}; details[name] = detail
        print(f"[{name}] subjects={splits[name]} trials={len(y)} labels={Counter(y.tolist())}", flush=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    datasets = {}; model_note = ""; probe_info = {}

    if args.model == "eegnet":
        for name in raw:
            x = normalize_eeg(raw[name]["eeg"], args.eeg_normalization)
            datasets[name] = TensorDataset(torch.from_numpy(x), torch.from_numpy(raw[name]["y"]))
        model = EEGNet(channels=24, samples=4000, classes=2, dropout=args.dropout, kernel_length=64).to(device)
        probe = torch.from_numpy(normalize_eeg(raw["train"]["eeg"][:2], args.eeg_normalization)).to(device)
        with torch.no_grad(): outputs = model(probe)
        probe_info = {"input": list(probe.shape), "output": list(outputs.shape), "finite": bool(torch.isfinite(outputs).all())}
        model_note = f"EEGNet full-model training; {args.eeg_normalization}"
    elif args.model == "fnirst":
        train = raw["train"]["fnirs"]
        mean = train.mean(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32)
        std = np.maximum(train.std(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32), 1e-6)
        for name in raw:
            x = ((raw[name]["fnirs"] - mean) / std).astype(np.float32)
            datasets[name] = TensorDataset(torch.from_numpy(x), torch.from_numpy(raw[name]["y"]))
        model = fNIRS_T(n_class=2, sampling_point=250, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
        probe = datasets["train"].tensors[0][:2].to(device)
        with torch.no_grad(): outputs = model(probe)
        probe_info = {"input": list(probe.shape), "output": list(outputs.shape), "finite": bool(torch.isfinite(outputs).all())}
        model_note = "fNIRS-T on author-provided HbO/HbR; train-split channel normalization"
    else:
        for name in raw:
            eeg = normalize_eeg(raw[name]["eeg"], args.eeg_normalization)
            hbr = resample(raw[name]["fnirs"][:, 1], 200, axis=-1).astype(np.float32)
            hbr = zscore_trial(hbr)
            datasets[name] = WindowDataset(eeg, hbr, raw[name]["y"], epoch_start_s=0, window_seconds=3, stride_seconds=1)
        model = SHINDAMFNet(dropout=args.dropout, sensor_layout="project_all", eeg_input_nodes=24, hbr_input_nodes=24, n_classes=2).to(device)
        pe, ph, _, _ = datasets["train"][0]
        with torch.no_grad(): outputs = model(pe[None].to(device), ph[None].to(device))
        probe_info = {"eeg": [1, *pe.shape], "hbr": [1, *ph.shape], "outputs": [list(x.shape) for x in outputs], "finite": all(bool(torch.isfinite(x).all()) for x in outputs)}
        model_note = "DAMFNet EEG+HbR; EEG 24->8 learned projection, native 24 HbR nodes; 18 overlapping windows/trial"

    diagnostics = {"dataset": "HYGRIP", "task": TASK, "model": args.model, "model_note": model_note,
                   "split": splits, "seed": args.seed, "preprocessing": {
                       "epoch": "task onset 0-20 s", "eeg": f"prepared EEG; {args.eeg_normalization}",
                       "fnirs": "author-provided oxy/dxy mol; 0.01-0.1 Hz; 1 s pre-onset baseline; umol/L; no motion correction"},
                   "sets": {name: {"trials": len(raw[name]["y"]), "labels": dict(Counter(raw[name]["y"].tolist())), "details": details[name]} for name in raw},
                   "parameters": sum(p.numel() for p in model.parameters()), "forward": probe_info}
    write_json(args.output_dir / "diagnostics.json", diagnostics); print(json.dumps(probe_info), flush=True)
    if args.diagnose_only: return

    loaders = {name: make_loader(datasets[name], args.batch_size, name == "train", args.num_workers, args.seed) for name in datasets}
    criterion = nn.CrossEntropyLoss()
    if args.model == "fnirst":
        head = list(model.mlp_head.parameters()); ids = {id(p) for p in head}
        optimizer = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in ids], "lr": args.backbone_lr}, {"params": head, "lr": args.head_lr}], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); scheduler = None

    history = []; best = None; best_acc = -1.; stale = 0; started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = seen = 0
        for step, batch in enumerate(loaders["train"], 1):
            if args.max_batches and step > args.max_batches: break
            optimizer.zero_grad(set_to_none=True)
            if args.model == "damfnet":
                eeg, hbr, y, _ = batch; eeg, hbr, y = eeg.to(device).float(), hbr.to(device).float(), y.to(device)
                loss = damf_loss(model(eeg, hbr), y, criterion)
            else:
                x, y = batch; x, y = x.to(device).float(), y.to(device)
                raw_loss = criterion(model(x), y)
                if args.model == "fnirst":
                    flood = 0.45 if epoch <= 30 else 0.40
                    loss = (raw_loss - flood).abs() + flood
                else: loss = raw_loss
            loss.backward(); optimizer.step()
            if args.model == "eegnet": model.constrain_weights()
            loss_sum += float(loss.item()) * len(y); seen += len(y)
        val = evaluate_damf(model, loaders["val"], device, criterion) if args.model == "damfnet" else evaluate_single(model, loaders["val"], device, criterion)
        selection = val["trial"]["accuracy"] if args.model == "damfnet" else val["accuracy"]
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "val": val, "elapsed_seconds": time.time() - started}
        history.append(row); print(f"epoch {epoch:03d}/{args.epochs} loss={row['train_loss']:.5f} val_acc={selection:.4f}", flush=True)
        if selection > best_acc:
            best_acc, best, stale = selection, row, 0; torch.save({"model": model.state_dict(), "record": row}, args.output_dir / "best.pt")
        else: stale += 1
        if scheduler: scheduler.step()
        write_json(args.output_dir / "history.json", history)
        if args.patience and stale >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best['epoch']}", flush=True); break

    evaluator = evaluate_damf if args.model == "damfnet" else evaluate_single
    final_test = evaluator(model, loaders["test"], device, criterion)
    torch.save({"model": model.state_dict(), "test": final_test}, args.output_dir / "last.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"])
    best_test = evaluator(model, loaders["test"], device, criterion)
    summary = {"run_finished": datetime.now().astimezone().isoformat(timespec="seconds"), "model": args.model, "best": best,
               "best_test": best_test, "final_test": final_test, "history": history, "diagnostics": diagnostics,
               "schedule": {"epochs": args.epochs, "trained_epochs": history[-1]["epoch"], "batch_size": args.batch_size,
                            "lr": args.lr, "head_lr": args.head_lr, "backbone_lr": args.backbone_lr,
                            "weight_decay": args.weight_decay, "dropout": args.dropout, "patience": args.patience}}
    write_json(args.output_dir / "summary.json", summary)
    primary = best_test["trial"] if args.model == "damfnet" else best_test
    print(json.dumps({"best_epoch": best["epoch"], "best_val": best_acc, "best_test": primary}, indent=2), flush=True)


if __name__ == "__main__": main()
