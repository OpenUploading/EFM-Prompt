import argparse
import json
import os
from pathlib import Path

import mne
import numpy as np
import torch
from einops import rearrange
from scipy.io import loadmat
from scipy.signal import resample
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from timm.models import create_model
from torch.utils.data import DataLoader, TensorDataset

import modeling_finetune  # noqa: F401 - registers LaBraM models with timm
import utils


BCI2A_CH_NAMES = [
    "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P1", "PZ", "P2", "POZ",
]


def _subject_id(path: Path) -> str:
    return path.stem[:3]


def _load_true_labels(labels_dir: Path, stem: str) -> np.ndarray:
    label_path = labels_dir / f"{stem}.mat"
    mat = loadmat(label_path, squeeze_me=True)
    return np.asarray(mat["classlabel"], dtype=np.int64) - 1


def _extract_trials_from_gdf(path: Path, labels_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    sfreq = int(raw.info["sfreq"])
    eeg = raw.get_data(picks=np.arange(22), units="uV")
    eeg = np.nan_to_num(eeg)

    trial_code = event_id["768"]
    cue_codes = {event_id[str(code)]: code - 769 for code in (769, 770, 771, 772) if str(code) in event_id}
    trial_events = events[events[:, 2] == trial_code]

    if split == "test":
        labels = _load_true_labels(labels_dir, path.stem)
    else:
        cue_events = events[np.isin(events[:, 2], list(cue_codes.keys()))]
        labels_by_sample = {int(sample): cue_codes[int(code)] for sample, _, code in cue_events}
        labels = []
        for sample, _, _ in trial_events:
            following = cue_events[cue_events[:, 0] > sample]
            if len(following) == 0:
                labels.append(None)
            else:
                labels.append(labels_by_sample[int(following[0, 0])])

    samples = []
    targets = []
    start_offset = int(2.0 * sfreq)
    raw_window = int(4.0 * sfreq)
    out_window = 800

    for idx, event in enumerate(trial_events):
        if idx >= len(labels) or labels[idx] is None:
            continue
        start = int(event[0]) + start_offset
        stop = start + raw_window
        if stop > eeg.shape[1]:
            continue
        trial = eeg[:, start:stop]
        if sfreq != 200:
            trial = resample(trial, out_window, axis=1)
        samples.append(trial.astype(np.float32))
        targets.append(int(labels[idx]))

    return np.stack(samples), np.asarray(targets, dtype=np.int64)


def load_bci2a_split(data_root: Path, split: str, max_subjects: int | None, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    folder = data_root / split
    labels_dir = data_root / "true_labels"
    files = sorted(folder.glob("*.gdf"))
    if max_subjects:
        files = files[:max_subjects]

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{split}_{'-'.join(_subject_id(p) for p in files)}_22ch_4s_200hz.npz"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["X"], cached["y"]

    all_x = []
    all_y = []
    for path in files:
        x, y = _extract_trials_from_gdf(path, labels_dir, split)
        print(f"{split} {path.name}: {len(y)} trials")
        all_x.append(x)
        all_y.append(y)
    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def load_pretrained(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("module", checkpoint))
    state_dict = {k[8:] if k.startswith("student.") else k: v for k, v in state_dict.items()}
    model_state = model.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in state_dict and state_dict[key].shape != model_state[key].shape:
            del state_dict[key]
    for key in list(state_dict.keys()):
        if "relative_position_index" in key:
            del state_dict[key]
    utils.load_state_dict(model, state_dict)


def make_loaders(args):
    cache_dir = Path(args.output_dir) / "cache"
    X_train, y_train = load_bci2a_split(Path(args.data_root), "train", args.max_train_subjects, cache_dir)
    X_test, y_test = load_bci2a_split(Path(args.data_root), "test", args.max_test_subjects, cache_dir)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y_train))
    val_count = max(1, int(0.2 * len(idx)))
    val_idx = idx[:val_count]
    train_idx = idx[val_count:]

    train_ds = TensorDataset(torch.from_numpy(X_train[train_idx]), torch.from_numpy(y_train[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(X_train[val_idx]), torch.from_numpy(y_train[val_idx]))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
    )


def forward_batch(model, x, input_chans, device):
    x = x.float().to(device) / 100
    x = rearrange(x, "B N (A T) -> B N A T", T=200)
    return model(x, input_chans=input_chans)


def evaluate(model, loader, input_chans, device):
    model.eval()
    preds = []
    labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            logits = forward_batch(model, x, input_chans, device)
            total_loss += criterion(logits, y).item() * len(y)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=r"D:\BaiduNetdiskDownload\MI_BCI_IV_2a")
    parser.add_argument("--output_dir", default="outputs/bci2a_labram")
    parser.add_argument("--checkpoint", default="checkpoints/labram-base.pth")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_train_subjects", type=int, default=None)
    parser.add_argument("--max_test_subjects", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output_dir) / "mplconfig"))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, test_loader = make_loaders(args)
    device = torch.device(args.device)
    input_chans = utils.get_input_chans(BCI2A_CH_NAMES)

    model = create_model(
        "labram_base_patch200_200",
        pretrained=False,
        num_classes=4,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )
    load_pretrained(model, Path(args.checkpoint))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = torch.nn.CrossEntropyLoss()
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for step, (x, y) in enumerate(train_loader, start=1):
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, x, input_chans, device)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            seen += len(y)
            if step % 10 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {total_loss / seen:.4f}")

        val_stats = evaluate(model, val_loader, input_chans, device)
        test_stats = evaluate(model, test_loader, input_chans, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / seen,
            "val": val_stats,
            "test": test_stats,
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1, "history": history},
            Path(args.output_dir) / f"checkpoint-{epoch + 1}.pth",
        )

    with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
