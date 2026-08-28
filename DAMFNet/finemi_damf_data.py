"""FineMI paired EEG/HbR trials shaped for DAMFNet."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample


EEG_ROOT = Path(r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_paper_car_uv100_binary_1v6")
FNIRS_ROOT = Path(r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_prompt_fnirs_binary_1v6_0_12s")
TASKS = {
    "mi": {
        "name": "FineMI EEG-HbR motor imagery",
        "description": "paper binary event 1 (0) vs event 6 (1)",
        "labels": {"event_1": 0, "event_6": 1},
    }
}


def _load_all(eeg_root: Path, fnirs_root: Path):
    eeg_parts, hbr_parts, label_parts, subject_parts = [], [], [], []
    for subject in range(1, 19):
        with np.load(eeg_root / f"subject{subject:02d}_paired.npz", allow_pickle=False) as item:
            eeg = item["eeg"].astype(np.float32, copy=False)
            labels = item["labels"].astype(np.int64, copy=False)
            blocks = item["block_ids"]
        with np.load(fnirs_root / f"subject{subject:02d}_fnirs_prompt.npz", allow_pickle=False) as item:
            times = item["fnirs_times_s"]
            selected = (times >= 3.0 - 1e-6) & (times <= 7.0 + 1e-6)
            # DAMFNet has one fNIRS branch.  Match its SHIN use of HbR rather
            # than concatenating HbO/HbR and changing the branch semantics.
            hbr = item["fnirs_graph"][:, :, 1, selected].astype(np.float32, copy=False)
            if not np.array_equal(labels, item["labels"]) or not np.array_equal(blocks, item["block_ids"]):
                raise RuntimeError(f"FineMI subject{subject:02d}: EEG/fNIRS trial ordering differs")
        if eeg.shape != (80, 62, 800) or hbr.shape[0:2] != (80, 24):
            raise RuntimeError(f"FineMI subject{subject:02d}: unexpected EEG/HbR shapes {eeg.shape}/{hbr.shape}")
        # 3..7 s is a four-second window.  DAMFNet's fNIRS branch is 10 Hz,
        # hence 40 points, aligned to the 800 EEG points at 200 Hz.
        hbr = resample(hbr, 40, axis=-1).astype(np.float32)
        eeg_parts.append(eeg); hbr_parts.append(hbr); label_parts.append(labels)
        subject_parts.append(np.full(len(labels), subject, dtype=np.int16))
    return (np.concatenate(eeg_parts), np.concatenate(hbr_parts),
            np.concatenate(label_parts), np.concatenate(subject_parts))


def load_split(
    eeg_root: Path,
    fnirs_root: Path,
    subjects: list[int],
    split_name: str,
    task_key: str,
    cache_dir: Path,
    sensor_layout: str = "project_all",
    epoch_start_s: float = 0.0,
    epoch_stop_s: float = 4.0,
):
    del split_name, task_key, cache_dir, sensor_layout
    if epoch_start_s != 0.0 or epoch_stop_s != 4.0:
        raise ValueError("FineMI DAMFNet uses its existing complete 0..4-s EEG epoch")
    eeg, hbr, labels, subject_ids = _load_all(eeg_root, fnirs_root)
    selected = np.isin(subject_ids, np.asarray(subjects, dtype=np.int16))
    train_subjects = set(range(1, 13))
    train = np.isin(subject_ids, np.asarray(sorted(train_subjects), dtype=np.int16))
    mean = hbr[train].mean(axis=(0, 2), keepdims=True, dtype=np.float64)
    std = hbr[train].std(axis=(0, 2), keepdims=True, dtype=np.float64)
    hbr = ((hbr - mean) / np.maximum(std, 1e-12)).astype(np.float32)
    return eeg[selected], hbr[selected], labels[selected], [{
        "subjects": subjects,
        "trials": int(selected.sum()),
        "label_counts": dict(Counter(labels[selected].tolist())),
        "source": "EEG paper CAR, uV/100, 0..4 s; HbR 3..7 s, resampled to 10 Hz; HbR train-subject-only normalization",
    }]
