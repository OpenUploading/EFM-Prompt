"""Prepare/load compact FineMI EEG-fNIRS data for EFM prompt experiments.

The fNIRS cache is shared data only.  The three-component prompt and final
TMPA keep their own architectures and objectives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


EEG_ROOT = Path(
    r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_rawuv_binary_1v6"
)
FNIRS_ROOT = Path(
    r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_prompt_fnirs_binary_1v6_0_12s"
)
PROMPT_WINDOW_S = (3.0, 7.0)
SPLITS = {
    "train": list(range(1, 13)),
    "val": list(range(13, 16)),
    "test": list(range(16, 19)),
}


def validate_subject(subject: int) -> dict:
    path = FNIRS_ROOT / f"subject{subject:02d}_fnirs_prompt.npz"
    with np.load(path, allow_pickle=False) as item:
        graph = item["fnirs_graph"]
        times = item["fnirs_times_s"]
        labels = item["labels"]
        blocks = item["block_ids"]
    if graph.shape[:3] != (80, 24, 2) or graph.shape[-1] != len(times) or graph.dtype != np.float32:
        raise RuntimeError(f"{path}: invalid graph {graph.shape}/{graph.dtype}")
    if np.bincount(labels, minlength=2).tolist() != [40, 40] or not np.isfinite(graph).all():
        raise RuntimeError(f"{path}: invalid labels/content")
    with np.load(EEG_ROOT / f"subject{subject:02d}_paired.npz", allow_pickle=False) as eeg:
        if not np.array_equal(labels, eeg["labels"]) or not np.array_equal(blocks, eeg["block_ids"]):
            raise RuntimeError(f"subject{subject}: EEG/fNIRS ordering differs")
    return {"subject": subject, "shape": list(graph.shape), "bytes": path.stat().st_size}


def load_prompt_trials(
    subjects: list[int],
    normalize_from_subjects: list[int] | None = None,
    prompt_window_s: tuple[float, float] = PROMPT_WINDOW_S,
    eeg_root: Path = EEG_ROOT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    eeg_all, fnirs_all, labels_all, subject_all = [], [], [], []
    for subject in subjects:
        with np.load(eeg_root / f"subject{subject:02d}_paired.npz", allow_pickle=False) as item:
            eeg = item["eeg"].astype(np.float32, copy=False).reshape(-1, 62, 4, 200)
            labels = item["labels"].astype(np.int64, copy=False)
            eeg_blocks = item["block_ids"]
        with np.load(FNIRS_ROOT / f"subject{subject:02d}_fnirs_prompt.npz", allow_pickle=False) as item:
            times = item["fnirs_times_s"]
            peak = (times >= prompt_window_s[0] - 1e-6) & (times <= prompt_window_s[1] + 1e-6)
            fnirs = item["fnirs_graph"][..., peak].astype(np.float32, copy=False)
            if not np.array_equal(labels, item["labels"]) or not np.array_equal(eeg_blocks, item["block_ids"]):
                raise RuntimeError(f"subject{subject}: EEG/fNIRS ordering differs")
        eeg_all.append(eeg)
        fnirs_all.append(fnirs)
        labels_all.append(labels)
        subject_all.append(np.full(len(labels), subject, dtype=np.int16))
    eeg = np.concatenate(eeg_all)
    fnirs = np.concatenate(fnirs_all)
    labels = np.concatenate(labels_all)
    subject_ids = np.concatenate(subject_all)
    normalization = None
    if normalize_from_subjects is not None:
        train = fnirs[np.isin(subject_ids, normalize_from_subjects)]
        mean = train.mean(axis=(0, 3), keepdims=True, dtype=np.float64)
        std = train.std(axis=(0, 3), keepdims=True, dtype=np.float64)
        # Hb concentrations are stored in mol/L (typical std ~1e-7), so the
        # generic 1e-6 floor used by older SHIN code would suppress the signal.
        fnirs = ((fnirs - mean) / np.maximum(std, 1e-12)).astype(np.float32)
        normalization = "per node/chromophore, train subjects only, across trials/time"
    meta = {
        "dataset": "FineMI/Yi2025 binary event 1 vs 6",
        "eeg": f"{eeg_root}; [N,62,4,200]",
        "fnirs": f"OD -> MBLL HbO/HbR -> 0.01-0.1 Hz -> {prompt_window_s[0]:g}..{prompt_window_s[1]:g} s; train-only normalization",
        "fnirs_normalization": normalization,
        "subject_ids": subject_ids,
    }
    return eeg, fnirs, labels, meta
