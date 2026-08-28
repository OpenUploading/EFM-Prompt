"""Paired ds004022 EEG/fNIRS trials for DAMFNet.

EEG is deliberately reused from the established 4--40 Hz, CAR, 200-Hz,
five-second MI cache.  The raw fNIRS MATLAB files contain a MATLAB ``table``;
``mat-io`` decodes that MCOS object without requiring MATLAB.  Since the
released DS files do not document an HbO/HbR conversion, this loader retains
the signed 24 stored fNIRS channels, applies the SHIN 0.01--0.1-Hz filter,
pre-MI baseline correction, and per-trial z-score, and calls the branch
``nirs`` rather than incorrectly claiming it is HbR.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


EEG_CACHE = Path(r"D:\0senior student creation\braindecode_codebrain_prep\cache\ds004022_eeg_mi5s_4to40hz_200hz.npz")
TASKS = {
    "mi": {
        "name": "ds004022 EEG-fNIRS motor imagery",
        "description": "reach (0), grasp (1), lift (2), twist (3)",
        "labels": {"reach": 0, "grasp": 1, "lift": 2, "twist": 3},
    }
}
TASK_CODES = {3: 0, 4: 1, 5: 2, 6: 3}
MI_START_CODE = 8


def _unwrap(value):
    """Remove MATLAB's singleton ndarray wrappers produced by mat-io."""
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def _field(struct, name: str):
    value = struct[name] if isinstance(struct, np.void) else struct[name]
    return _unwrap(value)


def _mat_run(path: Path) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    try:
        from matio import load_from_mat
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install mat-io in the training environment") from exc
    root = load_from_mat(path)["nirs_data"]
    root = _unwrap(root)
    cnt = _field(_field(root, "cnt"), "cnt")
    mrk = _field(_field(root, "mrk"), "mrk")
    signal = _field(cnt, "x")
    if not hasattr(signal, "to_numpy"):
        raise RuntimeError(f"{path}: expected decoded MATLAB table, got {type(signal)!r}")
    signal = signal.to_numpy(dtype=np.float64, copy=True)
    fs = float(np.asarray(_field(cnt, "fs")).squeeze())
    positions = np.asarray(_field(mrk, "pos"), dtype=np.float64).reshape(-1)
    codes = np.asarray(_field(mrk, "toe"), dtype=np.int64).reshape(-1)
    if signal.ndim != 2 or signal.shape[1] != 24:
        raise RuntimeError(f"{path}: expected continuous fNIRS [time,24], got {signal.shape}")
    if abs(fs - 7.8125) > 1e-6:
        raise RuntimeError(f"{path}: expected 7.8125-Hz fNIRS, got {fs}")
    return signal, fs, positions, codes


def _normalise_trial(trial: np.ndarray) -> np.ndarray:
    mean = trial.mean(dtype=np.float64)
    std = trial.std(dtype=np.float64)
    if not np.isfinite(std) or std < 1e-8:
        raise RuntimeError(f"Invalid fNIRS trial standard deviation: {std}")
    return ((trial - mean) / std).astype(np.float32)


def _subject_nirs(dataset_root: Path, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trials, labels, keys = [], [], []
    b, a = butter(3, [0.01, 0.1], btype="bandpass", fs=7.8125)
    for session in (1, 2, 3):
        path = dataset_root / f"sub-{subject:02d}" / "fnirs" / (
            f"sub-{subject:02d}_task-motorimagery_run-{session}_nirs.mat"
        )
        signal, fs, positions, codes = _mat_run(path)
        filtered = filtfilt(b, a, signal, axis=0)
        mi_indices = np.flatnonzero(codes == MI_START_CODE)
        if len(mi_indices) != 40:
            raise RuntimeError(f"{path}: expected 40 MI-start markers, got {len(mi_indices)}")
        # One DS run (sub-02/run-1) has a duplicate task marker immediately
        # before trial 31.  The first task cue after the preceding MI period
        # agrees with the paired EEG event sequence; the later marker is the
        # anomalous duplicate and must not overwrite the trial label.
        paired = []
        previous_mi = -1
        for mi_index in mi_indices:
            candidates = [
                index for index in range(previous_mi + 1, int(mi_index))
                if codes[index] in TASK_CODES
            ]
            if not candidates:
                raise RuntimeError(f"{path}: no task marker before MI marker {mi_index}")
            cue_index = candidates[0]
            paired.append((TASK_CODES[int(codes[cue_index])], positions[mi_index]))
            previous_mi = int(mi_index)
        for trial_id, (label, mi_pos) in enumerate(paired):
            # MATLAB marker positions are one-based.  Five seconds at 7.8125 Hz
            # is resampled exactly (32/25) to 50 points at the DAMFNet 10-Hz rate.
            start = int(round(mi_pos)) - 1
            stop = start + int(round(5.0 * fs))
            base_start, base_stop = start - int(round(5.0 * fs)), start - int(round(2.0 * fs))
            if base_start < 0 or stop > len(filtered):
                raise RuntimeError(f"{path}: fNIRS MI/baseline window out of range")
            trial = filtered[start:stop] - filtered[base_start:base_stop].mean(axis=0, keepdims=True)
            trial = resample_poly(trial, 32, 25, axis=0)
            if trial.shape != (50, 24):
                raise RuntimeError(f"{path}: resampled fNIRS must be [50,24], got {trial.shape}")
            trials.append(_normalise_trial(trial.T))
            labels.append(label)
            # The established EEG cache numbers trials globally across the
            # complete dataset (sub-01: 0..119, sub-02: 120..239, ...).
            global_trial = (subject - 1) * 120 + (session - 1) * 40 + trial_id
            keys.append((subject, session, global_trial))
    return np.stack(trials), np.asarray(labels, dtype=np.int64), np.asarray(keys, dtype=np.int64)


def _all_nirs(dataset_root: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = cache_dir / "ds004022_signed_fnirs_0to5s_shin_filter_baseline_zscore_v4.npz"
    if cache_path.exists():
        item = np.load(cache_path, allow_pickle=False)
        return item["nirs"], item["labels"], item["keys"]
    arrays = [_subject_nirs(dataset_root, subject) for subject in range(1, 8)]
    nirs = np.concatenate([part[0] for part in arrays])
    labels = np.concatenate([part[1] for part in arrays])
    keys = np.concatenate([part[2] for part in arrays])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, nirs=nirs, labels=labels, keys=keys)
    return nirs, labels, keys


def load_split(
    eeg_root: Path,
    fnirs_root: Path,
    subjects: list[int],
    split_name: str,
    task_key: str,
    cache_dir: Path,
    sensor_layout: str = "project_all",
    epoch_start_s: float = 0.0,
    epoch_stop_s: float = 5.0,
):
    del eeg_root, task_key, sensor_layout
    if epoch_start_s != 0.0 or epoch_stop_s != 5.0:
        raise ValueError("ds004022 DAMFNet uses the complete documented 0..5-s MI epoch")
    eeg_item = np.load(EEG_CACHE, allow_pickle=True)
    eeg = eeg_item["eeg"].astype(np.float32)
    labels = eeg_item["labels"].astype(np.int64)
    eeg_keys = np.column_stack((eeg_item["subject_ids"], eeg_item["session_ids"], eeg_item["trial_ids"])).astype(np.int64)
    nirs, nirs_labels, nirs_keys = _all_nirs(fnirs_root, cache_dir)
    nirs_by_key = {tuple(key): (row, label) for key, row, label in zip(nirs_keys, nirs, nirs_labels)}
    selected = np.isin(eeg_keys[:, 0], np.asarray(subjects, dtype=np.int64))
    selected_keys = eeg_keys[selected]
    selected_hbr, selected_nirs_labels = [], []
    for key in selected_keys:
        item = nirs_by_key.get(tuple(key))
        if item is None:
            raise RuntimeError(f"Missing paired fNIRS trial for key={tuple(key)}")
        selected_hbr.append(item[0]); selected_nirs_labels.append(item[1])
    selected_hbr = np.stack(selected_hbr).astype(np.float32)
    selected_labels = labels[selected]
    if not np.array_equal(selected_labels, np.asarray(selected_nirs_labels, dtype=np.int64)):
        raise RuntimeError("EEG/fNIRS labels disagree after subject/session/trial pairing")
    return eeg[selected], selected_hbr, selected_labels, [{
        "source": "EEG CAR+4-40Hz+200Hz cache; signed fNIRS table -> 0.01-0.1Hz -> -5..-2s baseline -> 10Hz -> per-trial z-score",
        "subjects": subjects, "trials": int(selected.sum()), "label_counts": dict(Counter(selected_labels.tolist())),
    }]
