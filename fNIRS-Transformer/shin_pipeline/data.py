"""Direct SHIN MATLAB loader matching the paper's Dataset B preprocessing."""

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt


TASKS = {
    "mi": {
        "sessions": (0, 2, 4),
        "label_map": {"left_hand": 0, "right_hand": 1},
    },
    "ma": {
        "sessions": (1, 3, 5),
        "label_map": {"mental_arithmetic": 0, "baseline_rest": 1},
    },
}

# MNE/OMLC absorption coefficients at 760 and 850 nm, columns=[HbO, HbR].
# Coefficients include MNE's 0.2303 conversion factor.
ABSORPTION = np.asarray(
    [[134.9558, 356.624156], [243.6574, 159.210996]], dtype=np.float64
)


def _subject_dir(root: Path, subject: int) -> Path:
    return root / f"subject {subject:02d}"


def _validate_channel_order(clab) -> list[str]:
    names = [str(value) for value in np.asarray(clab).reshape(-1)]
    if len(names) != 72:
        raise ValueError(f"expected 72 wavelength channels, got {len(names)}")
    low = [name.removesuffix("lowWL") for name in names[:36]]
    high = [name.removesuffix("highWL") for name in names[36:]]
    if any(not name.endswith("lowWL") for name in names[:36]):
        raise ValueError("first 36 channels are not all low-wavelength channels")
    if any(not name.endswith("highWL") for name in names[36:]):
        raise ValueError("last 36 channels are not all high-wavelength channels")
    if low != high:
        raise ValueError("760/850 nm source-detector channel orders differ")
    return low


def intensity_to_haemoglobin(intensity: np.ndarray, distance_m=0.03, ppf=6.0):
    """Convert 760/850 nm continuous intensity to HbO/HbR in micromolar."""
    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.ndim != 2 or intensity.shape[1] != 72:
        raise ValueError(f"expected continuous intensity [time,72], got {intensity.shape}")

    # Match the standard MNE optical-density handling for non-positive samples.
    safe = np.abs(intensity)
    for channel in range(safe.shape[1]):
        positive = safe[:, channel][safe[:, channel] > 0]
        if not len(positive):
            raise ValueError(f"intensity channel {channel} contains no positive sample")
        safe[:, channel] = np.maximum(safe[:, channel], positive.min())
    optical_density = -np.log(safe / safe.mean(axis=0, keepdims=True))

    # delta_OD = extinction * distance * PPF * delta_concentration.
    inverse = np.linalg.pinv(ABSORPTION * float(distance_m) * float(ppf)) * 1e-3
    low, high = optical_density[:, :36], optical_density[:, 36:]
    concentration_molar = np.einsum(
        "ab,tcb->tca", inverse, np.stack((low, high), axis=-1)
    )
    # [time, channel, HbO/HbR] -> [time, 2, channel], expressed in micromolar.
    return concentration_molar.transpose(0, 2, 1) * 1e6


def load_subject(data_root: Path, subject: int, task: str = "ma"):
    if task not in TASKS:
        raise ValueError(f"unknown SHIN task: {task}")
    task_config = TASKS[task]
    folder = _subject_dir(Path(data_root), subject)
    cnt_path, mrk_path = folder / "cnt.mat", folder / "mrk.mat"
    if not cnt_path.exists() or not mrk_path.exists():
        raise FileNotFoundError(f"missing cnt.mat/mrk.mat under {folder}")
    cnt = loadmat(cnt_path, simplify_cells=True)["cnt"]
    mrk = loadmat(mrk_path, simplify_cells=True)["mrk"]

    continuous_parts, events = [], []
    channel_names, fs, offset = None, None, 0
    session_lengths = []
    for session_index in task_config["sessions"]:
        recording, markers = cnt[session_index], mrk[session_index]
        current_fs = float(recording["fs"])
        if fs is None:
            fs = current_fs
        if current_fs != fs or fs != 10.0:
            raise ValueError(f"subject {subject}: expected every fNIRS session at 10 Hz")
        current_names = _validate_channel_order(recording["clab"])
        if channel_names is None:
            channel_names = current_names
        if current_names != channel_names:
            raise ValueError(f"subject {subject}: channel order changes between sessions")
        x = np.asarray(recording["x"], dtype=np.float64)
        continuous_parts.append(x)
        session_lengths.append(int(len(x)))
        times_ms = np.asarray(markers["time"], dtype=np.float64).reshape(-1)
        descriptions = np.asarray(markers["event"]["desc"], dtype=np.int64).reshape(-1)
        if len(times_ms) != 20 or Counter(descriptions.tolist()) != Counter({1: 10, 2: 10}):
            raise ValueError(f"subject {subject}, session {session_index + 1}: invalid events")
        for time_ms, description in zip(times_ms, descriptions):
            sample = offset + int(round(time_ms * fs / 1000.0))
            # SHIN uses event codes 1 and 2 for both binary tasks.
            events.append((sample, int(description - 1), session_index + 1))
        offset += len(x)

    continuous = np.concatenate(continuous_parts, axis=0)
    haemoglobin = intensity_to_haemoglobin(continuous)
    b, a = butter(3, [0.01, 0.1], btype="bandpass", fs=fs)
    haemoglobin = filtfilt(b, a, haemoglobin, axis=0)

    baseline_start, baseline_stop = int(5 * fs), int(2 * fs)
    task_points = int(20 * fs)
    trials, labels = [], []
    for sample, label, _ in events:
        if sample - baseline_start < 0 or sample + task_points > len(haemoglobin):
            raise ValueError(f"subject {subject}: event window is out of bounds at {sample}")
        baseline = haemoglobin[sample - baseline_start : sample - baseline_stop].mean(
            axis=0, keepdims=True
        )
        trial = haemoglobin[sample : sample + task_points] - baseline
        trials.append(trial.transpose(1, 2, 0))  # [2,36,200]
        labels.append(label)

    x = np.asarray(trials, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.shape != (60, 2, 36, 200) or Counter(y.tolist()) != Counter({0: 30, 1: 30}):
        raise ValueError(f"subject {subject}: unexpected output {x.shape}, labels={Counter(y.tolist())}")
    if not np.isfinite(x).all():
        raise ValueError(f"subject {subject}: non-finite haemoglobin values")
    return x, y, {
        "subject": subject,
        "task": task,
        "label_map": task_config["label_map"],
        "sessions": [value + 1 for value in task_config["sessions"]],
        "session_continuous_samples": session_lengths,
        "sfreq": fs,
        "channels": channel_names,
        "output_shape": list(x.shape),
        "class_counts": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
        "processing": [
            "optical density",
            "modified Beer-Lambert (distance=0.03 m, PPF=6.0)",
            "Butterworth order-3 bandpass 0.01-0.1 Hz",
            "baseline -5 to -2 seconds",
            "task window 0 to 20 seconds",
        ],
    }


def normalize_trials(x: np.ndarray) -> np.ndarray:
    """Per-trial global z-score, matching the original Dataset wrapper."""
    mean = x.mean(axis=(1, 2, 3), keepdims=True, dtype=np.float64)
    std = x.std(axis=(1, 2, 3), keepdims=True, dtype=np.float64)
    return np.asarray((x - mean) / np.maximum(std, 1e-8), dtype=np.float32)


def load_subjects(data_root: Path, subjects, task: str = "ma"):
    xs, ys, infos = [], [], []
    for subject in subjects:
        x, y, info = load_subject(data_root, int(subject), task=task)
        xs.append(x)
        ys.append(y)
        infos.append(info)
        print(f"subject {subject:02d}: x={x.shape} labels={dict(Counter(y.tolist()))}")
    return normalize_trials(np.concatenate(xs)), np.concatenate(ys), infos
