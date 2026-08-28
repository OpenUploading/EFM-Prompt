"""Aligned SHIN EEG/HbR loader for DAMFNet's 3-second windows."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset


SHIN_EEG_CHANNELS = [
    "F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8",
    "AFF1h", "AFF2h", "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h",
    "CCP3h", "T7", "P7", "P3", "PPO1h", "POO1", "POO2", "PPO2h",
    "P4", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "P8", "T8",
]

# Fixed sensor layout used by the portable DAMFNet baseline. The eight EEG
# sensors are the bilateral FCC/CCP electrodes corresponding to its fixed
# eight-node input. The 24 fNIRS nodes are the same final 24 source-detector
# pairs used by that baseline. No discarded sensor is mixed into the model.
DAMF_FIXED_EEG_INDICES = (12, 13, 14, 15, 24, 25, 26, 27)
DAMF_FIXED_FNIRS_INDICES = tuple(range(12, 36))
SENSOR_LAYOUTS = {
    "project_all": {
        "eeg_indices": tuple(range(30)),
        "fnirs_indices": tuple(range(36)),
    },
    "damf_fixed": {
        "eeg_indices": DAMF_FIXED_EEG_INDICES,
        "fnirs_indices": DAMF_FIXED_FNIRS_INDICES,
    },
}

TASKS = {
    "mi": {
        "name": "EEG-fNIRS-MI",
        "description": "left_hand (0) vs right_hand (1)",
        "session_indices": (0, 2, 4),
        "session_names": ("ses-0imagery", "ses-2imagery", "ses-4imagery"),
        "labels": {"left_hand": 0, "right_hand": 1},
    },
    "ma": {
        "name": "EEG-fNIRS-MA",
        "description": "subtraction (0) vs rest (1)",
        "session_indices": (1, 3, 5),
        "session_names": ("ses-1arithmetic", "ses-3arithmetic", "ses-5arithmetic"),
        "labels": {"subtraction": 0, "rest": 1},
    },
}

# MNE/OMLC absorption coefficients at 760 and 850 nm, columns HbO/HbR.
ABSORPTION = np.asarray(
    [[134.9558, 356.624156], [243.6574, 159.210996]],
    dtype=np.float64,
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def exactly_one(folder: Path, pattern: str) -> Path:
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"{folder}: expected one {pattern}, found {len(paths)}")
    return paths[0]


def validate_fnirs_channels(clab) -> list[str]:
    names = [str(value) for value in np.asarray(clab).reshape(-1)]
    if len(names) != 72:
        raise ValueError(f"Expected 72 wavelength channels, got {len(names)}")
    low = [name.removesuffix("lowWL") for name in names[:36]]
    high = [name.removesuffix("highWL") for name in names[36:]]
    if any(not name.endswith("lowWL") for name in names[:36]):
        raise ValueError("First 36 channels are not low-wavelength channels")
    if any(not name.endswith("highWL") for name in names[36:]):
        raise ValueError("Last 36 channels are not high-wavelength channels")
    if low != high:
        raise ValueError("Low/high wavelength source-detector orders differ")
    return low


def intensity_to_hbr(intensity: np.ndarray, distance_m: float = 0.03, ppf: float = 6.0):
    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.ndim != 2 or intensity.shape[1] != 72:
        raise ValueError(f"Expected intensity [time,72], got {intensity.shape}")
    safe = np.abs(intensity)
    for channel in range(safe.shape[1]):
        positive = safe[:, channel][safe[:, channel] > 0]
        if not len(positive):
            raise ValueError(f"Intensity channel {channel} has no positive samples")
        safe[:, channel] = np.maximum(safe[:, channel], positive.min())
    optical_density = -np.log(safe / safe.mean(axis=0, keepdims=True))
    inverse = np.linalg.pinv(ABSORPTION * distance_m * ppf) * 1e-3
    concentration = np.einsum(
        "ab,tcb->tca",
        inverse,
        np.stack((optical_density[:, :36], optical_density[:, 36:]), axis=-1),
    )
    # Return HbR only, in micromolar: [time,36].
    return concentration[:, :, 1] * 1e6


def normalize_trial(x: np.ndarray) -> np.ndarray:
    mean = float(x.mean(dtype=np.float64))
    std = float(x.std(dtype=np.float64))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError(f"Invalid trial standard deviation: {std}")
    return np.asarray((x - mean) / std, dtype=np.float32)


def load_subject(
    eeg_root: Path,
    fnirs_root: Path,
    subject: int,
    task_key: str,
    sensor_layout: str = "project_all",
    epoch_start_s: float = 0.0,
    epoch_stop_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    import mne

    if sensor_layout not in SENSOR_LAYOUTS:
        raise ValueError(f"Unknown sensor layout: {sensor_layout}")
    if epoch_stop_s <= epoch_start_s:
        raise ValueError("epoch_stop_s must be greater than epoch_start_s")
    layout = SENSOR_LAYOUTS[sensor_layout]
    eeg_indices = layout["eeg_indices"]
    fnirs_indices = layout["fnirs_indices"]
    eeg_start_offset = int(round(epoch_start_s * 200.0))
    eeg_stop_offset = int(round(epoch_stop_s * 200.0))
    hbr_start_offset = int(round(epoch_start_s * 10.0))
    hbr_stop_offset = int(round(epoch_stop_s * 10.0))
    expected_eeg_samples = eeg_stop_offset - eeg_start_offset
    expected_hbr_samples = hbr_stop_offset - hbr_start_offset

    task = TASKS[task_key]
    fnirs_folder = fnirs_root / f"subject {subject:02d}"
    cnt = loadmat(fnirs_folder / "cnt.mat", simplify_cells=True)["cnt"]
    mrk = loadmat(fnirs_folder / "mrk.mat", simplify_cells=True)["mrk"]

    eeg_trials: list[np.ndarray] = []
    hbr_trials: list[np.ndarray] = []
    labels: list[int] = []
    details: list[dict] = []
    expected_fnirs_names: list[str] | None = None

    for session_index, session_name in zip(
        task["session_indices"], task["session_names"]
    ):
        eeg_dir = eeg_root / f"sub-{subject:02d}" / session_name / "eeg"
        bdf_path = exactly_one(eeg_dir, "*_eeg.bdf")
        events_path = exactly_one(eeg_dir, "*_events.tsv")
        channels_path = exactly_one(eeg_dir, "*_channels.tsv")
        eeg_names = [
            row["name"] for row in read_tsv(channels_path)
            if row.get("type", "").upper() == "EEG"
        ]
        if eeg_names != SHIN_EEG_CHANNELS:
            raise RuntimeError(f"Unexpected EEG channel order in {channels_path}")
        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
        if abs(float(raw.info["sfreq"]) - 200.0) > 1e-6:
            raise RuntimeError(f"Expected EEG at 200 Hz in {bdf_path}")
        continuous_eeg = np.nan_to_num(
            raw.get_data(picks=SHIN_EEG_CHANNELS, units="uV"),
            copy=False,
        )[list(eeg_indices)]
        raw.close()

        eeg_events: list[tuple[int, int]] = []
        for event in read_tsv(events_path):
            label_name = event.get("trial_type", "")
            if label_name in task["labels"]:
                eeg_events.append((int(event["sample"]), task["labels"][label_name]))

        recording, markers = cnt[session_index], mrk[session_index]
        if float(recording["fs"]) != 10.0:
            raise RuntimeError(f"Expected fNIRS at 10 Hz for subject {subject}")
        fnirs_names = validate_fnirs_channels(recording["clab"])
        if expected_fnirs_names is None:
            expected_fnirs_names = fnirs_names
        if fnirs_names != expected_fnirs_names:
            raise RuntimeError(f"fNIRS channel order changed for subject {subject}")
        hbr_continuous = intensity_to_hbr(np.asarray(recording["x"], dtype=np.float64))
        b, a = butter(3, [0.01, 0.1], btype="bandpass", fs=10.0)
        hbr_continuous = filtfilt(b, a, hbr_continuous, axis=0)[:, list(fnirs_indices)]

        times_ms = np.asarray(markers["time"], dtype=np.float64).reshape(-1)
        descriptions = np.asarray(markers["event"]["desc"], dtype=np.int64).reshape(-1)
        fnirs_events = [
            (int(round(time_ms * 10.0 / 1000.0)), int(description - 1))
            for time_ms, description in zip(times_ms, descriptions)
        ]
        eeg_sequence = [label for _, label in eeg_events]
        fnirs_sequence = [label for _, label in fnirs_events]
        if (
            len(eeg_events) != 20
            or len(fnirs_events) != 20
            or eeg_sequence != fnirs_sequence
            or Counter(eeg_sequence) != Counter({0: 10, 1: 10})
        ):
            raise RuntimeError(
                f"EEG/fNIRS event alignment failed for subject {subject}, {session_name}: "
                f"EEG={eeg_sequence}, fNIRS={fnirs_sequence}"
        )

        for (eeg_sample, label), (fnirs_sample, _) in zip(eeg_events, fnirs_events):
            eeg_start = eeg_sample + eeg_start_offset
            eeg_stop = eeg_sample + eeg_stop_offset
            if eeg_start < 0 or eeg_stop > continuous_eeg.shape[1]:
                raise RuntimeError(
                    f"EEG epoch out of range for subject {subject}: "
                    f"[{eeg_start}, {eeg_stop}) of {continuous_eeg.shape[1]}"
                )
            eeg_trial = continuous_eeg[:, eeg_start:eeg_stop]
            if eeg_trial.shape != (len(eeg_indices), expected_eeg_samples):
                raise RuntimeError(f"Bad EEG trial shape {eeg_trial.shape}")
            # Baseline correction follows the established SHIN fNIRS pipeline.
            baseline = hbr_continuous[fnirs_sample - 50:fnirs_sample - 20].mean(
                axis=0,
                keepdims=True,
            )
            hbr_start = fnirs_sample + hbr_start_offset
            hbr_stop = fnirs_sample + hbr_stop_offset
            if hbr_start < 0 or hbr_stop > hbr_continuous.shape[0]:
                raise RuntimeError(
                    f"HbR epoch out of range for subject {subject}: "
                    f"[{hbr_start}, {hbr_stop}) of {hbr_continuous.shape[0]}"
                )
            hbr_trial = hbr_continuous[hbr_start:hbr_stop] - baseline
            if hbr_trial.shape != (expected_hbr_samples, len(fnirs_indices)):
                raise RuntimeError(f"Bad HbR trial shape {hbr_trial.shape}")
            eeg_trials.append(normalize_trial(eeg_trial))
            hbr_trials.append(normalize_trial(hbr_trial.T))
            labels.append(label)
        details.append({
            "session": session_name,
            "session_index": session_index,
            "trials": 20,
            "event_sequence_aligned": True,
        })

    eeg = np.stack(eeg_trials)
    hbr = np.stack(hbr_trials)
    y = np.asarray(labels, dtype=np.int64)
    if (
        eeg.shape != (60, len(eeg_indices), expected_eeg_samples)
        or hbr.shape != (60, len(fnirs_indices), expected_hbr_samples)
        or Counter(y.tolist()) != Counter({0: 30, 1: 30})
    ):
        raise RuntimeError(
            f"Unexpected subject output: EEG={eeg.shape}, HbR={hbr.shape}, y={Counter(y.tolist())}"
        )
    return eeg, hbr, y, {
        "subject": subject,
        "task": task_key,
        "sessions": details,
        "eeg_shape": list(eeg.shape),
        "hbr_shape": list(hbr.shape),
        "sensor_layout": sensor_layout,
        "epoch_seconds": [epoch_start_s, epoch_stop_s],
        "eeg_channels": [SHIN_EEG_CHANNELS[index] for index in eeg_indices],
        "fnirs_channels": [expected_fnirs_names[index] for index in fnirs_indices],
        "processing": {
            "eeg": (
                f"physical microvolts, {epoch_start_s:g}..{epoch_stop_s:g} s, "
                "per-trial global z-score"
            ),
            "hbr": (
                "optical density, modified Beer-Lambert, 0.01-0.1 Hz, "
                f"baseline -5..-2 s, epoch {epoch_start_s:g}..{epoch_stop_s:g} s, "
                "per-trial global z-score"
            ),
        },
    }


def load_split(
    eeg_root: Path,
    fnirs_root: Path,
    subjects: list[int],
    split_name: str,
    task_key: str,
    cache_dir: Path,
    sensor_layout: str = "project_all",
    epoch_start_s: float = 0.0,
    epoch_stop_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(map(str, subjects))
    if (
        sensor_layout == "project_all"
        and epoch_start_s == 0.0
        and epoch_stop_s == 10.0
    ):
        cache_name = f"{task_key}_{split_name}_sub-{tag}_aligned_eeg-hbr_trials.npz"
    else:
        epoch_tag = (
            f"{epoch_start_s:g}_{epoch_stop_s:g}"
            .replace("-", "m")
            .replace(".", "p")
        )
        cache_name = (
            f"{task_key}_{split_name}_sub-{tag}_{sensor_layout}_"
            f"epoch-{epoch_tag}_aligned_eeg-hbr_trials.npz"
        )
    cache_path = cache_dir / cache_name
    if cache_path.is_file():
        item = np.load(cache_path, allow_pickle=False)
        eeg, hbr, y = item["eeg"], item["hbr"], item["y"]
        print(
            f"[{split_name}] cache {cache_path}: EEG={eeg.shape}, HbR={hbr.shape}, "
            f"y={Counter(y.tolist())}",
            flush=True,
        )
        return eeg, hbr, y, [{"subject": subject, "source": "cache"} for subject in subjects]
    eeg_parts, hbr_parts, y_parts, details = [], [], [], []
    for subject in subjects:
        eeg, hbr, y, detail = load_subject(
            eeg_root,
            fnirs_root,
            subject,
            task_key,
            sensor_layout,
            epoch_start_s,
            epoch_stop_s,
        )
        eeg_parts.append(eeg)
        hbr_parts.append(hbr)
        y_parts.append(y)
        details.append(detail)
        print(
            f"[{split_name}] sub-{subject}: EEG={eeg.shape}, HbR={hbr.shape}, "
            f"y={Counter(y.tolist())}",
            flush=True,
        )
    eeg_all = np.concatenate(eeg_parts)
    hbr_all = np.concatenate(hbr_parts)
    y_all = np.concatenate(y_parts)
    np.savez(cache_path, eeg=eeg_all, hbr=hbr_all, y=y_all)
    return eeg_all, hbr_all, y_all, details


class WindowDataset(Dataset):
    """Expose aligned fixed-length windows from a paired multimodal epoch."""

    def __init__(
        self,
        eeg: np.ndarray,
        hbr: np.ndarray,
        labels: np.ndarray,
        epoch_start_s: float = 0.0,
        window_seconds: float = 3.0,
        stride_seconds: float = 1.0,
    ) -> None:
        if eeg.shape[0] != hbr.shape[0] or eeg.shape[0] != len(labels):
            raise ValueError("EEG/HbR/label trial counts differ")
        self.eeg_window_samples = int(round(window_seconds * 200.0))
        self.hbr_window_samples = int(round(window_seconds * 10.0))
        self.eeg_stride_samples = int(round(stride_seconds * 200.0))
        self.hbr_stride_samples = int(round(stride_seconds * 10.0))
        if min(
            self.eeg_window_samples,
            self.hbr_window_samples,
            self.eeg_stride_samples,
            self.hbr_stride_samples,
        ) <= 0:
            raise ValueError("Window and stride must be positive")
        eeg_windows = (
            (eeg.shape[-1] - self.eeg_window_samples) // self.eeg_stride_samples
        ) + 1
        hbr_windows = (
            (hbr.shape[-1] - self.hbr_window_samples) // self.hbr_stride_samples
        ) + 1
        if eeg_windows <= 0 or eeg_windows != hbr_windows:
            raise ValueError(
                f"EEG/HbR window counts differ: EEG={eeg_windows}, HbR={hbr_windows}"
            )
        self.eeg = eeg
        self.hbr = hbr
        self.labels = labels
        self.epoch_start_s = float(epoch_start_s)
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.windows_per_trial = eeg_windows
        self.window_intervals = [
            [
                self.epoch_start_s + index * self.stride_seconds,
                self.epoch_start_s
                + index * self.stride_seconds
                + self.window_seconds,
            ]
            for index in range(self.windows_per_trial)
        ]

    def __len__(self) -> int:
        return len(self.labels) * self.windows_per_trial

    def __getitem__(self, index: int):
        trial, window = divmod(index, self.windows_per_trial)
        eeg_start = window * self.eeg_stride_samples
        hbr_start = window * self.hbr_stride_samples
        eeg_window = self.eeg[
            trial, :, eeg_start:eeg_start + self.eeg_window_samples
        ].T.copy()
        hbr_window = self.hbr[
            trial, :, hbr_start:hbr_start + self.hbr_window_samples
        ].T.copy()
        return (
            torch.from_numpy(eeg_window),
            torch.from_numpy(hbr_window),
            torch.tensor(self.labels[trial], dtype=torch.long),
            torch.tensor(trial, dtype=torch.long),
        )
