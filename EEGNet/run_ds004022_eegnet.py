"""Run the portable EEGNet training loop on the ds004022 four-class EEG cache."""
from collections import Counter
from pathlib import Path

import numpy as np

import run_shin_eegnet as portable


CACHE = Path(r"D:\0senior student creation\braindecode_codebrain_prep\cache\ds004022_eeg_mi5s_4to40hz_200hz.npz")
_TRAIN_MEAN: np.ndarray | None = None
_TRAIN_STD: np.ndarray | None = None


def load_split(root, name, subjects, cache_dir, task_key, task):
    """Strict subject-independent split with train-only normalization.

    The base runner calls train first, then validation and test.  The mean/std
    are consequently fitted on training participants only and reused unchanged
    for held-out participants.
    """
    global _TRAIN_MEAN, _TRAIN_STD
    item = np.load(CACHE, allow_pickle=True)
    mask = np.isin(item["subject_ids"], np.asarray(subjects, dtype=np.int64))
    x = item["eeg"][mask].astype(np.float32)
    y = item["labels"][mask].astype(np.int64)
    if name == "train":
        _TRAIN_MEAN = x.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
        _TRAIN_STD = x.std(axis=(0, 2), keepdims=True, dtype=np.float64)
    if _TRAIN_MEAN is None or _TRAIN_STD is None:
        raise RuntimeError("Training split must be loaded before validation/test")
    x = ((x - _TRAIN_MEAN) / (_TRAIN_STD + 1e-6)).astype(np.float32)
    print(f"[{name}] ds004022 cache: X={x.shape}, y={Counter(y.tolist())}", flush=True)
    return x, y, [{"subjects": subjects, "split": "subject-independent; all three runs retained per participant", "source": str(CACHE)}]


if __name__ == "__main__":
    portable.TASKS["mi"] = {
        "name": "ds004022-4class-MI",
        "description": "reach (0), grasp (1), lift (2), twist (3)",
        "sessions": (),
        "labels": {"reach": 0, "grasp": 1, "lift": 2, "twist": 3},
    }
    portable.load_split = load_split
    portable.main()
