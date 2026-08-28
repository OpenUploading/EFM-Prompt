"""Read-only signal-quality checks for the prepared HYGRIP EEG trials."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"D:\data\HYGRIP-Baselines\prepared")
SUBJECTS = list("ABCDEFGHIJKLMN")
FS = 200
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def features(x: np.ndarray, seconds: int) -> np.ndarray:
    x = np.asarray(x[:, :, : seconds * FS], dtype=np.float64)
    x -= x.mean(axis=1, keepdims=True)  # common-average reference
    sos = butter(4, (1, 45), btype="bandpass", fs=FS, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    freq, psd = welch(x, fs=FS, nperseg=400, noverlap=200, axis=-1)
    values = []
    for lo, hi in BANDS:
        mask = (freq >= lo) & (freq < hi)
        values.append(np.log(np.maximum(psd[..., mask].mean(axis=-1), 1e-20)))
    return np.concatenate(values, axis=1).astype(np.float32)


def classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, solver="liblinear", random_state=1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, default=ROOT)
    args = parser.parse_args()
    rows = []
    by_subject = {}
    for subject in SUBJECTS:
        data = loadmat(args.prepared_root / f"subject_{subject}_trials.mat", variable_names=["eeg_uv", "labels"])
        x = np.asarray(data["eeg_uv"], dtype=np.float32)
        y = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
        by_subject[subject] = (x, y)
        channel_std = x.std(axis=(0, 2), dtype=np.float64)
        rows.append(
            {
                "subject": subject,
                "shape": list(x.shape),
                "labels": y.tolist(),
                "mean_uv": float(x.mean(dtype=np.float64)),
                "std_uv": float(x.std(dtype=np.float64)),
                "abs_p50_uv": float(np.percentile(np.abs(x), 50)),
                "abs_p95_uv": float(np.percentile(np.abs(x), 95)),
                "abs_p99_uv": float(np.percentile(np.abs(x), 99)),
                "abs_max_uv": float(np.max(np.abs(x))),
                "min_channel_std_uv": float(channel_std.min()),
                "max_channel_std_uv": float(channel_std.max()),
                "flat_channels": int(np.sum(channel_std < 1e-6)),
            }
        )

    print("SIGNAL_STATS")
    print(json.dumps(rows, ensure_ascii=False))
    for seconds in (4, 10, 20):
        feats = {s: features(*by_subject[s][:1], seconds) for s in SUBJECTS}
        x_train = np.concatenate([feats[s] for s in SUBJECTS[:10]])
        y_train = np.concatenate([by_subject[s][1] for s in SUBJECTS[:10]])
        x_val = np.concatenate([feats[s] for s in SUBJECTS[10:12]])
        y_val = np.concatenate([by_subject[s][1] for s in SUBJECTS[10:12]])
        x_test = np.concatenate([feats[s] for s in SUBJECTS[12:]])
        y_test = np.concatenate([by_subject[s][1] for s in SUBJECTS[12:]])
        model = classifier()
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        within = {}
        for subject in SUBJECTS:
            y = by_subject[subject][1]
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
            within[subject] = float(cross_val_score(classifier(), feats[subject], y, cv=cv).mean())
        print(
            "FEATURE_RESULT",
            json.dumps(
                {
                    "seconds": seconds,
                    "feature": "CAR + zero-phase 1-45 Hz + channelwise log bandpower",
                    "cross_subject_val_accuracy": float(accuracy_score(y_val, val_pred)),
                    "cross_subject_test_accuracy": float(accuracy_score(y_test, test_pred)),
                    "cross_subject_test_balanced_accuracy": float(
                        balanced_accuracy_score(y_test, test_pred)
                    ),
                    "test_predictions": test_pred.tolist(),
                    "within_subject_5fold": within,
                    "within_subject_mean": float(np.mean(list(within.values()))),
                    "within_subject_median": float(np.median(list(within.values()))),
                }
            ),
        )


if __name__ == "__main__":
    main()
