"""Create one compact FineMI 0--12 s fNIRS cache for reusable prompt windows."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Numba/llvmlite initialization hangs in the bci4models Windows environment.
# MNE treats Numba as optional, and this fNIRS pipeline does not need it.
sys.modules["numba"] = None
import mne
import numpy as np


RAW_ROOT = Path(r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\FineMI")
LABEL_ROOT = Path(r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_200hz_no_car")
OUT_ROOT = Path(r"D:\0senior student creation\datasets\FineMI_Yi2025_raw\processed_prompt_fnirs_binary_1v6_0_12s")
EVENT_ID = {f"{value}.0": value for value in range(1, 9)}
PREFIX_DROPS = {(10, "block1"): 1, (10, "block2"): 1, (10, "block5"): 1,
                (10, "block6"): 1, (11, "block1"): 7, (11, "block2"): 1,
                (12, "block1"): 1, (14, "block6"): 2}


def subject_blocks(subject: int) -> list[str]:
    return ["block1-4", "block5", "block6", "block7", "block8"] if subject == 1 else [f"block{i}" for i in range(1, 9)]


def correct_annotations(raw, subject: int, block: str) -> None:
    if subject == 1 and block == "block1-4":
        raw.annotations.delete(np.arange(160, 200))
    if subject == 5 and block == "block6":
        raw.annotations.delete(0)
    drop = PREFIX_DROPS.get((subject, block), 0)
    if drop:
        raw.annotations.delete(np.arange(drop))


def build_subject(subject: int) -> dict:
    output = OUT_ROOT / f"subject{subject:02d}_fnirs_prompt.npz"
    hbo_parts, hbr_parts = [], []
    channel_names = times = None
    for block in subject_blocks(subject):
        raw = mne.io.read_raw_nirx(RAW_ROOT / f"subject{subject}" / "fNIRS" / block, preload=True, verbose="ERROR")
        correct_annotations(raw, subject, block)
        hb = mne.preprocessing.nirs.beer_lambert_law(mne.preprocessing.nirs.optical_density(raw, verbose="ERROR"))
        hb.filter(0.01, 0.1, method="iir", iir_params={"order": 6, "ftype": "butter", "output": "sos"}, verbose="ERROR")
        events, _ = mne.events_from_annotations(hb, event_id=EVENT_ID, verbose="ERROR")
        epochs = mne.Epochs(hb, events, EVENT_ID, tmin=0.0, tmax=12.0, baseline=None,
                            preload=True, reject_by_annotation=True, verbose="ERROR")
        hbo_parts.append(epochs.get_data(picks="hbo", copy=True).astype(np.float32))
        hbr_parts.append(epochs.get_data(picks="hbr", copy=True).astype(np.float32))
        current_names = [hb.ch_names[i] for i in mne.pick_types(hb.info, fnirs="hbo")]
        if channel_names is None:
            channel_names, times = current_names, epochs.times.astype(np.float32)
        elif channel_names != current_names or not np.array_equal(times, epochs.times.astype(np.float32)):
            raise RuntimeError(f"subject{subject} {block}: channel/time grid changed")
        raw.close()
    hbo, hbr = np.concatenate(hbo_parts), np.concatenate(hbr_parts)
    with np.load(LABEL_ROOT / f"subject{subject:02d}_paired.npz", allow_pickle=False) as item:
        labels_all = item["labels"].astype(np.int64)
        block_ids_all = item["block_ids"].astype(np.int16)
    if len(hbo) != len(labels_all):
        raise RuntimeError(f"subject{subject}: fNIRS/label trial count mismatch")
    keep = np.isin(labels_all, (0, 5))
    labels = (labels_all[keep] == 5).astype(np.int64)
    graph = np.stack((hbo[keep], hbr[keep]), axis=2).astype(np.float32)
    if graph.shape[:3] != (80, 24, 2) or not np.isfinite(graph).all():
        raise RuntimeError(f"subject{subject}: invalid cache {graph.shape}")
    np.savez_compressed(output, fnirs_graph=graph, fnirs_times_s=times, labels=labels,
                        subject_ids=np.full(80, subject, dtype=np.int16), block_ids=block_ids_all[keep],
                        channel_names=np.asarray(channel_names), chromophore_order=np.asarray(["HbO", "HbR"]),
                        fnirs_sfreq=np.asarray(7.8125, dtype=np.float32), fnirs_window_s=np.asarray([0.0, 12.0], dtype=np.float32))
    return {"subject": subject, "shape": list(graph.shape), "bytes": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_subject, subject): subject for subject in range(1, 19)}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"subject{record['subject']:02d}: {record['shape']}", flush=True)
    records.sort(key=lambda item: item["subject"])
    manifest = {"dataset": "FineMI/Yi2025 binary event 1 vs 6 reusable fNIRS",
                "preprocessing": "OD; MBLL HbO/HbR; Butterworth order 6 0.01-0.1 Hz; epoch 0..12 s",
                "prompt_window": "loader selects 6..10 s", "records": records}
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
