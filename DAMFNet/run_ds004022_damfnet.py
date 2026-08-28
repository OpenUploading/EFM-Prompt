"""DAMFNet four-class fusion baseline on ds004022.

This thin entry point intentionally reuses the SHIN runner's official DAMFNet
windowing, three-branch loss, validation selection and reporting code.  Only
the dataset backend and the 18/24-to-8/24, four-class model adapter differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import run_shin_damfnet as shared
from ds004022_damf_data import TASKS, load_split
from models.shin_damfnet import SHINDAMFNet


ROOT = Path(r"D:\0senior student creation\datasets\ds004022_orthopedic_mi_eeg_fnirs")
OUTPUT = Path(__file__).resolve().parent / "runs_ds004022"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAMFNet on paired ds004022 EEG/fNIRS")
    parser.add_argument("--eeg-root", type=Path, default=ROOT)
    parser.add_argument("--fnirs-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--task", choices=("mi",), default="mi")
    parser.add_argument("--train-subjects", default="1-5")
    parser.add_argument("--val-subjects", default="6")
    parser.add_argument("--test-subjects", default="7")
    parser.add_argument("--sensor-layout", choices=("project_all",), default="project_all")
    parser.add_argument("--epoch-start-s", type=float, default=0.0)
    parser.add_argument("--epoch-stop-s", type=float, default=5.0)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--window-stride-seconds", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--loss-w-eeg", type=float, default=1.0)
    parser.add_argument("--loss-w-hbr", type=float, default=1.0)
    parser.add_argument("--loss-w-fuse", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-subjects-per-split", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def model_factory(*, dropout: float, sensor_layout: str) -> SHINDAMFNet:
    return SHINDAMFNet(
        dropout=dropout, sensor_layout=sensor_layout, num_classes=4,
        eeg_input_nodes=18, hbr_input_nodes=24,
    )


def main() -> None:
    shared.arguments = arguments
    shared.TASKS = TASKS
    shared.load_split = load_split
    shared.SHINDAMFNet = model_factory
    shared.SHIN_EEG_CHANNELS = [f"DS-EEG-{index:02d}" for index in range(1, 19)]
    shared.main()


if __name__ == "__main__":
    main()
