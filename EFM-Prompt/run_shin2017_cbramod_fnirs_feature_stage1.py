"""Boundary conditional-prompt ablations for CBraMod on Shin2017.

The CBraMod encoder is frozen. fNIRS conditionally generates prompts only at
the patch-embedding output and/or final encoder output; no encoder layer is
modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from sgformer_mapped_prompt import (
    SGFormerMappedEncoder,
    load_fnirs_montage,
    load_sgformer_graph_trials,
    normalize_graph_from_train,
)
from mope_class_aware_ot import class_aware_ot_losses, pairwise_token_ot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CBraMod boundary conditional-prompt ablations")
    parser.add_argument("--portable-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prep-root", type=Path, default=Path(r"D:\0senior student creation\braindecode_codebrain_prep"))
    parser.add_argument(
        "--eeg-bids-root", type=Path,
        default=Path(r"D:\0senior student creation\datasets\shin2017_eeg_bids_bdf"),
    )
    parser.add_argument("--shin-root", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--task", choices=("mi", "ma"), default="mi")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 30)))
    parser.add_argument("--train-subjects", nargs="+", type=int, default=list(range(1, 20)))
    parser.add_argument("--val-subjects", nargs="+", type=int, default=list(range(20, 25)))
    parser.add_argument("--test-subjects", nargs="+", type=int, default=list(range(25, 30)))
    parser.add_argument("--rebuild-cache", action="store_true")
    # Keep the 10 one-second patches used by the CBraMod frozen baseline so
    # both conditions have the same EEG input shape and classifier head.
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=200)
    parser.add_argument("--eeg-scale", type=float, default=1.0)
    parser.add_argument("--eeg-offset", type=float, default=0.0)
    parser.add_argument("--trial-offsets", nargs="+", type=float, default=[0.0])
    parser.add_argument("--fnirs-window", type=float, default=10.0)
    parser.add_argument("--fnirs-offset", type=float, default=0.0)
    parser.add_argument(
        "--fnirs-conditioner", choices=("stats", "temporal"), default="temporal",
        help="Condition prompts from hand-crafted trial statistics or a learned raw fNIRS temporal encoder.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prompt-hidden", type=int, default=256)
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--prompt-rank", type=int, default=8)
    parser.add_argument("--prompt-family", choices=("legacy", "mope"), default="legacy")
    parser.add_argument("--prompt-source", choices=("conditional", "static"), default="conditional")
    parser.add_argument("--expert-count", type=int, default=16)
    parser.add_argument("--router-temperature", type=float, default=0.1)
    parser.add_argument("--router-noise-std", type=float, default=0.00390625)
    parser.add_argument("--importance-threshold", type=float, default=0.05)
    parser.add_argument("--importance-weight", type=float, default=0.01)
    parser.add_argument(
        "--mope-contrast-mode",
        choices=("none", "dynamic_mapped_class_ot"),
        default="none",
        help="Training-only class-aware EEG-to-prompt OT for routed dynamic plus mapped MoPE tokens.",
    )
    parser.add_argument("--ot-temperature", type=float, default=0.1)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--ot-pair-weight", type=float, default=0.1)
    parser.add_argument("--ot-class-weight", type=float, default=0.02)
    parser.add_argument("--dynamic-expert-mode", choices=("flat", "tap4x4"), default="flat")
    parser.add_argument("--tap-attribute-weight", type=float, default=0.1)
    parser.add_argument("--mope-drop-component", choices=("none", "static", "dynamic", "mapped"), default="none")
    parser.add_argument("--mapped-mode", choices=("mlp", "sgformer"), default="mlp")
    parser.add_argument("--sgformer-cache-path", type=Path, default=None)
    parser.add_argument("--sgformer-graph-dimension", type=int, default=128)
    parser.add_argument("--sgformer-attention-residual-weight", type=float, default=0.5)
    parser.add_argument("--sgformer-graph-weight", type=float, default=0.8)
    parser.add_argument("--mode", choices=("eeg_only", "pre", "post", "pre_post"), default="pre_post")
    parser.add_argument("--shuffle-fnirs", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-epoch", type=int, default=9999)
    parser.add_argument("--training-strategy", choices=("joint", "prompt_only"), default="joint")
    parser.add_argument("--head-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--experiment-note", default=None)
    return parser.parse_args()


SHIN_EEG_CHANNELS = [
    "F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8", "AFF1h", "AFF2h",
    "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h", "CCP3h", "T7", "P7", "P3", "PPO1h",
    "POO1", "POO2", "PPO2h", "P4", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "P8", "T8",
]
SHIN_TASKS = {
    "mi": {
        "name": "EEG-MI",
        "description": "left_hand (0) vs right_hand (1)",
        "sessions": (("ses-0imagery", 0), ("ses-2imagery", 2), ("ses-4imagery", 4)),
        "labels": {"left_hand": 0, "right_hand": 1},
    },
    "ma": {
        "name": "EEG-MA",
        "description": "subtraction (0) vs rest (1)",
        "sessions": (("ses-1arithmetic", 1), ("ses-3arithmetic", 3), ("ses-5arithmetic", 5)),
        "labels": {"subtraction": 0, "rest": 1},
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_fnirs_from_train(fnirs: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = fnirs[train_idx].mean(axis=0, keepdims=True)
    std = fnirs[train_idx].std(axis=0, keepdims=True)
    return ((fnirs - mean) / (std + 1e-6)).astype(np.float32)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def one_file(folder: Path, pattern: str) -> Path:
    matches = list(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern} under {folder}, found {len(matches)}")
    return matches[0]


def load_paired_bids_trial_cache(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Use the exact CBraMod BDF EEG path and pair each retained trial with NIRS."""
    cache_path = Path(args.cache_path)
    if cache_path.exists() and not args.rebuild_cache:
        try:
            cached = np.load(cache_path, allow_pickle=True)
            meta = cached["meta"].item()
            if ("fnirs_sequence" in cached and meta.get("task") == args.task
                    and float(meta.get("eeg_scale", -1)) == float(args.eeg_scale)):
                return cached["eeg"], cached["fnirs"], cached["fnirs_sequence"], cached["labels"], meta
        except (BadZipFile, EOFError, OSError, ValueError, KeyError) as error:
            print(f"[cache] Invalid cache at {cache_path}; rebuilding ({error}).", flush=True)

    import pyedflib

    eeg_root = Path(args.eeg_bids_root)
    nirs_root = Path(args.shin_root)
    task = SHIN_TASKS[args.task]
    eeg_trials, fnirs_trials, fnirs_sequences, labels, subject_ids, session_ids = [], [], [], [], [], []

    for subject in args.subjects:
        nirs_dir = nirs_root / "NIRS" / f"subject {subject:02d}"
        if not (nirs_dir / "cnt.mat").exists() or not (nirs_dir / "mrk.mat").exists():
            continue
        nirs_cnt = loadmat(nirs_dir / "cnt.mat", squeeze_me=True, struct_as_record=False)["cnt"]
        nirs_mrk = loadmat(nirs_dir / "mrk.mat", squeeze_me=True, struct_as_record=False)["mrk"]

        for bids_session, mat_session in task["sessions"]:
            eeg_dir = eeg_root / f"sub-{subject:02d}" / bids_session / "eeg"
            if not eeg_dir.exists():
                continue
            bdf_path = one_file(eeg_dir, "*_eeg.bdf")
            channels_path = one_file(eeg_dir, "*_channels.tsv")
            events_path = one_file(eeg_dir, "*_events.tsv")
            channel_rows = read_tsv(channels_path)
            eeg_names = [row["name"] for row in channel_rows if row.get("type", "").upper() == "EEG"]
            if eeg_names != SHIN_EEG_CHANNELS:
                raise ValueError(f"{channels_path}: expected the 30 non-EOG CBraMod EEG channels")

            reader = pyedflib.EdfReader(str(bdf_path))
            try:
                signal_labels = reader.getSignalLabels()
                picks = [signal_labels.index(name) for name in SHIN_EEG_CHANNELS]
                units = [reader.getPhysicalDimension(index) for index in picks]
                sfreqs = [float(reader.getSampleFrequency(index)) for index in picks]
                if any(unit != "uV" for unit in units) or any(abs(rate - 200.0) > 1e-6 for rate in sfreqs):
                    raise ValueError(f"{bdf_path}: expected 30 EEG channels in uV at 200 Hz")
                eeg_uv = np.stack([reader.readSignal(index) for index in picks]).astype(np.float32)
            finally:
                reader.close()

            eeg_events = [
                (int(row["sample"]), task["labels"][row["trial_type"]])
                for row in read_tsv(events_path) if row.get("trial_type") in task["labels"]
            ]
            nirs_x = nirs_cnt[mat_session].x.astype(np.float32)
            nirs_times = np.asarray(nirs_mrk[mat_session].time).reshape(-1)
            nirs_labels = np.asarray(nirs_mrk[mat_session].event.desc).reshape(-1).astype(int)
            nirs_events = [(time_ms, int(label) - 1) for time_ms, label in zip(nirs_times, nirs_labels) if label in (1, 2)]
            if len(eeg_events) != 20 or len(nirs_events) != 20:
                raise ValueError(f"sub-{subject:02d} {bids_session}: expected 20 paired MI events")
            if [label for _, label in eeg_events] != [label for _, label in nirs_events]:
                raise ValueError(f"sub-{subject:02d} {bids_session}: EEG/NIRS event labels are not aligned")

            nirs_sfreq = float(nirs_cnt[mat_session].fs)
            for (start, label), (nirs_time, _) in zip(eeg_events, nirs_events):
                stop = start + 2000
                if stop > eeg_uv.shape[1]:
                    raise ValueError(f"{bdf_path}: 10-second EEG trial exceeds recording length")
                nirs_start = int(round((float(nirs_time) / 1000.0 + args.fnirs_offset) * nirs_sfreq))
                nirs_stop = nirs_start + int(round(args.fnirs_window * nirs_sfreq))
                if nirs_start < 0 or nirs_stop > nirs_x.shape[0]:
                    raise ValueError(f"sub-{subject:02d} {bids_session}: fNIRS window exceeds recording length")
                nirs_segment = nirs_x[nirs_start:nirs_stop]
                fnirs_feature = np.concatenate((
                    nirs_segment.mean(axis=0), nirs_segment.std(axis=0), nirs_segment[-1] - nirs_segment[0],
                )).astype(np.float32)
                eeg_trials.append((eeg_uv[:, start:stop] / args.eeg_scale).reshape(30, 10, 200))
                fnirs_trials.append(fnirs_feature)
                fnirs_sequences.append(nirs_segment)
                labels.append(label)
                subject_ids.append(subject)
                session_ids.append(mat_session)

    eeg = np.stack(eeg_trials).astype(np.float32)
    fnirs = np.stack(fnirs_trials).astype(np.float32)
    fnirs_sequence = np.stack(fnirs_sequences).astype(np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    meta = {
        "dataset": "Shin2017 BIDS BDF EEG + original NIRS MAT",
        "task": args.task,
        "task_description": task["description"],
        "eeg_preprocessing": (
            "physical_uV -> float32; select named 30 EEG channels; exclude 2 EOG; "
            f"no CAR; no filtering; divide by {args.eeg_scale:g}"
        ),
        "eeg_scale": args.eeg_scale,
        "eeg_shape": list(eeg.shape),
        "fnirs_shape": list(fnirs.shape),
        "fnirs_sequence_shape": list(fnirs_sequence.shape),
        "subject_ids": subject_ids,
        "session_ids": session_ids,
        "pairing_note": "BDF EEG and MAT NIRS events are paired by subject, imagery session, event order, and label.",
        "fnirs_normalization": "Features or raw fNIRS sequences are normalized from training-fold statistics only.",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache_path = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
    np.savez_compressed(
        temporary_cache_path, eeg=eeg, fnirs=fnirs, fnirs_sequence=fnirs_sequence, labels=labels_array, meta=meta
    )
    temporary_cache_path.replace(cache_path)
    return eeg, fnirs, fnirs_sequence, labels_array, meta


def shuffle_fnirs_within_splits(fnirs: np.ndarray, splits: list[np.ndarray], seed: int) -> np.ndarray:
    """Break trial-wise EEG-fNIRS pairing while retaining each split's distribution."""
    shuffled = fnirs.copy()
    rng = np.random.default_rng(seed)
    for indices in splits:
        shuffled[indices] = fnirs[rng.permutation(indices)]
    return shuffled


class CBraModPromptDataset(Dataset):
    def __init__(self, eeg, fnirs, fnirs_graph, labels, indices) -> None:
        self.eeg = eeg
        self.fnirs = fnirs
        self.fnirs_graph = fnirs_graph
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        graph = (
            torch.from_numpy(self.fnirs_graph[index])
            if self.fnirs_graph is not None
            else torch.empty(0, dtype=torch.float32)
        )
        return (
            torch.from_numpy(self.eeg[index]),
            torch.from_numpy(self.fnirs[index]),
            graph,
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def shuffle_prompt_sources_within_splits(
    fnirs: np.ndarray,
    fnirs_graph: np.ndarray | None,
    splits: list[np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    shuffled_fnirs = fnirs.copy()
    shuffled_graph = fnirs_graph.copy() if fnirs_graph is not None else None
    rng = np.random.default_rng(seed)
    for indices in splits:
        permutation = rng.permutation(indices)
        shuffled_fnirs[indices] = fnirs[permutation]
        if shuffled_graph is not None:
            shuffled_graph[indices] = fnirs_graph[permutation]
    return shuffled_fnirs, shuffled_graph


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def metrics(y_true, y_pred, loss):
    return {
        "loss": float(loss),
        "acc": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


class BoundaryPrompt(nn.Module):
    """Static, dynamic, and mapping prompts with a low-rank feature expansion."""

    def __init__(self, fnirs_dim: int, d_model: int, prompt_count: int, rank: int, token_count: int,
                 hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.prompt_count = prompt_count
        self.d_model = d_model
        self.prior = nn.Sequential(
            nn.Linear(fnirs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.LayerNorm(d_model),
        )
        self.static = nn.Parameter(torch.empty(prompt_count, d_model))
        self.dynamic = nn.Linear(d_model, prompt_count * d_model)
        self.mapping = nn.Linear(d_model, prompt_count * d_model)
        self.to_rank = nn.Linear(prompt_count * d_model, rank)
        self.token_basis = nn.Parameter(torch.empty(rank, token_count, d_model))
        self.alpha = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.static, std=0.02)
        nn.init.normal_(self.dynamic.weight, std=1e-3)
        nn.init.zeros_(self.dynamic.bias)
        nn.init.normal_(self.mapping.weight, std=1e-3)
        nn.init.zeros_(self.mapping.bias)
        nn.init.normal_(self.token_basis, std=0.02)

    def forward(self, fnirs: torch.Tensor) -> torch.Tensor:
        prior = self.prior(fnirs)
        prompts = self.static.unsqueeze(0)
        prompts = prompts + self.dynamic(prior).view(-1, self.prompt_count, self.d_model)
        prompts = prompts + self.mapping(prior).view(-1, self.prompt_count, self.d_model)
        coefficients = self.to_rank(prompts.flatten(1))
        return self.alpha * torch.einsum("br,rtd->btd", coefficients, self.token_basis)


class FnirsTemporalEncoder(nn.Module):
    """Encode a trial's raw fNIRS sequence [B, time, channels] into one condition vector."""

    def __init__(self, channels: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = min(128, max(64, output_dim // 2))
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.Conv1d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim)
        )

    def forward(self, fnirs: torch.Tensor) -> torch.Tensor:
        if fnirs.ndim != 3:
            raise ValueError(f"Expected raw fNIRS [B,time,channels], got {tuple(fnirs.shape)}")
        return self.project(self.temporal(fnirs.transpose(1, 2)))


class CBraModBoundaryPrompt(nn.Module):
    def __init__(self, cbramod_cls, channels: int, patch_count: int, fnirs_dim: int,
                 hidden_dim: int, prompt_count: int, prompt_rank: int, mode: str, dropout: float,
                 fnirs_conditioner: str, prompt_source: str, prompt_family: str = "legacy",
                 expert_count: int = 16, router_temperature: float = 0.1,
                 router_noise_std: float = 0.00390625, importance_threshold: float = 0.05,
                 mope_drop_component: str = "none", mapped_mode: str = "mlp",
                 dynamic_expert_mode: str = "flat",
                 graph_montage: dict | None = None, sgformer_graph_dimension: int = 128,
                 sgformer_attention_residual_weight: float = 0.5,
                 sgformer_graph_weight: float = 0.8) -> None:
        super().__init__()
        if channels != 30 or patch_count != 10:
            raise ValueError(
                "Stage-1 comparison is defined for the CBraMod SHIN baseline "
                "input [B,30,10,200]; use --seq-len 10."
            )
        self.channels = channels
        self.patch_count = patch_count
        self.d_model = 200
        self.mode = mode
        self.backbone = cbramod_cls(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=channels, n_layer=12, nhead=8,
        )
        # Exactly the official CBraMod all_patch_reps head used by the baseline:
        # Flatten(30*10*200) -> 2000 -> 200 -> 2.
        self.classifier = nn.Sequential(
            nn.Linear(30 * 10 * self.d_model, 10 * self.d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_count * self.d_model, self.d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 2),
        )
        token_count = channels * patch_count
        self.fnirs_conditioner = fnirs_conditioner
        self.prompt_source = prompt_source
        self.prompt_family = prompt_family
        self.mapped_mode = mapped_mode
        self.condition_dim = hidden_dim if fnirs_conditioner == "temporal" else fnirs_dim
        self.fnirs_encoder = (
            FnirsTemporalEncoder(fnirs_dim, self.condition_dim, dropout)
            if fnirs_conditioner == "temporal" and mode != "eeg_only" else None
        )
        self.pre_prompt = None
        self.post_prompt = None
        self.graph_encoder = None
        self.attribute_encoder = None
        prompt_class = BoundaryPrompt
        extra_prompt_arguments = {}
        if prompt_family == "mope":
            from cbramod_mope_boundary import MoPEBoundaryPrompt, TAPFourAttributeFnirsEncoder

            prompt_class = MoPEBoundaryPrompt
            extra_prompt_arguments = {
                "expert_count": expert_count,
                "temperature": router_temperature,
                "router_noise_std": router_noise_std,
                "importance_threshold": importance_threshold,
                "drop_component": mope_drop_component,
                "mapped_mode": mapped_mode,
                "dynamic_expert_mode": dynamic_expert_mode,
            }
        if mode in {"pre", "pre_post"}:
            self.pre_prompt = prompt_class(
                self.condition_dim, self.d_model, prompt_count, prompt_rank, token_count, hidden_dim, dropout,
                **extra_prompt_arguments,
            )
        if mode in {"post", "pre_post"}:
            self.post_prompt = prompt_class(
                self.condition_dim, self.d_model, prompt_count, prompt_rank, token_count, hidden_dim, dropout,
                **extra_prompt_arguments,
            )
        if (
            prompt_family == "mope"
            and mapped_mode == "sgformer"
            and mode != "eeg_only"
            and mope_drop_component != "mapped"
        ):
            if graph_montage is None:
                raise ValueError("SGFormer mapped mode requires the SHIN fNIRS montage")
            self.graph_encoder = SGFormerMappedEncoder(
                positions_3d=torch.as_tensor(graph_montage["positions_3d"]),
                edge_index=torch.as_tensor(graph_montage["edge_index"]),
                prompt_dimension=200,
                graph_dimension=sgformer_graph_dimension,
                dropout=dropout,
                attention_residual_weight=sgformer_attention_residual_weight,
                graph_weight=sgformer_graph_weight,
            )
        if (
            prompt_family == "mope"
            and dynamic_expert_mode == "tap4x4"
            and mode != "eeg_only"
            and mope_drop_component != "dynamic"
        ):
            if graph_montage is None:
                raise ValueError("TAP 4x4 dynamic prompts require the SHIN fNIRS montage")
            self.attribute_encoder = TAPFourAttributeFnirsEncoder(
                positions_3d=torch.as_tensor(graph_montage["positions_3d"]),
                edge_index=torch.as_tensor(graph_montage["edge_index"]),
                output_dim=self.condition_dim,
                dropout=dropout,
            )

    @staticmethod
    def static_reference(fnirs: torch.Tensor) -> torch.Tensor:
        """Use a deterministic template with the same shape, never the measured fNIRS values."""
        if fnirs.ndim == 2:
            return torch.linspace(-1.0, 1.0, fnirs.shape[1], device=fnirs.device, dtype=fnirs.dtype).expand_as(fnirs)
        time = torch.linspace(-1.0, 1.0, fnirs.shape[1], device=fnirs.device, dtype=fnirs.dtype).view(1, -1, 1)
        channel = torch.linspace(-1.0, 1.0, fnirs.shape[2], device=fnirs.device, dtype=fnirs.dtype).view(1, 1, -1)
        return (time + channel).expand_as(fnirs)

    @staticmethod
    def static_graph_reference(fnirs_graph: torch.Tensor) -> torch.Tensor:
        node = torch.linspace(-1.0, 1.0, fnirs_graph.shape[1], device=fnirs_graph.device, dtype=fnirs_graph.dtype).view(1, -1, 1, 1)
        chromophore = torch.linspace(-0.5, 0.5, fnirs_graph.shape[2], device=fnirs_graph.device, dtype=fnirs_graph.dtype).view(1, 1, -1, 1)
        time = torch.linspace(-1.0, 1.0, fnirs_graph.shape[3], device=fnirs_graph.device, dtype=fnirs_graph.dtype).view(1, 1, 1, -1)
        return (node + chromophore + time).expand_as(fnirs_graph)

    def features(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        fnirs_graph: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        feats = self.backbone.patch_embedding(eeg)
        boundary_pairs = []
        condition_input = fnirs if self.prompt_source == "conditional" else self.static_reference(fnirs)
        condition = self.fnirs_encoder(condition_input) if self.fnirs_encoder is not None else condition_input
        mapped_nodes = None
        attribute_conditions = None
        if self.graph_encoder is not None or self.attribute_encoder is not None:
            if fnirs_graph.ndim != 4:
                raise ValueError("Structured fNIRS prompts require graph trials [B,36,2,T]")
            graph_input = (
                fnirs_graph
                if self.prompt_source == "conditional"
                else self.static_graph_reference(fnirs_graph)
            )
            if self.graph_encoder is not None:
                mapped_nodes = self.graph_encoder(graph_input)
            if self.attribute_encoder is not None:
                attribute_conditions = self.attribute_encoder(graph_input)
        if self.pre_prompt is not None:
            if self.prompt_family == "mope" and return_aux:
                prompt, aux = self.pre_prompt(
                    condition, mapped_nodes, attribute_conditions, return_aux=True
                )
                boundary_pairs.append((feats.flatten(1, 2).detach(), aux["contrast_tokens"]))
            else:
                prompt = (
                    self.pre_prompt(condition, mapped_nodes, attribute_conditions)
                    if self.prompt_family == "mope"
                    else self.pre_prompt(condition)
                )
            feats = feats + prompt.view_as(feats)
        feats = self.backbone.encoder(feats)
        feats = self.backbone.proj_out(feats)
        if self.post_prompt is not None:
            if self.prompt_family == "mope" and return_aux:
                prompt, aux = self.post_prompt(
                    condition, mapped_nodes, attribute_conditions, return_aux=True
                )
                boundary_pairs.append((feats.flatten(1, 2).detach(), aux["contrast_tokens"]))
            else:
                prompt = (
                    self.post_prompt(condition, mapped_nodes, attribute_conditions)
                    if self.prompt_family == "mope"
                    else self.post_prompt(condition)
                )
            feats = feats + prompt.view_as(feats)
        features = feats.flatten(start_dim=1)
        return (features, boundary_pairs) if return_aux else features

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        fnirs_graph: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        features = self.features(eeg, fnirs, fnirs_graph, return_aux=return_aux)
        if not return_aux:
            return self.classifier(features)
        flat_features, boundary_pairs = features
        return self.classifier(flat_features), boundary_pairs

    def importance_loss(self) -> torch.Tensor:
        losses = [
            prompt.importance_loss()
            for prompt in (self.pre_prompt, self.post_prompt)
            if prompt is not None and hasattr(prompt, "importance_loss")
        ]
        return torch.stack(losses).mean() if losses else next(self.parameters()).new_zeros(())

    def attribute_loss(self, target: torch.Tensor) -> torch.Tensor:
        losses = [
            prompt.attribute_loss(target)
            for prompt in (self.pre_prompt, self.post_prompt)
            if prompt is not None and hasattr(prompt, "attribute_loss")
        ]
        return torch.stack(losses).mean() if losses else next(self.parameters()).new_zeros(())

    def routing_statistics(self) -> dict[str, float]:
        statistics = {}
        for name, prompt in (("pre", self.pre_prompt), ("post", self.post_prompt)):
            if prompt is not None and hasattr(prompt, "routing_statistics"):
                statistics.update({f"{name}_{key}": value for key, value in prompt.routing_statistics().items()})
        return statistics


def load_pretrained(model: CBraModBoundaryPrompt, checkpoint: Path) -> dict:
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(source, dict) and "model" in source:
        source = source["model"]
    result = model.backbone.load_state_dict(source, strict=True)
    model.backbone.proj_out = nn.Identity()
    return {
        "checkpoint": str(checkpoint.resolve()),
        "loaded_keys": len(source),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "proj_out_after_loading": "Identity",
    }


def load_classifier_checkpoint(
    model: CBraModBoundaryPrompt, checkpoint: Path, task: str, expected_seed: int | None = None
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    classifier_state = {
        key.removeprefix("classifier."): value for key, value in state.items() if key.startswith("classifier.")
    }
    if not classifier_state:
        raise ValueError(f"{checkpoint} does not contain an official CBraMod classifier state.")
    saved_task = payload.get("args", {}).get("task") if isinstance(payload, dict) else None
    if saved_task is not None and saved_task != task:
        raise ValueError(f"Head checkpoint task is {saved_task}, but this run requests {task}.")
    saved_seed = payload.get("args", {}).get("seed") if isinstance(payload, dict) else None
    if expected_seed is not None and saved_seed is not None and int(saved_seed) != int(expected_seed):
        raise ValueError(
            f"Head checkpoint seed is {saved_seed}, but this prompt run uses seed {expected_seed}. "
            "Train/load the matching EEG-only head for this seed."
        )
    result = model.classifier.load_state_dict(classifier_state, strict=True)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "saved_task": saved_task,
        "saved_seed": saved_seed,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def set_trainable(model: CBraModBoundaryPrompt, train_backbone: bool, train_classifier: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model.pre_prompt is not None:
        for parameter in model.pre_prompt.parameters():
            parameter.requires_grad = True
    if model.post_prompt is not None:
        for parameter in model.post_prompt.parameters():
            parameter.requires_grad = True
    if model.fnirs_encoder is not None:
        for parameter in model.fnirs_encoder.parameters():
            parameter.requires_grad = True
    if model.graph_encoder is not None:
        for parameter in model.graph_encoder.parameters():
            parameter.requires_grad = True
    if model.attribute_encoder is not None:
        for parameter in model.attribute_encoder.parameters():
            parameter.requires_grad = True
    if train_classifier:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    if train_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True


def make_optimizer(model: CBraModBoundaryPrompt, args: argparse.Namespace, train_backbone: bool, train_classifier: bool):
    groups = []
    if train_classifier:
        groups.append({"params": list(model.classifier.parameters()), "lr": args.head_lr, "name": "head"})
    if model.pre_prompt is not None:
        groups.insert(0, {"params": list(model.pre_prompt.parameters()), "lr": args.feature_lr, "name": "pre_prompt"})
    if model.post_prompt is not None:
        groups.insert(0, {"params": list(model.post_prompt.parameters()), "lr": args.feature_lr, "name": "post_prompt"})
    if model.fnirs_encoder is not None:
        groups.insert(0, {"params": list(model.fnirs_encoder.parameters()), "lr": args.feature_lr, "name": "fnirs_encoder"})
    if model.graph_encoder is not None:
        groups.insert(0, {"params": list(model.graph_encoder.parameters()), "lr": args.feature_lr, "name": "sgformer_graph_encoder"})
    if model.attribute_encoder is not None:
        groups.insert(0, {"params": list(model.attribute_encoder.parameters()), "lr": args.feature_lr, "name": "tap_attribute_encoder"})
    if train_backbone:
        groups.insert(0, {"params": list(model.backbone.parameters()), "lr": args.backbone_lr, "name": "backbone"})
    if not groups:
        raise ValueError("No trainable parameters were selected.")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, seen, preds, labels = 0.0, 0, [], []
    for eeg, fnirs, fnirs_graph, y in data_loader:
        eeg = eeg.to(device, non_blocking=True).float()
        fnirs = fnirs.to(device, non_blocking=True).float()
        fnirs_graph = fnirs_graph.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        logits = model(eeg, fnirs, fnirs_graph)
        loss = criterion(logits, y)
        total += float(loss.item()) * len(y)
        seen += len(y)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return metrics(np.concatenate(labels), np.concatenate(preds), total / seen)


def train_epoch(model, data_loader, optimizer, device):
    model.train()
    if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
        model.backbone.eval()
    criterion = nn.CrossEntropyLoss()
    total, seen = 0.0, 0
    for eeg, fnirs, fnirs_graph, y in data_loader:
        eeg = eeg.to(device, non_blocking=True).float()
        fnirs = fnirs.to(device, non_blocking=True).float()
        fnirs_graph = fnirs_graph.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(eeg, fnirs, fnirs_graph), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(y)
        seen += len(y)
    return total / seen


def train_epoch_mope(
    model, data_loader, optimizer, device,
    importance_weight: float, attribute_weight: float,
    contrast_mode: str = "none", ot_temperature: float = 0.1,
    sinkhorn_epsilon: float = 0.1, sinkhorn_iterations: int = 20,
    ot_pair_weight: float = 0.1, ot_class_weight: float = 0.02,
) -> dict:
    model.train()
    if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
        model.backbone.eval()
    criterion = nn.CrossEntropyLoss()
    totals = {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "importance_loss": 0.0,
        "attribute_loss": 0.0,
        "ot_pair_loss": 0.0,
        "ot_class_loss": 0.0,
    }
    routing_totals, seen = {}, 0
    for eeg, fnirs, fnirs_graph, y in data_loader:
        eeg = eeg.to(device, non_blocking=True).float()
        fnirs = fnirs.to(device, non_blocking=True).float()
        fnirs_graph = fnirs_graph.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if contrast_mode == "dynamic_mapped_class_ot":
            logits, boundary_pairs = model(eeg, fnirs, fnirs_graph, return_aux=True)
            if not boundary_pairs:
                raise RuntimeError("MoPE OT contrast did not receive any enabled prompt boundary.")
            boundary_losses = [
                class_aware_ot_losses(
                    pairwise_token_ot(
                        eeg_anchor, prompt_tokens, sinkhorn_epsilon, sinkhorn_iterations
                    ),
                    y,
                    ot_temperature,
                )
                for eeg_anchor, prompt_tokens in boundary_pairs
            ]
            pair_loss = torch.stack([losses[0] for losses in boundary_losses]).mean()
            class_loss = torch.stack([losses[1] for losses in boundary_losses]).mean()
        else:
            logits = model(eeg, fnirs, fnirs_graph)
            pair_loss = logits.new_zeros(())
            class_loss = logits.new_zeros(())
        classification_loss = criterion(logits, y)
        importance_loss = model.importance_loss()
        attribute_loss = model.attribute_loss(y)
        loss = (
            classification_loss
            + importance_weight * importance_loss
            + attribute_weight * attribute_loss
            + ot_pair_weight * pair_loss
            + ot_class_weight * class_loss
        )
        loss.backward()
        optimizer.step()

        batch_size = len(y)
        totals["total_loss"] += float(loss.item()) * batch_size
        totals["classification_loss"] += float(classification_loss.item()) * batch_size
        totals["importance_loss"] += float(importance_loss.item()) * batch_size
        totals["attribute_loss"] += float(attribute_loss.item()) * batch_size
        totals["ot_pair_loss"] += float(pair_loss.item()) * batch_size
        totals["ot_class_loss"] += float(class_loss.item()) * batch_size
        for key, value in model.routing_statistics().items():
            routing_totals[key] = routing_totals.get(key, 0.0) + value * batch_size
        seen += batch_size
    return {
        **{key: value / seen for key, value in totals.items()},
        "routing": {key: value / seen for key, value in routing_totals.items()},
    }


def make_report(args, summary, counts) -> str:
    best, test, final = summary["best"], summary["best_test"], summary["final"]
    prompt_components = (
        "static + dense expert-routed dynamic + mapped"
        if args.prompt_family == "mope" else "static + dynamic + mapping"
    )
    mope_rows = (
        f"| MoPE experts / temperature | {args.expert_count} / {args.router_temperature:g} |\n"
        f"| Importance weight / threshold | {args.importance_weight:g} / {args.importance_threshold:g} |\n"
        if args.prompt_family == "mope" else ""
    )
    contrast_rows = (
        f"| MoPE OT contrast | {args.mope_contrast_mode} |\n"
        f"| OT temperature / Sinkhorn | {args.ot_temperature:g} / eps={args.sinkhorn_epsilon:g}, iter={args.sinkhorn_iterations} |\n"
        f"| OT pair / class weight | {args.ot_pair_weight:g} / {args.ot_class_weight:g} |\n"
        if args.prompt_family == "mope" else ""
    )
    return f"""# Boundary Conditional Prompt for CBraMod

## Design

{args.experiment_note}

The CBraMod encoder is frozen. fNIRS conditions static, dynamic, and mapping
prompts at the EFM boundaries only; no encoder layer is modified.

## Parameters

| Item | Value |
|---|---:|
| Backbone | CBraMod Base |
| Task | {SHIN_TASKS[args.task]['name']}; {SHIN_TASKS[args.task]['description']} |
| Mode | {args.mode} |
| Prompt family | {args.prompt_family} |
| Dynamic experts | {args.dynamic_expert_mode} |
| Mapped prompt | {args.mapped_mode} |
| Prompt source | {args.prompt_source} |
| Training strategy | {args.training_strategy} |
| Frozen-head checkpoint | {str(args.head_checkpoint) if args.head_checkpoint else "-"} |
| fNIRS pairing | {"shuffled within each data split" if args.shuffle_fnirs else "trial aligned"} |
| fNIRS conditioner | {args.fnirs_conditioner} |
| fNIRS input | {"raw 10-second sequence" if args.fnirs_conditioner == "temporal" else "mean / std / endpoint delta"} |
| Prompt components | {prompt_components} |
| Prompt count / rank | {args.prompt_count} / {args.prompt_rank} |
{mope_rows}{contrast_rows}| Feature lr | {args.feature_lr:g} |
| Head lr | {args.head_lr:g} |
| Backbone lr | {args.backbone_lr:g} |
| Backbone unfreeze epoch | {args.unfreeze_epoch} |
| Epochs | {args.epochs} |
| Batch size | {args.batch_size} |
| Seed | {args.seed} |
| Trainable prompt/conditioner params | {counts['prompt'] + counts['post_prompt'] + counts['fnirs_encoder'] + counts['sgformer_graph_encoder'] + counts['tap_attribute_encoder']} |
| Trainable head params | {counts['head']} |
| Backbone params | {counts['backbone']} |

## Results

| Checkpoint | Epoch | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Test Kappa |
|---|---:|---:|---:|---:|---:|---:|
| Best validation | {best['epoch']} | {best['val']['acc']:.4f} | {best['val']['f1_macro']:.4f} | {test['acc']:.4f} | {test['f1_macro']:.4f} | {test['kappa']:.4f} |
| Last | {final['epoch']} | {final['val']['acc']:.4f} | {final['val']['f1_macro']:.4f} | {final['test']['acc']:.4f} | {final['test']['f1_macro']:.4f} | {final['test']['kappa']:.4f} |

Best-test confusion matrix: `{test['confusion_matrix']}`.
"""


def main() -> None:
    args = parse_args()
    if args.mope_contrast_mode != "none":
        if args.prompt_family != "mope":
            raise ValueError("--mope-contrast-mode requires --prompt-family mope.")
        if args.mode == "eeg_only":
            raise ValueError("--mope-contrast-mode requires an enabled prompt boundary.")
        if args.prompt_source != "conditional":
            raise ValueError("--mope-contrast-mode requires trial-aligned conditional fNIRS prompts.")
        if args.mope_drop_component in {"dynamic", "mapped"}:
            raise ValueError(
                "dynamic_mapped_class_ot requires both dynamic and mapped prompt components."
            )
        if args.ot_temperature <= 0 or args.sinkhorn_epsilon <= 0 or args.sinkhorn_iterations < 1:
            raise ValueError("Invalid OT contrast temperature or Sinkhorn settings.")
    args.portable_root = args.portable_root.resolve()
    args.prep_root = args.prep_root.resolve()
    prep_scripts = args.prep_root / "scripts"
    cbramod_root = args.portable_root / "CBraMod"
    sys.path.insert(0, str(prep_scripts))
    sys.path.insert(0, str(cbramod_root))

    from models.cbramod import CBraMod

    if args.shin_root is None:
        args.shin_root = args.prep_root / "datasets" / "shin2017_eeg_nirs_left_right_hand_mi"
    if args.cache_path is None:
        args.cache_path = (
            Path(__file__).resolve().parent / "cache" / f"shin2017_{args.task}_bids_nirs_paired_sub01-sub29_10patch.npz"
        )
    if args.sgformer_cache_path is None:
        cache_path = Path(args.cache_path)
        args.sgformer_cache_path = cache_path.with_name(
            f"{cache_path.stem}_sgformer_hbo-hbr_graph.npz"
        )
    if args.checkpoint is None:
        args.checkpoint = cbramod_root / "pretrained_weights" / "pretrained_weights.pth"
    if args.experiment_note is None:
        args.experiment_note = (
            "Evaluate boundary conditional prompts with a frozen CBraMod encoder and a matched "
            "official downstream classifier."
        )
    if args.training_strategy == "prompt_only":
        if args.mode == "eeg_only":
            raise ValueError("prompt_only requires a prompt mode: pre, post, or pre_post.")
        if args.head_checkpoint is None:
            raise ValueError("prompt_only requires --head-checkpoint from a matching EEG-only run.")
        if args.unfreeze_epoch <= args.epochs:
            raise ValueError("prompt_only keeps CBraMod frozen; set --unfreeze-epoch greater than --epochs.")
    if args.prompt_source == "static" and args.shuffle_fnirs:
        raise ValueError("--shuffle-fnirs is only valid for conditional prompts.")
    if args.prompt_family == "mope" and args.mode != "eeg_only":
        if args.expert_count < 2 or args.router_temperature <= 0:
            raise ValueError("MoPE requires --expert-count >= 2 and --router-temperature > 0.")
        if (
            args.router_noise_std < 0 or args.importance_threshold < 0
            or args.importance_weight < 0 or args.tap_attribute_weight < 0
        ):
            raise ValueError("MoPE noise, thresholds, and loss weights must be non-negative.")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "mplconfig"))
    seed_everything(args.seed)

    args.shin_root = str(Path(args.shin_root).resolve())
    args.eeg_bids_root = str(Path(args.eeg_bids_root).resolve())
    args.cache_path = str(Path(args.cache_path).resolve())
    eeg, fnirs_stats, fnirs_sequence, labels, meta = load_paired_bids_trial_cache(args)
    if "subject_ids" not in meta or eeg.ndim != 4 or eeg.shape[2] != args.seq_len:
        args.rebuild_cache = True
        eeg, fnirs_stats, fnirs_sequence, labels, meta = load_paired_bids_trial_cache(args)
    if tuple(eeg.shape[1:]) != (30, 10, 200):
        raise ValueError(f"Expected BDF CBraMod input [B,30,10,200], got {tuple(eeg.shape)}")
    labels = labels.astype(np.int64)
    subject_ids = np.asarray(meta["subject_ids"], dtype=np.int64)

    available = set(subject_ids.tolist())
    train_subjects = [s for s in args.train_subjects if s in available]
    val_subjects = [s for s in args.val_subjects if s in available]
    test_subjects = [s for s in args.test_subjects if s in available]
    split_sets = [set(train_subjects), set(val_subjects), set(test_subjects)]
    if not all(split_sets) or any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train/val/test subjects must be non-empty and disjoint")

    train_idx = np.flatnonzero(np.isin(subject_ids, train_subjects))
    val_idx = np.flatnonzero(np.isin(subject_ids, val_subjects))
    test_idx = np.flatnonzero(np.isin(subject_ids, test_subjects))
    fnirs = fnirs_sequence if args.fnirs_conditioner == "temporal" else fnirs_stats
    fnirs = normalize_fnirs_from_train(fnirs, train_idx)
    needs_mapped_graph = (
        args.mapped_mode == "sgformer" and args.mope_drop_component != "mapped"
    )
    needs_attribute_graph = (
        args.dynamic_expert_mode == "tap4x4" and args.mope_drop_component != "dynamic"
    )
    needs_graph = (
        args.prompt_family == "mope" and args.mode != "eeg_only"
        and (needs_mapped_graph or needs_attribute_graph)
    )
    fnirs_graph = None
    graph_meta = None
    graph_montage = None
    if needs_graph:
        fnirs_graph, graph_labels, graph_subjects, graph_meta = load_sgformer_graph_trials(
            shin_root=Path(args.shin_root),
            subjects=args.subjects,
            task_sessions=SHIN_TASKS[args.task]["sessions"],
            fnirs_window=args.fnirs_window,
            fnirs_offset=args.fnirs_offset,
            cache_path=Path(args.sgformer_cache_path),
            rebuild_cache=args.rebuild_cache,
        )
        if not np.array_equal(graph_labels, labels):
            raise ValueError("SGFormer graph labels do not match the existing paired prompt cache")
        if not np.array_equal(graph_subjects.astype(np.int64), subject_ids):
            raise ValueError("SGFormer graph subject order does not match the existing paired prompt cache")
        fnirs_graph = normalize_graph_from_train(fnirs_graph, train_idx)
        graph_montage = load_fnirs_montage(Path(args.shin_root))
    if args.shuffle_fnirs:
        if args.mode == "eeg_only":
            raise ValueError("--shuffle-fnirs is meaningful only for a conditional-prompt mode")
        fnirs, fnirs_graph = shuffle_prompt_sources_within_splits(
            fnirs,
            fnirs_graph,
            [train_idx, val_idx, test_idx],
            args.seed + 1009,
        )

    diagnostics = {
        "stage": "boundary_mope_prompt" if args.prompt_family == "mope" else "boundary_conditional_prompt",
        "backbone": "CBraMod",
        "task": {"key": args.task, "name": SHIN_TASKS[args.task]["name"],
                 "description": SHIN_TASKS[args.task]["description"]},
        "encoder_internal_prompt": False,
        "mode": args.mode,
        "shuffle_fnirs": args.shuffle_fnirs,
        "prompt_source": args.prompt_source,
        "prompt_family": args.prompt_family,
        "mapped_mode": args.mapped_mode,
        "dynamic_expert_mode": args.dynamic_expert_mode,
        "fnirs_conditioner": args.fnirs_conditioner,
        "prompt_components": ["static", "expert_routed_dynamic", "mapped"] if args.prompt_family == "mope"
        else ["static", "dynamic", "mapping"],
        "mope": {
            "expert_count": args.expert_count,
            "routing": "dense_softmax",
            "temperature": args.router_temperature,
            "router_noise_std": args.router_noise_std,
            "importance_threshold": args.importance_threshold,
            "importance_weight": args.importance_weight,
            "tap_attribute_weight": args.tap_attribute_weight,
            "contrast_mode": args.mope_contrast_mode,
            "ot_temperature": args.ot_temperature,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "ot_pair_weight": args.ot_pair_weight,
            "ot_class_weight": args.ot_class_weight,
            "drop_component": args.mope_drop_component,
            "paper_faithfulness": (
                "paper prompt decomposition and routing with a CBraMod boundary residual projection; "
                "not per-layer token concatenation"
            ),
        } if args.prompt_family == "mope" else None,
        "training_strategy": args.training_strategy,
        "uses_fnirs": args.prompt_source == "conditional",
        "eeg_shape": list(eeg.shape),
        "fnirs_shape": list(fnirs.shape),
        "fnirs_graph_shape": list(fnirs_graph.shape) if fnirs_graph is not None else None,
        "structured_fnirs_graph": ({
            "cache": str(Path(args.sgformer_cache_path).resolve()),
            "preprocessing": graph_meta["preprocessing"],
            "chromophore_order": graph_meta["chromophore_order"],
            "montage_path": graph_montage["path"],
            "edge_count": graph_montage["edge_count"],
            "graph_method": graph_montage["graph_method"],
            "graph_dimension": args.sgformer_graph_dimension,
            "attention_residual_weight": args.sgformer_attention_residual_weight,
            "graph_weight": args.sgformer_graph_weight,
        } if needs_graph else None),
        "sgformer_graph": ({
            "cache": str(Path(args.sgformer_cache_path).resolve()),
            "graph_dimension": args.sgformer_graph_dimension,
            "attention_residual_weight": args.sgformer_attention_residual_weight,
            "graph_weight": args.sgformer_graph_weight,
        } if needs_graph and needs_mapped_graph else None),
        "subjects": {"train": train_subjects, "val": val_subjects, "test": test_subjects},
        "trials": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "fnirs_input": (
            "fixed deterministic reference sequence; measured fNIRS values are not used"
            if args.prompt_source == "static"
            else
            "raw fNIRS sequence [trial,time,channels] normalized from training subjects"
            if args.fnirs_conditioner == "temporal"
            else "mean/std/endpoint_delta normalized from training subjects"
        ),
        "source_meta": meta,
    }
    write_json(args.output_dir / "diagnostics.json", diagnostics)
    if args.diagnose_only:
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    train_loader = DataLoader(
        CBraModPromptDataset(eeg, fnirs, fnirs_graph, labels, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CBraModPromptDataset(eeg, fnirs, fnirs_graph, labels, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        CBraModPromptDataset(eeg, fnirs, fnirs_graph, labels, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    _, channels, patch_count, _ = eeg.shape
    model = CBraModBoundaryPrompt(
        CBraMod,
        channels=channels,
        patch_count=patch_count,
        fnirs_dim=fnirs.shape[-1],
        hidden_dim=args.prompt_hidden,
        prompt_count=args.prompt_count,
        prompt_rank=args.prompt_rank,
        mode=args.mode,
        dropout=args.dropout,
        fnirs_conditioner=args.fnirs_conditioner,
        prompt_source=args.prompt_source,
        prompt_family=args.prompt_family,
        expert_count=args.expert_count,
        router_temperature=args.router_temperature,
        router_noise_std=args.router_noise_std,
        importance_threshold=args.importance_threshold,
        mope_drop_component=args.mope_drop_component,
        mapped_mode=args.mapped_mode,
        dynamic_expert_mode=args.dynamic_expert_mode,
        graph_montage=graph_montage,
        sgformer_graph_dimension=args.sgformer_graph_dimension,
        sgformer_attention_residual_weight=args.sgformer_attention_residual_weight,
        sgformer_graph_weight=args.sgformer_graph_weight,
    )
    diagnostics["pretrained_load"] = load_pretrained(model, Path(args.checkpoint).resolve())
    if args.training_strategy == "prompt_only":
        diagnostics["classifier_load"] = load_classifier_checkpoint(
            model, args.head_checkpoint.resolve(), args.task, expected_seed=args.seed
        )
    counts = {
        "prompt": sum(p.numel() for p in model.pre_prompt.parameters()) if model.pre_prompt is not None else 0,
        "post_prompt": sum(p.numel() for p in model.post_prompt.parameters()) if model.post_prompt is not None else 0,
        "fnirs_encoder": sum(p.numel() for p in model.fnirs_encoder.parameters()) if model.fnirs_encoder is not None else 0,
        "sgformer_graph_encoder": sum(p.numel() for p in model.graph_encoder.parameters()) if model.graph_encoder is not None else 0,
        "tap_attribute_encoder": sum(p.numel() for p in model.attribute_encoder.parameters()) if model.attribute_encoder is not None else 0,
        "head": sum(p.numel() for p in model.classifier.parameters()),
        "backbone": sum(p.numel() for p in model.backbone.parameters()),
    }
    diagnostics["parameters"] = counts
    write_json(args.output_dir / "diagnostics.json", diagnostics)
    model.to(device)

    train_classifier = args.training_strategy == "joint"
    train_backbone = train_classifier and args.unfreeze_epoch <= 1
    set_trainable(model, train_backbone, train_classifier)
    optimizer = make_optimizer(model, args, train_backbone, train_classifier)
    diagnostics["trainable_parameters"] = {
        name: sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        for name, module in {
            "classifier": model.classifier, "pre_prompt": model.pre_prompt,
            "post_prompt": model.post_prompt, "fnirs_encoder": model.fnirs_encoder,
            "sgformer_graph_encoder": model.graph_encoder,
            "tap_attribute_encoder": model.attribute_encoder,
            "backbone": model.backbone,
        }.items() if module is not None
    }
    write_json(args.output_dir / "diagnostics.json", diagnostics)

    history, best_record, best_acc = [], None, -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_classifier and epoch == args.unfreeze_epoch and not train_backbone:
            train_backbone = True
            set_trainable(model, True, True)
            optimizer = make_optimizer(model, args, True, True)
            print(f"[train] epoch {epoch}: unfreezing CBraMod backbone", flush=True)

        if args.prompt_family == "mope" and args.mode != "eeg_only":
            train_statistics = train_epoch_mope(
                model, train_loader, optimizer, device,
                args.importance_weight, args.tap_attribute_weight,
                args.mope_contrast_mode, args.ot_temperature,
                args.sinkhorn_epsilon, args.sinkhorn_iterations,
                args.ot_pair_weight, args.ot_class_weight,
            )
            train_loss = train_statistics["total_loss"]
        else:
            train_statistics = None
            train_loss = train_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "stage": (
                f"{args.mode}_prompt_only" if not train_classifier
                else args.mode if not train_backbone else f"{args.mode}_plus_finetune"
            ),
            "train_loss": train_loss,
            "val": val,
            "elapsed_seconds": time.time() - started,
        }
        if train_statistics is not None:
            record["train_loss_components"] = {
                "classification": train_statistics["classification_loss"],
                "importance": train_statistics["importance_loss"],
                "importance_weight": args.importance_weight,
                "attribute": train_statistics["attribute_loss"],
                "attribute_weight": args.tap_attribute_weight,
                "ot_pair": train_statistics["ot_pair_loss"],
                "ot_pair_weight": args.ot_pair_weight,
                "ot_class": train_statistics["ot_class_loss"],
                "ot_class_weight": args.ot_class_weight,
                "mope_contrast_mode": args.mope_contrast_mode,
            }
            record["routing"] = train_statistics["routing"]
        history.append(record)
        message = (
            f"epoch {epoch:03d}/{args.epochs} stage={record['stage']} "
            f"train_loss={train_loss:.4f} val_acc={val['acc']:.4f} val_f1={val['f1_macro']:.4f}"
        )
        if train_statistics is not None:
            message += f" imp={train_statistics['importance_loss']:.4f}"
            if args.mope_contrast_mode != "none":
                message += (
                    f" ot_pair={train_statistics['ot_pair_loss']:.4f}"
                    f" ot_class={train_statistics['ot_class_loss']:.4f}"
                )
        print(message, flush=True)
        if val["acc"] > best_acc:
            best_acc, best_record = val["acc"], record
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       args.output_dir / "best_model.pth")
        write_json(args.output_dir / "history.json", history)

    final_test = evaluate(model, test_loader, device)
    final = {"epoch": args.epochs, "val": history[-1]["val"], "test": final_test}
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args)},
               args.output_dir / "last_model.pth")
    best_checkpoint = torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    best_test = evaluate(model, test_loader, device)
    summary = {
        "best": best_record,
        "best_test": best_test,
        "final": final,
        "elapsed_seconds": time.time() - started,
        "seed": args.seed,
        "mapped_mode": args.mapped_mode,
        "dynamic_expert_mode": args.dynamic_expert_mode,
        "mope_contrast_mode": args.mope_contrast_mode,
        "ot": {
            "temperature": args.ot_temperature,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "pair_weight": args.ot_pair_weight,
            "class_weight": args.ot_class_weight,
        },
        "experiment_note": args.experiment_note,
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "EXPERIMENT_RECORD.md").write_text(make_report(args, summary, counts), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
