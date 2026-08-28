"""SGFormer-enhanced mapped token for the foundation boundary MoPE runner.

The graph encoder is migrated from the standalone SHIN SGFormer experiment.
It preserves independent HbO/HbR temporal encoders, the 36-node montage,
one-layer SGFormer global attention, and the parallel local GCN.  Only the
readout is specific to MoPE: 36 graph nodes are attention-pooled to the single
mapped token expected by the existing prompt concatenation path.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.spatial import Delaunay, QhullError
import torch
from torch import nn
import torch.nn.functional as F


ABSORPTION = np.asarray(
    [[134.9558, 356.624156], [243.6574, 159.210996]],
    dtype=np.float64,
)


def normalized_adjacency(node_count: int, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index [2,E], got {tuple(edge_index.shape)}")
    adjacency = torch.zeros(node_count, node_count, dtype=torch.float32)
    adjacency[edge_index[0], edge_index[1]] = 1.0
    adjacency[edge_index[1], edge_index[0]] = 1.0
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


class DenseGraphConv(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(dimension, dimension)
        self.neighbor_projection = nn.Linear(dimension, dimension, bias=False)
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("ij,bjd->bid", adjacency, x)
        update = self.self_projection(x) + self.neighbor_projection(neighbors)
        return self.norm(x + self.dropout(F.gelu(update)))


class SGFormerGraphEncoder(nn.Module):
    """One-layer SGFormer used by the standalone 36-node SHIN graph model."""

    def __init__(
        self,
        dimension: int,
        dropout: float,
        attention_residual_weight: float = 0.5,
        graph_weight: float = 0.8,
    ) -> None:
        super().__init__()
        if not 0.0 <= attention_residual_weight <= 1.0:
            raise ValueError("SGFormer attention residual weight must be in [0,1]")
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError("SGFormer graph weight must be in [0,1]")
        self.attention_residual_weight = float(attention_residual_weight)
        self.graph_weight = float(graph_weight)
        self.input_layer = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.global_norm = nn.LayerNorm(dimension)
        self.local_gcn = DenseGraphConv(dimension, dropout)
        self.output_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _frobenius_normalize(x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True)
        return x / norm.clamp_min(torch.finfo(x.dtype).eps)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected SGFormer input [B,N,D], got {tuple(x.shape)}")
        nodes = x.shape[1]
        z0 = self.input_layer(x)
        query = self._frobenius_normalize(self.query(z0))
        key = self._frobenius_normalize(self.key(z0))
        value = self.value(z0)
        key_value = torch.einsum("bnd,bne->bde", key, value)
        numerator = torch.einsum("bnd,bde->bne", query, key_value) + nodes * value
        key_sum = key.sum(dim=1)
        denominator = torch.einsum("bnd,bd->bn", query, key_sum).unsqueeze(-1) + nodes
        global_attention = numerator / denominator
        beta = self.attention_residual_weight
        global_output = self.global_norm(beta * global_attention + (1.0 - beta) * z0)
        global_output = self.dropout(global_output)
        local_output = self.local_gcn(z0, adjacency)
        alpha = self.graph_weight
        return self.output_norm((1.0 - alpha) * global_output + alpha * local_output)


class SGFormerMappedEncoder(nn.Module):
    """Map a preprocessed HbO/HbR trial to 36 node tokens of width 200."""

    def __init__(
        self,
        positions_3d: torch.Tensor,
        edge_index: torch.Tensor,
        prompt_dimension: int = 200,
        graph_dimension: int = 128,
        dropout: float = 0.1,
        attention_residual_weight: float = 0.5,
        graph_weight: float = 0.8,
    ) -> None:
        super().__init__()
        if positions_3d.shape != (36, 3):
            raise ValueError(f"Expected 36 fNIRS positions, got {tuple(positions_3d.shape)}")
        if graph_dimension % 2 != 0 or (graph_dimension // 2) % 8 != 0:
            raise ValueError("graph_dimension/2 must be divisible by 8")
        geometry = positions_3d.float()
        geometry = (geometry - geometry.mean(dim=0, keepdim=True)) / (
            geometry.std(dim=0, keepdim=True).clamp_min(1e-6)
        )
        self.register_buffer("positions_3d", geometry)
        self.register_buffer("adjacency", normalized_adjacency(36, edge_index.long()))
        branch_dimension = graph_dimension // 2

        def make_chromophore_encoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=5, padding=2),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv1d(64, branch_dimension, kernel_size=5, padding=2),
                nn.GroupNorm(8, branch_dimension),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )

        self.hbo_temporal_encoder = make_chromophore_encoder()
        self.hbr_temporal_encoder = make_chromophore_encoder()
        self.geometry_encoder = nn.Sequential(
            nn.Linear(3, graph_dimension),
            nn.GELU(),
            nn.Linear(graph_dimension, graph_dimension),
        )
        self.node_embedding = nn.Embedding(36, graph_dimension)
        self.sgformer = SGFormerGraphEncoder(
            dimension=graph_dimension,
            dropout=dropout,
            attention_residual_weight=attention_residual_weight,
            graph_weight=graph_weight,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(graph_dimension),
            nn.Linear(graph_dimension, prompt_dimension),
        )

    def forward(self, fnirs: torch.Tensor) -> torch.Tensor:
        if fnirs.ndim != 4 or tuple(fnirs.shape[1:3]) != (36, 2):
            raise ValueError(f"Expected fNIRS [B,36,2,T], got {tuple(fnirs.shape)}")
        batch, nodes, _, samples = fnirs.shape
        hbo = fnirs[:, :, 0, :].reshape(batch * nodes, 1, samples)
        hbr = fnirs[:, :, 1, :].reshape(batch * nodes, 1, samples)
        hbo_features = self.hbo_temporal_encoder(hbo).squeeze(-1)
        hbr_features = self.hbr_temporal_encoder(hbr).squeeze(-1)
        temporal = torch.cat((hbo_features, hbr_features), dim=-1).reshape(batch, nodes, -1)
        geometry = self.geometry_encoder(self.positions_3d).unsqueeze(0)
        node_ids = torch.arange(nodes, device=fnirs.device)
        features = temporal + geometry + self.node_embedding(node_ids).unsqueeze(0)
        return self.projection(self.sgformer(features, self.adjacency))


class SGFormerMappedReadout(nn.Module):
    """Attention-pool 36 graph nodes to the existing one-token mapped slot."""

    def __init__(self, d_model: int = 200, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.mean_projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self._attention_weights: torch.Tensor | None = None
        nn.init.normal_(self.query, std=0.02)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 3 or nodes.shape[1:] != (36, 200):
            raise ValueError(f"Expected mapped nodes [B,36,200], got {tuple(nodes.shape)}")
        query = self.query.expand(nodes.shape[0], -1, -1)
        attended, weights = self.attention(query, nodes, nodes, need_weights=True)
        self._attention_weights = weights
        graph_mean = self.mean_projection(nodes.mean(dim=1, keepdim=True))
        return self.output(self.norm(attended + graph_mean))

    @torch.no_grad()
    def attention_statistics(self) -> dict[str, float]:
        if self._attention_weights is None:
            return {}
        weights = self._attention_weights.detach().squeeze(1)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
        return {
            "mapped_node_entropy": float(entropy.mean().item()),
            "mapped_node_max_weight": float(weights.max(dim=-1).values.mean().item()),
        }


def _validate_wavelength_channels(clab: object) -> None:
    names = [str(value) for value in np.asarray(clab).reshape(-1)]
    if len(names) != 72:
        raise ValueError(f"Expected 72 wavelength channels, got {len(names)}")
    low = [name.removesuffix("lowWL") for name in names[:36]]
    high = [name.removesuffix("highWL") for name in names[36:]]
    if any(not name.endswith("lowWL") for name in names[:36]):
        raise ValueError("First 36 fNIRS channels are not low-wavelength channels")
    if any(not name.endswith("highWL") for name in names[36:]):
        raise ValueError("Last 36 fNIRS channels are not high-wavelength channels")
    if low != high:
        raise ValueError("Low/high wavelength source-detector orders differ")


def intensity_to_hbo_hbr(
    intensity: np.ndarray,
    distance_m: float = 0.03,
    ppf: float = 6.0,
) -> np.ndarray:
    """Return HbO/HbR concentrations as [time,36,2] in micromolar."""
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
    return concentration * 1e6


def _fnirs_data_root(shin_root: Path) -> Path:
    nested = shin_root / "NIRS"
    return nested if nested.is_dir() else shin_root


def load_fnirs_montage(shin_root: Path, subject: int = 1) -> dict:
    root = _fnirs_data_root(Path(shin_root))
    path = root / f"subject {subject:02d}" / "mnt.mat"
    montage = loadmat(path, simplify_cells=True)["mnt"]
    positions = np.asarray(montage["pos_3d"], dtype=np.float64).T
    xy = np.column_stack((
        np.asarray(montage["x"], dtype=np.float64),
        np.asarray(montage["y"], dtype=np.float64),
    ))
    sd = np.asarray(montage["sd"], dtype=np.int64)
    if positions.shape != (36, 3) or xy.shape != (36, 2) or sd.shape != (36, 2):
        raise RuntimeError(
            f"Unexpected montage shapes: pos={positions.shape}, xy={xy.shape}, sd={sd.shape}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(xy).all():
        raise RuntimeError(f"Non-finite fNIRS montage coordinates in {path}")
    edge_set: set[tuple[int, int]] = set()
    graph_method = "delaunay_2d"
    try:
        for triangle in Delaunay(xy).simplices:
            for left, right in ((0, 1), (1, 2), (2, 0)):
                edge_set.add(tuple(sorted((int(triangle[left]), int(triangle[right])))))
    except QhullError:
        graph_method = "knn_3d_fallback_k4"
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        for node in range(36):
            for neighbor in np.argsort(distances[node])[1:5]:
                edge_set.add(tuple(sorted((node, int(neighbor)))))
    sd_zero = sd - 1 if sd.min() == 1 else sd.copy()
    for left in range(36):
        for right in range(left + 1, 36):
            if sd_zero[left, 0] == sd_zero[right, 0] or sd_zero[left, 1] == sd_zero[right, 1]:
                edge_set.add((left, right))
    edge_index = np.asarray(sorted(edge_set), dtype=np.int64).T
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RuntimeError("Failed to construct fNIRS graph edges")
    return {
        "path": str(path.resolve()),
        "positions_3d": positions.astype(np.float32),
        "edge_index": edge_index,
        "edge_count": int(edge_index.shape[1]),
        "graph_method": graph_method + "+shared_optode",
    }


def load_sgformer_graph_trials(
    shin_root: Path,
    subjects: list[int],
    task_sessions: tuple[tuple[str, int], ...],
    fnirs_window: float,
    fnirs_offset: float,
    cache_path: Path | None = None,
    rebuild_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load SGFormer HbO/HbR trials in the same subject/session/event order."""
    expected_samples = int(round(fnirs_window * 10.0))
    expected_cache = {
        "subjects": list(map(int, subjects)),
        "task_sessions": [int(session) for _, session in task_sessions],
        "fnirs_window": float(fnirs_window),
        "fnirs_offset": float(fnirs_offset),
    }
    if cache_path is not None and cache_path.is_file() and not rebuild_cache:
        cached = np.load(cache_path, allow_pickle=True)
        meta = cached["meta"].item()
        values = cached["fnirs_graph"]
        if (
            all(meta.get(key) == value for key, value in expected_cache.items())
            and tuple(values.shape[1:]) == (36, 2, expected_samples)
        ):
            return values, cached["labels"], cached["subjects"], meta
    root = _fnirs_data_root(Path(shin_root))
    graph_trials: list[np.ndarray] = []
    labels: list[int] = []
    subject_ids: list[int] = []
    session_ids: list[int] = []
    for subject in subjects:
        folder = root / f"subject {subject:02d}"
        cnt = loadmat(folder / "cnt.mat", simplify_cells=True)["cnt"]
        mrk = loadmat(folder / "mrk.mat", simplify_cells=True)["mrk"]
        for _, session_index in task_sessions:
            recording, markers = cnt[session_index], mrk[session_index]
            sampling_rate = float(recording["fs"])
            if abs(sampling_rate - 10.0) > 1e-6:
                raise ValueError(f"Expected fNIRS at 10 Hz for subject {subject}")
            _validate_wavelength_channels(recording["clab"])
            continuous = intensity_to_hbo_hbr(np.asarray(recording["x"], dtype=np.float64))
            b, a = butter(3, [0.01, 0.1], btype="bandpass", fs=sampling_rate)
            continuous = filtfilt(b, a, continuous, axis=0)
            times_ms = np.asarray(markers["time"], dtype=np.float64).reshape(-1)
            descriptions = np.asarray(markers["event"]["desc"], dtype=np.int64).reshape(-1)
            events = [
                (int(round(time_ms * sampling_rate / 1000.0)), int(description - 1))
                for time_ms, description in zip(times_ms, descriptions)
                if description in (1, 2)
            ]
            if len(events) != 20 or Counter(label for _, label in events) != Counter({0: 10, 1: 10}):
                raise RuntimeError(f"Unexpected fNIRS events for subject {subject}, session {session_index}")
            for event_sample, label in events:
                start = event_sample + int(round(fnirs_offset * sampling_rate))
                stop = start + expected_samples
                baseline_start, baseline_stop = event_sample - 50, event_sample - 20
                if baseline_start < 0 or stop > continuous.shape[0]:
                    raise RuntimeError(
                        f"fNIRS epoch out of range for subject {subject}, session {session_index}"
                    )
                baseline = continuous[baseline_start:baseline_stop].mean(axis=0, keepdims=True)
                trial = continuous[start:stop] - baseline
                if trial.shape != (expected_samples, 36, 2) or not np.isfinite(trial).all():
                    raise RuntimeError(f"Invalid graph trial shape/content: {trial.shape}")
                graph_trials.append(trial.transpose(1, 2, 0).astype(np.float32))
                labels.append(label)
                subject_ids.append(subject)
                session_ids.append(session_index)
    values = np.stack(graph_trials).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    subject_array = np.asarray(subject_ids, dtype=np.int16)
    meta = {
        "preprocessing": "intensity -> OD -> MBLL -> HbO/HbR -> 0.01-0.1 Hz -> -5..-2 s baseline",
        "shape": list(values.shape),
        **expected_cache,
        "session_ids": session_ids,
        "chromophore_order": ["HbO", "HbR"],
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
        np.savez_compressed(
            temporary,
            fnirs_graph=values,
            labels=label_array,
            subjects=subject_array,
            meta=meta,
        )
        temporary.replace(cache_path)
    return values, label_array, subject_array, meta


def normalize_graph_from_train(values: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    if values.ndim != 4 or values.shape[1] < 1 or values.shape[2] != 2:
        raise ValueError(f"Expected graph trials [N,nodes,2,T], got {values.shape}")
    train = values[train_indices]
    mean = train.mean(axis=(0, 3), keepdims=True, dtype=np.float64)
    std = train.std(axis=(0, 3), keepdims=True, dtype=np.float64)
    normalized = (values - mean) / np.maximum(std, 1e-6)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized SGFormer graph trials")
    return normalized.astype(np.float32)
