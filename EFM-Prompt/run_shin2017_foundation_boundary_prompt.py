"""Boundary MoPE prompts for CodeBrain and CSBrain on the SHIN dataset.

Both adapters inject prompts after native patch embedding and/or after the
backbone output. The raw EEG input is never modified and encoder weights stay
frozen unless an explicit joint fine-tuning stage is requested.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from cbramod_mope_boundary import (  # noqa: E402
    MoPEBoundaryPrompt,
    TAPFourAttributeFnirsEncoder,
)
from sgformer_mapped_prompt import (  # noqa: E402
    SGFormerMappedEncoder,
    load_fnirs_montage,
    load_sgformer_graph_trials,
    normalize_graph_from_train,
)
from run_shin2017_cbramod_fnirs_feature_stage1 import (  # noqa: E402
    SHIN_TASKS,
    FnirsTemporalEncoder,
    load_paired_bids_trial_cache,
    metrics,
    normalize_fnirs_from_train,
)


DEFAULT_PREP_ROOT = Path(r"D:\0senior student creation\braindecode_codebrain_prep")
DEFAULT_EEG_ROOT = Path(r"D:\0senior student creation\datasets\shin2017_eeg_bids_bdf")
DEFAULT_CODEBRAIN_CHECKPOINT = Path(
    r"D:\0senior student creation\2026-06-27_MI_BCI_IV_2a_4models_experiment_log\repos\CodeBrain\Checkpoints\CodeBrain.pth"
)
DEFAULT_CSBRAIN_CHECKPOINT = Path(
    r"D:\0senior student creation\2026-06-27_MI_BCI_IV_2a_4models_experiment_log\repos\CSBrain\pth_downloaded\pth\CSBrain.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CodeBrain/CSBrain SHIN boundary prompt runner")
    parser.add_argument("--portable-root", type=Path, default=SCRIPT_ROOT.parent)
    parser.add_argument("--prep-root", type=Path, default=DEFAULT_PREP_ROOT)
    parser.add_argument("--eeg-bids-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--shin-root", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=("codebrain", "csbrain"), required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, default=None)
    parser.add_argument("--task", choices=("mi", "ma"), default="mi")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 30)))
    parser.add_argument("--train-subjects", nargs="+", type=int, default=list(range(1, 20)))
    parser.add_argument("--val-subjects", nargs="+", type=int, default=list(range(20, 25)))
    parser.add_argument("--test-subjects", nargs="+", type=int, default=list(range(25, 30)))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=200)
    parser.add_argument("--eeg-scale", type=float, default=100.0)
    parser.add_argument("--fnirs-window", type=float, default=10.0)
    parser.add_argument("--fnirs-offset", type=float, default=0.0)
    parser.add_argument("--fnirs-conditioner", choices=("stats", "temporal"), default="temporal")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prompt-hidden", type=int, default=256)
    parser.add_argument("--prompt-count", type=int, default=6)
    parser.add_argument("--prompt-rank", type=int, default=8)
    parser.add_argument("--prompt-source", choices=("conditional", "static"), default="conditional")
    parser.add_argument("--expert-count", type=int, default=16)
    parser.add_argument("--router-temperature", type=float, default=0.1)
    parser.add_argument("--router-noise-std", type=float, default=0.00390625)
    parser.add_argument("--importance-threshold", type=float, default=0.05)
    parser.add_argument("--importance-weight", type=float, default=0.01)
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
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FoundationTrialDataset(Dataset):
    def __init__(
        self,
        eeg: np.ndarray,
        fnirs: np.ndarray,
        fnirs_graph: np.ndarray | None,
        labels: np.ndarray,
        indices: np.ndarray,
    ) -> None:
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


def clean_state(source):
    if isinstance(source, dict):
        source = source.get("state_dict", source.get("model", source))
    return {key.removeprefix("module."): value for key, value in source.items()}


def load_compatible(module: nn.Module, checkpoint: Path) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint}")
    clean = clean_state(torch.load(checkpoint, map_location="cpu", weights_only=False))
    current = module.state_dict()
    matched = {
        key: value for key, value in clean.items()
        if key in current and current[key].shape == value.shape
    }
    if not matched:
        raise RuntimeError(f"No compatible backbone tensors found in {checkpoint}")
    result = module.load_state_dict(matched, strict=False)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_tensor_count": len(clean),
        "matched_tensor_count": len(matched),
        "matched_parameter_count": int(sum(value.numel() for value in matched.values())),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


class CodeBrainBoundaryEncoder(nn.Module):
    """CodeBrain SSSM with prompt hooks around its native patch embedding."""

    def __init__(self, portable_root: Path, checkpoint: Path, dropout: float, n_layer: int = 8):
        super().__init__()
        codebrain_root = portable_root / "CodeBrain" / "external" / "CodeBrain-source"
        sys.path.insert(0, str(codebrain_root))
        from Models.SSSM import SSSM

        self.backbone = SSSM(
            in_channels=200,
            res_channels=200,
            skip_channels=200,
            out_channels=200,
            num_res_layers=n_layer,
            diffusion_step_embed_dim_in=200,
            diffusion_step_embed_dim_mid=200,
            diffusion_step_embed_dim_out=200,
            s4_lmax=570,
            s4_d_state=64,
            s4_dropout=dropout,
            s4_bidirectional=True,
            s4_layernorm=True,
            codebook_size_t=4096,
            codebook_size_f=4096,
            if_codebook=False,
        )
        self.pretrained_report = load_compatible(self.backbone, checkpoint)

    def encode(self, eeg: torch.Tensor, pre_prompt=None, post_prompt=None) -> torch.Tensor:
        batch, channels, patches, _ = eeg.shape
        tokens = self.backbone.patch_embedding(eeg)
        if pre_prompt is not None:
            tokens = tokens + pre_prompt

        # This is SSSM.forward after patch embedding, kept explicit so the
        # prompt is placed in CodeBrain's native [channel, patch, 200] grid.
        x = tokens.permute(0, 3, 1, 2).reshape(batch, 200, channels * patches)
        x = self.backbone.init_conv(x)
        x = self.backbone.residual_layer(x)
        x = self.backbone.final_conv(x)
        features = x.reshape(batch, 200, channels, patches).permute(0, 2, 3, 1)
        features = self.backbone.norm(features)
        if post_prompt is not None:
            features = features + post_prompt
        return features


class CSBrainBoundaryEncoder(nn.Module):
    """Official CSBrain with hooks after patch embedding and final features."""

    def __init__(self, portable_root: Path, checkpoint: Path, dropout: float, n_layer: int = 12):
        super().__init__()
        csbrain_root = portable_root / "CSBrain"
        sys.path.insert(0, str(csbrain_root))
        from models.CSBrain import CSBrain
        from models.model_for_shin import SHIN_BRAIN_REGIONS, shin_sorted_indices

        self.sorted_indices = shin_sorted_indices()
        self.backbone = CSBrain(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=n_layer,
            nhead=8,
            brain_regions=SHIN_BRAIN_REGIONS,
            sorted_indices=self.sorted_indices,
        )
        self.pretrained_report = load_compatible(self.backbone, checkpoint)
        self.backbone.proj_out = nn.Identity()

    def encode(self, eeg: torch.Tensor, pre_prompt=None, post_prompt=None) -> torch.Tensor:
        x = eeg[:, self.sorted_indices, :, :]
        tokens = self.backbone.patch_embedding(x)
        if pre_prompt is not None:
            tokens = tokens + pre_prompt

        for layer_idx in range(self.backbone.encoder.num_layers):
            tokens = self.backbone.TemEmbedEEGLayer(tokens) + tokens
            tokens = self.backbone.BrainEmbedEEGLayer(tokens, self.backbone.area_config) + tokens
            tokens = self.backbone.encoder.layers[layer_idx](tokens, self.backbone.area_config)
        features = self.backbone.proj_out(tokens)
        if post_prompt is not None:
            features = features + post_prompt
        return features


class FoundationBoundaryPrompt(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        fnirs_shape: tuple[int, ...],
        checkpoint: Path,
        graph_montage: dict | None = None,
    ):
        super().__init__()
        if args.backbone == "codebrain":
            self.encoder = CodeBrainBoundaryEncoder(args.portable_root, checkpoint, args.dropout)
            sys.path.insert(0, str(args.portable_root / "CodeBrain" / "scripts"))
            from shin_linear_head import OfficialClassificationHead

            self.classifier = OfficialClassificationHead(30, 10, 200, 2, args.dropout)
        else:
            self.encoder = CSBrainBoundaryEncoder(args.portable_root, checkpoint, args.dropout)
            self.classifier = nn.Sequential(
                nn.Linear(30 * 10 * 200, 10 * 200),
                nn.ELU(),
                nn.Dropout(args.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(),
                nn.Dropout(args.dropout),
                nn.Linear(200, 2),
            )

        self.mode = args.mode
        self.prompt_source = args.prompt_source
        self.fnirs_encoder = None
        self.graph_encoder = None
        self.attribute_encoder = None
        self.pre_prompt = None
        self.post_prompt = None
        if args.mode != "eeg_only":
            condition_dim = args.prompt_hidden if args.fnirs_conditioner == "temporal" else fnirs_shape[-1]
            if args.fnirs_conditioner == "temporal":
                self.fnirs_encoder = FnirsTemporalEncoder(fnirs_shape[-1], condition_dim, args.dropout)
            token_count = 30 * 10
            prompt_kwargs = dict(
                condition_dim=condition_dim,
                d_model=200,
                prompt_count=args.prompt_count,
                rank=args.prompt_rank,
                token_count=token_count,
                hidden_dim=args.prompt_hidden,
                dropout=args.dropout,
                expert_count=args.expert_count,
                temperature=args.router_temperature,
                router_noise_std=args.router_noise_std,
                importance_threshold=args.importance_threshold,
                drop_component=args.mope_drop_component,
                mapped_mode=args.mapped_mode,
                dynamic_expert_mode=args.dynamic_expert_mode,
            )
            if args.mode in {"pre", "pre_post"}:
                self.pre_prompt = MoPEBoundaryPrompt(**prompt_kwargs)
            if args.mode in {"post", "pre_post"}:
                self.post_prompt = MoPEBoundaryPrompt(**prompt_kwargs)
            if args.mapped_mode == "sgformer" and args.mope_drop_component != "mapped":
                if graph_montage is None:
                    raise ValueError("SGFormer mapped mode requires the SHIN fNIRS montage")
                self.graph_encoder = SGFormerMappedEncoder(
                    positions_3d=torch.as_tensor(graph_montage["positions_3d"]),
                    edge_index=torch.as_tensor(graph_montage["edge_index"]),
                    prompt_dimension=200,
                    graph_dimension=args.sgformer_graph_dimension,
                    dropout=args.dropout,
                    attention_residual_weight=args.sgformer_attention_residual_weight,
                    graph_weight=args.sgformer_graph_weight,
                )
            if args.dynamic_expert_mode == "tap4x4" and args.mope_drop_component != "dynamic":
                if graph_montage is None:
                    raise ValueError("TAP 4x4 dynamic prompts require the SHIN fNIRS montage")
                self.attribute_encoder = TAPFourAttributeFnirsEncoder(
                    positions_3d=torch.as_tensor(graph_montage["positions_3d"]),
                    edge_index=torch.as_tensor(graph_montage["edge_index"]),
                    output_dim=condition_dim,
                    dropout=args.dropout,
                )

    @staticmethod
    def static_reference(fnirs: torch.Tensor) -> torch.Tensor:
        if fnirs.ndim == 2:
            return torch.linspace(-1.0, 1.0, fnirs.shape[1], device=fnirs.device, dtype=fnirs.dtype).expand_as(fnirs)
        time_axis = torch.linspace(-1.0, 1.0, fnirs.shape[1], device=fnirs.device, dtype=fnirs.dtype).view(1, -1, 1)
        channel_axis = torch.linspace(-1.0, 1.0, fnirs.shape[2], device=fnirs.device, dtype=fnirs.dtype).view(1, 1, -1)
        return (time_axis + channel_axis).expand_as(fnirs)

    def condition(self, fnirs: torch.Tensor) -> torch.Tensor | None:
        source = fnirs if self.prompt_source == "conditional" else self.static_reference(fnirs)
        return self.fnirs_encoder(source) if self.fnirs_encoder is not None else source

    @staticmethod
    def static_graph_reference(fnirs_graph: torch.Tensor) -> torch.Tensor:
        node_axis = torch.linspace(
            -1.0, 1.0, fnirs_graph.shape[1], device=fnirs_graph.device, dtype=fnirs_graph.dtype
        ).view(1, -1, 1, 1)
        chromophore_axis = torch.linspace(
            -0.5, 0.5, fnirs_graph.shape[2], device=fnirs_graph.device, dtype=fnirs_graph.dtype
        ).view(1, 1, -1, 1)
        time_axis = torch.linspace(
            -1.0, 1.0, fnirs_graph.shape[3], device=fnirs_graph.device, dtype=fnirs_graph.dtype
        ).view(1, 1, 1, -1)
        return (node_axis + chromophore_axis + time_axis).expand_as(fnirs_graph)

    def features(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        fnirs_graph: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.condition(fnirs)
        graph_source = None
        if self.graph_encoder is not None or self.attribute_encoder is not None:
            if fnirs_graph.ndim != 4:
                raise ValueError("Structured fNIRS prompts require graph trials [B,36,2,T]")
            graph_source = (
                fnirs_graph
                if self.prompt_source == "conditional"
                else self.static_graph_reference(fnirs_graph)
            )
        mapped_nodes = self.graph_encoder(graph_source) if self.graph_encoder is not None else None
        attribute_conditions = (
            self.attribute_encoder(graph_source) if self.attribute_encoder is not None else None
        )
        pre = self.pre_prompt(condition, mapped_nodes, attribute_conditions).view(eeg.shape[0], 30, 10, 200) if self.pre_prompt is not None else None
        post = self.post_prompt(condition, mapped_nodes, attribute_conditions).view(eeg.shape[0], 30, 10, 200) if self.post_prompt is not None else None
        return self.encoder.encode(eeg, pre_prompt=pre, post_prompt=post)

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        fnirs_graph: torch.Tensor,
    ) -> torch.Tensor:
        features = self.features(eeg, fnirs, fnirs_graph)
        if hasattr(self.classifier, "flatten"):
            features = self.classifier.flatten(features)
            return self.classifier(features)
        return self.classifier(features.flatten(1))

    def importance_loss(self) -> torch.Tensor:
        losses = [
            prompt.importance_loss()
            for prompt in (self.pre_prompt, self.post_prompt)
            if prompt is not None
        ]
        return torch.stack(losses).mean() if losses else next(self.parameters()).new_zeros(())

    def attribute_loss(self, target: torch.Tensor) -> torch.Tensor:
        losses = [
            prompt.attribute_loss(target)
            for prompt in (self.pre_prompt, self.post_prompt)
            if prompt is not None
        ]
        return torch.stack(losses).mean() if losses else next(self.parameters()).new_zeros(())

    def routing_statistics(self) -> dict[str, float]:
        result = {}
        for name, prompt in (("pre", self.pre_prompt), ("post", self.post_prompt)):
            if prompt is not None:
                result.update({f"{name}_{key}": value for key, value in prompt.routing_statistics().items()})
        return result

    def set_backbone_eval(self) -> None:
        if any(parameter.requires_grad for parameter in self.encoder.backbone.parameters()):
            self.encoder.backbone.train()
        else:
            self.encoder.backbone.eval()


def load_classifier_checkpoint(model: FoundationBoundaryPrompt, checkpoint: Path, task: str, seed: int) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    classifier_state = {
        key.removeprefix("classifier."): value for key, value in state.items() if key.startswith("classifier.")
    }
    if not classifier_state:
        raise ValueError(f"{checkpoint} has no classifier state")
    saved_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    saved_task = saved_args.get("task")
    saved_seed = saved_args.get("seed")
    if saved_task is not None and saved_task != task:
        raise ValueError(f"Head checkpoint task is {saved_task}, but this run requests {task}.")
    if saved_seed is not None and int(saved_seed) != int(seed):
        raise ValueError(f"Head checkpoint seed is {saved_seed}, but this prompt run uses seed {seed}.")
    result = model.classifier.load_state_dict(classifier_state, strict=True)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "saved_task": saved_task,
        "saved_seed": saved_seed,
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def set_trainable(model: FoundationBoundaryPrompt, train_classifier: bool, train_backbone: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (
        model.pre_prompt, model.post_prompt, model.fnirs_encoder,
        model.graph_encoder, model.attribute_encoder,
    ):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True
    if train_classifier:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    if train_backbone:
        for parameter in model.encoder.backbone.parameters():
            parameter.requires_grad = True


def make_optimizer(model: FoundationBoundaryPrompt, args: argparse.Namespace, train_classifier: bool, train_backbone: bool):
    groups = []
    for name, module, lr in (
        ("pre_prompt", model.pre_prompt, args.feature_lr),
        ("post_prompt", model.post_prompt, args.feature_lr),
        ("fnirs_encoder", model.fnirs_encoder, args.feature_lr),
        ("sgformer_graph_encoder", model.graph_encoder, args.feature_lr),
        ("tap_attribute_encoder", model.attribute_encoder, args.feature_lr),
    ):
        if module is not None:
            groups.append({"params": list(module.parameters()), "lr": lr, "name": name})
    if train_classifier:
        groups.append({"params": list(model.classifier.parameters()), "lr": args.head_lr, "name": "head"})
    if train_backbone:
        groups.append({"params": list(model.encoder.backbone.parameters()), "lr": args.backbone_lr, "name": "backbone"})
    if not groups:
        raise ValueError("No trainable parameters selected")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, seen, predictions, labels = 0.0, 0, [], []
    for eeg, fnirs, fnirs_graph, target in loader:
        eeg = eeg.to(device).float()
        fnirs = fnirs.to(device).float()
        fnirs_graph = fnirs_graph.to(device).float()
        target = target.to(device)
        logits = model(eeg, fnirs, fnirs_graph)
        total += float(criterion(logits, target).item()) * len(target)
        seen += len(target)
        predictions.append(logits.argmax(1).cpu().numpy())
        labels.append(target.cpu().numpy())
    return metrics(np.concatenate(labels), np.concatenate(predictions), total / seen)


def train_epoch(model, loader, optimizer, device, importance_weight: float, attribute_weight: float):
    model.train()
    model.set_backbone_eval()
    criterion = nn.CrossEntropyLoss()
    totals = {"loss": 0.0, "classification": 0.0, "importance": 0.0, "attribute": 0.0}
    seen = 0
    for eeg, fnirs, fnirs_graph, target in loader:
        eeg = eeg.to(device).float()
        fnirs = fnirs.to(device).float()
        fnirs_graph = fnirs_graph.to(device).float()
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        classification = criterion(model(eeg, fnirs, fnirs_graph), target)
        importance = model.importance_loss()
        attribute = model.attribute_loss(target)
        loss = classification + importance_weight * importance + attribute_weight * attribute
        loss.backward()
        optimizer.step()
        count = len(target)
        totals["loss"] += float(loss.item()) * count
        totals["classification"] += float(classification.item()) * count
        totals["importance"] += float(importance.item()) * count
        totals["attribute"] += float(attribute.item()) * count
        seen += count
    return {key: value / seen for key, value in totals.items()}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.portable_root = args.portable_root.resolve()
    args.prep_root = args.prep_root.resolve()
    if args.tap_attribute_weight < 0:
        raise ValueError("--tap-attribute-weight must be non-negative")

    if args.shin_root is None:
        args.shin_root = args.prep_root / "datasets" / "shin2017_eeg_nirs_left_right_hand_mi"
    if args.cache_path is None:
        args.cache_path = SCRIPT_ROOT / "cache" / f"shin2017_{args.task}_bids_nirs_paired_sub01-sub29_10patch.npz"
    if args.sgformer_cache_path is None:
        args.sgformer_cache_path = args.cache_path.with_name(
            f"{args.cache_path.stem}_sgformer_hbo-hbr_graph.npz"
        )
    if args.backbone_checkpoint is None:
        args.backbone_checkpoint = DEFAULT_CODEBRAIN_CHECKPOINT if args.backbone == "codebrain" else DEFAULT_CSBRAIN_CHECKPOINT
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)

    loader_args = argparse.Namespace(
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
        task=args.task,
        eeg_scale=args.eeg_scale,
        subjects=args.subjects,
        eeg_bids_root=args.eeg_bids_root,
        shin_root=args.shin_root,
        fnirs_window=args.fnirs_window,
        fnirs_offset=args.fnirs_offset,
    )
    eeg, fnirs_stats, fnirs_sequence, labels, meta = load_paired_bids_trial_cache(loader_args)
    if tuple(eeg.shape[1:]) != (30, 10, 200):
        raise ValueError(f"Expected [B,30,10,200], got {tuple(eeg.shape)}")
    subject_ids = np.asarray(meta["subject_ids"], dtype=np.int64)
    train_idx = np.flatnonzero(np.isin(subject_ids, args.train_subjects))
    val_idx = np.flatnonzero(np.isin(subject_ids, args.val_subjects))
    test_idx = np.flatnonzero(np.isin(subject_ids, args.test_subjects))
    if not all(len(index) for index in (train_idx, val_idx, test_idx)):
        raise ValueError("train/val/test subject split is empty")
    fnirs = fnirs_sequence if args.fnirs_conditioner == "temporal" else fnirs_stats
    fnirs = normalize_fnirs_from_train(fnirs, train_idx)
    needs_mapped_graph = (
        args.mapped_mode == "sgformer" and args.mope_drop_component != "mapped"
    )
    needs_attribute_graph = (
        args.dynamic_expert_mode == "tap4x4" and args.mope_drop_component != "dynamic"
    )
    needs_graph = args.mode != "eeg_only" and (needs_mapped_graph or needs_attribute_graph)
    fnirs_graph = None
    graph_meta = None
    graph_montage = None
    if needs_graph:
        fnirs_graph, graph_labels, graph_subjects, graph_meta = load_sgformer_graph_trials(
            shin_root=args.shin_root,
            subjects=args.subjects,
            task_sessions=SHIN_TASKS[args.task]["sessions"],
            fnirs_window=args.fnirs_window,
            fnirs_offset=args.fnirs_offset,
            cache_path=args.sgformer_cache_path,
            rebuild_cache=args.rebuild_cache,
        )
        expected_subjects = np.asarray(meta["subject_ids"], dtype=np.int64)
        if not np.array_equal(graph_labels, labels):
            raise ValueError("SGFormer graph labels do not match the existing paired prompt cache")
        if not np.array_equal(graph_subjects.astype(np.int64), expected_subjects):
            raise ValueError("SGFormer graph subject order does not match the existing paired prompt cache")
        fnirs_graph = normalize_graph_from_train(fnirs_graph, train_idx)
        graph_montage = load_fnirs_montage(args.shin_root)
    if args.shuffle_fnirs:
        if args.mode == "eeg_only":
            raise ValueError("--shuffle-fnirs requires a prompt mode")
        fnirs, fnirs_graph = shuffle_prompt_sources_within_splits(
            fnirs,
            fnirs_graph,
            [train_idx, val_idx, test_idx],
            args.seed + 1009,
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    model = FoundationBoundaryPrompt(
        args,
        tuple(fnirs.shape[1:]),
        args.backbone_checkpoint,
        graph_montage=graph_montage,
    )
    diagnostics = {
        "backbone": args.backbone,
        "task": args.task,
        "mode": args.mode,
        "mapped_mode": args.mapped_mode,
        "dynamic_expert_mode": args.dynamic_expert_mode,
        "tap_attribute_weight": args.tap_attribute_weight,
        "prompt_source": args.prompt_source,
        "training_strategy": args.training_strategy,
        "prompt_location": "after native patch embedding and after backbone output",
        "raw_eeg_prompt": False,
        "fNIRS_conditioner": args.fnirs_conditioner,
        "fNIRS_pairing": "shuffled within split" if args.shuffle_fnirs else "trial aligned",
        "eeg_shape": list(eeg.shape),
        "fnirs_shape": list(fnirs.shape),
        "fnirs_graph_shape": list(fnirs_graph.shape) if fnirs_graph is not None else None,
        "structured_fnirs_graph": ({
            "cache": str(args.sgformer_cache_path.resolve()),
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
            "cache": str(args.sgformer_cache_path.resolve()),
            "graph_dimension": args.sgformer_graph_dimension,
            "attention_residual_weight": args.sgformer_attention_residual_weight,
            "graph_weight": args.sgformer_graph_weight,
        } if needs_graph and needs_mapped_graph else None),
        "subjects": {"train": args.train_subjects, "val": args.val_subjects, "test": args.test_subjects},
        "trials": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "pretrained_load": model.encoder.pretrained_report,
        "parameters": {"total": sum(parameter.numel() for parameter in model.parameters())},
    }
    if args.diagnose_only:
        write_json(args.output_dir / "diagnostics.json", diagnostics)
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
        return

    if args.training_strategy == "prompt_only":
        if args.mode == "eeg_only" or args.head_checkpoint is None:
            raise ValueError("prompt_only requires a matching EEG-only --head-checkpoint")
        diagnostics["classifier_load"] = load_classifier_checkpoint(model, args.head_checkpoint, args.task, args.seed)

    counts = {
        "classifier": sum(parameter.numel() for parameter in model.classifier.parameters()),
        "prompt": sum(parameter.numel() for module in (model.pre_prompt, model.post_prompt) if module is not None for parameter in module.parameters()),
        "fnirs_encoder": sum(parameter.numel() for parameter in model.fnirs_encoder.parameters()) if model.fnirs_encoder is not None else 0,
        "sgformer_graph_encoder": sum(parameter.numel() for parameter in model.graph_encoder.parameters()) if model.graph_encoder is not None else 0,
        "tap_attribute_encoder": sum(parameter.numel() for parameter in model.attribute_encoder.parameters()) if model.attribute_encoder is not None else 0,
        "backbone": sum(parameter.numel() for parameter in model.encoder.backbone.parameters()),
    }
    diagnostics["parameters"] = counts
    model.to(device)
    train_classifier = args.training_strategy == "joint"
    train_backbone = train_classifier and args.unfreeze_epoch <= 1
    set_trainable(model, train_classifier, train_backbone)
    diagnostics["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    write_json(args.output_dir / "diagnostics.json", diagnostics)

    train_loader = DataLoader(FoundationTrialDataset(eeg, fnirs, fnirs_graph, labels, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(FoundationTrialDataset(eeg, fnirs, fnirs_graph, labels, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(FoundationTrialDataset(eeg, fnirs, fnirs_graph, labels, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    optimizer = make_optimizer(model, args, train_classifier, train_backbone)
    history, best_record, best_acc = [], None, -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_classifier and epoch == args.unfreeze_epoch and not train_backbone:
            train_backbone = True
            set_trainable(model, True, True)
            optimizer = make_optimizer(model, args, True, True)
        train_stats = train_epoch(
            model, train_loader, optimizer, device,
            args.importance_weight, args.tap_attribute_weight,
        )
        val = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train": train_stats, "val": val}
        history.append(record)
        print(f"epoch {epoch:03d}/{args.epochs} train={train_stats['loss']:.4f} val_acc={val['acc']:.4f}", flush=True)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_record = record
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, args.output_dir / "best_model.pth")
        write_json(args.output_dir / "history.json", history)

    final_test = evaluate(model, test_loader, device)
    best_payload = torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    best_test = evaluate(model, test_loader, device)
    summary = {
        "backbone": args.backbone,
        "task": args.task,
        "mapped_mode": args.mapped_mode,
        "dynamic_expert_mode": args.dynamic_expert_mode,
        "seed": args.seed,
        "best": best_record,
        "best_test": best_test,
        "final_test": final_test,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "diagnostics.json", diagnostics)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
