# EFM Prompt Experiments

This folder keeps prompt-learning experiments outside individual foundation-model
source trees. Each script should expose a small adapter around one backbone while
sharing the same SHIN split, metrics, and reporting style.

New prompt-run outputs are written to `EFM-Prompt/runs/`; paired EEG-fNIRS
caches are written to `EFM-Prompt/cache/`. External paths are used only as raw
data sources.

## CBraMod

The active path is boundary conditional-prompt ablation:

`run_shin2017_cbramod_fnirs_feature_stage1.py` freezes the CBraMod encoder and
uses fNIRS-derived temporal or statistical features to generate static,
dynamic, and mapping prompts at the patch-embedding output and/or final encoder
output. The Transformer internals are untouched. Its default prompt family is
`legacy`, preserving the completed experiments.

The separate MoPE runner is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_cbramod_mope_boundary_gpu.ps1
```

It implements static, dense expert-routed dynamic, and mapped prompts together
with the thresholded importance loss. MoPE outputs are isolated under
`EFM-Prompt/runs_mope/`. See `CONDITIONAL_MOPE_METHOD.md` for the paper method
and the exact CBraMod boundary adaptation.

The original 16-expert dynamic prompt remains the default (`-DynamicExpertMode
flat`).  `-DynamicExpertMode tap4x4` optionally replaces only that component
with four fNIRS-specific paths (HbO, HbR, spatial graph, and temporal), each
routing four experts. Static and mapped prompts and the downstream low-rank
boundary projection are unchanged. TAP runs use an auxiliary per-attribute
classification loss controlled by `-TAPAttributeWeight` (default `0.1`).

Stage 2 candidate:

`run_shin_cbramod_prompt.py` inserts a learnable prompt after CBraMod patch
embedding and before the criss-cross Transformer encoder.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_cbramod_prompt_gpu.ps1
```

Run the five matched ablations sequentially:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_cbramod_boundary_ablation_gpu.ps1
```

The runner executes `eeg_only`, `pre`, `post`, `pre_post`, and `pre_post` with
fNIRS shuffled within each split. All five conditions use the 10-patch
`all_patch_reps` classifier and frozen CBraMod encoder.

## CodeBrain/CSBrain optional SGFormer mapped token

Foundation boundary runs retain the historical MLP mapped prompt by default.
Select the enhanced implementation with `-MappedMode sgformer`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_foundation_boundary_prompt_gpu.ps1 `
  -Backbone codebrain -Task mi -Mode pre_post -TrainingStrategy prompt_only `
  -MappedMode sgformer
```

The enhanced mapper preserves independent HbO/HbR temporal encoders, the
36-node geometry and identity embeddings, one-layer SGFormer global attention,
and the parallel local GCN from the standalone SGFormer package. An independent
pre/post attention readout reduces the shared 36 graph nodes to the existing
single `[B,1,200]` mapped slot. The downstream MoPE concatenation, `to_rank`,
`token_basis`, and zero-initialized residual scale are unchanged.

All existing modes and controls remain supported: `eeg_only`, `pre`, `post`,
`pre_post`, aligned/shuffled fNIRS, `joint`/`prompt_only`, and component drops.
Use `-MappedMode mlp` to explicitly select the original implementation.
