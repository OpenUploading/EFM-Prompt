# EFM-Prompt

Research code for EEG-fNIRS multimodal prompt learning and cross-dataset transfer.

The repository contains the portable experiment code used to compare EEG-only
baselines with fNIRS-conditioned prompt methods across SHIN, FineMI, ds004022,
and HYGRIP. It also includes the local adaptations of EEGNet, CBraMod,
CodeBrain, CSBrain, DAMFNet, LaBraM, and fNIRS-Transformer used by the project.

## Prompt methods

- Three-component boundary prompt: static, dynamic expert-routed, and mapped
  components.
- Deep conditional prompt: fNIRS-conditioned residual prompts injected at
  multiple depths.
- TMPA variants: multimode prompts with token-level and prompt-level alignment.
- Hierarchical cross-attention and bidirectional contrast variants.
- PromptMoPE and component-ablation implementations.

The primary implementations and launch scripts are in [`EFM-Prompt/`](EFM-Prompt/).

## Repository layout

| Directory | Contents |
| --- | --- |
| `EFM-Prompt/` | Prompt modules and SHIN/FineMI/HYGRIP experiment entry points |
| `EEGNet/` | EEGNet baselines and dataset-specific protocols |
| `CBraMod/` | CBraMod baselines and frozen/full-finetuning entry points |
| `DAMFNet/` | EEG-fNIRS fusion baselines |
| `CodeBrain/`, `CSBrain/`, `LaBraM/` | Foundation-model baselines |
| `fNIRS-Transformer/` | fNIRS-only baseline |
| `HYGRIP/` | HYGRIP preprocessing and protocol helpers |

## Data and checkpoints

Datasets, processed arrays, pretrained weights, checkpoints, caches, and full
training outputs are intentionally excluded from Git. Obtain each dataset and
upstream pretrained checkpoint from its official source, then pass local paths
through the corresponding command-line arguments.

Some scripts retain Windows paths used during the original experiments as
defaults. Replace them with paths on your machine or provide the matching CLI
options before running.

## Environments

Each model directory contains its own `requirements.txt` and, where available,
an `ENVIRONMENT.txt` recording the original Python environment. Create separate
environments for models whose dependency versions conflict.

The earlier Chinese packaging notes are available in
[`README_CN.md`](README_CN.md).

## Licenses

Upstream model code remains subject to the license included in its respective
subdirectory. No dataset or pretrained-model license is granted by this
repository.

