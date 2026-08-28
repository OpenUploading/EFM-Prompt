# SHIN 七模型官方分类头代码便携包

本压缩包包含 SHIN 数据适配后的七个模型源码，以及从本机实际 Conda 环境导出的逐模型 `requirements.txt`。

## 模型与环境

| 模型目录 | Conda 环境 | Python | 分类头 |
|---|---|---:|---|
| `LaBraM` | `LaBraM` | 3.11.15 | 官方 mean pooling + `Linear(200,2)` |
| `CBraMod` | `cbramod_env` | 3.9.25 | 官方 `all_patch_reps` 三层 MLP |
| `CodeBrain` | `codebrain-bcic2a` | 3.11.15 | 官方三层下游 MLP |
| `CSBrain` | `csbrain-bcic2a` | 3.10.20 | 官方三层下游 MLP |
| `fNIRS-Transformer` | `codebrain-bcic2a` | 3.11.15 | 官方 `LayerNorm + Linear` |
| `EEGNet` | `csbrain-bcic2a` | 3.10.20 | 官方紧凑线性层与 max-norm |
| `DAMFNet` | `csbrain-bcic2a` | 3.10.20 | 官方 EEG/HbR 辅助头和融合主头 |

每个模型目录内均包含：

- `requirements.txt`：从对应实际 Conda 环境的 `site-packages` 导出；
- `ENVIRONMENT.txt`：环境名、Python 版本和包数量；
- 原项目 README、许可证和 SHIN 运行脚本（源目录中存在时）。

## 安装说明

建议优先按 `ENVIRONMENT.txt` 中的 Python 版本新建 Conda 环境，再安装对应 `requirements.txt`：

```powershell
conda create -n <新环境名> python=<对应版本> -y
conda activate <新环境名>
python -m pip install -r requirements.txt
```

`torch`、`torchvision` 或 `torchaudio` 带有 `+cu128` 等 CUDA 标签时，可能需要使用与该版本对应的 PyTorch wheel 索引。若目标机器 CUDA 或驱动版本不同，应先按目标机器安装匹配的 PyTorch，再安装其余依赖。

## 未打包内容

为了保持代码包可传输，本包不包含：

- SHIN 数据集；
- 历史训练结果、缓存和 checkpoint；
- 官方预训练权重；
- `.git`、IDE 配置及 `__pycache__`；
- `.pt`、`.pth`、`.npz`、`.npy`、`.mat`、BDF/EDF 等大型二进制文件。

原机器的主要预训练权重位置：

| 模型 | 原权重位置 |
|---|---|
| LaBraM | `...\LaBraM-main\checkpoints\labram-base.pth` |
| CBraMod | `D:\CBraMod-main\CBraMod-main\pretrained_weights\pretrained_weights.pth` |
| CodeBrain | `...\bcic-iv-2a-eeg-cuda-train\external\CodeBrain\Checkpoints\CodeBrain.pth` |
| CSBrain | `...\work\CSBrain-official\pth\CSBrain.pth` |

EEGNet、fNIRS-Transformer 和当前 DAMFNet SHIN 适配不依赖上述基础模型预训练权重。

## 数据路径

各运行脚本仍保留本机实验使用的数据默认路径，例如：

- EEG：`D:\DataSets\SHIN\v1.0.1`
- fNIRS：`D:\DataSets\SHIN\NIRS_01-29`

换机器后可通过脚本参数修改数据路径和输出目录。

## 复现说明

- 当前实验默认跨受试者划分：训练 1–19，验证 20–24，测试 25–29。
- 后续实验统一随机种子：`seed=1`；DAMFNet 合作版超参数对照的历史实验曾使用 `seed=0`。
- 本包只整理代码和环境依赖，没有自动启动新训练。

## HYGRIP 适配（2026-08-28）

HYGRIP 的公共数据处理、协议说明和依赖文件位于 `HYGRIP/`；EEGNet、DAMFNet、fNIRS-Transformer、CBraMod、CodeBrain、CSBrain 的 HYGRIP 训练入口及 `.cmd` 已分别放入对应模型目录。详细映射见 `HYGRIP/README.md`。
