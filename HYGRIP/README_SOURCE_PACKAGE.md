# HYGRIP 数据处理与模型适配复用包

本包整理了 HYGRIP 左/右手动态握力二分类实验的数据处理脚本、模型适配脚本及 Windows 启动文件。默认协议为跨受试划分：A–J 训练、K–L 验证、M–N 测试；只根据验证集准确率选择 `best.pt`，测试集不参与选模。

## 1. 包含内容

```text
work/hygrip_baselines/
  prepare_hygrip_trials.m              第一步：提取对齐的 EEG、HbO、HbR trial
  prepare_hygrip_trials_v2.m           第二步：修正并重新处理 EEG，保留第一步的 fNIRS
  diagnose_hygrip_eeg_signal.py        EEG 信号质量诊断
  run_hygrip_baseline.py               EEGNet、fNIRS-T、DAMFNet 适配入口
  run_hygrip_cbramod.py                CBraMod 分类头/全微调入口
  run_hygrip_codebrain_csbrain.py      CodeBrain、CSBrain 分类头/全微调入口
  RUN_HYGRIP_*.cmd                     已使用过的启动参数
work/hefmi_baselines/                   运行上述适配入口所需的本地模型辅助代码
licenses/                               随包代码中可获得的上游许可证
```

不包含原始数据、处理后的 trial、缓存、训练结果或预训练权重。

## 2. 环境

- Windows 10/11、NVIDIA CUDA GPU。
- MATLAB，需 Signal Processing Toolbox。
- Python 使用既有 `csbrain-bcic2a` 环境：`D:\miniconda\envs\csbrain-bcic2a\python.exe`。
- Python 主要依赖见根目录 `requirements.txt`。

如果机器路径不同，先修改各 `.cmd` 中的 `PYTHON`、MATLAB、数据和输出路径。启动文件使用自身目录作为工作目录，解压后可以整体移动。

## 3. 数据要求

原始文件默认为：

```text
D:\DataSets\HYGRIP\hygrip.h5
```

数据集包含 A–N 共 14 名受试者。任务标签为左手动态握力 `0`、右手动态握力 `1`。

## 4. 推荐数据处理顺序

### 第一步：提取 EEG 与 fNIRS

运行：

```text
RUN_HYGRIP_PREPARE.cmd
```

输出默认为 `D:\data\HYGRIP-Baselines\prepared`。其中：

- trial 为任务开始后 0–20 秒；
- fNIRS 直接读取作者提供的 `oxy`/`dxy`，分别作为 HbO/HbR；原始单位为 mol；
- 连续 HbO/HbR 先做 0.01–0.1 Hz 三阶 Butterworth 零相位滤波；
- 每个 trial 使用 onset 前 1 秒做基线校正，再乘 `1e6` 输出为 μmol/L；
- 不重复执行光密度转换或 MBLL，也未做 fNIRS 运动伪迹校正。

### 第二步：生成 EEG v2（正式 EEG 版本）

运行：

```text
RUN_HYGRIP_PREPARE_EEG_V2.cmd
```

输出默认为 `D:\data\HYGRIP-Baselines\prepared_eeg_v2`。该步骤从原始 HDF5 重新处理 EEG，同时原样复制第一步已经核验的 HbO/HbR 与标签：

- 按连续信号从 1000 Hz 抗混叠重采样到 200 Hz；
- 根据作者绘图代码和数据量级，将存储值按 V 解释并转换为 μV；
- 每通道 12-MAD 极端尖峰裁剪、CAR；
- 12.5/25/37.5 Hz 陷波（Q=50）；
- 1–45 Hz 四阶 Butterworth 零相位滤波；
- 最后截取 onset 后 0–20 秒。

注意：第一步脚本中的旧 EEG 单位解释已被 v2 修正。正式 EEGNet、DAMFNet、CBraMod、CodeBrain、CSBrain 实验应使用 `prepared_eeg_v2`；第一步目录主要作为已核验 fNIRS 的来源。

每名受试者输出：

```text
subject_A_trials.mat ... subject_N_trials.mat
  eeg_uv:    [trial, 24, 4000]
  fnirs_um:  [trial, 2, 24, 250]，第 2 维依次为 HbO、HbR
  labels:    [trial]
  meta:      预处理说明
```

## 5. 模型和正式入口

| 模型 | 推荐启动文件 | 输入 | 训练设置 |
|---|---|---|---|
| EEGNet | `RUN_HYGRIP_EEGNET_V2.cmd` | EEG 24×4000 | 全模型，100 epoch，batch 8，lr 1e-3 |
| fNIRS-T | `RUN_HYGRIP_FNIRST.cmd` | HbO+HbR 2×24×250 | 50 epoch，batch 32，head/backbone lr 1e-4/1e-3 |
| DAMFNet | `RUN_HYGRIP_DAMFNET_V2.cmd` | EEG+HbR | 40 epoch，batch 40，lr 1e-4 |
| CBraMod 头部 | `RUN_HYGRIP_CBRAMOD_V2_SUBJECT_HOLDOUT_HEADONLY50.cmd` | EEG 前 10 秒 | 冻结骨干，50 epoch，head lr 1e-4 |
| CBraMod 全微调 | `RUN_HYGRIP_CBRAMOD_V2_SUBJECT_HOLDOUT_FULLFT30.cmd` | EEG 前 10 秒 | 30 epoch，head/backbone lr 1e-4/1e-5 |
| CodeBrain 头部 | `RUN_HYGRIP_CODEBRAIN_V2_SUBJECT_HOLDOUT_HEADONLY50.cmd` | EEG 前 10 秒 | 冻结骨干，50 epoch |
| CodeBrain 全微调 | `RUN_HYGRIP_CODEBRAIN_V2_SUBJECT_HOLDOUT_FULLFT30.cmd` | EEG 前 10 秒 | 30 epoch，head/backbone lr 1e-4/1e-5 |
| CSBrain 头部 | `RUN_HYGRIP_CSBRAIN_V2_SUBJECT_HOLDOUT_HEADONLY50.cmd` | EEG 前 10 秒 | 冻结骨干，50 epoch |
| CSBrain 全微调 | `RUN_HYGRIP_CSBRAIN_V2_SUBJECT_HOLDOUT_FULLFT30.cmd` | EEG 前 10 秒 | 30 epoch，head/backbone lr 1e-4/1e-5 |

推荐的 v2、分类头和全微调 `.cmd` 会在结束后保留窗口，便于查看退出码；少数保留作对照的旧版入口可能直接返回。输出目录必须为空或不存在，防止覆盖已有实验。

## 6. 预训练模型要求

EEGNet、fNIRS-T 和 DAMFNet 所需模型实现已经随包提供，不需要预训练权重。三个基础模型入口保持原实验协议。

CBraMod 需要官方仓库和权重。默认位置为：

```text
D:\CBraMod-main\CBraMod-main
D:\CBraMod-main\CBraMod-main\pretrained_weights\pretrained_weights.pth
```

也可在 Python 命令中传入：

```text
--cbramod-root <官方仓库根目录> --checkpoint <预训练权重路径>
```

CodeBrain 的必要官方模型源码已随包放在预期相对目录，但权重没有打包。通过 `--checkpoint` 指定 `CodeBrain.pth`；如改用别处源码，可同时传 `--codebrain-root`。

CSBrain 的必要模型源码已随包放在预期相对目录，但权重没有打包。通过 `--checkpoint` 指定 `CSBrain.pth`；如改用别处源码，可同时传 `--csbrain-root`。

## 7. 快速检查

训练前建议先执行以下检查：

```powershell
cd work\hygrip_baselines
D:\miniconda\envs\csbrain-bcic2a\python.exe -m py_compile run_hygrip_baseline.py run_hygrip_cbramod.py run_hygrip_codebrain_csbrain.py diagnose_hygrip_eeg_signal.py
D:\miniconda\envs\csbrain-bcic2a\python.exe diagnose_hygrip_eeg_signal.py --prepared-root D:\data\HYGRIP-Baselines\prepared_eeg_v2
```

CBraMod 可先运行 `RUN_HYGRIP_CBRAMOD_DIAGNOSE.cmd`。CodeBrain/CSBrain 可在对应训练命令末尾临时加入 `--diagnose-only`，确认权重加载和前向传播后再正式训练。

## 8. 结果文件

各训练入口会保存 `args.json`、`diagnostics.json`、训练历史、`summary.json` 和最佳权重。选择最佳 epoch 时只读取验证集指标；最终测试指标在最佳验证权重确定后计算。

本包版本：2026-08-26。
