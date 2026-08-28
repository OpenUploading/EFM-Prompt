# HYGRIP 在 portable 中的目录分配

本目录保存 HYGRIP 数据集的公共处理代码；各模型的训练入口和启动命令已放入 portable 中对应的模型目录。原压缩包未删除，也未包含原始数据、处理缓存或训练结果。

## 公共文件

- `preprocessing/prepare_hygrip_trials.m`：提取 EEG、HbO、HbR trial。
- `preprocessing/prepare_hygrip_trials_v2.m`：生成正式 EEG v2 数据。
- `preprocessing/diagnose_hygrip_eeg_signal.py`：信号质量诊断。
- `preprocessing/RUN_HYGRIP_PREPARE*.cmd`：两阶段预处理命令。
- `hefmi_within_subject_folds.py`：CBraMod 五折协议使用的公共划分工具。
- `README_SOURCE_PACKAGE.md`：收到的原始复用包说明。

## 模型入口位置

| 模型 | Python 入口 | 启动命令 |
|---|---|---|
| EEGNet | `../EEGNet/run_hygrip_eegnet.py` | `../EEGNet/RUN_HYGRIP_EEGNET*.cmd` |
| DAMFNet | `../DAMFNet/run_hygrip_damfnet.py` | `../DAMFNet/RUN_HYGRIP_DAMFNET*.cmd` |
| fNIRS-Transformer | `../fNIRS-Transformer/run_hygrip_fnirst.py` | `../fNIRS-Transformer/RUN_HYGRIP_FNIRST.cmd` |
| CBraMod | `../CBraMod/run_hygrip_cbramod.py` | `../CBraMod/RUN_HYGRIP_CBRAMOD*.cmd` |
| CodeBrain | `../CodeBrain/run_hygrip_codebrain.py` | `../CodeBrain/RUN_HYGRIP_CODEBRAIN*.cmd` |
| CSBrain | `../CSBrain/run_hygrip_csbrain.py` | `../CSBrain/RUN_HYGRIP_CSBRAIN*.cmd` |

适配入口已经改为从 portable 的同级模型目录读取模型源码。CBraMod、CodeBrain 和 CSBrain 默认读取各自目录已有的 `pretrained_weights`，不再依赖原压缩包中写死的外部源码路径。

## 数据和环境

正式 EEG 实验应使用 `prepared_eeg_v2`。默认数据路径和 Python/MATLAB 路径仍沿用原始复用包中的本机配置；换机器或环境时，需要修改相应 `.cmd`。完整预处理协议、数据形状和超参数见 `README_SOURCE_PACKAGE.md`。
