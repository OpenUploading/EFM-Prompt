# EEGNet 官方架构 × SHIN 适配

## 来源

- 上游仓库：`https://github.com/vlawhern/arl-eegmodels`
- 固定提交：`4a512e503198db2010848813ead9afbf8cd54c97`
- 原始实现：`EEGModels.py` 中的新版 EEGNet-8,2
- 上游运行时：TensorFlow/Keras 2.0–2.3
- 本地运行时：`csbrain-bcic2a` 中的 PyTorch + CUDA

上游代码完整保留。`eegnet_pytorch.py` 是逐层对应的 PyTorch 移植，保留时间卷积、跨通道深度卷积、可分离卷积、两级平均池化、Dropout以及深度卷积/分类层的max-norm约束。PyTorch版本输出logits，交叉熵内部完成softmax。

## SHIN 数据适配

| 项目 | 设置 |
|---|---|
| EEG通道 | 30，严格检查原始顺序 |
| 采样率 | 200 Hz |
| Trial | 完整10秒，2000点 |
| 输入 | `[batch,30,2000]` |
| 首层时间核 | 100点，约0.5秒；由官方128 Hz默认64点按采样率缩放 |
| MI标签 | left_hand=0，right_hand=1 |
| MA标签 | subtraction=0，rest=1 |
| 标准化 | 物理微伏后逐trial全局z-score |
| 受试者划分 | train 1–19 / val 20–24 / test 25–29 |
| Seed | 1 |
| 预训练 | 无；EEGNet从随机初始化训练 |

## 训练默认值

EEGNet不含预训练骨干和独立分类头，因此不使用“冻结/解冻”和双学习率。默认使用作者示例常见的Adam设置：

```text
epochs=100
batch_size=8
lr=1e-3
weight_decay=0
dropout=0.5
kernel_length=100
seed=1
```

正式训练脚本为 `RUN_SHIN_MI.cmd` 和 `RUN_SHIN_MA.cmd`。训练结果写入 `D:\data\EEGNet-SHIN`，完成并复核后再追加到 `D:\data\SHIN_EXPERIMENT_LOG.md` 的19/5/5新表。
