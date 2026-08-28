# DAMFNet × SHIN 适配

## 来源与模型身份

- 用户指定仓库：`https://github.com/useflf/DAMFNet`
- 固定提交：`86ce33d4925d5e5603ccb2d7f4833d430f37ae2e`
- 模型：完整EEG–fNIRS双分支、空间/时间双节点融合、残差CTAM
- 预训练权重：仓库未提供，模型从随机初始化训练

## 为什么不能只接入EEG

DAMFNet是多模态融合模型，不是单独的EEG骨干。SHIN同时具有严格对应的EEG BIDS数据和原始fNIRS数据，因此本适配同时读取两种模态，并逐session核对20个事件的类别序列。只有事件序列完全一致才允许生成配对trial。

## SHIN输入

| 项目 | 设置 |
|---|---|
| 任务 | MI：左手/右手；MA：减法/静息 |
| 划分 | train 1–19 / val 20–24 / test 25–29 |
| Seed | 1 |
| EEG trial | 30通道，200 Hz，0–10秒 |
| fNIRS trial | HbR，36节点，10 Hz，0–10秒 |
| fNIRS处理 | 光密度、MBLL、0.01–0.1 Hz、-5至-2秒基线 |
| 滑窗 | 每trial产生8个3秒窗，步长1秒 |
| EEG窗口 | `[600,30]` |
| HbR窗口 | `[30,36]` |
| 模型选择 | 8个window logits求均值后的trial级验证准确率 |

官方公开数据配置要求EEG 8节点、HbR 24节点。新增两个可学习节点投影：

```text
SHIN EEG 30节点 -> Linear(30,8) -> 官方EEG分支
SHIN HbR 36节点 -> Linear(36,24) -> 官方HbR分支
```

投影后保持官方融合主干及注意力张量尺寸。仓库中`models/fusion_net.py`错误引用不存在的`model`包，已改为相对导入；同时移除了未使用的`torchvision`导入。

## 默认训练

沿用仓库默认超参数：

```text
epochs=40
batch_size=40
Adam lr=1e-4
weight_decay=0
dropout=0.4
EEG/HbR/Fusion辅助损失权重=1/1/1
seed=1
```

运行 `RUN_SHIN_MI.cmd` 或 `RUN_SHIN_MA.cmd`。结果写入`D:\data\DAMFNet-SHIN`；复核后追加到`D:\data\SHIN_EXPERIMENT_LOG.md`的19/5/5新表。

## 固定DAMF 8/24通道、包含引导期的-2～10秒滑窗版本

这一版本不再把全部30个EEG通道和36个HbR节点投影到8/24节点，而是直接保留合作版DAMFNet使用的固定传感器：

```text
EEG: FCC5h, FCC3h, CCP5h, CCP3h, FCC4h, FCC6h, CCP4h, CCP6h
HbR: SHIN源探测器节点索引12..35（共24个）
```

其余EEG和fNIRS通道全部丢弃，模型内部不包含节点投影层。EEG与HbR均提取事件相对时间-2～10秒，其中-2～0秒为引导期；以3秒窗口和1秒步长生成：

```text
[-2,1], [-1,2], [0,3], [1,4], [2,5],
[3,6], [4,7], [5,8], [6,9], [7,10]
```

每个trial共有10个窗口，trial级预测对10个融合logits求平均。最大训练40轮，早停patience为30。运行：

```text
RUN_SHIN_MI_FIXED8_24_NEG2_10_PAT30.cmd
RUN_SHIN_MA_FIXED8_24_NEG2_10_PAT30.cmd
```

结果写入`D:\data\DAMFNet-SHIN-Fixed8-24-Neg2To10-Pat30`。
