# 官方 CSBrain × SHIN 独立结果页

本页只记录官方 CSBrain 代码及官方基础预训练权重实验，与旧的 `CSBrain-Fallback` 表格分开。

## 固定实验协议

| 参数 | 值 |
|---|---|
| 数据集 | SHIN EEG |
| 受试者划分 | train sub-1–19 / val sub-20–24 / test sub-25–29 |
| Seed | 1 |
| Epoch | 100 |
| Batch size | 8 |
| 分类头学习率 | 1e-4 |
| 主干学习率 | 1e-5 |
| Weight decay | 1e-4 |
| 主干解冻 | 第91 epoch，最后10 epoch微调 |
| 提前解冻 | 本轮不做 |
| 输入 | 30通道 × 10 patch × 200采样点 |
| 分类头 | 官方30通道/10-patch三层MLP：60000→2000→200→2 |

## 结果

| 任务 | 最佳Epoch | 最佳Val Acc | Test Acc | Test Macro-F1 | Test Kappa | 最终Test Acc | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---|
| EEG-MI | 5 | 0.6500 | **0.7267** | 0.7239 | 0.4533 | 0.7000 | `D:\data\CSBrain-Official-SHIN\20260725-084957_mi_official_ep100_headlr1e-4_backbonelr1e-5_unfreeze91_seed1` |
| EEG-MA | 25 | **0.6933** | **0.7300** | **0.7278** | **0.4600** | 0.6767 | `D:\data\CSBrain-Official-SHIN\20260725-090918_ma_official_ep100_headlr1e-4_backbonelr1e-5_unfreeze91_seed1` |

## 曲线与解冻观察

| 任务 | 冻结阶段最佳 | 解冻阶段最佳 | 最终训练Loss | 最终验证Loss | 判断 |
|---|---|---|---:|---:|---|
| EEG-MI | Epoch 5，Val Acc 0.6500 | Epoch 94，Val Acc 0.6367 | 0.0557 | 6.4133 | 解冻未刷新最佳；分类头很早开始过拟合 |
| EEG-MA | Epoch 25，Val Acc 0.6933 | Epoch 94，Val Acc 0.6767 | 0.0270 | 5.1372 | 解冻未刷新最佳；后期过拟合明显 |

- MI 最佳测试混淆矩阵为 `[[94,56],[26,124]]`。left-hand 召回率为 0.6267，right-hand 召回率为 0.8267，模型偏向 right-hand。
- MA 最佳测试混淆矩阵为 `[[123,27],[54,96]]`。subtraction 召回率为 0.8200，rest 召回率为 0.6400，模型偏向 subtraction。
- MI 最终模型比最佳模型测试准确率下降 0.0267；MA 下降 0.0533。必须使用验证集最佳检查点，不能使用第100轮模型。
- 两个最佳检查点都产生于冻结主干阶段，说明当前官方全 patch 分类头已经能利用预训练特征，但其 1.204亿参数相对于1140个训练 trial 过大。第91轮解冻没有改善验证上限。
- 本轮按约定不做提前解冻。下一轮若优化，应优先缩小分类头或增加正则化，而不是单纯提前解冻主干。
