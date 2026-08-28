# LaBraM 运行 SHIN EEG

入口为 `run_shin_gpu.ps1`，使用 conda 环境 `LaBraM` 和本目录的
`checkpoints/labram-base.pth`。默认正式实验参数如下：

| 参数 | 默认值 |
|---|---:|
| epoch | 100 |
| batch size | 8 |
| 分类头学习率 | 1e-4 |
| 骨干学习率 | 1e-5 |
| 骨干解冻 | 第 91 epoch |
| seed | 1 |
| 输出根目录 | `D:\data\LaBraM-SHIN` |

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_shin_gpu.ps1
```

每次自动创建带时间戳的新目录，其中包含 `best_model.pth`、
`last_model.pth`、`history.json`、`summary.json`、`diagnostics.json` 和中文
`EXPERIMENT_RECORD.md`。脚本拒绝写入非空输出目录，以免覆盖既有实验。

## 适配摘要

- 只使用 30 个 EEG 通道，排除 2 个 EOG 通道。
- 使用完整 10 秒 trial（2000 点），重排为 LaBraM 所需的 10 个 200 点 patch。
- 18 个 SHIN 10-5 半步电极名称做唯一的近邻 10-20 位置映射；完整映射写入每次实验的 `diagnostics.json`。
- 使用物理微伏除以 100，遵循 LaBraM 原有下游适配代码。
- 被试独立划分与 CodeBrain 相同：sub-1~23 / sub-24~26 / sub-27~29。
