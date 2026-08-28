# CBraMod 运行 SHIN EEG

默认使用 `cbramod_env`，以 `pretrained_weights/pretrained_weights.pth` 为骨干初始化，
执行 100 epoch、batch size 8、分类头学习率 1e-4、骨干学习率 1e-5、
第 91 epoch 解冻、seed 1 的被试独立实验。

```powershell
powershell -ExecutionPolicy Bypass -File .\run_shin_gpu.ps1
```

输出位于 `D:\data\CBraMod-SHIN` 下的新时间戳目录。每次实验保存最优/最终权重、
训练历史、数据诊断、汇总 JSON 和中文 `EXPERIMENT_RECORD.md`。

输入适配：30 个 EEG 通道、200 Hz、完整 10 秒 trial，重排为 `30×10×200`；
排除 2 个 EOG；无需电极名称映射。遵循仓库 BCIC-IV-2a 处理方式，使用物理微伏、
公共平均参考和 0.3–50 Hz 五阶 Butterworth 滤波。分类头为平均池化加
`Linear(200, 2)`。

使用官方全 patch 一层分类头运行时：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_shin_gpu.ps1 -HeadType full_patch_onelayer
```

该分类头把 `30×10×200` 展平后接 `Linear(60000, 2)`，共有 120,002 个参数。
