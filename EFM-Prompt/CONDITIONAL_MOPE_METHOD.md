# Conditional Prompt Tuning 与 CBraMod-MoPE 实现

## 1. 原论文方法

论文：Ruixiang Jiang, Lingbo Liu, Changwen Chen, *Conditional Prompt Tuning for Multimodal Fusion*, arXiv:2312.03734。

设主模态为 `x`，辅助模态为 `y`，对应编码器为 `E_x` 和 `E_y`。方法采用顺序融合：

1. 先由辅助模态编码器得到全局表征：`psi_y = E_y(y)`。
2. 使用 `psi_y` 生成实例相关的动态 prompt 和映射 prompt。
3. 将这些 prompt 与数据集级静态 prompt 一起注入冻结的主模态编码器。
4. 使用主模态编码结果完成下游预测。

论文将普通 prompt 分解为三部分：

- 静态 prompt `P_s in R^(l x d)`：所有样本共享，表示数据集级任务适配信息。
- 动态 prompt `P_d in R^(l x d)`：由辅助模态表征控制，表示样本级偏移。
- 映射 prompt `P_m in R^d`：两层 GELU MLP 将 `psi_y` 映射为单个 prompt，直接注入辅助模态的细粒度信息。

对于主模态编码器第 `i` 层，论文的输入为：

```text
x_hat_i = [x_cls_i, P_s_i, P_d_i, P_m_i, T_(i-1)]
```

其中方括号表示 token 维拼接。主模态预训练参数保持冻结，router、mapper、prompt 和任务分类头参与训练。论文还使用普通 prompt tuning 调整辅助模态编码器。

## 2. Mixture of Prompt Experts

每个被提示的主模态层都有独立的 `k` 个 prompt 专家：

```text
E_i,j in R^(l x d),  j = 1,...,k
```

该层的 router 根据辅助模态表征产生 dense routing 权重：

```text
r_i = softmax(W_r,i psi_y / tau + epsilon)
```

论文默认温度 `tau=0.1`，训练时加入噪声 `epsilon` 以增加路由多样性。原论文未明确噪声幅度；当前 `k=16` 默认采用作者后续公开实现中的标准差 `1/16^2=0.00390625`，并允许通过命令行调整。动态 prompt 是全部专家的凸组合：

```text
P_d,i = sum_j r_i,j E_i,j
```

论文不采用 Top-K 稀疏路由。所有 prompt 专家都是参数节点，dense 融合不会产生传统 FFN-MoE 的高前向计算成本，并且实验结果略优于 Top-1 路由。

## 3. Importance loss

对一个训练批次 `Y`，专家 `j` 的重要性定义为无噪声路由概率之和：

```text
Imp(E_j) = sum_(y in Y) softmax(W_r psi_y / tau)_j
```

路由正则为各专家重要性的变异系数平方：

```text
L_imp = (std(Imp) / mean(Imp))^2
```

当变异系数低于 `gamma=0.05` 时停止该项梯度，避免实例级小批量路由被过度约束。总训练目标为：

```text
L = L_cls + lambda_imp L_imp
```

当前实现默认 `lambda_imp=0.01`，与作者公开 MoPE 实现的示例配置一致；该权重在原始预印本方法节中未明确给定，因此作为显式可调超参数保存。

## 4. 论文默认配置

- prompt 长度 `l=6`；
- prompt 专家数 `k=16`；
- router 温度 `tau=0.1`；
- importance 阈值 `gamma=0.05`；
- mapper 为两层 MLP，中间使用 GELU；
- 主模态所有 Transformer 层使用独立 prompt 和 router；
- 主模态预训练参数冻结，分类头参与训练；
- 辅助模态使用普通 prompt tuning。

## 5. CBraMod 边界适配

CBraMod 的隐藏状态是固定的 `[B, 30, 10, 200]` 时空网格。其 Criss-Cross Attention 分别沿通道轴和时间 patch 轴计算注意力，直接增加任意 prompt token 会改变网格尺寸和注意力语义。因此当前版本不修改 CBraMod 内部层，而是在边界完成论文组件的适配：

```text
fNIRS [B,100,72]
  -> temporal conditioner
  -> psi_fnirs [B,256]
  -> {P_s, dense-MoPE P_d, two-layer mapped P_m}
  -> concatenate prompts
  -> low-rank projection to [B,30,10,200]
  -> add before and/or after frozen CBraMod encoder
```

前置和后置模块各自拥有独立的静态 prompt、专家池、router、mapper 和低秩投影。残差缩放参数 `alpha` 初始化为零，使模型从未调制的 CBraMod 表征开始优化。

这保留了论文的 prompt 分解、实例级 dense expert routing、路由噪声和 importance loss，但注入位置是 CBraMod 边界残差，不是论文的逐层 token 拼接。实验报告会明确记录该差异。

## 6. 训练与对照

`joint`：冻结 CBraMod，同时训练分类头、fNIRS conditioner、prompt experts、router 和 mapper。

`prompt_only`：加载同任务 EEG-only 分类头并冻结，只训练 fNIRS conditioner、prompt experts、router、mapper和边界投影。

建议至少保留以下对照：

1. EEG-only；
2. legacy conditional prompt；
3. MoPE conditional prompt；
4. MoPE conditional prompt，打乱 EEG-fNIRS 配对；
5. 参数结构相同但不使用真实 fNIRS 的 MoPE static 对照；
6. `lambda_imp=0` 的 MoPE 消融。

每轮训练在 `history.json` 中记录分类损失、importance loss、路由熵、专家重要性变异系数及最大/最小平均使用率。

## 7. 代码入口

- 现有版本：`run_cbramod_fnirs_feature_stage1_gpu.ps1`，默认 `prompt-family=legacy`。
- MoPE 版本：`run_cbramod_mope_boundary_gpu.ps1`，结果默认写入 `runs_mope`。
- MoPE 模块：`cbramod_mope_boundary.py`。

原论文：https://arxiv.org/abs/2312.03734

作者后续官方实现：https://github.com/songrise/MoPE
