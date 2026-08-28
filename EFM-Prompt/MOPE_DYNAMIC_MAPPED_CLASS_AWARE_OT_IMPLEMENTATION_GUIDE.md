# MoPE Dynamic + Mapped 类别感知 OT 新模式实施手册

## 1. 给 5.6terria 的任务指令

请在当前 `EFM-Prompt` 工程中，为 CBraMod 三分量 MoPE Prompt **新增一种可选训练模式**：

`dynamic_mapped_class_ot`

该模式在现有三分量 Prompt（static + expert-routed dynamic + mapped）基础上，增加 EEG 与 fNIRS 条件 Prompt 之间的类别感知 OT 对比损失。

必须遵守以下约束：

1. 保留所有已有 Python/PowerShell 文件、模式、参数、结果和运行行为。
2. 不覆盖或重命名 `legacy`、`mope`、`flat`、`tap4x4`、`mlp`、`sgformer` 等已有模式。
3. 新功能必须由一个新参数显式开启；默认关闭。不开启时，数值路径和旧代码行为应保持不变。
4. 不修改 CBraMod 主体结构、官方分类头、数据划分、预处理及现有 Prompt 注入位置。
5. static Prompt 不参与类别 OT；只使用路由后的 dynamic tokens 和 mapped token。
6. OT 只作为训练期辅助损失。验证和测试只根据单样本 logits 推理，不依赖 batch 内其他样本。
7. 不对 16 个原始专家的全部 `16 x 6` tokens 直接做 OT；OT 对象是每个样本路由后实际使用的 Prompt。

## 2. 当前结构与新增结构

当前 flat MoPE 专家库为：

```text
prompt_experts: [K, L, D]
K = 16 experts
L = 6 tokens per expert（常用实验设置）
D = 200
```

样本 `i` 的路由权重和动态 Prompt：

\[
r_i=\operatorname{softmax}(g(c_i)/\tau),
\qquad
P_i^{dyn}=\sum_{k=1}^{K}r_{ik}E_k
\]

因此 `P_dyn` 的形状是 `[B,L,D]`，不是 `[B,K*L,D]`。

mapped Prompt 为：

\[
P_i^{map}=M(c_i)\in\mathbb{R}^{1\times D}
\]

新模式构造：

\[
Q_i=[P_i^{dyn};P_i^{map}]\in\mathbb{R}^{(L+1)\times D}
\]

常用设置下，`Q` 形状为 `[B,7,200]`。static Prompt 是所有样本共享参数，不能表达样本或类别差异，因此不放入 `Q`。

在每个已启用的注入边界，取对应 EEG token：

```text
pre:  patch_embedding 后、加入 pre Prompt 前的 EEG tokens
post: encoder/proj_out 后、加入 post Prompt 前的 EEG tokens
```

将 EEG 网格展平为：

```text
H: [B, 30*10, 200] = [B,300,200]
```

OT 本身允许两侧 token 数不同，因此不要为了凑成 7 个 token 改写 CBraMod 特征结构。

## 3. 为什么不能做 Prompt 自身与自身的 OT

不要将强正样本写成：

\[
OT(Q_i,Q_i)
\]

它天然接近零，几乎没有有效学习信号。正确的强正样本是同一 trial 的跨模态配对：

\[
H_i^{EEG}\leftrightarrow Q_i^{fNIRS\rightarrow Prompt}
\]

batch 内距离矩阵定义为：

\[
D_{ij}=OT(H_i,Q_j)
\]

关系定义：

```text
i == j                         强正样本
i != j 且 y_i == y_j           弱正样本
y_i != y_j                     负样本
```

## 4. OT 计算

先对 token 在特征维做 L2 normalization，再使用 cosine cost：

\[
C_{ab}=1-\hat h_a^T\hat q_b
\]

用带熵正则的 Sinkhorn 求传输计划：

\[
T^*=\arg\min_T\langle T,C\rangle-\epsilon H(T)
\]

OT 距离：

\[
D(H,Q)=\langle T^*,C\rangle
\]

建议创建一个独立的小模块，例如：

```text
mope_class_aware_ot.py
```

其中放置：

```text
sinkhorn_from_cost(cost, epsilon, iterations)
pairwise_token_ot(eeg_tokens, prompt_tokens, epsilon, iterations)
class_aware_ot_losses(distance_matrix, labels, temperature)
```

可以参考当前 `foundation_tmpa_token_alignment.py` 中经过验证的 Sinkhorn 实现，但不要让 MoPE 模块 import 整个 TMPA 训练入口，也不要改变 TMPA 原文件。

实现必须保持可微，不能把 Prompt tokens、cost 或 distance 转成 NumPy，也不能对 Prompt 分支调用 `detach()`。EEG anchor 可以 `detach()`，使 OT 只推动 fNIRS 编码器、router、experts 和 mapped mapper；分类损失仍按现有路径训练允许训练的模块。

## 5. 类别感知损失

令：

\[
S_{ij}=-D_{ij}/T
\]

### 5.1 同 trial 强正损失

使用对称 InfoNCE，矩阵对角线为正确匹配：

\[
L_{pair}=\frac{1}{2}
\left[
CE(S,[0,\ldots,B-1])+CE(S^T,[0,\ldots,B-1])
\right]
\]

### 5.2 同类别弱正损失

对每个 anchor，将同类且非自身的样本作为正样本，其他类别作为负样本。使用 supervised contrastive 的 log-softmax 形式。若某个 anchor 在当前 batch 中没有同类其他样本，则跳过该 anchor，不能产生 NaN。

分别按行和按列计算后取平均，得到 `L_class`。

### 5.3 总损失

保留旧损失，新增两项：

\[
L=L_{cls}
+\lambda_{imp}L_{importance}
+\lambda_{attr}L_{attribute}
+\lambda_{pair}L_{pair}
+\lambda_{class}L_{class}
\]

第一组建议参数：

```text
ot_temperature       = 0.1
sinkhorn_epsilon     = 0.1
sinkhorn_iterations  = 20
ot_pair_weight       = 0.1
ot_class_weight      = 0.02
```

`L_pair` 表示强正关系；较小的 `L_class` 权重表示同类其他 trial 只是弱正关系。

当 batch 中只有一个样本，或者不存在有效弱正样本时，`L_class` 必须返回同设备、同 dtype 的标量 0。

## 6. 推荐的代码改动边界

### 6.1 `cbramod_mope_boundary.py`

在 `MoPEBoundaryPrompt` 中增加一种**不破坏旧调用**的辅助输出方式。

推荐方案：

```python
forward(..., return_aux=False)
```

默认 `False` 时仍只返回原来的 residual，保证旧模式完全不变。`True` 时返回：

```text
residual
aux = {
    "dynamic_tokens": [B,L,D],
    "mapped_tokens":  [B,1,D],
    "contrast_tokens": cat(dynamic, mapped, dim=1),
}
```

注意事项：

1. flat 模式使用路由混合后的 `[B,L,D]`。
2. tap4x4 模式使用当前实际注入前的、四属性聚合后的 `[B,L,D]`。
3. mapped MLP 与 SGFormer 都统一整理为 `[B,1,D]`。
4. 新模式必须要求 dynamic 和 mapped 都存在；若 `drop_component` 删除其中之一，应明确报错。
5. 原有 residual 的计算顺序、参数初始化及输出不得改变。

### 6.2 `run_shin2017_cbramod_fnirs_feature_stage1.py`

新增参数，名称建议：

```text
--mope-contrast-mode        choices: none, dynamic_mapped_class_ot; default none
--ot-temperature           float; default 0.1
--sinkhorn-epsilon         float; default 0.1
--sinkhorn-iterations      int; default 20
--ot-pair-weight           float; default 0.1
--ot-class-weight          float; default 0.02
```

只有满足以下条件才启用新损失：

```text
prompt_family == "mope"
mope_contrast_mode == "dynamic_mapped_class_ot"
mode != "eeg_only"
prompt_source == "conditional"
```

模型 forward 在新模式下需要额外返回：

```text
logits
boundary_pairs = [(eeg_anchor_pre, prompt_tokens_pre), ...]
```

边界规则：

```text
mode=pre       只计算 pre OT
mode=post      只计算 post OT
mode=pre_post  分别计算 pre/post OT，再对相应 loss 取平均
```

训练循环中仅新模式使用 auxiliary output。`evaluate()` 继续调用普通 logits 路径，不计算 batch OT。

`history.json`、`summary.json`、终端输出和 `EXPERIMENT_RECORD.md` 至少增加：

```text
mope_contrast_mode
ot_pair_loss
ot_class_loss
ot_pair_weight
ot_class_weight
ot_temperature
sinkhorn_epsilon
sinkhorn_iterations
```

旧模式的历史字段可以不增加，或者记录 mode=`none`，但不能改变旧指标含义。

### 6.3 `run_cbramod_mope_boundary_gpu.ps1`

新增 PowerShell 参数并传递给 Python：

```text
-MoPEContrastMode none|dynamic_mapped_class_ot
-OTTemperature 0.1
-SinkhornEpsilon 0.1
-SinkhornIterations 20
-OTPairWeight 0.1
-OTClassWeight 0.02
```

默认 `MoPEContrastMode=none`。新模式的输出目录名必须包含明确标识，例如：

```text
contrast-dynamic_mapped_class_ot
```

这样不会覆盖或混淆原有运行结果。

## 7. 不允许进行的修改

1. 不删除、替换或重命名已有模式。
2. 不把 `dynamic_mapped_class_ot` 设为默认值。
3. 不改数据划分、seed、分类头、CBraMod checkpoint 加载方式或 best epoch 选择标准。
4. 不把 OT 用于验证集或测试集的 batch 内推理。
5. 不使用测试标签或测试统计量参与训练。
6. 不直接对静态 Prompt 做类别对比。
7. 不将同一 Prompt 与自身的 OT 当作强正损失。
8. 不保存新的 `last_model.pth` 变体，也不删除已有结果。
9. 不顺便重构无关模块。

## 8. 验收检查

完成后必须执行以下检查，但不要直接进行正式长训练。

### 8.1 旧模式兼容

使用原参数且不开启新模式：

```text
-MoPEContrastMode none
```

确认：

```text
forward 输出类型不变
训练损失组成不变
旧 checkpoint 可加载
PowerShell 旧命令仍可运行
```

### 8.2 新模式形状

以 `B=8,L=6,D=200` 检查：

```text
dynamic_tokens = [8,6,200]
mapped_tokens = [8,1,200]
contrast_tokens = [8,7,200]
eeg_anchor = [8,300,200]
distance_matrix = [8,8]
pair_loss/class_loss/total_loss = scalar
```

### 8.3 梯度

单个 mini-batch 反向传播后确认：

```text
prompt_experts 有有限且非零梯度
router 有有限且非零梯度
mapper 有有限且非零梯度
fNIRS encoder 有有限梯度
冻结 CBraMod 主体无梯度
prompt_only 下冻结分类头无梯度
```

### 8.4 数值稳定性

检查：

```text
没有 NaN/Inf
缺少同类弱正样本时 class loss 为 0
Sinkhorn 不在 fp16 下发生指数溢出；必要时 OT cost/iterations 使用 float32
```

### 8.5 最小 smoke test

只运行 1 epoch 或少量 batch，确认日志和结果文件完整。不要替用户启动 50 epoch 正式实验。

## 9. 建议的新模式运行形式

实现完成后，应能使用类似下面的命令手动运行：

```powershell
.\run_cbramod_mope_boundary_gpu.ps1 `
  -Task mi `
  -Mode pre_post `
  -PromptSource conditional `
  -TrainingStrategy prompt_only `
  -DynamicExpertMode flat `
  -MappedMode mlp `
  -MoPEContrastMode dynamic_mapped_class_ot `
  -OTTemperature 0.1 `
  -SinkhornEpsilon 0.1 `
  -SinkhornIterations 20 `
  -OTPairWeight 0.1 `
  -OTClassWeight 0.02 `
  -Epochs 50 `
  -BatchSize 8 `
  -PromptCount 6 `
  -ExpertCount 16 `
  -Seed 1
```

这只是目标接口示例。实现模型应在完成修改后，根据实际参数名输出一条 1-epoch smoke-test 命令和一条正式实验命令，但不要自行启动正式实验。

## 10. 完成后汇报格式

请简洁列出：

1. 修改过的文件及每个文件的作用。
2. 新增参数及默认值。
3. dynamic/mapped/EEG token 的实际形状。
4. 总损失的精确公式和代码对应位置。
5. 旧模式兼容性检查结果。
6. smoke test 结果。
7. 用户可以手动运行的新命令。

