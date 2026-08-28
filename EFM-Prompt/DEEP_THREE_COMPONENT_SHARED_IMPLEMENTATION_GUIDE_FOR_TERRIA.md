# 给 Terria 的修改手册：共享式深层三分量 Prompt

## 1. 目标

在当前 `EFM-Prompt` 工程中新增一种深层 Prompt 模式：

```text
deep_three_component_shared
```

新模式将三分量 Prompt 注入冻结 EFM 的多个中间层：

```text
static prompt + expert-routed dynamic prompt + mapped prompt
```

核心设计是“一套共享生成器 + 每层轻量低秩投影”，不能为每个注入层复制完整 fNIRS 编码器、router、16 个专家或 mapped MLP。

必须保留当前所有代码、模式、参数、checkpoint 和结果。当前 `DeepConditionalPrompt` 及其 3 个注入点必须保持原行为；新功能只能通过新参数显式开启。

本阶段只实现深层三分量 Prompt，损失仍为分类交叉熵。不要同时加入 TMPA、OT、TAP、SGFormer 或其他新机制，避免无法判断效果来自哪里。

## 2. 当前代码事实

相关文件：

```text
foundation_deep_prompt.py
run_shin2017_foundation_deep_prompt.py
```

当前 `DeepConditionalPrompt`：

```text
共享 fNIRS token encoder
每阶段 static query
fNIRS cross-attention
EEG-to-prompt cross-attention
每阶段 gate
```

当前注入点：

```text
CBraMod：12 层中的第 6、9、12 层前
CSBrain：12 层中的第 6、9、12 层前
CodeBrain：8 个 residual SSSM block 中的第 4、6、8 个 block 前
```

当前 EFM 主体和 EEG-only 分类头均冻结，只优化 `model.prompt.parameters()`。这一训练协议在新模式中保持不变。

## 3. 新结构

### 3.1 共享 fNIRS 表示

沿用现有 `FnirsTokenEncoder`：

\[
Z_i=E_{fNIRS}(F_i)\in\mathbb{R}^{N_f\times d_p}
\]

对 token 求均值，得到路由条件：

\[
c_i=\operatorname{Mean}(Z_i)\in\mathbb{R}^{d_p}
\]

`E_fNIRS` 在所有注入层之间共享，而且每次模型 forward 只计算一次。禁止在每个层调用时重复编码完整 fNIRS。

建议由模型在进入骨干循环前调用：

```python
prompt_context = prompt.encode_fnirs(fnirs)
```

各层调用：

```python
tokens = prompt.inject(tokens, prompt_context, stage)
```

不要继续让每个 `prompt(..., fnirs, stage)` 内部重复运行 fNIRS encoder。

### 3.2 static Prompt

每个注入层有自己的 static Prompt：

\[
P_s^{(l)}\in\mathbb{R}^{L\times d_p}
\]

参数形状：

```text
[stage_count, prompt_tokens, prompt_dim]
```

static Prompt 很小，允许层间独立。

### 3.3 共享动态专家 Prompt

只建立一套专家库：

\[
E\in\mathbb{R}^{K\times L\times d_p}
\]

建议：

```text
K = 16
L = 6
prompt_dim = 128
```

只建立一个共享 router：

\[
r_i=\operatorname{softmax}(W_rc_i/\tau)
\]

路由后的动态 Prompt：

\[
P_{d,i}=\sum_{k=1}^{K}r_{ik}E_k
\in\mathbb{R}^{L\times d_p}
\]

`P_dynamic` 在一次 forward 中只生成一次，然后供所有注入层共享。不要创建：

```text
[stage_count, expert_count, prompt_tokens, prompt_dim]
```

专家库正确形状只能是：

```text
[expert_count, prompt_tokens, prompt_dim]
```

保留现有 MoPE 使用的 dense softmax 路由思想。初始建议：

```text
expert_count = 16
router_temperature = 0.1
router_noise_std = 0.00390625
```

训练时可以添加小路由噪声，验证和测试时不得加噪声。

保留 importance balance loss，防止专家长期只使用少数几个：

\[
L=L_{CE}+\lambda_{imp}L_{importance}
\]

建议 `importance_weight=0.01`。这是已有 MoPE 的损失，不属于 TMPA/OT。

### 3.4 共享 mapped Prompt

使用一个共享 MLP：

\[
P_{m,i}=MLP(c_i)\in\mathbb{R}^{1\times d_p}
\]

建议结构：

```text
Linear(prompt_dim, prompt_hidden)
GELU
Dropout
Linear(prompt_hidden, prompt_dim)
```

默认 `prompt_hidden=256`。该 MLP 在所有层之间共享。

### 3.5 每层三分量组合

第 `l` 个注入位置使用：

\[
Q_i^{(l)}=
[P_s^{(l)};P_{d,i};P_{m,i}]
\]

当 `L=6` 时，共有：

```text
6 static + 6 dynamic + 1 mapped = 13 prompt tokens
```

dynamic 和 mapped 跨层共享，只有 static 随层变化。

### 3.6 共享 Prompt-to-EEG interaction

将当前层 EEG token 展平：

\[
X_i^{(l)}\in\mathbb{R}^{N_l\times 200}
\]

先使用一个跨层共享的 EEG 输入投影：

\[
\hat X_i^{(l)}=W_{in}X_i^{(l)}\in\mathbb{R}^{N_l\times d_p}
\]

使用一个跨层共享的 cross-attention：

\[
R_i^{(l)}=
CrossAttention(\hat X_i^{(l)},Q_i^{(l)},Q_i^{(l)})
\]

cross-attention、输入 LayerNorm 和输入投影全部共享，不得按层复制。

### 3.7 每层 rank-8 投影和门控

每个注入层只拥有独立的低秩输出投影：

\[
U_l\in\mathbb{R}^{d_p\times r},
\qquad
V_l\in\mathbb{R}^{r\times200}
\]

其中：

```text
rank = 8
```

层输出：

\[
\Delta X_i^{(l)}=V_l(U_l(R_i^{(l)}))
\]

注入方式：

\[
X_i^{(l)}\leftarrow
X_i^{(l)}+alpha\tanh(g_l)\Delta X_i^{(l)}
\]

其中：

```text
prompt_scale alpha = 0.05
gate g_l 初始值 = 0
```

零初始化 gate 必须保证新模块初始化时与冻结 EEG-only 模型函数完全一致。

低秩投影必须实现为每层独立的 `Linear(prompt_dim, rank, bias=False)` 和 `Linear(rank, eeg_dim, bias=False)`，不能使用完整的每层 `Linear(200,200)`。

## 4. 注入层

新模式使用 4 个均匀分布的注入位置。

### CBraMod

12 层 Transformer，Python 零基索引：

```text
{2: 0, 5: 1, 8: 2, 11: 3}
```

即第 3、6、9、12 层原生 block 前。

### CSBrain

12 层 encoder，Python 零基索引：

```text
{2: 0, 5: 1, 8: 2, 11: 3}
```

必须保留 CSBrain 原有 `TemEmbedEEGLayer`、`BrainEmbedEEGLayer` 和 `area_config` 调用顺序。Prompt 放在原生 encoder block 前，与现有深层 Prompt 位置语义一致。

### CodeBrain

8 个 residual SSSM blocks，Python 零基索引：

```text
{1: 0, 3: 1, 5: 2, 7: 3}
```

即第 2、4、6、8 个 residual block 前。必须保留 CodeBrain 的 `[B,200,C*P]` 隐状态和原始 skip accumulation，只在注入时临时恢复为 `[B,C,P,200]`，注入后再还原。

旧模式仍使用原来的 3 个注入点，不得改变旧模式映射。

## 5. 文件修改要求

### 5.1 `foundation_deep_prompt.py`

保留 `DeepConditionalPrompt` 原样。新增独立类：

```python
class SharedDeepThreeComponentPrompt(nn.Module):
    method_name = "deep_three_component_shared_low_rank"
```

推荐接口：

```python
context = prompt.encode_fnirs(fnirs)
tokens = prompt.inject(eeg_tokens, context, stage)
importance_loss = prompt.importance_loss()
routing_stats = prompt.routing_statistics()
```

`context` 至少包含：

```text
dynamic_tokens [B,L,prompt_dim]
mapped_tokens  [B,1,prompt_dim]
routing_scores [B,K]
```

不要用可变的模块内部缓存保存带梯度 context；由当前 forward 显式传递，避免 batch 间状态污染。

### 5.2 `run_shin2017_foundation_deep_prompt.py`

新增参数：

```text
--deep-prompt-mode
    conditional
    three_component_shared
default: conditional

--expert-count             default 16
--router-temperature       default 0.1
--router-noise-std         default 0.00390625
--importance-threshold     default 0.05
--importance-weight        default 0.01
--prompt-rank              default 8
--prompt-hidden            default 256
```

注意：现有 `--prompt-tokens` 默认值为 4。为了不改变旧模式默认行为：

```text
conditional 模式未显式传参时仍为 4
three_component_shared 正式命令显式传 --prompt-tokens 6
```

不要悄悄把全局默认值由 4 改成 6。

`FrozenDeepPromptModel` 根据 `deep_prompt_mode` 选择：

```text
conditional            -> 原 DeepConditionalPrompt，原 3 层映射
three_component_shared -> 新 SharedDeepThreeComponentPrompt，新 4 层映射
```

三个 encoder 的 forward 应在骨干循环前只调用一次 `encode_fnirs()`，然后在指定层调用 `inject()`。

训练循环：

```text
conditional:
    total_loss = classification_loss

three_component_shared:
    total_loss = classification_loss
               + importance_weight * importance_loss
```

验证和测试只计算分类结果；importance loss 和路由噪声只用于训练。

仍只将 `model.prompt.parameters()` 放入 optimizer。EFM 与分类头继续冻结。

### 5.3 运行入口

如果已有深层 Prompt PowerShell 包装脚本，则给它增加对应参数；如果没有，不要为了包装而复制训练逻辑，可直接使用 Python runner。

新结果目录必须包含：

```text
deep-three-component-shared
```

不得覆盖现有深层 Prompt 结果。

## 6. 参数量控制与记录

在 `diagnostics.json` 中记录：

```text
deep_prompt_mode
stage_map
prompt_dim
prompt_tokens
expert_count
prompt_rank
shared_parameters
stage_specific_parameters
trainable_parameters
backbone_parameters
trainable/backbone ratio
backbone_frozen = true
classifier_frozen = true
```

`shared_parameters` 包括：

```text
fNIRS encoder
router
expert bank
mapped MLP
shared EEG input projection
shared cross-attention
```

`stage_specific_parameters` 只能包括：

```text
4 组 static prompts
4 组 rank-8 down/up projections
4 个 gates
```

验收时检查模型中只有一份 `prompt_experts`、一份 `router`、一份 `mapper` 和一份 `fnirs_encoder`。

## 7. 结果记录

`history.json` 每轮增加：

```text
classification_loss
importance_loss
importance_weight
routing_entropy
routing_cv
active_experts（如已有统计方式可实现）
```

`summary.json` 增加：

```text
method
deep_prompt_mode
stage_map
best_epoch
best_val
test_at_best_epoch
parameter_counts
args
```

best epoch 仍按照当前验证集 Accuracy 选择，不得查看 test 后选择模型。

## 8. 不允许修改的内容

1. 不删除或改写现有 `DeepConditionalPrompt`。
2. 不改变旧模式的 3 个注入点。
3. 不改变 EEG/fNIRS 数据、subject-independent 划分和预处理。
4. 不改变三个 EFM 的预训练权重加载和官方 EEG-only 分类头。
5. 不解冻 EFM 或分类头。
6. 不为每层复制完整专家库、router、mapped MLP 或 fNIRS encoder。
7. 不在本阶段加入 OT、TMPA、TAP、SGFormer、额外分类头或多任务 loss。
8. 不删除、覆盖旧 checkpoint 和实验结果。
9. 不自动启动正式 50 epoch 实验。

## 9. 验收测试

### 9.1 旧模式兼容

使用默认：

```text
--deep-prompt-mode conditional
```

确认模型类型、3 个注入点、参数量和 forward 输出与修改前一致，旧 checkpoint 仍可加载。

### 9.2 新模式形状

以 `B=2,L=6,d_p=128,K=16` 检查：

```text
fNIRS encoded once
dynamic_tokens  [2,6,128]
mapped_tokens   [2,1,128]
static_tokens   [2,6,128]（每层不同）
combined_tokens [2,13,128]
routing_scores  [2,16]
输出 EEG shape 与输入完全相同
```

### 9.3 共享性

检查 4 个 stage 的 module/parameter：

```text
专家库对象相同
router 对象相同
mapped MLP 对象相同
fNIRS encoder 对象相同
low-rank projection 和 static prompt 按 stage 不同
```

### 9.4 冻结与梯度

一个 mini-batch 反向传播后确认：

```text
EFM backbone 无梯度
classifier 无梯度
fNIRS encoder 有有限梯度
router 和专家库有有限梯度
mapped MLP 有有限梯度
使用到的 stage static prompt、low-rank projection 和 gate 有有限梯度
```

注意零 gate 初始化时，第一步可能主要只有 gate 获得分类梯度；importance loss 应能给 router 提供梯度。可在完成一个 optimizer step 后再次检查 Prompt 路径梯度。

### 9.5 参数量

输出新旧两种深层 Prompt 的可训练参数量，并证明没有乘以 4 复制专家生成器。

### 9.6 smoke test

分别对 CBraMod、CodeBrain、CSBrain 做参数诊断和最多 1 epoch smoke test。不得启动正式长实验。

## 10. 建议的实现后命令形式

实现后应能运行类似：

```powershell
D:\Anaconda\envs\bci4models\python.exe `
  .\run_shin2017_foundation_deep_prompt.py `
  --backbone cbramod `
  --task mi `
  --deep-prompt-mode three_component_shared `
  --head-checkpoint "<匹配的 EEG-only best_model.pth>" `
  --output-dir ".\runs_deep\cbramod_mi_deep-three-component-shared_seed1" `
  --prompt-dim 128 `
  --prompt-tokens 6 `
  --expert-count 16 `
  --prompt-rank 8 `
  --prompt-hidden 256 `
  --router-temperature 0.1 `
  --importance-weight 0.01 `
  --prompt-scale 0.05 `
  --epochs 50 `
  --batch-size 8 `
  --prompt-lr 3e-4 `
  --weight-decay 1e-4 `
  --dropout 0.1 `
  --seed 1
```

完成实现后，Terria 应根据工程中的真实 checkpoint 路径，分别给出 CBraMod、CodeBrain、CSBrain 的 MI/MA 手动运行命令，但不要替用户启动正式实验。

## 11. 完成后的汇报要求

请简洁报告：

1. 修改文件和新增类。
2. 新旧模式如何通过参数切换。
3. 三个模型各自的 4 个注入位置。
4. 共享模块和层独立模块分别有哪些。
5. 新旧可训练参数量及占 backbone 的比例。
6. EFM/分类头冻结检查结果。
7. 形状、梯度和 1 epoch smoke test 结果。
8. 六条正式手动命令（3 个模型 × MI/MA）。

