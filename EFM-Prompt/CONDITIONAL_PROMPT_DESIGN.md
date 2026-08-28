# Boundary Conditional Prompt for EEG-fNIRS Foundation Models

## Reference Principle

This design follows Jiang, Liu, and Chen, *Conditional Prompt Tuning for
Multimodal Fusion* (arXiv:2312.03734): encode one modality first and use its
sample-specific representation as a prior to generate prompts for the frozen
encoder of the other modality. The original decomposition is retained, but the
prompting location is changed from every encoder layer to the two EFM
boundaries:

\[
\hat X_l = [P_s^l;\ R(\psi_N);\ f_m(\psi_N);\ X_{l-1}],
\qquad l=1,\ldots,L.
\]

Here, EEG is the prompted modality and fNIRS is the conditioning modality.
The EEG foundation encoder remains frozen. This is a boundary-conditioned
variant of the original deep prompting method, rather than a claim of identical
layer-wise prompting.

## Model-Independent Interface

All three backbones consume EEG patches and expose a feature tensor with the
same semantic axes:

\[
X\in\mathbb{R}^{B\times C\times P\times D},
\]

where `C` is the EEG-channel axis, `P` is the one-second patch axis, and `D`
is the latent dimension. For the SHIN configuration, all three use
`[B, 30, 10, 200]` at the backbone input/output boundary.

| Backbone | Native spatiotemporal mechanism | Prompt insertion boundary |
|---|---|---|
| CBraMod | patch embedding followed by criss-cross Transformer attention over channel-patch tokens | after `patch_embedding`, before each frozen encoder layer |
| CSBrain | patch embedding, temporal embedding, brain-region embedding, then region-aware Transformer layers | after temporal/region embedding and before each frozen encoder layer |
| CodeBrain | patch embedding followed by S4 temporal-state and local-attention residual blocks | additive feature prompt at every residual-block input; it has no token-concatenation Transformer boundary |

The conditional module therefore outputs a compact tensor in the backbone's
latent dimension. Each adapter is responsible only for placing that tensor at
the architecture-appropriate layer boundary.

## Boundary Conditional Prompt

The full model is

\[
H_0=E_{patch}(X_{EEG}),\qquad
\tilde H_0=H_0+\alpha_{in}U_{in}(P^{in}(z_N)),
\]

\[
Z=F_{EFM}^{frozen}(\tilde H_0),\qquad
\tilde Z=Z+\alpha_{out}U_{out}(P^{out}(z_N)),
\]

\[
\hat y=\operatorname{Head}_{fixed}(\operatorname{Flatten}(\tilde Z)).
\]

`E_patch` is the native EEG patch embedding, `F_EFM` is CBraMod, CSBrain, or
CodeBrain, and the fixed head is identical across every ablation. The two
boundary prompts have distinct roles:

| Boundary | Conditioning target | Function |
|---|---|---|
| Pre-EFM | patch embeddings | use the fNIRS prior to select or suppress EEG channel-patch evidence before frozen spatiotemporal encoding |
| Post-EFM | encoded EEG features | correct the task representation after frozen encoding without changing the classifier's input dimension |

The residual scales `alpha_in` and `alpha_out` are learnable scalars initialized
to zero. Thus, at initialization, the model exactly reduces to the frozen EEG
baseline with the same classifier.

## Conditional Prompt Generator

### fNIRS prior

Use the paired fNIRS trial to compute its statistical representation
`z_N in R^216` (mean, standard deviation, and endpoint change; normalized
using training-subject statistics only). A small frozen-safe encoder produces

\[
\psi_N = \operatorname{LN}(W_N z_N+b_N)\in\mathbb{R}^{D}.
\]

The fNIRS branch is intentionally shallow: its role is to provide a
sample-level physiological prior, not to replace the EEG encoder.

### Three prompt components

Generate `m` prompt vectors of width `D` at each boundary `q in {in,out}`:

\[
P^q(z_N)=P_s^q+P_d^q(z_N)+P_m^q(z_N).
\]

- `P_s^q in R^(m x D)`: learned static prompt. It represents task-level EEG-fNIRS common structure shared across trials.
- `P_d^q(z_N)`: instance-level dynamic prompt. In phase 2 it is a direct MLP projection of `psi_N`; in phase 3 it is replaced by MoPE.
- `P_m^q(z_N)=reshape(W_m^q psi_N)`: fNIRS-to-EEG mapping prompt. It carries fine-grained sample-specific haemodynamic information into the EEG representation space.

The pre- and post-EFM generators do not share their final projections:
`P^in` must shape the encoder input, whereas `P^out` must shape the
task-discriminative representation. They may share the small fNIRS prior
encoder that produces `psi_N`.

Initialize the final projections producing `P_d` and `P_m` near zero. The
initial network is then numerically close to the frozen pretrained model,
which is important for subject-independent SHIN training.

## Boundary Mapping and Placement

Flatten spatial and temporal EEG features to `T=C*P` positions only inside the
boundary adapter:

\[
H\in\mathbb{R}^{B\times T\times D},\qquad
\Delta^q=U_q(P^q(z_N))\in\mathbb{R}^{B\times T\times D}.
\]

Use a low-rank mapping instead of concatenating new tokens into the EFM:

\[
U_q(P)=A_q(P)B_q,\qquad \operatorname{rank}(U_q)\ll\min(T,D).
\]

This preserves the original conditional-prompt decomposition while avoiding
any sequence-length, positional-encoding, or attention-mask modification
inside the frozen EFM. Reshape `Delta^q` back to `[B,C,P,D]` and apply it as a
residual addition at the pre- or post-EFM boundary.

For CBraMod and CSBrain, `Delta^in` is added immediately after patch embedding
and `Delta^out` after the final projection. For CodeBrain, the corresponding
boundaries are its patch embedding output and final normalized feature output.
No backbone block or layer is modified.

## Training Protocol

Use a fixed downstream classifier in every condition:

\[
\operatorname{Flatten}(30\times10\times200)
\rightarrow 2000\rightarrow200\rightarrow2,
\]

with ELU and dropout `0.1`, identical to the CBraMod official
`all_patch_reps` head. Keep the EEG window `[0,10] s`, subject split, data
normalization, checkpoint selection, and seed fixed.

Train only the fNIRS encoder, the two static prompts, dynamic/mapping prompt
projections, low-rank boundary maps, residual scales, and the fixed-form
classifier. The EEG backbone is frozen.

Required comparisons are: EEG-only frozen baseline, static prompt only,
pre-EFM only, post-EFM only, pre-plus-post conditional prompt, and
shuffled-fNIRS pre-plus-post prompt. The shuffled condition tests whether a
gain truly depends on trial-aligned fNIRS.

## Phase 3: MoPE

Replace each direct dynamic boundary prompt with `K` dense-routed experts:

\[
r=\operatorname{softmax}(W_r\psi_N/\tau),\qquad
P_d^q=\sum_{k=1}^{K}r_kE_{k}^q,\qquad q\in\{in,out\}.
\]

Use the importance penalty from the reference paper to prevent routing collapse:

\[
\mathcal L=\mathcal L_{cls}+\lambda\left(\frac{\operatorname{std}(\operatorname{Imp})}
{\operatorname{mean}(\operatorname{Imp})}\right)^2,
\qquad \operatorname{Imp}_k=\sum_b r_{b,k}.
\]

Start with `m=4` prompt vectors, rank `r=8`, `K=4` experts, dense routing, and
a small importance coefficient. MoPE should only be enabled after boundary
conditional prompts outperform the matched EEG-only baseline across multiple
seeds.
