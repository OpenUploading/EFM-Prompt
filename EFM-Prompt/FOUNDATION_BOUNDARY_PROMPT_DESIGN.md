# CodeBrain and CSBrain Boundary Prompt

The CodeBrain and CSBrain adapters use the same boundary definition as the
CBraMod prompt experiments. The raw EEG input is not modified.

```text
EEG [B,30,10,200]
  -> native patch embedding
  -> + P_pre(fNIRS)                         optional
  -> frozen native encoder
  -> + P_post(fNIRS)                        optional
  -> official 30-channel/10-patch classifier
```

The prompt output is a residual grid with shape `[B,30,10,200]`. The MoPE
prompt contains a static component, dense softmax-routed expert prompts, and a
fNIRS-mapped component. The encoder and classifier are frozen in
`prompt_only`; the classifier is trained jointly in `joint`.

Use `run_foundation_boundary_prompt_gpu.ps1` with `-Backbone codebrain` or
`-Backbone csbrain`. The official backbone checkpoints are passed separately
from the portable prompt code.
