# SHIN Dataset B pipeline

This folder keeps the upstream scripts unchanged and adds an isolated SHIN
pipeline under `shin_pipeline/`.

## Data mapping

- Source: `D:\DataSets\SHIN\NIRS_01-29\subject XX\cnt.mat` and `mrk.mat`
- Sessions: mental-arithmetic sessions 2, 4, and 6
- Labels: mental arithmetic = 0, baseline/rest = 1
- Preprocessing: optical density, modified Beer-Lambert, 0.01-0.1 Hz
  Butterworth filtering, -5..-2 s baseline correction, and 0..20 s task window
- Model input: `[trial, 2, 36, 200]` (HbO/HbR, channels, time)
- Split: subjects 1-23 / 24-26 / 27-29

## Run

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_shin.ps1
```

Defaults are 50 epochs, batch size 128, head learning rate `1e-4`, and
the upstream AdamW backbone learning rate `1e-3`. Output is written to a new timestamped folder
under `D:\data\fNIRS-Transformer-SHIN`.

The default random seed is `1`. Every completed run writes
`EXPERIMENT_RECORD.md`, `summary.json`, and `diagnostics.json`; the Markdown
record contains the experiment idea, a parameter table, and a result table.
The timestamped output directory name also contains the seed.
