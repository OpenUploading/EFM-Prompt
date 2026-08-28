# SHIN experiment records

The default random seed for all new SHIN runs is `1`. The PowerShell runner
passes it explicitly and includes it in the timestamped output directory name.

Every completed training run writes:

- `EXPERIMENT_RECORD.md`: experiment idea, parameter table, and result table;
- `summary.json`: machine-readable configuration, seed, history, and metrics;
- `diagnostics.json`: data provenance, split, seed, and input validation;
- `best.pt` and `last.pt`: best-validation and final-epoch checkpoints.

Use `-ExperimentNote` to record the hypothesis or reason for a run and `-Seed`
only when an explicitly different-seed experiment is required.
