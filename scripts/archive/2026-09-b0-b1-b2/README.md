# Archived B0/B1/B2-era scripts (2026-09 restructuring)

These import deleted modules (`tapestry.models.manager`, `.season_cv`,
`.b1_seasons`, `.b2_scenarios`, `tapestry.evaluation.epibench/sweep/compare/
hubs.export_b0/nowcast/decisive/forward`, `tapestry.model_data.*`) or launch
Slurm commands that no longer exist (`manager compare`, `manager decisive`,
`tapestry.models.season_cv`). None of them run against the current codebase.
They plotted/audited results from the B0/B1/B2 sweeps and the EpiBench/R
scoring path, both superseded by the unified `Model`/`Scenario` and the
pure-Python WIS scorer -- see
[docs/design/restructure-2026-unified.md](../../../docs/design/restructure-2026-unified.md).

Kept for historical reference only. The current, working scripts are the
ones still in `scripts/`: `explore_covariates.py`, `export_hubverse_csv.py`,
`pull_covariates.py`, `update_published_explorer.sh`, `jlessler.sbatch`,
`notify.sbatch`, `b01_notify.py` -- see
[docs/workflows/training.md](../../../docs/workflows/training.md).
