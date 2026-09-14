# B0 experiment recipes

Run from the repository root with the package installed. These are research
recipes for the existing B0 protocol; reusable training lives in
`src/influpaintx/models/` and evaluation in `src/influpaintx/evaluation/`.

- `run_b0_experiments.py`: staged representation, history, dynamics, and loss-weight
  comparisons, then seed repeats. Frozen inputs and output directory are pinned
  in the file to the September 14 protocol. Completed summaries are reused.
- `audit_b0_holdout.py`: perturb held-out data and verify training inputs/scales;
  audit saved run artifacts. Uses the same pinned experiment directory.
- `report_b0_experiments.py`: summarize that completed staged protocol.
- `report_b0_cv.py`: report one supplied CV run without refitting.

```bash
PYTHONPATH=src python experiments/b0/run_b0_experiments.py
PYTHONPATH=src python experiments/b0/audit_b0_holdout.py
PYTHONPATH=src python experiments/b0/report_b0_experiments.py
PYTHONPATH=src python experiments/b0/report_b0_cv.py data/experiments/b0_season_cv
```

The first three commands target an existing dated protocol, not a generic new
experiment. For a new run use `python -m influpaintx.models.season_cv --output ...`
and `python -m influpaintx.evaluation.sweep` as described in the
[current workflows](../../docs/workflows/training.md).

Source files moved here from `scripts/`; stored results and their historical
provenance were not rewritten. Forecast validation, frozen-task matching,
aggregation, and objectives are shared through `evaluation/scoring.py`.
Reports now live in `docs/results/`. Historical report prose remains specific
to that experiment, including fixed interpretations; it is not a general report template.
