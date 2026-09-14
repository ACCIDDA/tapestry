# B0 experiment recipes

For new comparisons, use the [named experiment manager](../../docs/workflows/experiment-manager.md):

```bash
.venv/bin/python -m tapestry.models.manager list
.venv/bin/python -m tapestry.models.manager plan -e b0-next
.venv/bin/python -m tapestry.models.manager run -e b0-next
.venv/bin/python -m tapestry.models.manager compare -e b0-next
```

The focused suite has 14 configurations, 42 seeded CV runs, and 126 season fits.
Readable scenario strings key saved results; rerunning resumes completed work.
The scripts below are preserved for the older dated comparison.

Run from the repository root with the package installed. These are research
recipes for the existing B0 protocol; reusable training lives in
`src/tapestry/models/` and evaluation in `src/tapestry/evaluation/`.

- `run_b0_experiments.py`: staged representation, history, dynamics, and loss-weight
  comparisons, then seed repeats. Frozen inputs and output directory are pinned
  in the file to the September 14 protocol. Saved forecasts are scored through
  the full EpiBench config command; identical completed EpiBench outputs are reused.
- `audit_b0_holdout.py`: perturb held-out data and verify training inputs/scales;
  audit saved run artifacts. Uses the same pinned experiment directory.
- `report_b0_experiments.py`: summarize that completed staged protocol.

```bash
PYTHONPATH=src python experiments/b0/run_b0_experiments.py
PYTHONPATH=src python experiments/b0/audit_b0_holdout.py
PYTHONPATH=src python experiments/b0/report_b0_experiments.py
```

The first three commands target an existing dated protocol, not a generic new
experiment. For a new run use `python -m tapestry.models.season_cv --output ...`
and `python -m tapestry.evaluation.sweep` as described in the
[current workflows](../../docs/workflows/training.md).

Source files moved here from `scripts/`; stored results and their historical
provenance were not rewritten. Forecast validation, frozen-task matching,
aggregation, and objectives are shared through `evaluation/scoring.py`.
Reports now live in `docs/results/`. Historical report prose remains specific
to that experiment, including fixed interpretations; it is not a general report template.

The old Python persistence report and generator were removed. The canonical
[B0 evaluation report](../../docs/results/b0-configuration-comparison.md) uses
full EpiBench scores; [reproduction and integration gaps](../../docs/workflows/configuration-evaluation.md)
document its exact inputs and remaining Tapestry aggregation.
