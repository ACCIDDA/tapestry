# Tapestry

Probabilistic nowcasting and four-week forecasting of influenza, COVID-19 and RSV
admissions and ED visits. Independently fitted stages share one model interface
and connect through sampled reconstructed histories.

- [Documentation](https://accidda.github.io/tapestry/)
- [Architecture and reflection](docs/architecture.md)
- [Workflow](docs/workflow.md): acquire, build, plan, launch and rank
- [Model runs](docs/experiments/index.md): dated forecasting reports; nowcasting section
- [Data](docs/data/index.md) and [interactive staircase](https://accidda.github.io/tapestry/data/staircase/)
- [Setup](docs/setup.md)

```bash
uv sync
.venv/bin/python -m tapestry.dataset.build show
.venv/bin/python -m tapestry.experiment.planner status -e b-2-t0
```

`planner rank` is the canonical analysis path. Reports live in `docs/experiments/`;
model artifacts live in `data/experiments/`. The two-stage nowcasting/forecasting
pipeline remains available through `task=pipeline`; standalone tasks use
`task=nowcast` or `task=forecast`. See the workflow for full manager commands.
