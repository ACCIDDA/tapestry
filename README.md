# Tapestry

Tapestry is a research project for multi-disease epidemic nowcasting and forecasting.
It is in active development.

* [Documentation](https://accidda.github.io/tapestry/)
* [Data explorer with revisions](https://accidda.github.io/tapestry/explorer/)

## Current experiment

[Fresh forecasting covariate comparison](docs/design/forecast-covariates.md):
matched standalone forecasters comparing raw, smoothed, summarized and
encoded covariates across independent, pooled and attention-based geography, trained from scratch on finalized values masked by Wednesday availability.
Nowcasting is trained separately.

```bash
bash experiments/forecast-covariates.sh
.venv/bin/python -m tapestry.experiment.planner status -e forecast-geography-v2
```

## One workflow

The shared dataset is `data/processed/panel.npz`: six admission/ED targets across
states, DC and native US, with covariates, frozen reference labels and historical
Wednesday as-of observations. Independently fitted nowcasting and forecasting
models connect through sampled reconstructed histories.

From the repository root, using an existing panel:

```bash
uv sync
.venv/bin/python -m tapestry.dataset.build show --dataset data/processed/panel.npz
.venv/bin/python -m tapestry.experiment.planner plan -e my-experiment -s 'task=pipeline,input_mode=vintaged,nowcast.covariate_set=inpatient+ilinet,nowcast.lookback=12,forecast.lookback=8' --seeds 42 --device cpu
.venv/bin/python -m tapestry.experiment.planner run -e my-experiment
.venv/bin/python -m tapestry.experiment.planner status -e my-experiment
.venv/bin/python -m tapestry.experiment.planner rank -e my-experiment
```

Use `task=nowcast` or `task=forecast` for standalone fits through the same manager.
Use `nowcast.<field>` and `forecast.<field>` for independent pipeline stage settings.
The epochs, architecture, covariates and lookback can differ by stage.

[Model interface and manager commands](docs/design/nowcast-forecast.md) ·
[Training and source acquisition](docs/workflows/training.md) ·
[Environment setup](docs/getting-started.md) ·
[Longleaf setup](docs/longleaf-setup.md)

## Refresh the data

Training uses the saved panel without downloading sources. To acquire or refresh
sources and rebuild it, follow the [training guide](docs/workflows/training.md).
The current data catalog uses Delphi and Hub source names; old CDC source commands
and the former finalized/B1/B2 panel files are no longer the training interface.

Pipeline inputs are entirely as-of; labels use the frozen reference snapshot.
Season cross-validation is retrospective and is not a claim of historical
real-time deployment performance. Native-unit scoring is implemented in Python.

## Explore

```bash
.venv/bin/python -m tapestry.explorer serve
```

[Explorer guide](docs/explorer/overview.md). Historical B0/B1/B2 research notes
remain under `docs/legacy-v0/`; their commands are not the current workflow.
