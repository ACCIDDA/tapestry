# Emily’s EpiBench configs

Yes: these ten configs provide explicit forecast calendars and data-vintage rules for influenza, COVID-19 and RSV. They are **`epibench create` inputs**, not scoring configs or complete library scorecards.

All ten were executed successfully with the actual `epibench create --config-path` command against pinned local hub snapshots: **345 reference dates and 345 vintage-history files**. Per the user’s instruction, these files are parked and are not used for B0 training or scoring. Their data-quality review is deferred.

## What they define

Every file uses `vintaging: TRUE`, `vintaging_method: as_of`, and `vintaging_offset: -3`. The offset is in **days**: a Saturday reference date gets the latest available revisions through the preceding Wednesday. These histories can be used as forecast inputs; scoring future targets still needs a separate truth policy and scoring config.

| File | Target | First origin | Last origin | Created origins |
|---|---|---|---|---:|
| `covid_inchosp_24-25.yaml` | wk inc covid hosp | 2024-11-23 | 2025-08-30 | 41 |
| `covid_inchosp_25-26.yaml` | wk inc covid hosp | 2025-09-13 | 2026-08-15 | 49 |
| `covid_propedvisits_25-26.yaml` | wk inc covid prop ed visits | 2025-09-13 | 2026-08-15 | 49 |
| `flu_inchosp-25-26.yaml` | wk inc flu hosp | 2025-11-15 | 2026-05-30 | 29 |
| `flu_inchosp_23-24.yaml` | wk inc flu hosp | 2023-09-30 | 2024-05-25 | 35 |
| `flu_inchosp_24-25.yaml` | wk inc flu hosp | 2024-11-09 | 2025-05-31 | 30 |
| `flu_propedvisits_25-26.yaml` | wk inc flu prop ed visits | 2025-12-20 | 2026-05-30 | 24 |
| `rsv_inchosp_24-25.yaml` | wk inc rsv hosp | 2024-12-07 | 2025-05-31 | 26 |
| `rsv_inchosp_25-26.yaml` | wk inc rsv hosp | 2025-11-01 | 2026-05-30 | 31 |
| `rsv_propedvisits_25-26.yaml` | wk inc rsv prop ed visits | 2025-11-01 | 2026-05-30 | 31 |

Our current challenges remain unversioned custom configs for `epibench score`, using
the existing finalized evaluation truth. Emily’s created artifacts stay separate.

## Consequences for the current B0 report

The current [B0 report](../results/b0-configuration-comparison.md) uses the full EpiBench scoring command with exactly **0.025, 0.25, 0.5, 0.75, 0.975**. Its nine frozen evaluation task sets are unchanged. All new predictions and current Hubverse exports save only those five quantiles. Historical source forecasts retain their original quantiles for reproducibility.

The user explicitly requested that challenge ground truth not be used yet. Emily’s origins have not replaced the scoring calendar. In particular, the RSV 2024–25 config successfully creates historical inputs, but that alone does not supply an official-ensemble comparison on the current frozen task set. Switching B0 to these provisional inputs would require constructing and fitting a vintage-aware experiment; the existing B0 fits use finalized data.

These configs do not set forecast horizons, locations, quantiles, baseline/ensemble reference, scoring-truth releases, or scorecard functions. Those choices still need a scoring definition. The five quantiles are now shared with the bundled EpiBench challenges, while dates and reference models can still differ.

## Details to reconcile

- The provided COVID ED file is **2025–26**, not 2024–26; there is no separate 2024–25 COVID ED config in the folder.
- `covid_inchosp_25-26.yaml` sends output to `../covid19/prop-ed`. Its target is correctly hospital admissions, so this is an output-folder naming inconsistency. Prepared copies use distinct per-config output directories.
- EpiBench derives its date-library key from the local hub directory name. Prepared snapshots retain `FluSight-forecast-hub`, `covid19-forecast-hub`, and `rsv-forecast-hub` so date validation is enabled.

## Local artifacts and reproduction

Original files under `fromEmily/configs/` are unchanged. Only hub and output paths are adapted in `data/epibench/emily/configs/`. Full source hashes, commands, pinned commits, statuses and task-list paths are recorded in `data/epibench/emily/summary.json`; each config has a corresponding log. The initial path-check outputs are kept separately under `unvalidated_created/`; the canonical results are under `created/`.

`scripts/prepare_emily_challenges.py` prepares snapshots and invokes all ten configs. EpiBench refuses to overwrite an existing challenge output, so choose a fresh output root in that recipe for a repeat. No hub downloads or modifications to Emily’s originals are required.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_emily_challenges.py
```
