# Proposed follow-up to the B0 `crosses` calibration

Status: **proposal, nothing run.** This complements
[the crosses calibration](../results/b0-crosses.md) and is designed so its runs
can be scored and ranked *jointly* with the existing 153, not as a replacement.

## What this has to answer

The crosses experiment left three specific things unresolved.

1. **The fixed 50-epoch arms are undertrained.** In 75% of folds the training
   loss minimum falls in the last five epochs, with a median 4.2% relative
   improvement over the final ten. The ranking is therefore "best architecture
   at 50 epochs", which is not necessarily the converged ranking.
2. **The early-stopping result is confounded three ways.** `ep300:pat20` changes
   the cap, adds validation-based checkpoint selection, *and* removes ~20% of
   the training episodes for validation (inner 85/80/81 vs 104/99/99). It wins
   anyway (`anchor` −0.024 combined, best US score in the experiment), but the
   reason is not identified. `best_epoch` exceeds 50 in 81% of folds, median 80,
   max 173 — so most of the gain may simply be more epochs.
3. **The US is bad and the obvious fix backfires.** US carries ~half of all
   admissions WIS from 1.9% of tasks. Only 6 of 51 configurations beat the
   ensemble there against 40 of 51 on states/DC, and the states and US orderings
   are *negatively* rank-correlated (ρ = −0.324). The failure is overconfidence,
   not level: dispersion 0.819× the ensemble at US versus 1.008× at states, 50%
   coverage 37.5% versus 41.3%. Separate state/US heads make it worse
   (−0.288 combined to switch them *off* on `conv`).

## Track A — epoch ladder (no code change, hours)

Add one suite. `epochs` and `patience` are already `TrainingScenario` fields, so
this is expressible today and produces no duplicate of an existing crosses run.

```python
# src/tapestry/models/scenarios.py
def epochs_ladder():
    """Unconfounded training-length arms around the three crosses references."""
    scenarios = dict(CROSS_ANCHORS)                      # ep50:pat0, already run
    for name, ref in CROSS_ANCHORS.items():
        for label, variant in ofat(ref, epochs=(100, 300)).items():
            scenarios[f'{name}__{label}'] = variant      # ep100:pat0, ep300:pat0
    return scenarios

SUITES = {..., 'epochs': epochs_ladder}
```

**6 new configurations × 3 seeds = 18 runs = 54 season fits.** The three
references themselves are already done in `b0-crosses`, so only the new arms
need fitting. Fits took 1.5–7 s each in the crosses run, so this is minutes of
GPU, not hours.

Every arm holds `patience=0`, so the only thing that varies is training length —
no validation split, no checkpoint selection. Read against the existing
`ep50:pat0` and `ep300:pat20` runs this gives a clean decomposition:

| Comparison | Isolates |
|---|---|
| `ep50:pat0` → `ep100:pat0` → `ep300:pat0` | training length alone |
| `ep300:pat0` → `ep300:pat20` | validation selection + the 20% held-out cost |

### What each outcome would mean

- **Ordering of the three references stable across 50/100/300** — the crosses
  screen stands as it is, and no rerun of the 51 is warranted.
- **Ordering changes** — rerun the full `crosses` suite once at the converged
  setting. Do not patch the existing ranking; run it fresh under the new policy.
- **`ep300:pat0` matches `ep300:pat20`** — early stopping was buying epochs, not
  regularisation, and the validation split is pure cost.
- **`ep300:pat20` still wins at equal epochs** — checkpoint selection is doing
  real work and should become the default policy.

Prediction worth recording in advance: the small-capacity configurations are the
undertrained ones (Spearman(params, residual improvement) = −0.287; latent-32,
residual2, conv and attention arms all sit at 1.4–2.3% residual improvement
versus 4.3–4.4% for their smaller counterparts). Longer training should
therefore help the `raw` family most and *narrow* its deficit rather than widen
it. If instead the big configurations gain most, the capacity conclusion in the
crosses write-up is wrong and should be withdrawn.

## Track B — two US interventions

Both target the measured defect (intervals ~18% too narrow nationally), not the
architecture. They are deliberately the two cheapest things that could work.

### B1. Per-geography interval calibration — no refitting at all

`validation_forecasts.npz` already exists for every early-stopping run and is
currently unused. It holds `quantiles (23, 40, 4, 6, 52)`, `truth`, `mask`,
`locations` and `channels` — out-of-sample forecasts from validation blocks
*inside* the training seasons, so the held-out season is never touched.

Fit one multiplier per (geography, channel, horizon) that rescales each
quantile's distance from the median:

```text
q'(l) = median + k[geo, channel, horizon] * (q(l) - median)
```

choosing `k` to minimise validation WIS (or to hit nominal coverage — report
both, they are not the same objective). Apply to the held-out season's saved
forecasts and rescore with `tapestry.evaluation.totals.score_run`. No refitting,
no new fits, minutes of CPU.

This is a genuine test of *whether the US problem is calibration at all*. If a
single width multiplier near 1.2 recovers most of the US gap, the model is fine
and the uncertainty head is the problem. If it does not, the national error is
misplaced rather than mis-scaled and B2 is the real fix.

**Constraint to plan around:** only `patience > 0` runs write
`validation_forecasts.npz`, so today only the three `stopping_300_20`
configurations can be calibrated. Track A's `ep300:pat0` arms will not produce
them either. Either add a validation-only pass for the configurations to be
calibrated, or accept that B1 is evaluated on the early-stopping arms alone and
say so.

### B2. Correlated national error — needs a model change

US is currently a routed location inside the same tensor (`is_us` flag,
`loc == 'US'`), sharing the global latent draw with all 52 locations. State
errors therefore cancel when the national prediction forms, and US uncertainty
ends up close to an average of state uncertainties rather than a correlated
national one. That is the mechanism behind a 0.819 dispersion ratio.

Two variants, each a new scenario field so they stay inside the existing
grammar and ranking:

- **`us_aggregate`** — do not predict US directly. Sum the 51 state sample paths
  per member, so national uncertainty inherits whatever cross-state correlation
  the shared latent induces, and compare against the routed head.
- **`us_shared_factor`** — keep the routed head but add a per-episode national
  factor shared across states, so a common national shock is representable.

Run each as a one-factor change on all three references: **6 configurations ×
3 seeds = 18 runs.** Report states/DC and US separately and never as a single
combined number, because the two orderings disagree.

Honest expectation: B1 is likely to recover part of the gap cheaply; B2 is the
one that could change the model, and it is the only item here requiring code.

## What this deliberately does not do

- **No architecture search.** Grouped by family against the median seed SD of
  0.0294, only representation clears the noise floor (mean |Δ| 0.0544, best
  −0.131); capacity (0.0190) and data/context (0.0169) average *below* it. Five
  of the top ten configurations are *removals* from their reference. A
  28,481-parameter model where deleting components helps is information-limited,
  not capacity-limited.
- **No covariates yet.** They are the right next direction — nothing in the six
  channels can signal a second wave, which is the 2024-2025 double-peak failure
  (B0 predicts down at 77.7% of state rebound origins against the ensemble's
  56.2%). But covariates are a larger programme; this proposal is the cheap work
  that makes the covariate comparison interpretable.
- **No extra seeds.** Worth stating plainly: seeds remain the binding
  constraint, and neither track fixes that. A different configuration wins on
  every single seed, and the states/DC top-10 spread (0.0385) barely exceeds the
  median seed SD (0.0342). Undertraining does **not** explain this — configs
  nearer convergence have *higher* seed SD (mean 0.0478 versus 0.0263;
  Spearman −0.404), and validation stopping raised seed SD on two of three
  references. Undertraining is a bias; seed spread is a variance. Before any
  ranking is published, run 5–10 seeds on the top ~10 configurations.

## Layout and joint analysis

Keep the new runs in their own experiment folder so `b0-crosses` is never
mutated, and use the **same frozen support** so the totals are poolable:

```bash
.venv/bin/python -m tapestry.models.manager plan -e b0-epochs --suite epochs --device cuda
.venv/bin/python -m tapestry.models.manager run  -e b0-epochs
```

`data/experiments/b0-epochs/` then sits beside `data/experiments/b0-crosses/`
with identical `experiment.json` settings (`frozen:
data/evaluation/b0_hub_comparison_q23`, 2,048 evaluation draws).

For the joint ranking, `tapestry.evaluation.totals.rank` accepts an arbitrary
list of `{config_id, seed, path}`, so runs from both folders can be ranked
together without copying anything:

```python
from tapestry.evaluation.totals import rank
runs = [*crosses_runs, *epoch_runs]        # any mix of experiment folders
rank(runs, Path('data/experiments/joint-ranking-crosses-epochs'))
```

Both experiments must use the same frozen directory and the same 23-quantile
grid, or `frozen_cases` will refuse them — which is the intended guard.

## Cost

| Track | Configurations | Runs | Code change |
|---|---:|---:|---|
| A — epoch ladder | 6 | 18 | one suite function |
| B1 — calibration | 0 | 0 | a scoring script |
| B2 — US error structure | 6 | 18 | new scenario field + model |

Roughly 36 new fits against the 153 already done. The expensive item is B2's
model work, not compute.

## Assumptions

- Finalized retrospective CV throughout, selected on the same folds that score
  it; all three seasons inform development. Exploratory, not validation.
- Scoring stays the §10.3 total-WIS ratio on the frozen 23-quantile support, so
  nothing here is comparable to EpiBench relative WIS.
- B1 assumes validation blocks inside the training seasons are representative of
  the held-out season. That is exactly the assumption to check, and it is why
  the calibrated result must be reported beside the uncalibrated one.
