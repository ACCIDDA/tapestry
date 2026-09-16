# B0.1 — Architecture crosses

**Experiment label:** B0.1 · **Stored experiment ID:** `B0.1` · **Status:** complete —
172 configurations × 3 seeds = 516 season-CV runs, 1,548 season fits.
Fitted 2026-09-15 22:06 – 2026-09-16 04:28 local on `g1803jles01` (4 × L40) and
`g1803jles02` (2 × H100), eight lanes per card; about 319 GPU-hours, median 36
minutes per seed fit. Ranking artifact `ranking-509b07b0d243`.

This page reports the [B0.1 specification](../../design/b0.1.md): six
prespecified reference architectures, one-factor changes around each, two
interaction panels, and matched independent-model and convolution controls. All
models are sample-based; direct quantile prediction was removed. Scoring uses
the [adopted objective](../../design/architecture.md#82-design-choices-for-b0-loss-and-weights)
— equal seasons, 80% equal-weight states/DC plus 20% native US, and
location-relative model/ensemble WIS ratios — so scores are directly comparable
across this page but **not** to the [B0.0 page](../b0-crosses/index.md), which
retains its original pooled, target-first score.

!!! note "One-factor screen, not a factorial"
    Each cross changes one factor around its own reference. Effects are local to
    that reference, and the two interaction panels identify only the
    interactions they specify. The references are contrasting hypotheses, not
    winners selected from outer-season scores.

## Headline

**36 of 172 configurations beat the hub ensemble** (combined score < 1), and the
leader reaches **0.883**. The result is dominated by two facts: the decoder
choice decides the outcome far more than any information-exchange question the
experiment was designed to ask, and every configuration is **badly
under-dispersed**.

| | Value |
|---|---:|
| Configurations beating the ensemble | 36 of 172 |
| Best combined score | 0.883 |
| Worst combined score | 4.946 |
| Median seed SD | 0.120 |
| Range across the top 10 | 0.058 |
| Mean 50% coverage, states/DC (nominal 50) | ~40% |
| Mean 95% coverage, states/DC (nominal 95) | ~81% |

### The six references

| Reference | Combined score | Rank of 172 |
|---|---:|---:|
| `local_mlp` | 1.004 | 38 |
| `spatial_conv` | 1.034 | 56 |
| `target_multiscale` | 1.228 | 114 |
| `pathogen_multiscale` | 1.230 | 115 |
| `joint_multiscale` | 1.433 | 139 |
| `joint_trend` | 4.104 | 164 |

The ordering is the reverse of the experiment's architectural hypothesis. The
**simplest reference wins**, and every added mechanism — pathogen or target
heads, joint location–target exchange, and above all the stochastic trend
decoder — costs score at its own reference.

## Ranking

![Combined score for all 172 configurations](figures/ranking-combined.png)

Filled dots are the mean over seeds 42/43/44; open dots are the individual
seeds. The ensemble parity line at 1 sits inside the upper third of the table.
The axis stops at 2.0 so the region around parity — where the experiment is
actually decided — is legible; the 28 configurations worse than that are clipped
to the right edge and labelled **catastrophic** with their true score. All 28 are
`joint_trend` variants except two (`spatial_conv__decoder_stochastic_trend` and
`target_multiscale__decoder_stochastic_trend`), which is the same finding from a
different angle: the stochastic trend decoder, wherever it appears.

![Score distribution by reference family](figures/family-spread.png)

Family membership, not the one-factor crosses within a family, explains most of
the spread. `independent` and `local_mlp` cluster at parity; `joint_trend`
occupies a separate regime three to five times worse, with no overlap.

### Leaders

| Rank | Configuration | Combined | Seed SD | States/DC | US |
|---:|---|---:|---:|---:|---:|
| 1 | `independent__encoder_mlp__fit_partition_target__epochs_100` | 0.883 | 0.025 | 0.906 | 0.793 |
| 2 | `independent__encoder_mlp__fit_partition_pathogen__epochs_300` | 0.895 | 0.032 | 0.917 | 0.806 |
| 3 | `independent__encoder_mlp__fit_partition_target__epochs_300` | 0.895 | 0.047 | 0.917 | 0.807 |
| 4 | `independent__encoder_mlp__fit_partition_pathogen__epochs_100` | 0.897 | 0.019 | 0.922 | 0.796 |
| 5 | `independent__encoder_multiscale_conv__fit_partition_target__epochs_100` | 0.909 | 0.039 | 0.925 | 0.844 |
| 6 | `local_mlp__exchange_joint_location_target__head_sharing_pathogen` | 0.911 | 0.035 | 0.921 | 0.870 |

The top five are all **independent-model controls** — separately fitted models
per target or per pathogen, with no cross-target parameter sharing at all. These
were included as a baseline to beat, and they were not beaten. Their spread
(0.883–0.909) is half the median seed SD, so their internal order is not
resolved by three seeds.

## Fan plots

Four-week fans at every third origin, for the three leading configurations at
their median seed against the hub ensemble, for the United States and North
Carolina.

![Influenza admissions, 2024-2025](figures/fans-flu_hosp-2024-2025.png)

The B0 fans track the influenza wave closely at the nowcast and the shape is
right, but the 50% band is visibly too narrow through the ascending limb and
around the peak — the truth leaves the inner band repeatedly. The hub ensemble's
own median lags the ascent more than B0's does, which is where B0's score
advantage comes from; the ensemble buys its calibration with much wider bands.
Note also that B0 forecasts from earlier origins than the hub, which has no
submissions before December.

![Influenza admissions, 2023-2024](figures/fans-flu_hosp-2023-2024.png)
![Influenza admissions, 2025-2026](figures/fans-flu_hosp-2025-2026.png)
![COVID-19 admissions, 2024-2025](figures/fans-covid_hosp-2024-2025.png)
![COVID-19 admissions, 2025-2026](figures/fans-covid_hosp-2025-2026.png)
![RSV admissions, 2025-2026](figures/fans-rsv_hosp-2025-2026.png)
![Influenza ED visits, 2025-2026](figures/fans-flu_prop_ed_visits-2025-2026.png)
![COVID-19 ED visits, 2025-2026](figures/fans-covid_prop_ed_visits-2025-2026.png)
![RSV ED visits, 2025-2026](figures/fans-rsv_prop_ed_visits-2025-2026.png)

## Forecast quality

![Quality pairplot](figures/quality-pairplot.png)

WIS decomposes exactly into dispersion, underprediction and overprediction, so
the panels above compare the score against its own components plus interval
coverage. Each point is one configuration on states/DC tasks. Three things read
straight off it:

- **Calibration is the score.** 50% and 95% coverage both correlate with the
  combined score at ρ ≈ −0.7, and the dispersion share at ρ = −0.72. Nothing
  else comes close. Configurations do not lose because their medians are wrong;
  they lose because their intervals are too narrow.
- **No configuration reaches nominal.** The entire 50% coverage distribution
  sits between 14% and 45%, against a nominal 50; 95% coverage tops out near
  86%. Even the leaders are under-dispersed — the red nominal lines are outside
  the point cloud everywhere. The ensemble, by contrast, achieves 50.4% and
  90.4% on the same tasks.
- **`joint_trend` collapses, it does not merely underperform.** The red points
  form a separate arm at 14–31% coverage with a dispersion share near zero and
  bias down to −1.0, meaning essentially all of their WIS is underprediction
  with no interval width at all. That is a degenerate predictive distribution,
  not a poorly tuned one.

The bias panel is the one weak relationship (ρ = −0.04 against the score): the
main cluster is mildly negative, centred near −0.15, so models underpredict
slightly on average, but within the healthy cluster that bias does not separate
good from bad. Underprediction dominates the collapsed arm only.

![Coverage by target](figures/coverage.png)

The shortfall is worst for the ED-visit proportions — COVID-19 ED visits reach
only about 28% 50%-coverage at states/DC and 19% at US — and these are also the
only targets where the leaders score worse than the ensemble (COVID-19 ED visits
1.143 for the top five). Admissions targets, where the fourth-root transform and
the Q95 scaling are better matched to the data, are where the skill is.

## Performance discussion

**The experiment's main hypothesis did not survive.** B0.1 was designed around
spatial and cross-target information exchange, pathogen-versus-target parameter
sharing, and explicit level/growth uncertainty. Averaged over the references it
applies to, the one-factor effect of each mechanism is:

### Verdict on every factor

The table below gives each one-factor change, averaged over the references it
was applied to. **`joint_trend` is excluded**: its baseline is already collapsed,
so almost any change "improves" it by partially undoing the trend decoder, which
would otherwise contaminate every row (`count_transform_sqrt`, for instance,
looks like −0.45 with `joint_trend` included and −0.03 without).

A factor is **clear** only when every reference moves the same direction *and*
the mean exceeds the 0.120 median seed SD. Anything that changes sign between
references is **complicated** — the effect depends on the architecture it is
applied to, and this screen cannot resolve it.

| Factor | Mean Δ | Range | Same sign | Verdict |
|---|---:|---|:--:|---|
| `decoder_stochastic_trend` | **+1.259** | +0.10 … +2.34 | 4/4 | **CLEAR — badly harmful** |
| `encoder_mlp` | **−0.324** | −0.50 … −0.24 | 3/3 | **CLEAR — helpful** |
| `exchange_none` | **−0.272** | −0.29 … −0.26 | 2/2 | **CLEAR — helpful** |
| `encoder_conv` | **−0.139** | −0.22 … −0.04 | 4/4 | **CLEAR — helpful** |
| `ed_transform_linear` | **−0.124** | −0.42 … −0.01 | 5/5 | **CLEAR — helpful** |
| `noise_global` | −0.090 | −0.20 … −0.01 | 3/3 | consistent but under noise |
| `epochs_100` | −0.069 | −0.11 … −0.04 | 5/5 | consistent but under noise |
| `head_sharing_pathogen` | −0.023 | −0.02 … −0.02 | 2/2 | consistent but under noise |
| `exchange_pathogen_spatial` | +0.054 | +0.03 … +0.08 | 2/2 | consistent but under noise |
| `ed_transform_fourth_root` | −0.127 | −0.42 … +0.06 | 4/5 | complicated |
| `exchange_shared_spatial` | −0.094 | −0.16 … +0.01 | 2/3 | complicated |
| `head_sharing_shared` | −0.080 | −0.23 … +0.06 | 2/3 | complicated |
| `us_heads_separate` | −0.068 | −0.22 … +0.04 | 4/5 | complicated |
| `lookback_26` | −0.049 | −0.29 … +0.19 | 4/5 | complicated |
| `count_transform_log1p` | −0.049 | −0.22 … +0.09 | 3/5 | complicated |
| `lookback_8` | −0.046 | −0.31 … +0.17 | 3/5 | complicated |
| `dynamics_False` | −0.041 | −0.13 … +0.03 | 4/5 | complicated |
| `shared_factor_True` | −0.030 | −0.13 … +0.02 | 3/5 | complicated |
| `count_transform_sqrt` | −0.028 | −0.14 … +0.15 | 3/5 | complicated |
| `annual_calendar_False` | −0.024 | −0.30 … +0.24 | 2/5 | complicated |
| `location_embedding_8` | −0.021 | −0.12 … +0.08 | 3/5 | complicated |
| `encoder_multiscale_conv` | −0.017 | −0.07 … +0.03 | 1/2 | complicated |
| `noise_global_local` | −0.009 | −0.02 … +0.00 | 1/2 | complicated |
| `exchange_target_spatial` | +0.009 | −0.26 … +0.18 | 2/3 | complicated |
| `width_128` | +0.014 | −0.25 … +0.37 | 2/5 | complicated |
| `exchange_joint_location_target` | +0.091 | −0.10 … +0.33 | 2/3 | complicated |
| `head_sharing_target` | +0.101 | −0.03 … +0.24 | 2/3 | complicated |

Negative is better. Reading it:

- **Five factors are clear.** One is badly harmful (the trend decoder); four are
  helpful, and all four point the same way — *toward the simpler model*. A plain
  MLP encoder beats multiscale convolution, turning information exchange **off**
  beats every form of it, and the plain linear ED transform beats the fancier
  ones. Nothing that adds machinery is clearly good.
- **Four are consistent but too small to bank.** They never change sign, but the
  mean is under the seed-noise floor, so they are suggestive at best. `epochs_100`
  being mildly *better* than `epochs_300` belongs here and is discussed under
  overfitting below.
- **Eighteen are complicated**, i.e. the majority. Their effect flips sign
  depending on the reference, which means the one-factor screen has done its job
  and returned "it depends". `width_128` (−0.25 to +0.37) and
  `annual_calendar_False` (−0.30 to +0.24) are the clearest examples: these
  interact with the architecture and cannot be settled here.

The honest summary is that **one factor dominates and four point toward
simplicity; everything else is either noise-limited or architecture-dependent.**

### The `joint_trend` crosses are rescue attempts, not effects

Every cross around `joint_trend` is best read as "how much of the trend
decoder's damage does this undo". The baseline is 4.104; the best any single
change achieves is 1.559 (`count_transform_sqrt`, −2.545). None reach parity.
This is why the family is excluded from the table above, and it is independent
evidence that the decoder — not the recipe around it — is what is broken.

**The stochastic trend decoder is the finding.** Swapping only the decoder onto
an otherwise unchanged reference costs +0.99 at `local_mlp`, +1.60 at
`spatial_conv` and +2.34 at `target_multiscale`. Combined with the pairplot's
collapsed arm — near-zero dispersion share, bias near −1 — this is not a tuning
problem to be fixed with a wider prior. The decoder is producing near-degenerate
sample trajectories, and `joint_trend` inherits it, which is the entire reason
that family occupies its own regime. No `joint_trend` variant with a legacy
decoder was run, so the specification confounds the trend decoder with the rest
of the `joint_trend` recipe; that control is the obvious missing cell.

### Is the leader overfitting?

The leading configurations are visually appealing — the fans track the wave
closely and the medians are well placed — which is exactly when overfitting is
worth checking rather than assuming. Two different things could be meant, and
they have different answers.

**Overfitting the training seasons: no clear sign.** Early stopping is doing real
work and the fits are not running away:

| Diagnostic | Leaders | Whole suite |
|---|---|---|
| Selected epoch (mean) | 67–87 | 74 at cap 100, 114 at cap 300 |
| Fits hitting the epoch cap | 75 of 216 folds | 4% at cap 300 |
| Validation rebound after best epoch | 13–16% | 8.7% median |

Every fit selects an interior epoch by validation loss rather than training to
its budget, and at cap 300 only 4% of fits reach the cap at all — the models stop
because validation stops improving, not because they run out of epochs. The
train/validation gap at the selected epoch (~60% for the leaders) is *lower* than
the suite median (~64%), so the leaders are not the configurations fitting their
training data hardest.

The strongest evidence is the direct experiment: **raising the budget from 100 to
300 epochs does not help.** Across the six recipes where both caps were run, the
mean change is +0.009 and cap 300 is worse in 4 of 6. If the leaders were
underfitting, more epochs would help; if they were badly overfitting, more epochs
would hurt a lot. Neither happens — they are near the flat optimum where early
stopping is already finding the right place.

Note also that the leaders are *under*-dispersed, not over-confident in the
classic overfitting sense of memorised training points. Their failure mode is
intervals that are too narrow everywhere, including on the training seasons.

**Overfitting the model-selection process: yes, and it is the real risk.** This
is the version that should worry you. With 172 configurations ranked on three
outer seasons and a median seed SD of 0.120 against a top-10 spread of 0.058, the
apparent winner is substantially a draw from noise. "Rank 1 of 172" on this
evidence is not a reproducible claim — a fourth seed could reorder the top ten.
The defensible statement is that the independent-model cluster sits near
0.88–0.91, not that any single recipe is best.

A structural caveat reinforces this: the outer season CV has only three folds,
the seasons differ in which targets exist at all, and normalisers and stopping
exclude the held-out season but the *configuration choice* is informed by all
three. Selecting a winner on these scores and then reporting those same scores as
its performance would be optimistic. Treat the leaders as a shortlist to
re-evaluate, not as a measured ranking.

**Seed noise limits what can be concluded at the top.** The median seed SD is
0.120 while the whole top 10 spans 0.058. Within the leading group the ranking
is not identified by three seeds, and the leader's 0.883 should be read as "the
independent-model cluster sits near 0.88–0.91", not as a selected winner.

![Seed instability](figures/seed-instability.png)

Seed SD grows with the score: the collapsed configurations are both bad and
erratic, while the leaders are relatively stable (SD 0.019–0.047). That is
convenient — the configurations whose order matters are the ones measured most
precisely — but it does not rescue the within-top-10 ordering.

**States and US now agree.** The per-location scaling fix carried into B0.1:
the leaders score *better* at US (0.79–0.84) than at states/DC (0.91–0.93), and
the two rankings no longer point in opposite directions.

![States versus US](figures/states-vs-us.png)

![Skill by horizon](figures/horizon.png)

Skill is concentrated at the nowcast and decays outward, consistent with B0.0.

### What this implies for B1

1. **Fix dispersion before adding data.** Under-coverage is systematic across
   all 172 configurations and is the single largest lever on the score. A model
   that merely reached nominal coverage would likely beat every configuration
   here. Post-hoc calibration of the sample spread, or an explicit observation
   noise term, should be tried before any architectural elaboration.
2. **Drop or repair the stochastic trend decoder.** It is the largest effect in
   the experiment and it is negative everywhere it appears. If it is retained,
   run the missing `joint_trend` + legacy-decoder control first to separate the
   decoder from the rest of that recipe.
3. **Treat cross-target sharing as unsupported.** Independent per-target and
   per-pathogen models lead the table. Sharing has to earn its place against
   that baseline, and on this evidence it does not — which is a meaningful
   negative result for the B1 design, not just a null.
4. **Do not tune on differences below ~0.12.** More seeds, not more
   configurations, are what would resolve the top of the table. The epoch budget
   is settled: 100 is enough, and early stopping already finds the right epoch.
5. **Re-evaluate the shortlist rather than crowning a winner.** The leaders were
   selected on the same three seasons used to score them. Confirm the
   independent-model cluster on fresh seeds or a held-out season before treating
   any single recipe as best.

## Reproducing

The general procedure is in
[Postprocessing an experiment](../../workflows/experiment-postprocessing.md).

```bash
.venv/bin/python -m tapestry.models.manager rank -e B0.1
.venv/bin/python scripts/plot_b01_crosses.py -e B0.1 -r ranking-509b07b0d243
```

Ranking tables are in `data/experiments/B0.1/ranking-509b07b0d243/`
(`configuration_ranking.csv`, `run_scores.csv`, `season_scores.csv`,
`season_composite_scores.csv`, `manifest.json`). The per-configuration
calibration table behind the pairplot is saved alongside this page as
[`quality-metrics.csv`](quality-metrics.csv).
