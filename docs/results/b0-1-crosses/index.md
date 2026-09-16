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

| One-factor change | Mean Δ combined | n |
|---|---:|---:|
| `decoder_stochastic_trend` | **+1.259** | 4 |
| `head_sharing_target` | +0.101 | 3 |
| `lookback_26` | +0.100 | 6 |
| `exchange_joint_location_target` | +0.091 | 3 |
| `epochs_100` | +0.033 | 6 |
| `width_128` | +0.032 | 6 |
| `location_embedding_8` | −0.022 | 6 |
| `exchange_none` | −0.139 | 3 |
| `encoder_mlp` | −0.259 | 4 |
| `head_sharing_shared` | −0.626 | 4 |

Negative is better. Every exchange mechanism is at best neutral and
`exchange_none` is an improvement; the multiscale encoder is worse than a plain
MLP; sharing heads beats splitting them. The dominant term is an order of
magnitude larger than the rest and is not an architecture-of-interest at all: it
is the **decoder**.

**The stochastic trend decoder is the finding.** Swapping only the decoder onto
an otherwise unchanged reference costs +0.99 at `local_mlp`, +1.60 at
`spatial_conv` and +2.34 at `target_multiscale`. Combined with the pairplot's
collapsed arm — near-zero dispersion share, bias near −1 — this is not a tuning
problem to be fixed with a wider prior. The decoder is producing near-degenerate
sample trajectories, and `joint_trend` inherits it, which is the entire reason
that family occupies its own regime. No `joint_trend` variant with a legacy
decoder was run, so the specification confounds the trend decoder with the rest
of the `joint_trend` recipe; that control is the obvious missing cell.

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
   configurations, are what would resolve the top of the table.

## Reproducing

```bash
.venv/bin/python -m tapestry.models.manager rank -e B0.1
.venv/bin/python scripts/plot_b01_crosses.py -e B0.1 -r ranking-509b07b0d243
```

Ranking tables are in `data/experiments/B0.1/ranking-509b07b0d243/`
(`configuration_ranking.csv`, `run_scores.csv`, `season_scores.csv`,
`season_composite_scores.csv`, `manifest.json`). The per-configuration
calibration table behind the pairplot is saved alongside this page as
[`quality-metrics.csv`](quality-metrics.csv).
