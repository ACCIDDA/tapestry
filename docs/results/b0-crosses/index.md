# B0 calibration: the `crosses` experiment (post-scaling-fix)

**Experiment:** `b0-us-cross-4` · **Suite:** `crosses` · **Status:** complete —
60 configurations × 3 seeds = 180 season-CV runs, 540 season fits.
Fitted 2026-09-15 15:39–17:52 local on the patron GPU node `g1803jles01`
(4 × L40, six configuration lanes per card).

This is the rerun of the `crosses` calibration **after the per-location input
scaling fix** (commit `11a23ea`). The previous run is preserved, unchanged and
marked superseded, at
[Crosses calibration (legacy)](../b0-crosses-legacy/index.md); its fit artifacts
have been deleted, so it cannot be re-ranked.

It screens **one-factor changes around three pre-chosen reference recipes** — it
does not estimate interactions, and the references were design choices, not
winners picked from a sweep.

| Reference | Recipe | Combined score | Rank |
|---|---|---:|---:|
| `raw` | Historical raw-count B0, 8 weeks, no geography/dynamics | 1.0199 | 46 |
| `anchor` | Fourth-root, geography, dynamics, 12 weeks, shared MLP, latent 16 | 0.9701 | 24 |
| `conv` | Anchor + convolution, residual decoder, latent 32, spatial attention, local noise, separate heads, logit ED | 0.9215 | 4 |

## What the scaling fix changed

The fix normalizes each location by its own history instead of applying one
pooled scale. Under the old pooled scale the US sat at roughly 40× a state's
model units, which collapsed its intervals. The effect on the experiment is
large and is the main reason this page exists:

| | Legacy (`b0-crosses`) | This run (`b0-us-cross-4`) |
|---|---:|---:|
| Configurations beating the ensemble, US tasks | 6 of 51 | **5 of 60** |
| Best US score in the suite | 1.19 | **0.888** |
| Spearman ρ, states-only vs US-only ranking | −0.324 | −0.127 |
| US dispersion ratio vs ensemble | 0.819 | 0.868 |
| Leading recipe family | `conv` | **`anchor` (plain MLP)** |

The count of US-beating configurations barely moved, but the *magnitude* of the
US failure did: the leader now scores **0.888** on US tasks where the old
leader's family sat at 1.19–1.25. The US problem is no longer "every good
configuration is terrible nationally"; it is now concentrated in the `conv`
family, which still runs 1.16–1.37 at US while winning on states.

!!! warning "The two rankings still disagree, just less"
    ρ = −0.127 between the states-only and US-only orderings is closer to
    independent than to aligned. What helps states still does not reliably help
    the US; the fix removed a scaling artifact, not the underlying tension.

## Score definition

The selection score follows [architecture §10.3](../../design/architecture.md)
and is computed by `tapestry.evaluation.totals` from each run's `totals.csv`:

1. **Per target and season:** total model WIS ÷ total ensemble WIS over identical
   frozen tasks (every location including US, every reference date, horizons 0–3,
   on the hub's 23 quantiles). No per-task or per-location ratio is averaged.
2. **Per target:** mean of its season ratios, each season equal.
3. **Combined:** `(2 × (flu + COVID + RSV admissions) + (flu + COVID + RSV ED)) / 9`.
4. **Per configuration:** mean and seed SD over the three seeds.

**Lower is better; 1.0 means parity with the official hub ensemble.**

Ranking artifacts live in the experiment folder (not tracked in Git):
`data/experiments/b0-us-cross-4/ranking-2739af8db682/` —
`configuration_ranking.csv`, `run_scores.csv`, `season_scores.csv`,
`manifest.json`.

## Leaderboard

![Combined score for all 60 configurations, with individual seeds](figures/ranking-combined.png)

**37 of 60 configurations beat the official ensembles on the combined score**
(states/DC: 40 of 60; US: 5 of 60). The top six:

| Rank | Configuration | Combined (mean ± seed SD) | States/DC | US |
|---:|---|---:|---:|---:|
| 1 | `anchor__stopping_300_0` | 0.8924 ± 0.0327 | 0.9062 | **0.8879** |
| 2 | `conv__decoder_leg` | 0.9069 ± 0.0233 | 0.9003 | 1.1895 |
| 3 | `anchor__stopping_100_0` | 0.9150 ± 0.0366 | 0.9351 | 0.9374 |
| 4 | `conv` (reference) | 0.9215 ± 0.0480 | 0.8988 | 1.2507 |
| 5 | `conv__us_error_shf` | 0.9291 ± 0.0674 | 0.9282 | 1.1770 |
| 6 | `anchor__stopping_300_20` | 0.9293 ± 0.0269 | 0.9593 | 0.9824 |

Only rank 1 and rank 3 are below parity on *both* geographies. Both are
`anchor` with longer training and nothing else changed.

!!! warning "The leading group is not separated"
    The top-10 range is 0.0606 against a median seed SD of 0.0287 — barely two
    seed SDs across ten configurations. Three seeds cannot resolve this.

![Seed noise against leaderboard spread, and per-seed rank reshuffling](figures/seed-instability.png)

## The headline result is a season-specific overfit

`anchor__stopping_300_0` wins the combined score, but **not** because longer
training is uniformly better. Its per-season means, averaged over targets:

| Configuration | 2023-2024 | 2024-2025 | 2025-2026 |
|---|---:|---:|---:|
| `anchor` (50 epochs) | 0.961 | 0.949 | 1.008 |
| `anchor__stopping_100_0` | 1.229 | 0.865 | 0.920 |
| `anchor__stopping_300_0` | **1.688** | **0.832** | **0.860** |
| `anchor__stopping_300_20` | 0.946 | 0.778 | 1.008 |

Longer training buys real gains on 2024-2025 and 2025-2026 and **blows up on
2023-2024 flu**, the only target that season contributes. On influenza
admissions 2023-2024 the ladder runs 0.961 → 1.229 → 1.688 as epochs increase,
monotonically worse, and consistently so across seeds (1.539 / 1.875 / 1.650).

Because 2023-2024 contributes one target of nine weighted cells while the other
two seasons contribute the rest, the combined score rewards the trade. **Adding
early stopping removes the collapse** (`stopping_300_20`: 0.946 on 2023-2024)
and gives up much of the gain elsewhere, landing at rank 6.

!!! danger "Do not read rank 1 as 'train longer'"
    The honest reading is: 300 epochs without early stopping overfits to the
    seasons that dominate the objective. `anchor__stopping_300_20` — 300 epochs
    *with* patience 20 — is the configuration that is good everywhere without a
    catastrophic season, and it is the safer default despite ranking 6th.
    Patience was set to 0 in the references, so this is a defect of the
    reference design, not of long training as such.

![Training-length ladder by reference family](figures/epoch-ladder.png)

The ladder also **reverses between families**, which is the clearest
interaction this one-factor screen exposes:

| Family | ep50 | ep100 | ep300 | ep300 + patience 20 |
|---|---:|---:|---:|---:|
| `raw` | 1.0199 | 1.0391 | 1.0360 | 1.0202 |
| `anchor` | 0.9701 | 0.9150 | **0.8924** | 0.9293 |
| `conv` | **0.9215** | 0.9666 | 0.9939 | 0.9530 |

Longer training is the best change to `anchor` (−0.078), the *worst* structural
change to `conv` (+0.072), and does nothing for `raw`. The small MLP is
underfitted at 50 epochs; the convolutional model with a residual decoder and
latent 32 is already at capacity and degrades.

## One-factor effects

Deltas are against each configuration's **own** reference; negative is better.

### On `anchor` (0.9701) — the productive reference

| Change | Δ combined | Δ states | Δ US |
|---|---:|---:|---:|
| `stopping_300_0` | **−0.0777** | −0.0811 | −0.1773 |
| `stopping_100_0` | −0.0550 | −0.0522 | −0.1277 |
| `stopping_300_20` | −0.0408 | −0.0280 | −0.0827 |
| `us_error_shf` | −0.0256 | −0.0211 | −0.0430 |
| `decoder_res2` | −0.0183 | −0.0193 | −0.0461 |
| `ed_transform_4rt` | −0.0097 | −0.0149 | −0.0627 |
| `count_transform_raw` | +0.0214 | +0.0153 | −0.0608 |
| `count_transform_rate` | +0.0205 | +0.0144 | −0.0617 |

### On `conv` (0.9215) — almost nothing helps

| Change | Δ combined | Δ states | Δ US |
|---|---:|---:|---:|
| `decoder_leg` | **−0.0145** | +0.0015 | −0.0612 |
| `us_error_shf` | +0.0076 | +0.0294 | −0.0737 |
| `encoder_mlp` | +0.0303 | +0.0415 | −0.0707 |
| `heads_sh` | +0.0458 | +0.0699 | **−0.1651** |
| `geography_0` | +0.0997 | +0.0755 | +0.1226 |
| `ed_transform_lin` | +0.1024 | +0.1014 | −0.0387 |
| `dynamics_0` | +0.1150 | +0.1539 | +0.0553 |
| `ed_transform_4rt` | +0.1159 | +0.1331 | −0.1176 |

Only *simplifying* the decoder helps `conv`. Every representation change hurts
it, and its logit ED transform is load-bearing: reverting to linear costs +0.102.

### On `raw` (1.0199) — representation still dominates

| Change | Δ combined | Δ states | Δ US |
|---|---:|---:|---:|
| `lookback_12` | **−0.0558** | −0.0365 | −0.0962 |
| `latent_32` | −0.0414 | −0.0372 | −0.0647 |
| `count_transform_log1p` | −0.0346 | −0.0199 | −0.0225 |
| `count_transform_4rt` | −0.0309 | −0.0217 | −0.0122 |
| `encoder_conv` | +0.0287 | +0.0275 | +0.0579 |
| `noise_loc` | +0.0313 | +0.0260 | +0.0385 |

**Effects are not transferable between references.** `noise_loc` is the worst
change to `raw` (+0.031) and helps `anchor` (−0.006). `encoder_conv` hurts
`raw` (+0.029) and `anchor` (+0.008), yet the `conv` reference — which bundles
convolution with four other changes — is rank 4. `count_transform_raw` and
`count_transform_rate` hurt the combined score on `anchor` while *improving* its
US score by ~0.06. This is why the suite crosses three recipes rather than one.

Family placement, by rank: `anchor` median 18 (range 1–35), `conv` median 31.5
(2–55), `raw` median 46.5 (16–60).

## Per-season ranking, and the ensemble placed among the configurations

The combined score averages three seasons that do not contain the same targets:
2023-2024 has influenza admissions only, 2024-2025 adds COVID admissions, and
only 2025-2026 has all six. Scoring each season separately shows how much of the
ranking survives that aggregation.

![Weighting comparison and per-season ranks](figures/weighting-and-seasons.png)

Score, with rank within that season in brackets (60 configurations):

| Configuration | 2023-2024 | 2024-2025 | 2025-2026 | Overall |
|---|---|---|---|---|
| `anchor__stopping_300_0` | 1.688 (51) | 0.832 (7) | 0.837 (1) | 0.8924 (1) |
| `conv__decoder_leg` | 1.145 (21) | 0.811 (3) | 0.925 (3) | 0.9069 (2) |
| `anchor__stopping_100_0` | 1.229 (30) | 0.865 (13) | 0.889 (2) | 0.9150 (3) |
| `conv` | 0.966 (10) | 0.865 (12) | 0.969 (28) | 0.9215 (4) |
| `conv__us_error_shf` | 1.107 (20) | 0.846 (8) | 0.962 (17) | 0.9291 (5) |
| `anchor__stopping_300_20` | 0.946 (5) | 0.778 (1) | 0.960 (14) | 0.9293 (6) |
| `anchor__us_error_shf` | 0.985 (12) | 0.848 (11) | 0.965 (22) | 0.9445 (7) |
| `anchor__decoder_res2` | 0.879 (1) | 0.905 (21) | 0.960 (15) | 0.9518 (8) |
| `conv__encoder_mlp` | 1.213 (27) | 0.830 (6) | 0.967 (23) | 0.9518 (9) |
| `conv__stopping_300_20` | 1.172 (26) | 0.848 (10) | 1.014 (45) | 0.9530 (10) |
| `anchor__ed_transform_4rt` | 0.896 (2) | 0.976 (45) | 0.972 (30) | 0.9604 (11) |
| `conv__count_transform_log1p` | 1.514 (39) | 0.925 (25) | 0.927 (4) | 0.9613 (12) |

**Season ranks are essentially uncorrelated with each other.** Spearman ρ between
season rankings: 2023-24 vs 2024-25 **−0.016**, 2023-24 vs 2025-26 **0.130**,
2024-25 vs 2025-26 **0.024**. Each correlates with the overall rank only because
each contributes to it (0.57, 0.40, 0.66). A configuration's rank in one season
carries almost no information about its rank in another.

### Where the ensemble falls

The hub ensemble scores 1.0 by construction, so it can be inserted directly into
each ranking:

| Ranking | Ensemble position | Configurations beating it |
|---|---:|---:|
| 2023-2024 | 15 of 61 | 14 of 60 |
| 2024-2025 | **51 of 61** | 50 of 60 |
| 2025-2026 | 41 of 61 | 40 of 60 |
| Overall (WIS-sum) | 38 of 61 | 37 of 60 |
| Overall (equal-weight 52) | **10 of 61** | **9 of 60** |

The ensemble is easy to beat in 2024-2025 and hard to beat in 2023-2024 — the
same season asymmetry that drives the overfit above. **Only 11 of 60
configurations beat the ensemble in all three seasons**, and none of the top
three do it:

| Configuration | 2023-2024 | 2024-2025 | 2025-2026 | Overall |
|---|---:|---:|---:|---:|
| `conv` | 0.966 | 0.865 | 0.969 | 0.9215 |
| `anchor__stopping_300_20` | 0.946 | 0.778 | 0.960 | 0.9293 |
| `anchor__us_error_shf` | 0.985 | 0.848 | 0.965 | 0.9445 |
| `anchor__decoder_res2` | 0.879 | 0.905 | 0.960 | 0.9518 |
| `anchor__ed_transform_4rt` | 0.896 | 0.976 | 0.972 | 0.9604 |
| `anchor__latent_32` | 0.993 | 0.964 | 0.948 | 0.9615 |
| `anchor__heads_su` | 0.943 | 0.941 | 0.964 | 0.9669 |
| `anchor__dynamics_0` | 0.911 | 0.995 | 0.964 | 0.9671 |
| `anchor` | 0.961 | 0.949 | 0.974 | 0.9701 |
| `anchor__lookback_8` | 0.964 | 0.978 | 0.969 | 0.9750 |
| `anchor__encoder_conv` | 0.947 | 0.956 | 0.980 | 0.9781 |

Nine of the eleven are `anchor` variants, and `anchor__stopping_300_20` — the
patience configuration — is the best of them. **This list is a better shortlist
than the combined-score top three**, because every member is above the ensemble
everywhere rather than on average.

## Weighting the US as one location among 52

The combined score sums WIS without reweighting by location, so the US carries
**47.2%** of influenza-admissions WIS despite being 1 location in 52. Scoring
each location separately and averaging the 52 per-location WIS ratios equally
gives the US a 1.9% share instead. Computed by
`scripts/score_b0_per_location.py`; the states/DC and US sums it produces
reproduce `season_scores.csv` to 1e-14.

| Configuration | Equal-weight 52 (± seed SD) | Rank | WIS-sum | Rank | Shift |
|---|---|---:|---:|---:|---:|
| `conv__decoder_leg` | 0.9591 ± 0.0201 | 1 | 0.9069 | 2 | +1 |
| `conv` | 0.9601 ± 0.0253 | 2 | 0.9215 | 4 | +2 |
| `anchor__stopping_300_0` | 0.9620 ± 0.0255 | 3 | 0.8924 | 1 | −2 |
| `conv__count_transform_log1p` | 0.9832 ± 0.0500 | 4 | 0.9613 | 12 | +8 |
| `conv__stopping_300_20` | 0.9844 ± 0.0370 | 5 | 0.9530 | 10 | +5 |
| `anchor__stopping_100_0` | 0.9870 ± 0.0177 | 6 | 0.9150 | 3 | −3 |
| `conv__stopping_300_0` | 0.9938 ± 0.0323 | 7 | 0.9939 | 36 | **+29** |
| `conv__us_error_shf` | 0.9957 ± 0.0702 | 8 | 0.9291 | 5 | −3 |
| `conv__stopping_100_0` | 0.9996 ± 0.0661 | 9 | 0.9666 | 18 | +9 |
| `conv__encoder_mlp` | 1.0006 ± 0.0270 | 10 | 0.9518 | 9 | −1 |

Two things change. **Every score moves toward 1** — the suite's apparent margin
over the ensemble was substantially a US effect, and only 9 of 60 configurations
beat the ensemble once locations are equally weighted, against 37 under the
WIS-sum. And the `conv` family rises: it holds 7 of the top 10 here against 4
under the WIS-sum, because its state-level strength is no longer diluted by the
US penalty. Spearman between the two orderings is 0.844, so the rankings agree
broadly, but `conv__stopping_300_0` moves 29 places.

Per-season, equal-weight-52:

| Configuration | 2023-2024 | 2024-2025 | 2025-2026 | Overall |
|---|---|---|---|---|
| `conv__decoder_leg` | 1.205 (28) | 0.836 (2) | 0.968 (3) | 0.9591 (1) |
| `conv` | 0.986 (7) | 0.875 (9) | 1.000 (7) | 0.9601 (2) |
| `anchor__stopping_300_0` | 1.774 (60) | 0.828 (1) | 0.908 (1) | 0.9620 (3) |
| `conv__count_transform_log1p` | 1.376 (39) | 0.928 (27) | 0.962 (2) | 0.9832 (4) |
| `conv__stopping_300_20` | 1.160 (25) | 0.872 (7) | 1.025 (22) | 0.9844 (5) |
| `anchor__stopping_100_0` | 1.257 (34) | 0.881 (10) | 0.969 (4) | 0.9870 (6) |

Under equal weighting `anchor__stopping_300_0` ranks **60th of 60** in 2023-2024
while ranking 1st in both other seasons — the overfit is starker here than under
the WIS-sum. **No configuration beats the ensemble in all three seasons under
equal weighting** (against 11 under the WIS-sum), which is the most sobering
number on this page.

!!! note "Which weighting is right depends on the question"
    The WIS-sum is the §10.3 selection score and is the right answer if national
    forecasts matter proportionally to their magnitude. Equal weighting is the
    right answer if each jurisdiction's forecast matters equally. They disagree
    enough (ρ = 0.844, one 29-place move) that a chosen model should be checked
    under both, and this page now reports both rather than picking.

## Geography

| Target | US | States/DC |
|---|---:|---:|
| Flu admissions | 1.108 | 1.027 |
| COVID admissions | 0.960 | 0.971 |
| RSV admissions | 0.797 | 0.863 |
| Flu ED visits | 0.981 | 0.894 |
| COVID ED visits | 2.162 | 1.330 |
| RSV ED visits | 0.978 | 0.800 |

![States/DC versus US combined score, by family](figures/states-vs-us.png)

The US is **one location in 52** but still carries roughly **42–47% of all
admissions WIS** (flu 47.2%, RSV 43.6%, COVID 41.9%), because its per-task WIS
is far larger than a state's. The selection score sums WIS without reweighting
by location, so for admissions the headline ranking remains close to half a
national ranking. ED targets are proportions, so US carries only 1.3–1.8% there.

Per-target win counts across the 60 configurations: RSV admissions 60/60, RSV ED
60/60, flu ED 58/60, COVID admissions 41/60, flu admissions 16/60, **COVID ED
0/60** (mean ratio 1.338). COVID ED visits remains a systematic failure, as it
was in the legacy run.

## Calibration is still the outstanding defect

![Coverage by target and geography against nominal levels](figures/coverage.png)

Coverage is short of nominal nearly everywhere, averaged over all 180 runs:

| Target | 50% states (model/ens) | 95% states | 50% US | 95% US |
|---|---|---|---|---|
| Flu admissions | 36.7 / 51.2 | 77.7 / 88.0 | 39.5 / 49.7 | 80.3 / 86.8 |
| COVID admissions | 38.5 / 52.3 | 79.3 / 94.2 | 42.8 / 60.7 | 87.6 / 97.6 |
| RSV admissions | 38.7 / 42.8 | 77.5 / 88.9 | 41.8 / 49.1 | 84.0 / 92.2 |
| Flu ED visits | 40.7 / 53.9 | 84.9 / 90.3 | 41.7 / 67.0 | 78.0 / 89.3 |
| COVID ED visits | 34.1 / 52.0 | 80.8 / 93.5 | 31.6 / 78.2 | 82.6 / 100.0 |
| RSV ED visits | 46.2 / 48.1 | 88.2 / 89.4 | 45.0 / 65.8 | 82.2 / 97.3 |

Dispersion ratios are 0.903 (states/DC) and 0.868 (US) — intervals are roughly
the right order of width but ~10–13% too narrow, so the shortfall is placement
plus mild overconfidence rather than the gross collapse the legacy reference B0
showed (2.7–5.7% at the 50% level). **No calibration has been applied**, and the
saved early-stopping validation forecasts remain available for one.

## Skill by horizon

![Total WIS ratio by horizon](figures/horizon.png)

The legacy run reported skill *improving* with horizon. **That has reversed.**
`anchor__stopping_300_0` runs 0.861 → 0.902 → 0.936 → 0.999 across horizons 0–3
on states/DC, and 0.781 → 0.917 → 0.969 → 1.042 at US: strongest at the nowcast,
decaying to parity by four weeks. Pooled over all 60 configurations the profile
is nearly flat (states 1.015/1.001/0.997/1.012; US 1.059/1.059/1.053/1.069).

This is the expected pattern and the opposite of the legacy result, which
pointed at the nowcast as the defect. After the scaling fix the nowcast is the
model's strength and the **dynamics** are what fade — a different place to look
next. `conv__decoder_leg` is the exception, flat at 0.89–0.96 across all four
horizons on both geographies.

## Fan plots

One figure per season and target. **United States on the left, North Carolina on
the right.** Rows are the three best configurations at their median-scoring seed
(`anchor__stopping_300_0` s44, `conv__decoder_leg` s44,
`anchor__stopping_100_0` s44) and the hub ensemble. Every third forecast origin
is drawn; truth is the frozen black line.

### Influenza admissions

![Influenza admissions 2023-2024](figures/fans-flu_hosp-2023-2024.png)

![Influenza admissions 2024-2025](figures/fans-flu_hosp-2024-2025.png)

![Influenza admissions 2025-2026](figures/fans-flu_hosp-2025-2026.png)

The 2023-2024 panel is where rank 1 fails; compare its fans against
`conv__decoder_leg` in the same figure.

### COVID-19 admissions

![COVID-19 admissions 2024-2025](figures/fans-covid_hosp-2024-2025.png)

![COVID-19 admissions 2025-2026](figures/fans-covid_hosp-2025-2026.png)

### RSV admissions

![RSV admissions 2025-2026](figures/fans-rsv_hosp-2025-2026.png)

### ED visits, 2025-2026

![Influenza ED visits 2025-2026](figures/fans-flu_prop_ed_visits-2025-2026.png)

![COVID-19 ED visits 2025-2026](figures/fans-covid_prop_ed_visits-2025-2026.png)

![RSV ED visits 2025-2026](figures/fans-rsv_prop_ed_visits-2025-2026.png)

Rendered from saved `forecasts.npz` via `tapestry.evaluation.hubs.export_b0`
against the frozen truth, not through `manager compare`. EpiBench relative WIS
and its standard diagnostic plot set are **not** included here; the §10.3
selection score is the total-WIS ratio `rank` already computes.

### Member trajectories

Fans hide what individual ensemble members do. The same rank-1 configuration,
drawn as 50 of its 100 saved members per origin:

![Member trajectories, influenza admissions 2024-2025](figures/trajectories-flu_hosp-2024-2025.png)

Influenza 2024-2025 is chosen because it contains the within-season rebound and
because rank 1 scores well there (0.819), so this is not a strawman season. Two
things are visible that the quantile fans do not show. Members are **tight and
nearly parallel** — they share one episode-level latent, so they disagree about
level far more than about shape, which is the narrow-interval result above seen
directly. And at the February origins the whole bundle points **down** into the
second peak while truth climbs past it, at both US and NC.

### The double-peak failure persists

Measured on influenza 2024-2025 over every origin where truth fell for a week
then rose more than 10% within four weeks (183 episodes, 182 scored, only one
of them national), comparing the rank-1 configuration at its median seed:

| Influenza 2024-2025 rebounds, states/DC | B0 leader | Hub ensemble |
|---|---:|---:|
| Predicted *down* into the rebound | 85.7% | 84.0% |
| Four-week truth inside the 95% interval | **50.0%** | **68.1%** |
| Median miss at four weeks (admissions) | 28 | 13 |

Both models call the direction down at essentially the same rate, so this is not
a distinctive directional defect. The gap is **coverage and magnitude**: B0
covers the rebound outturn 50% of the time against the ensemble's 68%, with
roughly double the median miss. That is the calibration defect above, showing up
where it costs most, rather than a separate phenomenon.

## Against the wider hub field

The ensemble is not the strongest hub model, so beating it is a weaker claim than
it sounds. Placing `anchor__stopping_300_0` (mean per-task WIS over seeds)
against every other model on identical tasks:

| Task | B0 position | Ensemble position | Field |
|---|---:|---:|---:|
| Flu admissions 2023-2024 | **31** | 7 | 40 |
| Flu admissions 2024-2025 | **7** | 19 | 52 |
| Flu admissions 2025-2026 | **9** | 24 | 60 |
| Flu ED visits 2025-2026 | 8 | 9 | 19 |
| COVID admissions 2024-2025 | 6 | 6 | 20 |
| COVID admissions 2025-2026 | **5** | 7 | 19 |
| COVID ED visits 2025-2026 | 5 | 3 | 7 |
| RSV admissions 2025-2026 | **2** | 4 | 9 |
| RSV ED visits 2025-2026 | **2** | 3 | 6 |

Placing 7th of 52 and 9th of 60 in the large flu fields, and 2nd of 9 on RSV
admissions, is a real result for a six-channel model with no disease-specific
structure. The 2023-2024 flu placement (31st of 40) is the same overfit
documented above.

!!! note "Not the same B0 as the frozen leaderboards"
    The `Tapestry-B0-finalized-CV` entry inside
    `data/evaluation/b0_hub_comparison_q23/*/leaderboard.csv` is the **older
    reference run** with collapsed intervals (50% coverage of 1–6%). It is not a
    `crosses` configuration; do not read those rows as describing this
    experiment.

## Takeaways

**Model formulation.** Representation still dominates architecture, but the
scaling fix changed which architecture wins. A plain MLP (`anchor`) with correct
per-location scaling and enough training now beats the convolutional model with
spatial attention, residual decoder and latent 32 — and the `conv` family's only
improvement is *removing* its residual decoder. Depth and attention are not
paying for themselves at this data size. The reference-dependence of every
effect (`noise_loc`, `encoder_conv`, `count_transform_*` all flip sign between
references) means one-factor screens around a single recipe would have misled;
keep crossing at least two contrasting references.

**Training regime.** This is the largest single lever in the suite (−0.078 on
`anchor`) and the most dangerous. 50 epochs underfits the MLP; 300 epochs
without early stopping overfits to the two seasons carrying most of the
objective and destroys 2023-2024 flu (1.688). Early stopping with patience 20
fixes that season at the cost of the gains. **Patience 0 in all three references
was a design error** — the ladder should be rerun with patience as the default
and epochs as the free parameter, not the reverse.

**Lookback windows.** Small and reference-dependent, and no longer a
simplification story. 8 → 12 weeks is the best change to `raw` (−0.056), but
12 → 8 weeks is mildly *harmful* to `anchor` (+0.005) and clearly harmful to
`conv` (+0.050). Twelve weeks is the right default; there is no evidence here
for going longer, and 8 weeks is only adequate when the model has no geography
or dynamics features to use the extra context.

**Importance of seed.** Decisive, and the main constraint on every claim above.
Median seed SD is 0.0287 against a top-10 range of 0.0606 and a top-3 range of
0.0227 — the leading group is inside the noise. Per-seed orderings agree only at
Spearman ρ = 0.28–0.48 pairwise; on seed 43 alone the winner would be
`conv__us_error_shf`, on seeds 42 and 44 it is an `anchor__stopping_*`. Rank 1's
own seeds span 0.855–0.915. **Three seeds are not enough to rank configurations
at this resolution**; any follow-up that intends to select a model needs more
seeds, or differences larger than ~0.06, or paired per-seed contrasts rather
than mean ranks.

**Geography.** The scaling fix removed a genuine artifact — the best US score
went from 1.19 to 0.888 — but the states/US tension is structural, not fixed.
ρ = −0.127 between the two orderings, and the `conv` family still pays 1.16–1.37
at US to win on states.

**The mechanism is not aggregation.** The US is location 52 on the same axis as
the states: it is encoded, decoded and fit directly, with its own `is_us`
geography flag, and nothing in the forward pass sums states into a national
number. The US *mean* is fit like any other location.

Measured on the saved members (flu admissions, horizon 3, 2024-2025, seed 42,
100 members per origin), the defect is interval **width**, not correlation:

| Configuration | US 95% width | Width if states were summed | Ratio | US 95% coverage |
|---|---:|---:|---:|---:|
| `anchor` (reference) | 12,012 | 15,863 | 1.32 | 68.8% |
| `anchor__us_error_shf` | 11,470 | 12,508 | **1.09** | 60.4% |
| `anchor__stopping_300_0` (rank 1) | 4,927 | 11,863 | **2.41** | 47.9% |
| `conv` | 5,561 | 10,872 | 1.95 | 66.7% |

Member deviations are in fact *highly* correlated across locations — mean
corr(US, state) is 0.98 on the `anchor` reference — so the earlier hypothesis
that state errors cancel is wrong for that recipe. What distinguishes the
configurations is how much total spread they retain: rank 1's US interval is
**2.4× narrower** than the same model's own states imply, and its member
correlation drops to 0.56. Long training without early stopping shrinks the
ensemble spread, and it shrinks it hardest where coverage was already worst.
`us_error_shf` is the only switch that pulls the ratio back toward 1 (1.09),
which is consistent with its design and with its US gains (−0.043 on `anchor`,
−0.074 on `conv`).

**Where to look next**, in priority order: (1) rerun the stopping ladder with
patience on by default, since the current rank 1 is an artifact of its absence;
(2) apply interval calibration using the saved early-stopping validation
forecasts, which already exist and are unused — coverage is 35–46% at the 50%
level against a nominal 50; (3) more seeds before any further selection;
(4) COVID ED visits, which no configuration in either run has ever beaten.

## Assumptions and limitations

- **Exploratory, not validation.** All three seasons inform development, and the
  first two folds have later seasons in the fitting set. These are finalized
  retrospective CV results on assumed-truth NSSP values.
- **Selection on the same folds** used for ranking; the leaders are chosen on the
  data that scores them. The 2023-2024 overfit is visible *because* that season
  is in the objective, and would not have been caught by a held-out design.
- **Three seeds** cannot resolve the differences separating the leading group.
- **One-factor screen only.** Interactions were explicitly out of scope; the
  family-reversing effects documented above are evidence that interactions
  matter, not measurements of them.
- **Season coverage is unequal.** 2023-2024 contributes only influenza
  admissions; 2024-2025 adds COVID admissions; only 2025-2026 has all six
  targets. A season's influence on the combined score therefore varies, which is
  what makes the 2023-2024 trade profitable.
- **Provenance:** all 180 runs carry commit `11a23ea` (the scaling fix) or later.
  The five recorded commit/dirty states span `11a23ea` clean and dirty,
  `91dd267` dirty, `a77ee28` and `5d5eade`; the three later commits touch only
  `docs/`, `README.md` and `AGENTS.md`, so **no run differs in model code**.
  Verified by inspecting each commit's diffstat. Per architecture policy,
  mixed/dirty commits are acceptable only during exploration — paper results must
  be rerun clean.
- **`compare` was skipped by choice**, so this summary uses the total-WIS `rank`
  path only. EpiBench relative WIS and the standard diagnostic plots are not
  included. This limits what can be said *alongside* the ranking, not the
  ranking itself.
- **The rebound analysis uses one seed** (the median-scoring s44) of one
  configuration, not the seed mean, and horizon 3 only.

## Log

**2026-09-15 — state scaling fix and rerun.** The pooled input scale put the US
at ~40× a state's model units and collapsed its intervals. Fixed in `11a23ea` by
normalizing each location by its own history. `b0-us-cross-4` reran the full
crosses suite (60 configurations, three seeds) on that code. The legacy
`b0-crosses` page is retained as a superseded record; its fit artifacts were
deleted to reclaim disk.

Headline changes from the rerun: the leading family moved from `conv` to
`anchor`, best US score improved from 1.19 to 0.888, horizon profile reversed
(now best at the nowcast), and the winning configuration turns out to be a
season-specific overfit caused by `patience=0` in the reference designs.

## Reproducing

```bash
.venv/bin/python -m tapestry.models.manager status -e b0-us-cross-4
```

```bash
.venv/bin/python -m tapestry.models.manager rank -e b0-us-cross-4
```

```bash
.venv/bin/python scripts/plot_b0_crosses.py -e b0-us-cross-4 -r ranking-2739af8db682
```

`rank` regenerates `ranking-2739af8db682/` (it warns about the mixed commit
states discussed above). `plot_b0_crosses.py` writes every figure on this page
from the ranking artifacts plus each run's saved `forecasts.npz`; pass
`--skip-fans` for the summary figures alone. To add EpiBench relative WIS and
the standard plot set later, run `manager compare -e b0-us-cross-4` for a
shortlist from `rank`; R needs to be on `PATH` first
(`module load r/4.5.0`, or
`export PATH=/nas/longleaf/rhel9/apps/r/4.5.0/bin:$PATH`).
