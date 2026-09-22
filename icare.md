# Decisions and context the user cares strongly about

Read this file when planning, analyzing or changing the project. Record explicit
user priorities faithfully, with a date and enough context to apply them. Keep
user-provided premises distinct from measured results and hypotheses.

## NHSN reporting regimes — 2026-09-18

The user explicitly requires distinguishing these periods in NHSN analyses:

- **Before May 1, 2024:** a previous reporting regime.
- **May 1–October 31, 2024:** a pause in mandatory NHSN reporting; contributions
  were voluntary. Coverage is poor, and this is a bad example from which to
  learn the current reporting process.
- **From November 1, 2024:** a new reporting mandate. Treat this as a distinct
  regime when interpreting reporting coverage, revisions and nowcasting.

**User clarification:** “data in final form from November 2024 on” means
**use November 2024 onward as the relevant reporting regime**. It does not mean
those reports are revision-free. Keep earlier and voluntary-period observations
separate when drawing conclusions about nowcasting in the current regime.

**Intended use:** the model is for the next season. The relevant NHSN
nowcasting process is the regime beginning November 1, 2024. Judge the value of
nowcasting for that use on post-November-2024 genuine reports; do not use pooled
previous-regime results as the deciding evidence. Assuming the next season
continues this regime is the user's stated modeling premise, not a guarantee
about future reporting policy. This does not require discarding older epidemic
history from forecast training or treating NSSP as subject to the NHSN mandate.

These dates and descriptions are user-provided analysis premises. For descriptive
weekly plots, label periods by observation week-ending date, identify boundary
weeks as approximate, and do not apply NHSN policy labels to NSSP as its own policy.

**Scientific implication to examine:** training across reporting regimes may
hurt transfer of nowcasting models. This is a hypothesis, not an established
explanation of B1's weaker results. Compare the latest two report ages across
seasons and regimes, distinguish missing vintage archives from actual reporting
participation, and separate genuine preliminary reports from supplied finals.
Do not conclude that nowcasting is unhelpful from pooled cross-regime performance.

## Forward formulation benchmark — 2026-09-18

User decision: compare Direct B, joint gated forecasting/nowcasting, and an
independently trained nowcast-to-forecast pipeline for next-season use. Start
with target MLP, no artificial masking, matched forecast tasks/seeds/budgets,
and no large sweep. Retain older forecasting history without allowing earlier
NHSN regimes to supervise the current revision process. NSSP has its own vintage
availability. Distinguish visible revisions, missing-report reconstruction and
missing archives, particularly at four versus eleven days.

User requires a historical cutoff, frozen parameters through 2025–26,
training-only transformations/calibration, and no later-final filling of test
inputs. This is forward development because that season was already explored.
The pipeline's exact-training/estimated-deployment mismatch must be explicit;
sampling alone does not fix it. Predeclare near-zero denominator handling and
report stable scaled errors, coverage, scientific weighting and fitting
variability without treating seeds or places as independent seasons.

Implementation assumptions and exact dates are in
[the data specification](docs/data/forward-2025.md): July 26, 2025 UTC
cutoff and 28-day mature cutoff-final proxies. These are implementation
choices, not user assertions of certified finality. The user explicitly requested
acquiring missing archives; both Delphi acquisitions completed and passed checksum
verification. The previously prepared B1 data belongs to a different information
regime and remains available for its original retrospective analyses.

**Scope update from the user, September 18, 2026:** finish the data and return a
summary; defer all model work to a future step. This application needs revision
pairs only for the past two observation weeks at each Wednesday issuance
(four-day and eleven-day ages). Do not launch training or scoring in this step.

**Scope resumed by the user, September 18, 2026:** implement and run the three-way
forward benchmark using the completed paired data. Training and scoring are now
authorized. The prior data-only stop applied to the previous step. Exact fitting,
scoring and denominator decisions are predeclared in
[the experiment specification](docs/workflows/forward-2025.md). Fixed 100 epochs
and three seeds are implementation choices, not user-specified optimal budgets.

**Concurrency preference, September 18, 2026:** the user requests running all
benchmark runs concurrently and an explicit option to parallelize seeds. The
manager local runner now offers `--parallel-seeds` under its `--fit-workers` cap;
the shared Slurm dispatcher already schedules seeds independently. Worker count
controls actual concurrency. The benchmark was expanded from three workers on
one GPU to nine workers across two GPUs without changing fits or budgets.

## Validation-season question — 2026-09-19

The user asks whether 2025–26 is particularly bad for validation/coverage and
requests ways to improve coverage. This is a question to investigate, not a
user assertion that the season is anomalous or permission to discard it.
No new model was trained for that answer.

**Publication decision, September 19, 2026:** the user requests removing the
coverage-diagnosis and proposed-experiments page and publishing the completed
forward experiment with a clear place in the menu. The supplementary page,
its standalone season-comparison export and helper script were removed. The
benchmark retains its measured coverage results and has a dedicated
**Forward 2025–26** navigation section.

## One dataset array and one score — 2026-09-22

**User decisions, September 22, 2026:**

- **One dataset array.** `data/processed/panel.npz` (one weekly Saturday
  calendar, truth panel, and a Wednesday as-of overlay) replaces
  `finalized.npz` + `vintaged.npz`; lookback is chosen per scenario, not at
  build time. Vintaged episodes must see exactly what the previous vintaged
  builder gave them (as-of targets for the two latest context weeks).
- **Exactly one score.** Per target and season, the mean of per-location WIS
  ratios to the hub ensemble, states sharing 80% and the US 20% (the US share
  is an experiment setting, default 0.2), targets combined (2 x admissions +
  ED) / 9, seasons equal. The pooled total-WIS ratio is no longer a competing
  score; this supersedes the earlier total-WIS-ratio selection preference.
- **Cross-validation must not leak.** The refactor's episode-level split let
  training episodes carry held-out-season labels and kept validation weeks in
  inputs; the user asked to restore the pre-refactor policy by masking the
  array.

Implementation choice flagged, not a user decision: as-of covariates are kept
for 52 context weeks (not only the latest two) because the previous vintaged
builder resolved every covariate week as of the issuance. Details and the
decision log: [the unified design](docs/design/restructure-2026-unified.md).

## Exact as-of data, CV and score settings, figures — 2026-09-22 (later)

**User decisions, September 22, 2026:**

- **Exact as-of store.** For every Wednesday issuance, keep the value visible at
  its cutoff for *every* week from the calendar start, targets and covariates
  (state and national) alike, so any issuance's forecast can be reconstructed
  exactly as it was. Keep the file small. The user suggested float16.
- **How many recent context weeks are as-of is a scenario field**
  (`asof_weeks`); default 2 reproduces old B1; >= lookback is fully as-of.
- **CV settings belong in the scenario** (they change fits and run ids).
- **Score weights are rank-time options** (`rank --us-weight`), recorded in the
  ranking folder, not scenario and not `experiment.json`. This supersedes the
  earlier entry's "US share is an experiment setting". Scoring uses only each
  fold's held-out score episodes.
- **Four figures and nothing else:** CV layout, US/NC fans (1-2 configurations
  + hub ensemble), per-seed pairplot relative to the ensemble, location x
  season heatmap. Code simplification is the priority: one path build -> plan
  -> run/dispatch -> rank (which draws the figures); delete other plotting and
  dead code.
- **Dropped:** a `national_covariates=broadcast` option. National covariates
  stay US-only: Kinsa reaches only the US row, and with `spatial='none'`
  states cannot see it (current behaviour, not a planned option).

**Implementation choices (not user decisions), documented in
[the unified design](docs/design/restructure-2026-unified.md) §3-§6:**
float32 kept (float16 is exact only to 2048 and steps by 32 near 50,000, so US
admission counts would be rounded); on-disk storage as a revised-cell mask plus
values where the as-of differs from the truth (10 MB); `asof_weeks` applies to
targets only, vintaged covariates stay fully as-of (reproduces old behaviour
with one field); the old truth fallback for cells not yet visible is kept and
now extends to all `asof_weeks` weeks (flagged: fully as-of episodes are
truth-filled where nothing was visible); seasons stay a module constant; the
figure defaults (two best configurations at their lowest seed, quartile
reference dates per season, all six targets in the CV layout).
