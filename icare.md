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
Same-recipe, same-target historical coverage and proposed training-only
calibration experiments are documented in
[coverage notes](docs/results/Forward-2025/coverage-notes.md). The recommendations
are proposals, not fitted improvements; no new model was trained for this answer.
