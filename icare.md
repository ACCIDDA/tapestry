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
[the data specification](docs/legacy-v0/data/forward-2025.md): July 26, 2025 UTC
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
[the experiment specification](docs/legacy-v0/workflows/forward-2025.md). Fixed 100 epochs
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

## Report figures and appendix — 2026-09-22 (later)

**User requests, September 22, 2026**, for the automated report:

- **Fans:** many more reference dates (every 4 weeks of each held-out season by
  default; `--dates` overrides). Ensemble and each model on different subplots
  (columns: hub ensemble, then one per configuration; default the best ranked
  only, `--configs` adds/overrides). One row per disease; one figure per
  location x target kind (admissions and ED have different units); x = time
  across the held-out seasons; finalized truth in every panel; 50%/90% bands and
  median per reference date; same y limits within a row.
- **Heatmap:** per configuration, one panel per target, location x season WIS
  ratio (mean over seeds), log colour scale centred at 1, shared scale.
- **Dot plot:** every configuration and the hub ensemble as its own row, ranked by
  the main score; median over seeds opaque, seeds transparent; row lines as in the
  seaborn example; size adapts so labels and dots print cleanly.
- **Report:** an "Appendix: scenarios run" after the ranking table, linking to a
  generated key of all scenario fields (`docs/reference/scenario.md`), in the
  docs nav. Layout: figures, write-up (preserved), ranking table, appendix.

Implementation choices (not user decisions), in `evaluation/plots.py` and the
unified design §5: labels `C<k>` = ranking position plus the scenario string
wrapped at commas; `--configs` configurations each at their lowest seed; the
colour scale shared across all heatmap files of a call; the key regenerated by
`write_report` (no new command); field meanings (`scenario.MEANING`) taken from
existing docs, with pointers where a line is not enough.

## Hub vintages must not shadow released data; no truth fallback — 2026-09-22 (later)

**User premise, September 22, 2026:** for every hub round, the versioned truth
modelers had at that moment is in the hub's git repository — that is how hubs
distribute target data. So for any Wednesday issuance on which a hub round ran a
vintage MUST exist from git; a truth fallback there is our bug (a missing
acquisition or a shadowing rule), not missing data. Where no round ran (the 2025
federal-shutdown weeks 2025-10-08..11-12) there is genuinely nothing, and there
is also no ensemble to score.

**User decision, September 22, 2026:** cells that were genuinely not published at
the cutoff must be **unavailable**, not filled with final truth flagged
`known_final`. This supersedes the entry below ("Truth fallback kept") and
**changes vintaged results against old B1**; finalized mode is unaffected.

Measured, not user premises: nothing was missing from our acquisition. Three
defects hid data we already hold — a stale hub snapshot shadowed newer git/Delphi
releases (flu truth stopped at 2026-07-04, now 2026-09-05), FluSight's `as_of`
was the week-ending Saturday rather than the publication day until 2025-07
(leakage of up to six days), and the three NSSP channels ignored the hub ED
target files that sit in the same hub files as admissions. The RSV hub's empty
git target history is correct: that repository starts 2025-08-14 and never held a
non-`as_of` target file. After the fixes the 2025-26 whole-issuance gaps are
exactly the shutdown Wednesdays. Details and numbers:
[the unified design](docs/design/restructure-2026-unified.md) §2-§3 and its
decision log.

## Truth fallback kept, with its size documented — 2026-09-22 (superseded above)

**User decision, September 22, 2026:** in vintaged mode, KEEP the truth fallback:
target cells in the as-of window not yet published at a Wednesday cutoff are
filled with later final truth and flagged `known_final=True` (reproduces old B1).
Its size must be documented by a dataset analysis page, rerun whenever the panel
is regenerated: `python -m tapestry.dataset.analyze_dataset` writes
[docs/data/panel.md](docs/data/panel.md) (numbers computed, no hand-written claims).

Measured (2026-09-22 panel, not a user premise): the fallback is mostly whole
issuances without an archive rather than partially reported weeks; e.g. no NHSN
covid/RSV as-of values at any 2023-24 issuance, and in 2025-26 still 6 of 53
issuances fully truth-filled for NHSN covid/RSV (10 of 50 for flu).

## Review clean-up and one population file — 2026-09-22

**User decisions, September 22, 2026:** approved fixing every finding of the code
review (leakage test for validation inputs, re-plan code pinning, report
write-up protection, early-stopping guards, season-aligned validation weeks,
report written only for the complete default ranking, scenario-string labels in
the report, stale docs, dead code, duplicated constants, undocumented choices).
**One population file:** `data/metadata/locations.csv` is the only one kept
(`b1_locations.csv` was byte-identical); its sha256 is pinned in
`experiment.json` at plan time like the panel and frozen support, and runs refuse
to fit if it changed. It and the frozen support are git-ignored, so not synced to
Longleaf with the code (docs/longleaf-setup.md).

Implementation choices (not user decisions), in
[the unified design](docs/design/restructure-2026-unified.md) §4-§5 and its
decision log: loss scales from truth only (changes vintaged scales); heatmap
files named `heatmap-C<k>.png` (the scenario string is in title and heading);
`status` prints the resume commands.

## Regenerable wastewater indices and a legacy docs section — 2026-09-22 (later)

**User decisions, September 22, 2026:**

- **The derived wastewater indices must be regenerable.**
  `derived_nwss_state_indices` (`wval_like`, `pct_rank` per pathogen and state)
  had no rebuild command after the restructure. One command in the current
  workflow now rebuilds it from `delphi_nwss`/`delphi_nwss_aux`:
  `python -m tapestry.dataset.build nwss-indices --data-root data`. It must
  reproduce the existing snapshot, and the panel analysis page must plot the
  indices and their state coverage.
- **Docs history is kept, not deleted, but separated.** Every pre-restructure
  page (B0/B0.1/B1/B2 designs, their results, the archived workflows, pages
  describing removed modules) lives under `docs/legacy-v0/` in a "Legacy (v0)"
  nav section; the current pages stay in the main nav.
- **A failing test is fixed at its root cause**, code or expectation, with the
  reason stated.

Measured, not user premises: the restored builder reproduces every shared row of
the 2026-09-21 snapshot exactly and drops 32 rows (one US row per report time for
reference week 2020-02-29, `n_sites = 1`) that violate the >= 3 site rule; that
week precedes the calendar start, so `panel.npz` is unchanged array for array.
The explorer was right and the test wrong: the as-of API rounds a date back to
the preceding Wednesday/Saturday and never forward
([explorer overview](docs/explorer/overview.md)). Details:
[the unified design](docs/design/restructure-2026-unified.md) decision log.
