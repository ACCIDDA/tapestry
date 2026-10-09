# B7 three-recipe production fit and its 7 October forecast

Written 7 October 2026, 23:30 EDT.

<!-- model-choices:start -->
## Model choices

What this report's models were trained on, how errors and corrections were made, what they learned to predict, which seasons they were trained and evaluated on, and what they were scored on, for every configuration (generated 9 October 2026 from the saved scenario strings).

Each heading links to its explanation in [Model choices A–F](../../reference/model-choices.md). One column per group of configurations with identical choices.

| Choice | A_blocks3, B5_confirmed_candidate_1 | B_width256 |
|---|---|---|
| [Training histories (A)](../../reference/model-choices.md#a-training-histories) | final values with artificial reporting errors; corrected by the cross-fitted correction model | final values with artificial reporting errors; plus reconstruction labels |
| [Error source (B)](../../reference/model-choices.md#b-error-source) | prescribed 2025-26 process | prescribed 2025-26 process |
| [Error signals (C)](../../reference/model-choices.md#c-error-signals) | admissions and ED | admissions and ED |
| [Correction model (D)](../../reference/model-choices.md#d-correction-model) | tree on synthetic examples; newest 2 week(s) of admissions and ED | tree on synthetic examples; newest 2 week(s) of admissions and ED |
| [Evaluation inputs (E)](../../reference/model-choices.md#e-evaluation-inputs) | production: real operational reports of the 7 October 2026 issuance; no held-out evaluation | production: real operational reports of the 7 October 2026 issuance; no held-out evaluation |
| [Forecast view (F)](../../reference/model-choices.md#f-input-view) | **corrected** (the submitted forecast) | **corrected** (the submitted forecast) |
| [Prediction labels](../../reference/model-choices.md#labels-and-folds) | latest panel values, next 4 weeks | latest panel values, next 4 weeks + last 4 context weeks |
| [Evaluated season ← training seasons](../../reference/model-choices.md#labels-and-folds) | none held out ← 2022-23, 2023-24, 2024-25, 2025-26 | none held out ← 2022-23, 2023-24, 2024-25, 2025-26 |
<!-- model-choices:end -->
## Submission log

**8 October 2026, 00:23 EDT.** At the user's request, the B7 file replaced the on-time System2 file
in the open Hub PR (cdcepi/FluSight-forecast-hub#3764).
- Commit: `b1d0076b`, "update submission past deadline", on `jcblemai:main`.
- File name: `model-output/ACCIDDA-EpiLoom/2026-10-10-ACCIDDA-EpiLoom.csv` (unchanged).
- The submission window for reference date 10 October had closed at the end of 7 October, US Eastern.
  The previous on-time head was `2346eadf` (22:47 EDT, validation passed).
- The model metadata (`ACCIDDA-EpiLoom.yml`) was not changed. Its methods text still describes a
  two-recipe, five-seed, quantile-averaged ensemble, which matches neither file.

**Local record.** `production/submissions/2026-10-10/2026-10-10-ACCIDDA-EpiLoom.csv` is now the B7 file
(with its PDFs). The on-time System2 file is kept as
`production/submissions/2026-10-10/superseded/2026-10-10-ACCIDDA-EpiLoom.csv`
(MD5 `f4340c4d…`, identical to the PR's previous head).

**Expected effect.** This is measured over 55 past weeks (late November to May, 2024–25 and 2025–26,
Hub task grid, real reports):
- B7's state/DC log-admission WIS averaged 4% below the submitted System2's per week, with a
  week-to-week spread of ±11%; B7 was better in 69% of weeks;
- counts were even;
- ED was 2% worse.

Over a season, one week is worth about 0.1% of the average score.

**Hub precedent.** Seven earlier PRs were opened on time and re-pushed after the window. All were
merged despite failing the submission-time check, including UNC_IDD-InfluPaint
(cdcepi/FluSight-forecast-hub#2342, 22 November 2025 round), which is listed in that round's ensemble.

## Model

**Recipes.** There are three, chosen in
[the comparison with System2](../b7-folds-20261007/vs-system2/index.md):
- **A_blocks3:** width-96 MLP with Kinsa that outputs samples, with a three-block decoder.
- **B5_confirmed_candidate_1 (B5 ILI):** a sampled width-96 MLP with Kinsa, pretrained on
  pre-August-2022 state ILI.
- **B_width256:** a width-256 MLP that predicts quantiles directly and shares information between
  neighbouring locations.

**Training.** Each recipe was trained on all four completed seasons, 2022–23 to 2025–26, with ten
seeds (42–51):
- training admission and ED histories were artificially revised with the prescribed 2025–26
  reporting errors;
- the synthetic correction tree is fitted on the same histories;
- labels are latest future flu admissions and ED.

These are exactly the B7-fast settings. Only the fold changed: there is no held-out season, and
the artificial evaluation draws were switched off.

**Combination.** Equal weight per recipe and per seed; predictive distributions are mixed.

**Launch.** Slurm jobs 4220459 (H100) and 4220460 (L40), 23:02–23:23 EDT. A two-epoch check
(`b7-production-check-20261007`, job 4220458) fitted and replayed all three recipes first.

## 7 October forecast

All 30 checkpoints were replayed with `scripts/replay_b7_production.py`, which calls
`scripts/forecast_b6.py` on the corrected view:
- the same pinned operational panel as the submitted file (SHA256 prefix `68962ec56c82`);
- issuance 2026-10-07, context through 3 October.

The export used `scripts/export_b6.py` with
`output/b7/submission-20261007/definition.json`. It passes the same Python checks as the
submitted file: 9,568 rows, 52 locations, 23 quantiles, Hub bounds. No value needed the 0.25 ED cap.

## Against the submitted file

The submitted `2026-10-10-ACCIDDA-EpiLoom.csv` is the eight-recipe System2 without the FluSurv
member: mixed distributions, ten seeds per recipe, trained on 2022–23 to 2025–26 with each recipe's
B6 input treatment, on the same panel and issuance.

US flu admissions, median and 95% interval:

| Horizon (week ending) | Submitted | B7 three-recipe | Ratio of medians |
|---|---:|---:|---:|
| 0 (10 Oct) | 4,325 (2,362–9,376) | 4,133 (3,041–7,267) | 0.96 |
| 1 (17 Oct) | 5,158 (2,667–12,442) | 4,698 (3,356–8,743) | 0.91 |
| 2 (24 Oct) | 6,453 (2,734–17,224) | 5,631 (3,678–11,279) | 0.87 |
| 3 (31 Oct) | 8,259 (2,648–24,056) | 7,063 (3,912–15,672) | 0.86 |

What differs:
- **Growth.** The last six reported US weeks grew 34%, 24%, 34%, 28% and 24% per week, ending at
  3,246 on 3 October, a preliminary value. The submitted median keeps growing at about 25% a week;
  B7 slows to 14–25%.
- **US intervals.** B7's US 95% intervals are much narrower, and its lower bounds stay above the
  latest report.
- **States/DC.** The median state's medians are 4–10% lower in B7, with similar 95% widths.
- **Notable exceptions at horizon 3:** Iowa (×1.72; its current ED is unavailable), Vermont (×1.50),
  Nebraska (×0.66), Delaware (×0.67).
- **ED share.** Close at horizons 0–1, about 8% lower in B7 by horizon 3.

Which is right will be known within four weeks. The retrospective evidence is in the vs-System2
page; neither model was tested on a season this early.

Files: [summary.md](summary.md), [national.csv](national.csv), [comparison.png](comparison.png).
Per-location PDFs are in `output/b7/submission-20261007/`. Reproduce the comparison with
`scripts/compare_submissions.py SUBMITTED NEW OUT`.
