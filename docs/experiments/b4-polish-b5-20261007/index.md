# B4.polish and B5 — 2026–27 submission and new-option exploration (6–7 October 2026)

One page for the work of 6–7 October: settling the 2026–27 flu submission (B4.polish, production fits) and exploring new model options (B5.covariates, B5 exploration, B5 confirmation). Sections 1–3 summarize; the detailed record of each experiment (design, execution, full tables, commands) follows in sections 4–8. Data, designs and analysis tables stay in each experiment folder.

<!-- model-choices:start -->
## Model choices

What this report's models were trained on, how errors and corrections were made, what they learned to predict, which seasons they were trained and evaluated on, and what they were scored on, for every configuration (generated 9 October 2026 from the saved scenario strings).

### B4.polish

Each heading links to its explanation in [Model choices A–F](../../reference/model-choices.md). One row per group of configurations with identical choices.

| Configurations | [Training histories (A)](../../reference/model-choices.md#a-training-histories) | [Error source (B)](../../reference/model-choices.md#b-error-source) | [Error signals (C)](../../reference/model-choices.md#c-error-signals) | [Correction model (D)](../../reference/model-choices.md#d-correction-model) | [Evaluation inputs (E)](../../reference/model-choices.md#e-evaluation-inputs) | [Forecast view (F)](../../reference/model-choices.md#f-input-view) | [Prediction labels](../../reference/model-choices.md#labels-and-folds) | [Evaluated season ← training seasons](../../reference/model-choices.md#labels-and-folds) |
|---|---|---|---|---|---|---|---|---|
| A all | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| A recent2 | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24; 2025-26 ← 2023-24, 2024-25 |
| B all | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| B recent2 | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24; 2025-26 ← 2023-24, 2024-25 |
| B_finalized all | final values | each fold's latest training season | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| B_finalized recent2 | final values | each fold's latest training season | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24; 2025-26 ← 2023-24, 2024-25 |

### B4 production fits (the A + B submission candidate)

Each heading links to its explanation in [Model choices A–F](../../reference/model-choices.md). One column per group of configurations with identical choices.

| Choice | A | B |
|---|---|---|
| [Training histories (A)](../../reference/model-choices.md#a-training-histories) | final values with artificial reporting errors; corrected by the cross-fitted correction model | final values with artificial reporting errors; plus reconstruction labels |
| [Error source (B)](../../reference/model-choices.md#b-error-source) | each fold's latest training season | each fold's latest training season |
| [Error signals (C)](../../reference/model-choices.md#c-error-signals) | admissions | admissions |
| [Correction model (D)](../../reference/model-choices.md#d-correction-model) | tree on real examples; newest 2 week(s) of admissions | tree on synthetic examples; newest 2 week(s) of admissions |
| [Evaluation inputs (E)](../../reference/model-choices.md#e-evaluation-inputs) | production: no held-out evaluation | production: no held-out evaluation |
| [Forecast view (F)](../../reference/model-choices.md#f-input-view) | **corrected** | **corrected** |
| [Prediction labels](../../reference/model-choices.md#labels-and-folds) | latest panel values, next 4 weeks | latest panel values, next 4 weeks + last 4 context weeks |
| [Evaluated season ← training seasons](../../reference/model-choices.md#labels-and-folds) | none held out ← 2022-23, 2023-24, 2024-25, 2025-26 | none held out ← 2022-23, 2023-24, 2024-25, 2025-26 |

### B5 exploration

Each heading links to its explanation in [Model choices A–F](../../reference/model-choices.md). One row per group of configurations with identical choices.

| Configurations | [Training histories (A)](../../reference/model-choices.md#a-training-histories) | [Error source (B)](../../reference/model-choices.md#b-error-source) | [Error signals (C)](../../reference/model-choices.md#c-error-signals) | [Correction model (D)](../../reference/model-choices.md#d-correction-model) | [Evaluation inputs (E)](../../reference/model-choices.md#e-evaluation-inputs) | [Forecast view (F)](../../reference/model-choices.md#f-input-view) | [Prediction labels](../../reference/model-choices.md#labels-and-folds) | [Evaluated season ← training seasons](../../reference/model-choices.md#labels-and-folds) |
|---|---|---|---|---|---|---|---|---|
| 100 configurations: #1, #13, #17, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 90 configurations: #2, #3, #7, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 40 configurations: #4, #5, #24, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | all training seasons, recency-weighted | admissions, ED and covariates | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 103 configurations: #6, #26, #51, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | all training seasons, recency-weighted | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 11 configurations: #8, #50, #86, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions | tree on real examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 19 configurations: #9, #28, #29, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions | tree on synthetic examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 54 configurations: #10, #15, #27, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 23 configurations: #11, #87, #93, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions, ED and covariates | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 28 configurations: #12, #19, #166, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | all training seasons, recency-weighted | admissions | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 27 configurations: #14, #22, #37, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 31 configurations: #18, #20, #73, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 56 configurations: #21, #32, #66, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 16 configurations: #23, #78, #224, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions, ED and covariates | tree on real examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 39 configurations: #30, #33, #75, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions, ED and covariates | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 88 configurations: #31, #45, #55, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 8 configurations: #35, #129, #285, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | all training seasons, recency-weighted | admissions, ED and covariates | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 21 configurations: #36, #38, #43, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions, ED and covariates | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 11 configurations: #39, #168, #243, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions, ED and covariates | tree on synthetic examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 17 configurations: #40, #102, #170, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions | tree on synthetic examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 22 configurations: #41, #95, #136, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 28 configurations: #44, #58, #77, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions, ED and covariates | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 12 configurations: #47, #137, #177, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions, ED and covariates | tree on synthetic examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 27 configurations: #48, #119, #128, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 10 configurations: #52, #80, #114, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 13 configurations: #57, #67, #132, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on real examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 7 configurations: #62, #209, #221, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions, ED and covariates | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 8 configurations: #64, #185, #241, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions, ED and covariates | tree on synthetic examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 5 configurations: #85, #239, #596, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions, ED and covariates | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 23 configurations: #88, #104, #143, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions, ED and covariates | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 11 configurations: #98, #281, #311, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 25 configurations: #101, #105, #139, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | all training seasons, recency-weighted | admissions | tree on real examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| #108, #699 | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions, ED and covariates | tree on real examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 7 configurations: #124, #153, #206, … | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions, ED and covariates | tree on synthetic examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 12 configurations: #180, #279, #282, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | all training seasons, recency-weighted | admissions, ED and covariates | tree on real examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| #225, #486, #530, #894 | final values with artificial reporting errors; plus reconstruction labels | all training seasons, recency-weighted | admissions, ED and covariates | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| #424, #872 | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions, ED and covariates | tree on real examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |

### B5 covariates

Each heading links to its explanation in [Model choices A–F](../../reference/model-choices.md). One column per group of configurations with identical choices.

| Choice | 16 configurations: A_kinsa, A_kinsa, A_kinsa, … | 10 configurations: B_kinsa, B_kinsa, B_kinsa, … |
|---|---|---|
| [Training histories (A)](../../reference/model-choices.md#a-training-histories) | final values with artificial reporting errors; corrected by the cross-fitted correction model | final values with artificial reporting errors; plus reconstruction labels |
| [Error source (B)](../../reference/model-choices.md#b-error-source) | each fold's latest training season | each fold's latest training season |
| [Error signals (C)](../../reference/model-choices.md#c-error-signals) | admissions | admissions |
| [Correction model (D)](../../reference/model-choices.md#d-correction-model) | tree on real examples; newest 2 week(s) of admissions | tree on synthetic examples; newest 2 week(s) of admissions |
| [Evaluation inputs (E)](../../reference/model-choices.md#e-evaluation-inputs) | real archived Wednesday reports | real archived Wednesday reports |
| [Forecast view (F)](../../reference/model-choices.md#f-input-view) | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used |
| [Prediction labels](../../reference/model-choices.md#labels-and-folds) | latest panel values, next 4 weeks | latest panel values, next 4 weeks + last 4 context weeks |
| [Evaluated season ← training seasons](../../reference/model-choices.md#labels-and-folds) | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |

### B5 confirmation

Each heading links to its explanation in [Model choices A–F](../../reference/model-choices.md). One row per group of configurations with identical choices.

| Configurations | [Training histories (A)](../../reference/model-choices.md#a-training-histories) | [Error source (B)](../../reference/model-choices.md#b-error-source) | [Error signals (C)](../../reference/model-choices.md#c-error-signals) | [Correction model (D)](../../reference/model-choices.md#d-correction-model) | [Evaluation inputs (E)](../../reference/model-choices.md#e-evaluation-inputs) | [Forecast view (F)](../../reference/model-choices.md#f-input-view) | [Prediction labels](../../reference/model-choices.md#labels-and-folds) | [Evaluated season ← training seasons](../../reference/model-choices.md#labels-and-folds) |
|---|---|---|---|---|---|---|---|---|
| 5 configurations: A, A+actual0.5, A+log0.5, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 6 configurations: B, B+actual1, B+actual1+batch16, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| A B5 best 0.7764 | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| B B5 best 0.7804 | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on real examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| B B5 best 0.7875, Bs B5 best 0.7686 | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 3 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
<!-- model-choices:end -->
## Common protocol

Unless stated otherwise every model predicts **flu hospital admissions and flu ED visit proportions for the next four weeks**, learns **finalized** future values, and is fitted in two folds:

| Fold | Trained on | Evaluated on |
|---|---|---|
| Forward | 2022–23, 2023–24, 2024–25 | 2025–26 |
| Retrospective | 2022–23, 2023–24, 2025–26 | 2024–25 |

Epochs are chosen on internal validation weeks (3 of every 16 weeks of each training season hidden), then the model is refitted on all its training weeks. Evaluation uses each Hub's Wednesday reports (archived values, finalized values where nothing was archived), with the newest two flu-admission weeks corrected by the run's tree and ED left as reported. **Score: WIS relative to the frozen Hub ensemble on identical tasks, states/DC 80% and US 20%, seasons weighted equally; lower is better, 1 = ensemble.** "Combined" = admissions + 0.5 × ED on the native scale (2024–25 has no ED support, so it is admissions only there); "log" = admissions WIS on log(1 + count). Both evaluation seasons have been used for model selection since B2; nothing here is an untouched test.

The two B4 leaders:

- **Recipe A** — width-96 MLP on flu histories + national Kinsa, sampled output; training histories are artificially degraded reports corrected by cross-fitted trees fitted on genuine archived report-to-mature pairs.
- **Recipe B** — width-96 MLP on flu histories only, neighbor exchange, 23 direct quantiles; trained with artificial reporting errors plus an extra task reconstructing the 4 most recent finalized weeks.
- **Bs** (B5 only) — B with sampled output and an extra four-week-total WIS loss (weight 0.25).

## 1. Submission: A + B, five seeds each, trained on all four seasons

From [B4.polish](#4-b4polish-new-seeds-training-windows-stress-views-ensembles) (A, B, and B trained on finalized histories, each refitted on new seeds 44–48; 30 runs) and the ensemble stress test on saved forecasts:

| Forecast (seeds 44–48, B4 folds) | Combined | Log adm. | Newest admission week missing | Kinsa missing at evaluation |
|---|---:|---:|---:|---:|
| A, single run (mean of 5) | 0.829 | 0.878 | 0.963 | 0.787 |
| B, single run (mean of 5) | 0.857 | 0.870 | 0.998 | – |
| A, 5 seeds averaged | 0.796 | 0.827 | 0.932 | 0.749 |
| B, 5 seeds averaged | 0.818 | 0.820 | 0.944 | 0.818 |
| **A + B, 5 seeds each (equal recipe weight, quantile average)** | **0.795** | **0.803** | **0.919** | 0.754 |
| 25% A + 75% B | 0.803 | 0.807 | 0.928 | 0.778 |
| A + B + B-finalized | 0.813 | 0.810 | 0.921 | 0.781 |

Findings:

1. **The B4 leader scores were partly luck.** On new seeds A goes 0.780 → 0.829 and B 0.796 → 0.857 (single runs). Seed-to-seed spread is 0.04–0.10.
2. **A + B is the most stable choice**: it ties A alone on the combined score, is better on log admissions and with a late report, and single-seed A + B pairs range only 0.805–0.831 vs 0.783–0.891 for A alone.
3. **More training seasons help clearly.** Training only on the two seasons before the evaluated one scores 0.94–0.99 for every recipe; training with the later season in the 2024–25 fold helps too (A admissions 0.798 vs 1.048). Hence production models use all four seasons.
4. **Kinsa is a risk.** A scores better when Kinsa is made unavailable at evaluation (0.787 vs 0.829, all five seeds), entirely in 2024–25; ED gets worse. A degraded A (trained on finalized histories) at 50% weight costs the ensemble ~1% versus B alone.
5. **A late NHSN report costs ~15%** (newest admission week missing). An operational fallback is worth building.
6. Adding B-finalized, or giving A only 25%, does not help. Training-season calibration of interval width gives small or no gains.

**Production fits are done** ([page](#5-production-fits-for-202627)): A and B × seeds 44–48 trained on 2022–23 through 2025–26, checkpoints and correction trees saved on Longleaf. Still to do: a script that forecasts a new Wednesday issuance from these ten models and writes FluSight files.

## 2. What B5 tried and whether it works

[B5 exploration](#7-b5-exploration-vintages-stochastic-nowcasting-formulation): 400 random variations around A, B and Bs (cut from 1,000 for the deadline), seeds 42/43, analyzed by regression; then the leaders and the promising single changes were retrained on new seeds 44/45 ([confirmation](#8-b5-confirmation-on-new-seeds)). [B5.covariates](#6-b5covariates-covariate-representations): 26 matched configurations × 3 seeds.

| Idea | B5 regression (seeds 42/43; combined, negative = better) | New seeds 44/45 | Verdict |
|---|---|---|---|
| **Train on actual archived reports** (share of training episodes using the real archived report instead of an artificial error draw) | −4.7% at 0.5 (−3 to −5% per recipe) | **B: 0.815 vs 0.870**; A: 0.795 vs 0.797 | **Works for B** — adopt for B; neutral for A |
| Extra log(1 + count) admissions loss | −3.9% combined, −9 to −12% log | **A: 0.901 vs 0.797 (worse), log worse** | **Does not hold**; do not adopt |
| Batch size 16 | −2.6% | B + actual + batch 16 0.846 vs 0.815 without | Not confirmed |
| Stochastic nowcasting (forecast from many plausible corrected histories) | within-model −0.4 to +0.8% | – | Does not work |
| Damped-growth anchor (predict departures from a damped two-week trend) | +5.8% combined, +19% log | – | Hurts |
| Reporting errors also on ED and covariates | +6.6% | – | Hurts |
| Some uncorrected training examples | +3 to +5% | – | Hurts |
| Pooled multi-season reporting errors (2023–24 down-weighted) | +1.8% | – | Hurts slightly |
| Historical ILI pretraining (pre-2022 ILI rescaled to flu units) | −1.1%, n.s. (Bs −4%) | – | Inconclusive, small at best |
| Kinsa dropout during training | n.s. | – | No gain on normal inputs |
| Error strength, random/synchronous errors, tree penalty, correction window, several correction realizations, width, lookback, latent, members, noise, decoder, weight decay, multiscale features | within ±2%, n.s. | – | No clear effect |
| New covariate representations (recent growth, smoothing, learned encoder, multiscale) | B5.covariates: A's current Kinsa summary best (0.791); B best without covariates (0.822); growth helps only with ILINet added (0.813 vs 0.908) | – | No payoff |
| Best-of-400 configurations | 0.768–0.780 | regress to 0.797–1.078 | Selection luck |

## 3. Recommendation and next steps

1. **Submit A + B, five seeds each, equal recipe weight, quantile averaging**, trained on all four seasons (done).
2. **Replace B with "B trained on actual archived reports" (share 1.0)** once refitted on all four seasons (5 runs, ~30 min) and checked inside the A + B ensemble on seeds 44/45.
3. Build the operational forecast script and a fallback for a missing newest NHSN week.
4. Decide on Kinsa before the season (keep, or reduce A's reliance); one season's evidence is not enough to drop it.
5. Lower priority: further architecture/loss exploration. B5 showed most new formulations do not pay off here, and seed noise (±0.05) dominates small effects; future comparisons need ≥5 seeds.

## Execution notes

- Two B5 launches failed validation (a window rule that found no 2023–24 reporting errors; out-of-memory from historical-ILI pretraining batches); both were fixed before the main run. Out-of-memory failures on the 44 GB L40s were fixed mid-run by gradient accumulation (identical gradients).
- The Slurm controller was unreachable on 7 October ~07:00–08:30. Production fits and part of the confirmation ran by starting the queue manager directly inside existing allocations; the confirmation was resubmitted when Slurm returned.
- Everything ran from `/proj/jlessler/projects/tapestry-all/tapestry-b5b4polish-20261006`; job IDs are in `data/experiments/launch-20261006.json` there. Nothing is committed to Git yet.

## 4. B4.polish — new seeds, training windows, stress views, ensembles

Record folder: `docs/experiments/b4-polish-20261006/`.

Launched 6 October 2026. Merges the two production plans of 6 October (A+B as default; test whether it tolerates A's weaknesses; confirm on new seeds; test shorter training windows) before a final fit on all four seasons. Manager experiment `b4-polish-20261006`; ensemble stress test `data/experiments/b4-polish-ensembles` (remote checkout `/proj/jlessler/projects/tapestry-all/tapestry-b5b4polish-20261006`).

### Recipes (unchanged B4 scenarios)

All predict flu hospital admissions and flu ED proportions for the next four weeks and learn finalized labels.

- **A** — width-96 MLP, flu histories + Kinsa, sampled output; trained on artificially degraded reports corrected by cross-fitted trees fitted on genuine archived report-to-mature pairs.
- **B** — width-96 MLP, flu histories, neighbor exchange, 23 direct quantiles; trained with artificial reporting errors plus reconstruction of the 4 most recent finalized weeks.
- **B-finalized** — B's architecture trained on unchanged finalized histories (B4.refineTop2: 0.805 vs B's 0.796, combined native, two seeds). At evaluation it still uses B's tree correction; its uncorrected-input forecasts are the "raw" view.

### Part 1 — ensemble stress test on saved forecasts (no training; Slurm 4046731)

Members are the saved two-seed runs from B4.heads (A, B) and B4.refineTop2 (A trained on finalized histories as a deliberately degraded A; B-finalized). Weights by repetition. Groups: B alone; 25% A + 75% B; 50/50; degraded-A 50/50 and 25/75; B-finalized alone; B + B-finalized; A + B + B-finalized. Rules: quantile averaging and distribution mixture. Degraded A is a degradation experiment, not a Kinsa-outage simulation.

### Part 2 — new seeds and shorter training windows (30 runs; Slurm 4046724)

A, B and B-finalized, seeds 44–48 (new), two training windows each:

| `training_window` | Fold evaluated on 2025–26 trains on | Fold evaluated on 2024–25 trains on |
|---|---|---|
| all (B4 protocol) | 2022–23, 2023–24, 2024–25 | 2022–23, 2023–24, 2025–26 |
| recent2 | 2023–24, 2024–25 | 2022–23, 2023–24 (no later season) |

The two requested comparisons come from these without duplicate fits: for 2025–26, three seasons vs only the two latest; for 2024–25, training with the later 2025–26 season vs strictly earlier seasons only. Epochs are chosen on internal validation weeks of the permitted seasons, then refitted.

Every run also writes stress views of the same fitted model: **raw** (no admission correction), **delayed** (newest flu-admission week missing at the deadline, then corrected), **nokinsa** (A only: all covariates missing at evaluation; the missing-input path the model uses for unavailable feeds). These change evaluation inputs only, not the fitted model.

### Analysis plan

Admissions native and log relative WIS, ED, coverage, by season and horizon and by epidemic phase; ensemble value of A across seed subsets; stress views. Default production choice unless contradicted: **A + B, each refitted on 2022–23 through 2025–26, five seeds each, 50% weight per recipe, quantile averaging** (mixture kept if its coverage gain costs no material admissions WIS); no automatic spread calibration. Both evaluation seasons informed earlier selection; new seeds do not make them untouched.

### Commands

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b4_polish.py --plan   # = planner plan -e b4-polish-20261006 -s $(cat docs/experiments/b4-polish-20261006/scenarios.txt) --seeds 44 45 46 47 48 --device cuda
LANES=8 GPUS=1 sbatch --job-name=b4-polish-20261006 --array=0-0 --nodelist=g1803jles02 --cpus-per-task=8 --mem=200G --time=08:00:00 scripts/jlessler.sbatch b4-polish-20261006
.venv/bin/python -m tapestry.experiment.planner status -e b4-polish-20261006
.venv/bin/python -m tapestry.experiment.planner rank -e b4-polish-20261006 --no-plots
.venv/bin/python scripts/ensemble_b4.py --set polish --out data/experiments/b4-polish-ensembles
```

Validation `b4-polish-check-20261006` (seed 44, 2 epochs; excluded from results) Slurm 4046721; rank 4046729. Job IDs: `data/experiments/launch-20261006.json`.

### Results

#### Part 1 — ensemble stress test (completed, Slurm 4046731)

Members: saved two-seed runs (seeds 42/43) of A and B from B4.heads, and of A-finalized (A's architecture trained on unchanged finalized histories, the degraded member) and B-finalized from B4.refineTop2. All trained in the two B4 folds with finalized labels; evaluated on Wednesday reports with the newest 2 admission weeks corrected. Relative WIS vs Hub ensemble, lower is better; quantile averaging unless noted.

| Ensemble | Combined native | Admissions native | Admissions log | Forward ED | 95% coverage (adm.) |
|---|---:|---:|---:|---:|---:|
| **A 50% + B 50%** | **0.742** | **0.736** | 0.764 | **0.812** | 90.3% |
| A 25% + B 75% | 0.757 | 0.752 | **0.766** | 0.847 | 91.1% |
| A + B + B-finalized (equal) | 0.752 | 0.745 | 0.774 | 0.822 | 90.0% |
| B + B-finalized | 0.775 | 0.769 | 0.780 | 0.869 | 90.5% |
| B alone (two seeds) | 0.786 | 0.782 | 0.778 | 0.890 | 91.9% |
| B-finalized alone | 0.787 | 0.779 | 0.807 | 0.858 | 88.6% |
| degraded A 25% + B 75% | 0.779 | 0.776 | 0.776 | 0.876 | 90.5% |
| degraded A 50% + B 50% | 0.793 | 0.790 | 0.789 | 0.870 | 88.6% |

Per season, admissions native: A+B 0.697 (2024–25) / 0.774 (2025–26); B alone 0.697 / 0.866; A 25% + B 75% 0.688 / 0.815.

Findings: (1) **A+B 50/50 remains best overall**; 25/75 trades 2% combined score for slightly better 2024–25 and log scores. (2) **A broken like A-finalized costs about 1% at 50% weight** (0.793 vs B alone 0.786) and is roughly neutral at 25% (0.779) — the ensemble limits the damage but does not make a failing A free. (3) **B-finalized adds nothing** to A+B (0.752 vs 0.742) and is a weaker partner for B than A. (4) Mixture vs averaging: within 0.002 on combined score; mixture covers ~1.5 points more. (5) Without any admission correction (raw view), A+B scores 0.743 / 0.767 on native admissions by season vs 0.697 / 0.774 corrected: correction matters mainly in 2024–25.

#### Part 2 — new seeds 44–48 and training windows (30/30 runs complete, 4047376)

Single-model scores, mean over seeds 44–48, combined native relative WIS (admissions + 0.5 ED; lower is better; 1 = Hub ensemble), corrected view unless stated. Tables: `b4-polish-20261006/analysis/labeled-summary.csv`.

| Recipe | Training window | Combined native | Adm. native | Adm. log | Forward ED |
|---|---|---:|---:|---:|---:|
| A | all (B4 folds) | **0.829** | **0.826** | 0.878 | 0.873 |
| B | all | 0.857 | 0.863 | **0.870** | 0.876 |
| B-finalized | all | 0.890 | 0.886 | 0.878 | 0.913 |
| A | recent2 | 0.973 | 0.973 | 0.983 | 0.896 |
| B | recent2 | 0.986 | 0.998 | 0.968 | 0.980 |
| B-finalized | recent2 | 0.968 | 0.966 | 0.982 | 1.032 |

1. **Both leaders regress on new seeds:** A 0.780 on seeds 42/43 → 0.829 on 44–48; B 0.796 → 0.857. Seed SD of admissions native is 0.04–0.10 per season. The B4 leader scores were partly seed/selection luck, as anticipated; A remains ahead of B on the combined score and B ahead on log admissions.
2. **More training seasons clearly help.** Evaluated on 2025–26, training on 2022–23 through 2024–25 vs only 2023–24/2024–25: A admissions native 0.854 vs 0.899; B 0.910 vs 1.052. Evaluated on 2024–25, training with the later 2025–26 season vs strictly earlier seasons: A 0.798 vs 1.048; B 0.815 vs 0.944. Supports fitting production models on all four seasons.
3. **A is better with Kinsa missing at evaluation** (same fitted models; covariates set unavailable): combined 0.787 vs 0.829, better for all five seeds; the gain is entirely 2024–25 (admissions native 0.691 vs 0.798) while 2025–26 is unchanged (0.860 vs 0.854) and ED worsens (0.927 vs 0.873). Kinsa helps A when training but its evaluation-time values mislead in 2024–25. This is a warning for operational reliance on Kinsa and motivates the Kinsa-dropout arm in B5.
4. **A delayed admission report is costly:** with the newest flu-admission week missing, combined rises to 0.963 (A), 0.998 (B), 0.976 (B-finalized) — 13–16% worse. Worth an operational fallback (e.g. nowcast the missing week) if NHSN is late.
5. **No correction (raw view):** A 0.866, B 0.875 vs corrected 0.829/0.857 — correction is worth ~2–4%.
6. **Training-season calibration** on new seeds: small gains (A 0.826, B 0.848), unlike B4.refineTop2.

#### Part 2 — 5-seed ensembles (Slurm 4053164)

Quantile averaging over the seeds 44–48 runs above (equal weight per recipe; "A5 + B15" = 25% A / 75% B). Combined native relative WIS; lower is better.

| Ensemble (B4 folds) | Corrected inputs | Kinsa missing at evaluation | Newest admission week missing | Adm. log (corrected) | Adm. native 2024–25 / 2025–26 |
|---|---:|---:|---:|---:|---|
| A, 5 seeds | 0.796 | **0.749** | 0.932 | 0.827 | 0.763 / **0.824** |
| B, 5 seeds | 0.818 | 0.818 | 0.944 | 0.820 | 0.769 / 0.874 |
| **A + B, 5 seeds each** | **0.795** | 0.754 | **0.919** | **0.803** | **0.751** / 0.840 |
| 25% A + 75% B | 0.803 | 0.778 | 0.928 | 0.807 | 0.757 / 0.854 |
| A + B + B-finalized | 0.813 | 0.781 | 0.921 | 0.810 | 0.779 / 0.845 |
| B-finalized, 5 seeds | 0.862 | 0.862 | 0.949 | 0.845 | 0.849 / 0.869 |

With the two-season training window every ensemble scores 0.94–0.97 (more seasons help, confirmed).

Single-seed pairs (corrected, combined): A alone ranges 0.783–0.891 across seeds 44–48, B alone 0.799–0.880, **A + B pairs 0.805–0.831**. Pairing recipes removes most of the seed risk.

**Conclusions for the submission:**
1. **A + B, five seeds each, equal recipe weight, quantile averaging** stays the default: on new seeds it ties A-only on the combined score (0.795 vs 0.796), beats it on log admissions (0.803 vs 0.827) and on delayed reports, and is far more stable seed to seed. 25/75 and adding B-finalized are worse.
2. **More training seasons help** → fit production on 2022–23 through 2025–26.
3. **Kinsa is an open risk:** every A-containing ensemble is better with Kinsa missing at evaluation (A + B 0.754 vs 0.795), entirely from 2024–25, while ED worsens. Before submission, decide between keeping Kinsa, training with Kinsa dropout (B5 arm) or a better Kinsa representation ([B5.covariates](#6-b5covariates-covariate-representations)). Do not simply drop Kinsa at evaluation on the strength of one season.
4. **Late NHSN reports cost ~15%**; an operational fallback for a missing newest week is worth building.

## 5. Production fits for 2026–27

Record folder: `docs/experiments/b4-production-20261007/`.

Recipes A and B exactly as in B4 (see [B4.polish](#4-b4polish-new-seeds-training-windows-stress-views-ensembles)), seeds 44–48, **trained on all four seasons 2022–23 through 2025–26** (`evaluation_seasons=production`): epochs chosen on internal validation weeks (3 of every 16 weeks of each season hidden), then refitted on every week. Labels are finalized next-four-week flu admissions and ED. Each run also forecasts the first 2026–27 issuances available in the panel (August–September 2026) as an unscored sanity check; there is no frozen Hub support for 2026–27 and no matured 2026–27 reports, so no Hub scoring and no direct nowcast diagnostics.

Submission default from B4.polish: A + B, equal weight per recipe, quantile averaging over the ten runs; Kinsa risk noted there. Manager experiment `b4-production-20261007`. Queued as Slurm 4074997, but the Slurm controller was unreachable on 7 October morning, so at ~07:15 the queue manager was started directly on g1803jles01 inside the running B5 L40 allocation 4047546_0 (one L40, 4 workers, 1 CPU; log `output/slurm/b4-production-manual-l40.log`). The dispatcher's `squeue` reclaim check now times out after 20 s (it blocked during the outage). Cancel 4074997 once Slurm is back if the manual run has completed.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b4_production.py --plan   # = planner plan -e b4-production-20261007 -s $(cat docs/experiments/b4-production-20261007/scenarios.txt) --seeds 44 45 46 47 48 --device cuda
LANES=10 GPUS=1 sbatch --job-name=b4-production-20261007 --array=0-0 --nodelist=g1803jles02 --cpus-per-task=10 --mem=200G --time=04:00:00 scripts/jlessler.sbatch b4-production-20261007
.venv/bin/python -m tapestry.experiment.planner status -e b4-production-20261007
```

Code: `evaluation_seasons=production` in `model/scenario.py` (scored season 2026–27; training = all `TRAINING_SEASONS`), scoring skipped in `planner fit`, nowcast diagnostics skipped when no matured pairs exist (`pilot.fit`). Local 2-epoch check: both recipes fit on the four seasons and forecast 2026–27 inputs.

### Status (7 October, 08:35)

**All 10 production fits complete** (A and B × seeds 44–48; finished 07:39 on one L40). Each run saved, under `eval_2026-2027/`: the fitted forecaster (`model.pt`), the admission-correction tree (`nowcaster.pkl`), sanity forecasts for the first 2026–27 issuances (raw, corrected, half-mixture and training-season-calibrated views) and the calibration factors. Directories: A = `mlp-pathogen-scheduled_final-69f21e6aec42`, B = `mlp-pathogen-scheduled_final-91f17ac66760` (each `s44`–`s48/attempt-001`). Slurm 4074997 later started after the outage and found nothing left to run.

Not yet done: an operational forecast script that loads these ten checkpoints and correction trees for a new Wednesday issuance and writes the equal-recipe quantile average in FluSight format.

## 6. B5.covariates — covariate representations

Record folder: `docs/experiments/b5-covariates-20261006/`.

Launched 6 October 2026 (user request: test smoothed / slope / downstream-transformed covariates). Manager experiment `b5-covariates-20261006`: 26 configurations × seeds 42, 43, 44 = 78 runs, on the second H100 before it returns to B5. Remote checkout `/proj/jlessler/projects/tapestry-all/tapestry-b5b4polish-20261006`.

### What is compared

All models predict flu admissions + flu ED for the next four weeks, learn finalized labels, and use the B4 folds (train 2022–23, 2023–24, 2024–25 → evaluate 2025–26; train 2022–23, 2023–24, 2025–26 → evaluate 2024–25) with Wednesday-report evaluation and two-week admission correction.

Recipes (otherwise exactly the B4 scenarios): **A with Kinsa** (the B4 native leader), **A with Kinsa + ILINet**, **B with Kinsa added** (B4: summary Kinsa hurt B, 0.796 → 1.005). Reference: **B without covariates**.

Covariate representation (`covariate_encoder`), each with and without `signal_features=smooth_multiscale`:

| Representation | Per covariate and location |
|---|---|
| summary (current) | last value, last-3-week mean, linear slope over the whole 12-week window, SD, coverage, data age (standardized signed-log values) |
| smooth | the full 12-week history after a 3-week trailing mean |
| shared | learned 4-number encoding of the 12-week history + coverage, age |
| **growth (new)** | level at the newest report; 1-week and 2-week log growth and acceleration of log(raw + 10% of training SD), ending at the newest report actually available (handles source lags); coverage, age |

`smooth_multiscale` adds 3/6/12-week mean, slope and curvature (after 3-week smoothing) for **target histories and covariates**, so its effect is not covariate-only.

Hypothesis: a slope over 12 weeks mostly encodes season shape; a leading indicator's value is in its recent turn, which only `growth` exposes.

### Commands

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b5_covariates.py --plan   # = planner plan -e b5-covariates-20261006 -s $(cat docs/experiments/b5-covariates-20261006/scenarios.txt) --seeds 42 43 44 --device cuda
LANES=10 GPUS=1 sbatch --job-name=b5-covariates-20261006 --array=0-0 --nodelist=g1803jles02 --cpus-per-task=10 --mem=200G --time=06:00:00 scripts/jlessler.sbatch b5-covariates-20261006
.venv/bin/python -m tapestry.experiment.planner status -e b5-covariates-20261006
.venv/bin/python -m tapestry.experiment.planner rank -e b5-covariates-20261006 --no-plots
```

Validation `b5-covariates-check-20261006` Slurm 4053157; main 4053159; rank 4053160.

### Results (78/78 runs; validation 26/26; main 1 h 56 min on one H100)

Mean over seeds 42–44, corrected view, relative WIS vs Hub ensemble (lower is better). "sd" is the seed SD of the combined score. Full table: `b5-covariates-20261006/analysis/labeled-summary.csv`.

| Recipe | Covariate representation | Multiscale features | Combined native | Adm. native | Adm. log | Forward ED | sd |
|---|---|---|---:|---:|---:|---:|---:|
| A, Kinsa | **summary (current A)** | no | **0.791** | **0.780** | 0.877 | 0.823 | 0.018 |
| A, Kinsa | growth | no | 0.884 | 0.891 | 0.888 | 0.934 | 0.055 |
| A, Kinsa | shared | yes | 0.850 | 0.844 | **0.849** | 0.899 | 0.026 |
| A, Kinsa | shared | no | 0.929 | 0.923 | 0.962 | **0.795** | 0.153 |
| A, Kinsa | smooth | no / yes | 0.999 / 0.998 | | | | 0.17 |
| A, Kinsa | summary | yes | 1.026 | 1.029 | 1.052 | 0.945 | 0.072 |
| A, Kinsa + ILINet | **growth** | no | **0.813** | 0.799 | 0.853 | 0.915 | 0.021 |
| A, Kinsa + ILINet | summary | no | 0.908 | 0.900 | 0.944 | 0.876 | 0.192 |
| A, Kinsa + ILINet | smooth | no | 0.876 | 0.889 | 0.891 | 0.870 | 0.062 |
| B + Kinsa | summary | yes | 0.835 | 0.840 | 0.854 | 0.872 | 0.065 |
| B + Kinsa | shared | no | 0.855 | 0.852 | 0.856 | 0.926 | 0.014 |
| B + Kinsa | summary | no | 0.981 | 0.983 | 0.990 | 0.912 | 0.035 |
| B + Kinsa | growth | no | 0.917 | 0.919 | 0.914 | 0.842 | 0.008 |
| **B, no covariates** | – | no | **0.822** | 0.818 | 0.852 | 0.866 | 0.045 |
| B, no covariates | – | yes | 0.904 | 0.925 | 0.904 | 0.890 | 0.078 |

Conclusions:
1. **No new representation beats the current ones.** A's existing summary-Kinsa recipe (0.791) is best; B without covariates (0.822) beats every B + Kinsa variant.
2. **Growth features help only where ILINet is added** (A with Kinsa + ILINet: 0.813 vs 0.908 with summaries, and much more stable), still behind A with Kinsa only. With Kinsa alone they are worse (0.884).
3. **Smoothed multiscale features usually hurt** (6 of 8 covariate settings, and B without covariates 0.822 → 0.904), contrary to B2's weak positive signal.
4. **Seed variance is large** (SD up to 0.19 for the same configuration), so differences below ~0.05 with three seeds are not reliable. Combined with B4.polish (A better with Kinsa missing at evaluation in 2024–25), covariate engineering is **not** a high-payoff direction for this flu setup; robustness to covariate failure matters more than representation.

## 7. B5 exploration — vintages, stochastic nowcasting, formulation

Record folder: `docs/experiments/b5-explore-20261006/`.

Launched 6 October 2026 from the merged [B5 plan](../b5-plan-20261006.md) and the 6 October planning notes. Manager experiment `b5-explore-20261006`: **1,000 configurations × seeds 42/43 = 2,000 runs**, both B4 folds (train 2022–23, 2023–24, 2024–25 → evaluate 2025–26; train 2022–23, 2023–24, 2025–26 → evaluate 2024–25), flu admissions + flu ED, finalized next-four-week labels, Wednesday-report evaluation. Exploration only: no refinement is launched automatically from it.

### Design

Each configuration starts from one B4 leader and independently redraws every factor below; the leader's own value is the most frequent level, so configurations stay near a leader and main effects can be estimated by regression (as in [B4.refineTop2 item 1](../b4-refinetop2-20261006/index.md#item-1-which-sweep-settings-mattered)). The three anchors themselves are included unmodified.

| Anchor | Share | Definition |
|---|---:|---|
| A | 44% | Kinsa, sampled output, real-tree-corrected synthetic training histories |
| B | 37% | no covariates, neighbors, direct quantiles, artificial errors + reconstruction |
| Bs | 19% | B with sampled output and 0.25 × four-week-total WIS |

Design seed 20261007; exact strings in `b5-explore-20261006/design.json` / `b5-explore-20261006/scenarios.txt`; generator `scripts/plan_b5_explore.py`.

#### Factors (new code marked ★)

**Training vintages.**
- ★ `vintage_seasons` latest / all: artificial reporting errors (and real correction pairs) from the latest training season only, or from every archived training season with recency weights 1, ½, ¼. 2023–24 contributes 34 windows; 2022–23 has no archive. Older seasons only need flu admissions archived (2023–24 has no COVID/RSV archive).
- ★ `actual_share` 0 / .25 / .5 / 1: probability a training episode uses the actual archived report (finalized value where nothing was archived) instead of an artificial draw. For A it is then corrected like the synthetic reports.
- `reporting_strength` .5–2 (★ cap raised from 1 to 2), random strength on/off, synchronous (shared across states) vs local errors, transported missing-report patterns on/off, errors also on ED/covariates (`revision_signals=all`).
- ★ `correction_realizations` 1 / 2 / 4 (A): several artificial-report-then-correction realizations per training trajectory, one drawn per minibatch (every trajectory has the same number, so no season gains weight).
- ★ `uncorrected_share` 0 / .1 / .25 (A): sometimes train on the uncorrected artificial report (robustness to correction failure).

**Nowcasting.**
- ★ `nowcast_noise` 0 / .5 / 1: the tree also keeps a pool of its own out-of-fold errors (4 origin-block folds, labels in held contexts purged). A new **sampled** evaluation view forecasts from 512 (sampled head) or 16 (quantile head) plausible corrected histories per issuance — point correction plus a sampled error vector per location, newest weeks kept together — and pools the forecasts. The same run still writes the point-corrected view, so the effect is measured within each fitted model (zero-noise reference built in).
- ★ `nowcast_noise_train` (A): add the same sampled errors to corrected training histories each minibatch.
- Correction tree penalty 3–300, window 1–3 weeks, strength .75/1; B's evaluation tree synthetic vs real pairs.

**Formulation.**
- ★ `growth_anchor`: the MLP predicts departures from a damped extrapolation of the last two-week slope (weights 1, 1.5, 1.75, 1.875 for weeks 1–4, in model space) instead of the latest level; off where the newest or two-weeks-earlier cell is missing.
- ★ `log_loss_weight` 0 / .25 / .5 / 1: extra weekly flu-admission loss on the log(1 + count) scale (CRPS of log samples, or pinball on log quantiles), not divided by a level scale.
- ★ `covariate_dropout` 0 / .1 / .25 (A): hide all covariates (Kinsa) for a training episode.
- ★ historical ILI pretraining for the MLPs (25% of configurations, 1,200 updates of batch_size episodes, about 16 passes over 612 historical panel episodes): pre-August-2022 state ILI rescaled per location to flu admission/ED units by fit-only Q95 ratios; US absent; transfers epidemic shape only.
- Four-week-total WIS weight for sampled anchors.

**Untested knobs.** latent 8/16/32, batch 4/8/16, learning rate ×½/×2, weight decay 0/3e-5/1e-4, multiscale signal features, distance exchange, lookback 10/12/14, width 64/96/128; sampled heads also per-location noise, shared US noise factor, members 64/128/256, residual2 decoder.

### Assumptions

- Out-of-fold tree errors from the training seasons represent nowcast errors in the evaluated season; errors are independent across locations in the sampled view.
- Pooled donor seasons assume older reporting regimes remain informative after down-weighting (2023–24 revisions were much smaller than 2024–26).
- Main effects are estimated across random neighbors of three different leaders; interactions with the anchor are expected and will be analyzed per anchor.
- Both evaluation seasons were already used for model selection; this is development evidence.

### Execution

Remote checkout `/proj/jlessler/projects/tapestry-all/tapestry-b5b4polish-20261006`. Validation `b5-explore-check-20261006` (24 configurations, seed 42, 2 epochs; excluded) Slurm 4046723. Main: H100 array 4046725 (8 workers), L40 array 4046727 (4 × 4 workers), second H100 array 4046728 after B4.polish finishes (8 workers); about 32 concurrent runs, roughly 18–20 hours. Rank 4046730.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b5_explore.py --plan   # = planner plan -e b5-explore-20261006 -s $(cat docs/experiments/b5-explore-20261006/scenarios.txt) --seeds 42 43 --device cuda
LANES=8 GPUS=6 sbatch --job-name=b5-explore-20261006 --array=0-0 --nodelist=g1803jles02 --cpus-per-task=8 --mem=200G --time=2-00:00:00 scripts/jlessler.sbatch b5-explore-20261006
LANES=4 GPUS=6 sbatch --job-name=b5-explore-l40 --array=0-3 --nodelist=g1803jles01 --cpus-per-task=4 --mem=110G --time=2-00:00:00 scripts/jlessler.sbatch b5-explore-20261006
.venv/bin/python -m tapestry.experiment.planner status -e b5-explore-20261006
.venv/bin/python -m tapestry.experiment.planner rank -e b5-explore-20261006 --no-plots
```

### Execution notes

- First validation (4046723) failed: pooled vintages with lookback 14 reached windows before the panel start, and 2023–24 contributed no error windows because the window rule required COVID/RSV archives that 2023–24 never had. Fixed (partially archived seasons need flu admissions only; archive presence counts reports issued within the season). Default single-season behavior unchanged (35/48 windows as in B4).
- Second validation (4047375) ran out of GPU memory: historical-ILI pretraining used 64-episode batches. Now uses the run's batch size, 1,200 updates.
- Main run started 17:57 EDT. Observed pace ≈1.5 runs/min with 28 concurrent → about 21–24 h. Some configurations (256 members, batch 16, width 128) use ~27 GB and caused out-of-memory failures on the 44 GB L40s (10 of the first ~40 runs); failed runs will be retried on the H100s with `--retry-failed` at the end.
- 18:40: the second H100 allocation was given to [B5.covariates](#6-b5covariates-covariate-representations) for ~2 h, then returns to B5 (Slurm 4053162).

- 20:55: 69 of the first 233 finished runs had failed, all out-of-memory on the L40s. Fixed by gradient accumulation (minibatches split into ≤1,024 member-episode chunks; identical summed gradient; default 128 × 8 is a single chunk, so B4 configurations are unchanged; historical pretraining chunked the same way). Patched `training.py` in the checkout and in the pinned code (`data/experiments/b5-explore-20261006/code`, only that file); runs already completed are unaffected. The 69 failed runs were reset to pending (`dispatch.initialize(..., retry_failed=True)`) and rerun with the patch. Observed pace ≈57 runs/hour → about 30 more hours.

- 21:25: **cut to 400 configurations (800 runs) to deliver by 9:00 on 7 October** (user deadline). Kept: every configuration with a run started (160, including the three anchors) plus a random 240 of the rest (numpy seed 20261008); the other 600 configurations (1,200 runs) are marked `deferred` in `dispatch.json` and listed in `subset-20261006.json`. A random subset keeps main-effect estimates unbiased but wider. Rank job replaced by `rank --allow-incomplete` (Slurm 4069563). Analysis script: `scripts/analyze_b5.py` (regression effects pooled and per anchor, sampled-vs-point nowcast within run, best configurations per anchor).

- 05:57 7 October: last run finished (798/800 complete, 2 out-of-memory failures). The overnight watcher had a shell bug and did not fire; the idle allocations were found at 06:53. The Slurm controller then became unreachable (scancel/sbatch: "Unable to contact slurm controller"); ranking was run manually on the login node (`ranking-85bce1e60257`).

### Results (398 configurations with both seeds 42/43)

All models predict flu admissions + flu ED for four weeks, learn finalized labels, and are fitted in the B4 folds (train 2022–23, 2023–24, 2024–25 → evaluate 2025–26; train 2022–23, 2023–24, 2025–26 → evaluate 2024–25), evaluated on Wednesday reports with the newest admission weeks corrected. Scores: relative WIS vs the Hub ensemble, lower is better. The anchors refitted under this code reproduce B4 exactly: A 0.7803, B 0.7959, Bs 0.7965 (combined native).

#### Option effects (regression over the 398 configurations, anchor fixed effects)

% change in relative WIS when an option moves from the reference level, other options held fixed on average; negative = better; 95% robust interval for the combined score. Full table incl. per-anchor fits: `b5-explore-20261006/analysis/effects.csv`, `b5-explore-20261006/analysis/summary.txt`. Note: the anchors use `reporting_missingness=False`; the design drew True 3:1, so the regression reference is True.

| Option | Level vs reference | Combined native | Log admissions | Verdict |
|---|---|---:|---:|---|
| **Extra log-scale admissions loss** | 0.25 / 0.5 / 1 vs 0 | −2.9 / **−3.9** [−5.8, −1.9] / −2.7 | **−9.4 / −11.3 / −11.9** | **Works** — the largest gain, especially on log WIS (A: −4.3 to −4.8% combined) |
| **Train on actual archived reports** | share 0.25 / 0.5 / 1 vs 0 | −2.8 / **−4.7** [−6.6, −2.9] / −3.8 | −1.7 / −2.9 / +1.7 | **Works** for every anchor (B −3 to −5%) |
| Batch size | 16 vs 8 | −2.6 [−4.5, −0.7] | −2.5 | Works (A −4.9%) |
| Missing-report transport | off vs on | −3.3 [−5.1, −1.6] | −2.7 | Keep off (the anchors' setting) |
| Several correction realizations | 2 / 4 vs 1 | −1.8 / −1.9 (n.s.) | −2.2 | Possibly helps; not established |
| Historical ILI pretraining | vs none | −1.1 [−2.8, +0.5] | −1.4 | Inconclusive (Bs −4.3%) |
| Shared US noise factor | vs none | −1.4 (n.s.) | −2.1 | Inconclusive |
| **Damped-growth anchor** | on vs off | **+5.8** (off −5.5%) | **+19** | **Hurts** (A +10%) |
| Reporting errors also on ED/covariates | all vs admissions | **+6.6** [+4.6, +8.6] | +4.9 | Hurts |
| Uncorrected training examples | 0.1 / 0.25 vs 0 | +5.0 / +3.2 | +7.2 / +3.6 | Hurts |
| Pooled multi-season vintages | all vs latest | +1.8 [+0.2, +3.3] | +0.8 | Hurts slightly |
| Stochastic nowcast (training configs) | noise 1 vs 0 | +1.9 [+0.1, +3.8] | +1.8 | Hurts slightly |
| Kinsa dropout | 0.1 / 0.25 vs 0 | −0.5 / +2.2 (n.s.) | +0.6 / +1.7 | No gain on normal inputs |
| Error strength, random strength, synchronous errors, tree penalty, correction window, latent, members, width, lookback, weight decay, multiscale features, distance exchange, per-location noise | – | within ±2%, n.s. | – | No clear effect |

**Stochastic nowcasting within the same fitted model** (sampled-history view vs point correction): A −0.2 to −0.4%, B +0.2 to +0.8%, Bs −0.1 to −0.3% (`b5-explore-20261006/analysis/sampled-vs-corrected.csv`). **No meaningful gain**; not worth its evaluation cost.

#### Best configurations

No single random configuration clearly beats the A anchor on the combined score (best A variants 0.776–0.780 vs 0.780), but several beat B (best 0.780 vs 0.796) and Bs (0.768 vs 0.797), and some A variants gain a lot on log admissions (0.782 with log loss 1.0 vs 0.848). Best-of-400 scores are optimistic (selection on the same seasons). Lists: `b5-explore-20261006/analysis/summary.txt`.

#### Conclusions

1. Two new training options are worth adopting after confirmation: **the extra log-scale loss** (balances A's native-scale strength with log-scale accuracy) and **training on actual archived reports** (`actual_share` 0.5–1).
2. The ambitious formulations did not pay off here: damped growth, stochastic nowcast histories, pooled vintages, uncorrected-example robustness and ED/covariate errors all hurt or did nothing.
3. Confirmation on new seeds: [B5 confirmation](#8-b5-confirmation-on-new-seeds).

## 8. B5 confirmation on new seeds

Record folder: `docs/experiments/b5-confirm-top-20261007/`.

Retrains, on new seeds, the three anchors (B4 leaders A, B, Bs), each anchor plus the B5 options with clear helpful effects, and the two best B5 configurations per anchor: 15 configurations. Planned with seeds 44, 45, 46; **seeds 44 and 45 run first (30 runs)** to finish before 9:00. Same protocol as B5: flu admissions + ED, finalized labels, B4 folds, Wednesday-report evaluation with corrected admissions. Selection used the same two evaluation seasons; new seeds test seed luck, not new seasons.

| Label | Change from the anchor |
|---|---|
| A+actual0.5 | half of training episodes use the actual archived report (finalized where none) instead of an artificial draw |
| A+log0.5 | extra weekly flu-admission loss on log(1+count), weight 0.5 |
| A+actual0.5+log0.5+batch16 | both, and batch size 16 |
| B+actual1 | every training episode uses the actual archived report |
| B+actual1+batch16 | and batch size 16 |
| Bs+actual0.5+log0.25+random+penalty300 | actual share 0.5, log loss 0.25, random error strength, tree penalty 300 |
| best_b5 | two best B5 configurations per anchor by combined native score (seeds 42/43) |

**Execution.** The Slurm controller was unreachable at 07:00, so the queue manager was started directly on g1803jles02 inside the still-running B5 allocation 4047545_0 (one H100, 8 CPUs, 8 workers; owner label `4047545_0` so its runs are not reclaimed). Log: `output/slurm/b5-confirm-manual-h100.log`. Do not cancel 4047545 until this finishes. Script `scripts/plan_b5_confirm.py`; manager experiment `b5-confirm-top-20261007`.

```bash
.venv/bin/python -m tapestry.experiment.planner status -e b5-confirm-top-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b5-confirm-top-20261007 --allow-incomplete --no-plots
```

### Results (all 30 seed-44/45 runs; `ranking-201708e33945`)

Execution note: when the Slurm controller came back (~08:30) it ended the B5 allocations hosting the manual dispatcher; the 22 unfinished runs were resubmitted as Slurm 4119661 (2 H100) and 4119662 (4 L40). Seed 46 not run.

Mean over seeds 44 and 45 (one seed where marked), corrected view, relative WIS vs Hub ensemble, lower is better. Development scores on seeds 42/43 from B5 are in the labels of the "best" rows.

| Model | Seeds | Combined native | Adm. native | Adm. log | Forward ED |
|---|---:|---:|---:|---:|---:|
| A (anchor) | 2 | 0.797 | 0.790 | 0.866 | 0.850 |
| A + actual reports 0.5 | 2 | 0.795 | 0.788 | 0.898 | 0.843 |
| A + log loss 0.5 | 2 | 0.901 | 0.880 | 0.925 | 1.025 |
| A + actual 0.5 + log 0.5 + batch 16 | 2 | 0.858 | 0.845 | 0.873 | 0.903 |
| A, B5 best (0.776 on 42/43) | 2 | 0.797 | 0.793 | 0.857 | 0.797 |
| A, B5 2nd best (0.780) | 2 | 0.857 | 0.858 | 0.832 | 0.880 |
| B (anchor) | 2 | 0.870 | 0.865 | 0.915 | 0.893 |
| **B + actual reports 1.0** | 2 | **0.815** | **0.806** | **0.862** | 0.911 |
| B + actual 1.0 + batch 16 | 2 | 0.846 | 0.839 | 0.903 | 0.931 |
| B, B5 best (0.780) | 2 | 0.837 | 0.843 | 0.846 | 0.827 |
| B, B5 2nd best (0.788) | 2 | 0.861 | 0.855 | 0.885 | 0.896 |
| Bs (anchor) | 2 | 0.819 | 0.818 | 0.822 | 0.891 |
| Bs + actual 0.5 + log 0.25 + random + penalty 300 | 2 | 0.832 | 0.825 | 0.843 | 0.920 |
| Bs, B5 best (0.768) | 2 | 0.833 | 0.827 | 0.870 | 0.865 |
| Bs, B5 2nd best (0.769) | 2 | 1.078 | 1.109 | 1.172 | 0.887 |

Conclusions (two seeds; differences under ~0.05 are within seed noise):
1. **Training B on actual archived reports is confirmed**: 0.815 vs 0.870 (−6%), better on log admissions too. For A it is neutral (0.795 vs 0.797).
2. **The log-scale loss is not confirmed for A**: 0.901 vs 0.797, and log admissions also worse (0.925 vs 0.866). The B5 regression gain did not survive new seeds for this recipe; do not adopt it without more evidence.
3. **Best-of-400 picks regress strongly** (B5 best A 0.776 → 0.797, i.e. no better than A; best Bs 0.768 → 0.833 and 0.769 → 1.078), as expected from selection on the same seasons. Only constructed one-option changes are trustworthy.
4. Candidate for the submission: **A + (B trained on actual archived reports)**; to be checked as an ensemble once the remaining runs finish.

## Repository records

Each experiment folder keeps `design.json`, `scenarios.txt` and its analysis tables. Not committed (regenerable from the run folders under `/proj/jlessler/projects/tapestry-all/tapestry-b5b4polish-20261006/data/experiments/<experiment>/` with `planner rank`): per-location distribution diagnostics (`distribution-scores.csv`) and per-run raw-WIS tables (`pilot-raw-wis.csv`). Scripts: `scripts/plan_b4_polish.py`, `plan_b4_production.py`, `plan_b5_covariates.py`, `plan_b5_explore.py`, `plan_b5_confirm.py`, `ensemble_b4.py`, `summarize_b4_ensembles.py`, `analyze_b5.py`. Planning notes: [B5 plan](../b5-plan-20261006.md).
