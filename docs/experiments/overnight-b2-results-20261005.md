# End-of-run summary: forecasting with delayed reports

**Completed 5 October 2026.** The experiments found a small improvement from correcting recent hospital-admission reports. They did not find a convincing improvement from artificially vintaging the first two training seasons or from teaching one model to correct recent reports and forecast future weeks together.

The selected method improves the 2025–26 forecast score by **about 1.1%, with all three training seeds improving**. It keeps the strongest forecaster trained on completed data and combines predictions made from reported admission histories with predictions made from estimated corrected histories. This remains a development result requiring another season of validation.

## The models and the procedure

The **forecaster predicts the next four weeks** of hospital admissions and emergency-department visits. It uses a separate neural network for flu, COVID and RSV, with information shared across locations according to distance. It takes recent disease measurements, with no extra surveillance sources such as wastewater or Kinsa. The project calls this architecture C1 within the B2 experiment; those names are identifiers, not additional methods.

The **reporting-correction model estimates recent numbers that have not finished being reported**. It is a collection of small decision trees. It looks at the recent eight-week disease history, location and time of year, then estimates corrections for the four most recent weeks. The selected version has no extra surveillance sources. This is the model called a nowcaster.

For the 2025–26 evaluation:

1. Train the forecaster on completed 2022–23, 2023–24 and 2024–25 measurements. Both its historical inputs and the future answers it learns to predict are finalized.
2. Measure reporting errors in 2024–25 by comparing early reports with later completed numbers. Apply those patterns to the permitted historical training data to create artificial early reports.
3. Train the separate correction model to recover completed recent numbers from those artificial reports. It never receives completed 2025–26 answers as prediction inputs.
4. Keep the trained forecaster unchanged. At evaluation, generate half its possible future outcomes using the reported admission history and half using the correction model's estimated history. Leave emergency-department histories unchanged.
5. Combine those possible outcomes to calculate the final forecast and uncertainty intervals. The final check used 2,048 outcomes, split equally between the two histories.

The 50/50 mixture combines **predictions**, not input values or interval endpoints. For example, if a report says 100 admissions and the correction model estimates 120, one set of predictions uses the reported history and the other uses the estimated history. We do not replace both inputs with 110. These numbers are illustrative. Equal weights were an experimental choice, not an estimated probability that each history is correct.

## Training seasons, labels and evaluation

| Evaluation season | Seasons used to train the forecaster and correction model | Reporting-error examples taken only from |
|---|---|---|
| 2025–26 | 2022–23, 2023–24, 2024–25 | 2024–25 |
| 2024–25, retrospective | 2022–23, 2023–24, 2025–26 | 2025–26 |

The second evaluation intentionally uses a later season to predict an earlier one. It is not an operational forecast made in 2024–25. Forecasting labels always remain finalized future values. Correction models learn finalized recent values. Each main comparison uses seeds 42, 43 and 44: three repeats of training with different random choices, not three independent evaluation seasons.

Evaluation supplies Wednesday reports using the project's assumed source availability and reporting delays. Where an archived report is absent but the schedule assumes availability, we substitute the completed value. The model does not receive a flag identifying these substitutes. If the completed historical observation itself is absent, it remains unavailable; for example, we do not invent 2022–23 RSV admissions.

## Results for the selected method

The score measures forecast accuracy and the quality of uncertainty intervals. It is divided by the corresponding Hub ensemble score, so **lower is better and 1 means equal to that ensemble under our scoring rule**. It is not a percentage error in admissions. The table uses the same saved forecasters and matched H100 prediction settings.

| Evaluation inputs to the forecaster trained on completed data | 2025–26 | 2024–25 retrospective |
|---|---:|---:|
| Reported histories only | 0.928529 | 0.943373 |
| Equal mixture of predictions using reported versus corrected admission histories; ED unchanged | **0.918262** | **0.888403** |

The mean paired 2025–26 improvement is **1.10%**, with improvement in all three seeds. Repeating prediction on CPU gives a similar 1.11% improvement. Benefits vary by week; some weeks get worse. The three seeds are scored separately and their scores averaged, rather than combining three trained models into a new ensemble.

The 2025–26 score includes flu, COVID and RSV admissions and ED visits. The retrospective score includes only flu and COVID admissions. We therefore cannot interpret the larger retrospective improvement as a like-for-like seasonal comparison. Admissions receive twice the ED target weight; states/DC collectively receive 80% and the US receives 20%. Within each target and location, scores are relative to the Hub ensemble on identical tasks.

## What did not improve the strongest forecaster

**Artificially vintaging training inputs.** We changed completed historical inputs to resemble early reports, while leaving all future labels finalized. We varied which seasons and signals were changed, error strength, the fraction of examples changed, and whether errors were copied or generated by a learned model. No fully repeated version beat the strongest completed-input forecaster's mean 2025–26 score on the same L40 hardware. Its score was 0.939835; even the half-clean, half-artificial-report treatment scored 0.947646, which is worse.

**The first-two-season hypothesis.** On two weaker architectures, using archived recent-season inputs while leaving the first two seasons completed explained most apparent benefits. Adding learned artificial errors to the first two seasons did not improve their forward mean beyond that simpler treatment. Restricting copied errors to admissions helped one weaker model by 2.72% relative to its own completed-input reference, mainly through RSV admissions, but it did not beat the strongest forecaster. This is an improvement within a weaker architecture, not a new overall winner.

**Retraining behind the correction model.** We trained new forecasters on corrected artificial reports and evaluated them on corrected reports. For the strongest architecture, using half the estimated correction gave a 2025–26 score of 0.966752 after retraining, versus 0.924929 when the identical correction was given to the unchanged forecaster. Full correction gave 0.948652 after retraining versus 0.929750 without retraining. The matched reported-input reference was 0.931911. These comparisons use H100 prediction with 256 outcomes, so their absolute scores must not be mixed with the 2,048-outcome table above.

**A single model for correction and forecasting.** We trained neural networks to reconstruct completed recent weeks and predict completed future weeks together, varying how strongly each task affected training. This helped a weaker target-specific architecture with neighboring-state information and Kinsa, but did not improve the strongest disease-specific forecaster without additional surveillance sources.

## Interpretation and limits

The main conclusion is to retain the forecaster trained on completed data. The separate correction model plus reported/corrected prediction mixture is the leading small improvement. We have not demonstrated a large or operationally validated solution to reporting delays.

We repeatedly used these evaluation seasons to choose treatments. Another season is needed to assess generalization. Three seeds measure training variability, not all uncertainty. Archive gaps and completed-value substitutes make this an optimistic reconstruction of real-time information. Sparse reporting-error evidence, especially for ED visits, also limits what learned artificial-report generators can establish.

This is not the official CDC FluSight ranking score. On matched flu submissions, the selected method beats Google's model by about 3.1% under our admission-count scoring rule. Log-scale sensitivity checks instead favor Google. CDC uses log-transformed counts and a different aggregation, so these results do not establish an official Hub victory.

## Completed work and saved artifacts

- Node 1 completed 117 forecaster training tasks, each with both seasonal evaluations: 37 three-seed conditions and six single-seed pilots.
- Node 2 completed 65 forecaster training tasks and 117 evaluations of saved forecasters, each with both seasons. Some evaluations refitted only the separate correction model. Repeated checks are included in this count.
- Total overnight: **182 forecaster training tasks and 117 saved-forecaster evaluations**. All main conclusions above use three seeds.
- The earlier weekend sweep yielded **1,730 completed two-season training tasks**, which were also analyzed. Its strongest average treatment left half the examples unchanged and added weaker reporting errors to the rest; that benefit did not transfer to the strongest forecaster in the overnight test.
- The weekend code ignored its requested joint-model loss weights, so those weight comparisons are invalid. The overnight code fixed the wiring. Completed weekend results are retained; unfinished continuation jobs were cancelled.

Fitted models, predictions, scores, source snapshots and reproduction commands are retained on Longleaf. Experimental mechanisms remain in private research copies; no production forecasting default was changed. The shared joint-loss wiring repair is separate from this documentation commit.

Private research directories under `/proj/jlessler/projects/tapestry-all/`:

- `tapestry-vintage-overnight-20261005`
- `tapestry-nowcast-overnight-20261005`
- `tapestry-weekend-20261002`

The private directories' `data` links point to shared experiment storage. They must not be treated as disposable copies of the data. Full plan, launch, status and rank commands are saved in the respective private directories under `docs/experiments/vintage-overnight-20261005/manager-commands.md` and `docs/experiments/nowcast-overnight-20261005/commands.md`. Reproductions must use fresh experiment names, rather than overwrite fitted snapshots.

No training or inference jobs remain. Post-run cleanup removed 1,424 cached/temporary files and two unfinished fold checkpoints, freeing 27,228,657 bytes (about 26 MiB). All 2,033 completed run records and their result payloads were preserved, including smoke and historical runs outside the main comparison counts. Completed forecasts, fitted model files, logs and source snapshots remain available. The exact deletion manifest is `tapestry-nowcast-overnight-20261005/docs/experiments/longleaf-cleanup-20261005.json` on Longleaf.
