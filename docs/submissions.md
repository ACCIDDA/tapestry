# Chromantis submission log

See [Description and hub rules](reference/hub-scoring-2026-2027.md) for the three hubs' targets, submission formats, and scoring rules for 2026–27.

Chromantis records submissions across all pathogens and model recipes. Each dated entry identifies the pathogen, Hub, public model name, exact fitted recipe, training and operational inputs, submitted file, and publication status. Future recipes receive new entries; a change of model does not rename or overwrite the submission history.

The Hub model is **ACCIDDA-Chromantis**. It was submitted as ACCIDDA-EpiLoom for the 10 October 2026 round and renamed on the Hub on 8 October 2026 (PR 3774); the file contents did not change. COVID-19, RSV and other pathogens belong in the same log when submissions are recorded; none has been submitted yet.

Each entry must state: the recipe and its exact fitted checkpoints, the training seasons and training-input treatments, the prediction labels, the operational input panel (with hash), the submitted file hash, and the Hub PR, validation and merge record. The local record of each issuance is `production/submissions/<reference date>/`: the submitted CSV under its Hub name, its export record (`.json`: release, sources, hashes), the interval PDFs and the peer-comparison PDF drawn at submission time (`python -m chromantis.production record`); replaced versions are in its `superseded/` subfolder.

| Pathogen | Reference date | Public model | Recipe | Recorded status |
| --- | --- | --- | --- | --- |
| Influenza | 10 October 2026 | ACCIDDA-Chromantis (submitted as ACCIDDA-EpiLoom) | Three-recipe B7 mixture, 30 fits | Merged after the deadline; replaced the on-time eight-recipe forecast |
| COVID-19 | — | — | — | No submission entered in this log yet |
| RSV | — | — | — | No submission entered in this log yet |

Hub metadata changes are deferred until the recipe is settled, as decided on 8 October 2026.

## Influenza reference date 10 October 2026

**B7 was submitted and merged.** The Hub's merged CSV is byte-for-byte identical to the saved three-recipe B7 export and the local submission record. Verified on 8 October 2026 by hashing the file at the merge commit of [PR 3764](https://github.com/cdcepi/FluSight-forecast-hub/pull/3764) and the renamed file on Hub `main` after [PR 3774](https://github.com/cdcepi/FluSight-forecast-hub/pull/3774).

### Submission history

All times below are Eastern Daylight Time.

| Time | Event |
| --- | --- |
| 7 October, 10:47:26 pm | Commit `2346eadf83ab0fcd16e6d36d5c63d2cfd4069d35` contained the on-time eight-recipe, 80-fit mixture without FluSurv, called System2 during development. |
| 7 October, 11:00 pm | Weekly submission deadline. |
| 8 October, 12:23:20 am | Commit `b1d0076ba7fd08e71ed5be8ba705e4262dbe0393`, “update submission past deadline,” replaced that CSV with the three-recipe, 30-fit B7 mixture. The user explicitly requested the replacement and commit message. |
| 8 October, 12:25:40 am | The replacement's Hub validation check failed on one check only, `submission_time` (pushed at 04:25 UTC, outside the round's window). No format or value check failed. The earlier on-time head had passed. |
| 8 October, 7:06:03 am | PR 3764 merged as `1b168879a7cc46fbd62b716be4af242caec1061b`. Inclusion in the round's Hub ensemble is a separate fact and is not established by this log. |
| 8 October, 1:09 pm | The user renamed the model on the Hub: commit `6d20dff5`, “model-rename”, moved `ACCIDDA-EpiLoom` to `ACCIDDA-Chromantis` (metadata and `model-output/ACCIDDA-Chromantis/2026-10-10-ACCIDDA-Chromantis.csv`). Only `model_name`, `model_abbr` and the name in `methods_long` changed; the CSV is unchanged (same SHA256 as below). |
| 8 October, 1:36 pm | [PR 3774](https://github.com/cdcepi/FluSight-forecast-hub/pull/3774), “rename mode ACCIDDA_EpiLoom to ACCIDDA_Chromantis”, merged. |

### What the merged forecast contains

- Public name: **ACCIDDA-Chromantis** since 8 October (submitted as ACCIDDA-EpiLoom). `EpiLoomB7` and `System2` are local export names, not submitted models.
- Issuance: **7 October 2026**, with input context through **3 October**. Reference date: **10 October**.
- Targets: weekly incident influenza hospital admissions in counts and influenza ED visits as proportions from 0 to 1.
- Geography: 50 states, DC and native US, 52 locations total; no Puerto Rico.
- Horizons: 0–3, for weeks ending 10, 17, 24 and 31 October. No sample, peak or rate-change targets were submitted.
- Format: 23 quantiles per task, **9,568 rows** = 52 locations × 2 targets × 4 horizons × 23 quantiles.
- National admission medians: **4,133; 4,698; 5,631; 7,063**, in horizon order. These are forecasts, not observed outcomes.

### The model that produced it

All three neural-network recipes were **retrained on the four completed seasons 2022–23, 2023–24, 2024–25 and 2025–26**, with seeds **42–51** for each recipe. There are 30 fits. Production fitting did not hold out an epidemic season.

Admission and ED training histories were artificially made preliminary using the prescribed **2025–26 reporting-error process**. Future prediction labels remained the frozen research panel's latest flu admissions and ED values, described by the research protocol as the September 2026 reference values. Artificial revisions changed inputs, not future labels. Training and epoch selection used the existing normalized native loss plus **0.5 times the log-admission loss**; ED remained on its native scale.

| Recipe identifier | Model and training treatment |
| --- | --- |
| `A_blocks3` | Width-96 sampled MLP with a three-block decoder and Kinsa. Learns future labels from artificial admission/ED histories corrected by cross-fitted trees. |
| `B5_confirmed_candidate_1` | Width-96 sampled MLP with Kinsa and historical state ILI pretraining before August 2022. Uses the corrected-history training route, with its recipe-specific realizations and uncorrected share. ILI is a pretraining source, not a live input requirement. |
| `B_width256` | Width-256 direct-quantile MLP with neighboring-location exchange and no Kinsa. Learns from artificial preliminary histories, with recent-history reconstruction labels as an auxiliary objective and latest future labels. |

For the operational forecast, the **already fitted** models were replayed on real reported inputs from the pinned 7 October panel, with their saved two-week admission/ED correction trees. This replay did not retrain the neural networks. Kinsa was retained for the two sampled recipes, assumed available and non-revising. No FluSurv or other-pathogen histories were used. Calendar inputs remained enabled; the separate calendar-jitter and 2024 reporting-lapse masking proposal was **not this submitted model**.

The predictive distributions were mixed with equal recipe weights and equal seed weights within recipes: one third per recipe and one thirtieth per fit. This was **distribution mixing**, not averaging member quantiles. The Hub received marginal quantiles; this does not establish calibrated joint epidemic trajectories.

### Selection evidence and its scope

The final matched comparison replayed saved fold models; it did not retrain them. For evaluation on **2024–25**, both systems trained on **2022–23, 2023–24 and 2025–26**. For evaluation on **2025–26**, both trained on **2022–23 through 2024–25**. Both learned latest future admission/ED labels. B7 used artificial admission and ED revisions in training as described above; the eight-recipe System2 retained its B6 recipe-specific treatments, with ED mostly at latest values and no FluSurv member.

Both received the same archived Wednesday reports, including the existing substitutions with later values, and their own saved corrections. Scoring used identical FluSight tasks: reference dates 23 November 2024–31 May 2025 and 22 November 2025–30 May 2026, horizons 0–3. Admissions average those seasons equally; ED covers 2025–26 only. The table is for states/DC, using two seeds per recipe on both sides, not the 10-seed production fits.

| Measure with lower values better | Three-recipe B7 | Eight-recipe System2 without FluSurv |
| --- | ---: | ---: |
| WIS on log of admissions plus one | 0.2642 | 0.2737 |
| Admission-count WIS | 62.14 | 59.59 |
| ED-proportion WIS | 0.005041 | 0.005090 |

B7 traded approximately **3.5% lower log-admission WIS for 4.3% higher admission-count WIS**, with similar ED accuracy. That supported choosing it when season-long state log-admission accuracy was the priority. It does not quantify the expected benefit of the October 10 forecast. The comparison omitted October and early November, reused seasons for selection, and used 2025–26 reporting errors to construct B7's correction even when evaluating that season. The 2024–25 epidemic fold also used later-season training. These are exploratory, exchangeable-season comparisons, not an untouched prospective accuracy estimate.

### Exact artifacts and outstanding discrepancies

Both 10 October files have their interval PDFs (`*-hosp-plot50-80-95.pdf`, `*-ed-plot50-80-95.pdf`) and peer comparison (`*-peers.pdf`, peers merged by the Hub's 7 October 22:46 EDT commit `f66ce884`) in `production/submissions/2026-10-10/` and its `superseded/` folder. They were redrawn on 9 October 2026 from that same local Hub clone, which had not been updated since the submissions; the System2 file's interval PDFs had not been kept at the time.

The [merged Hub CSV](https://github.com/cdcepi/FluSight-forecast-hub/blob/1b168879a7cc46fbd62b716be4af242caec1061b/model-output/ACCIDDA-EpiLoom/2026-10-10-ACCIDDA-EpiLoom.csv), the renamed [`2026-10-10-ACCIDDA-Chromantis.csv`](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-output/ACCIDDA-Chromantis/2026-10-10-ACCIDDA-Chromantis.csv) on Hub `main`, `production/submissions/2026-10-10/2026-10-10-ACCIDDA-EpiLoom.csv`, and `output/b7/submission-20261007/2026-10-10-ACCIDDA-EpiLoomB7.csv` share SHA256:

```text
cdee271c326437b81049daed50c453d8b5210ebffa9690d8d593b39e06cf607a
```

The operational panel SHA256 is `68962ec56c8246469bb71c416f9e2abc593d7b68469dd5f4817caf71743029cc`. The export manifest records all 30 forecast sources and their hashes in `output/b7/submission-20261007/2026-10-10-ACCIDDA-EpiLoomB7.json`. Re-exporting those forecasts with the current code (`chromantis.production`, 8 October) reproduces the submitted CSV byte for byte. The production training design is `docs/experiments/b7-production-20261007/design.json`.

The superseded on-time CSV is `production/submissions/2026-10-10/superseded/2026-10-10-ACCIDDA-EpiLoom.csv` (as submitted on time), SHA256 `b68b34228bbd4feee5d03fe7ac1e3703f379d4c0dd9cc636f8584b3c7e5f7a70`.

The Hub metadata (`ACCIDDA-Chromantis.yml`) still describes two recipes, five seeds per recipe and averaged quantiles, which matches neither submitted version (B7: three recipes, ten seeds, mixed distributions). Its `ensemble_of_models: false` was set at the user's request at 10:13 pm on 7 October (commit `e8d8ec1d`, “fix ensemble”), when the submission was thought to be a seed ensemble of one recipe; a multi-recipe mixture may need `true`. The B7 export manifest also still says `published: false`, reflecting its pre-publication export state. The B7 release is `production/releases/b7-20261007.json` (all 30 checkpoints with the SHA256 of each `model.pt`, matching the hashes recorded when the forecast was made); the on-time System2 file can be regenerated from `production/releases/system2-20261007.json` (the 80 submitted checkpoints). Decision on 8 October 2026: defer Hub metadata changes until the recipe is settled. The log preserves the methods actually used for each issuance even when later submissions use a different recipe. This log records the actual merged model without changing Hub metadata or launching another submission.

Scientific sources: [production fit](experiments/b7-production-20261007/index.md), [matched comparison](experiments/b7-folds-20261007/vs-system2/index.md), [submission decision review](experiments/submission-choice-review-20261007.md).

The production-fit page reports a second, larger comparison (55 weeks, real reports): B7 state/DC log-admission WIS 4% lower than System2, counts even, ED 2% worse. The table above (two seeds per recipe, both systems) shows ED 1% better for B7. The two comparisons differ in seeds and support; neither is a prospective estimate. The earlier production-fit report uses “submitted” for the then-current System2 file; the chronology above resolves that ambiguity.
