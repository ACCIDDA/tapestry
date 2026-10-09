# Season peak study: whole-season paths vs peak loss (7 October 2026)

Question (user, 7 Oct): should a Chromantis peak model learn whole-season weekly paths,
only the peak, or both? Run locally on the M2 Max (CPU), launched 14:59 EDT.
Code: `scripts/peak_study.py` (study, climatology, scoring) and a small hook in
`training.fit_component` (`peak_weights`, `peak_crps`; inactive unless a Scenario
subclass sets `weekly_weight`/`peak_weight`). Scores: `peak-scores.csv` (per issuance),
`peak-overall.csv`, `peak-summary.csv` (by phase), `peak-scores.png`.

## What was trained

All arms use A's network and inputs: width-96 sampled MLP, flu admissions and flu ED
history plus Kinsa, sqrt counts, logit ED, b0 input scaling. They differ from production
A in four stated ways:
- no two-stage correction: training inputs are finalized values on the publication
  schedule ("scheduled_final"), not degraded-then-corrected histories; evaluation inputs
  are reports at the Hub deadline without correction trees;
- the prediction label is finalized flu admissions only (no ED), every week from the
  issuance to the last Saturday of May (horizons 1-39; weeks outside the issuance's
  October-May season masked);
- 32 sampled members per training episode (A: 128), to make CPU training feasible
  (0.6 s vs 6.6 s per step);
- a fixed 60 epochs, no early stopping (no validation split was built for peak losses).

Arms differ only in the loss:
- **season**: weekly fair CRPS on every remaining season week;
- **season_peak**: weekly CRPS + fair CRPS of each sampled path's remaining-season maximum;
- **peak**: remaining-season-maximum CRPS only (peak timing unsupervised).

Folds: each of 2022-23, 2023-24, 2024-25, 2025-26 held out in turn; training issuances
from 1 September to 31 March of the other three seasons, held-out season weeks
(CDC epiweek-31 seasons) masked from the training panel. Earlier-season folds therefore
train on later seasons: they are retrospective. Every season has already informed
earlier Chromantis choices.

References (no new training):
- **climatology**: each location's peak size and week from the other three seasons
  (finalized), one picked at random per path, size x lognormal(0, 0.3), week +-2 weeks;
- **rollout**: production-recipe A seeds 44-48 (trained on the other recent seasons,
  with correction, 128 members, early stopping) rolled forward four weeks at a time
  (see [rollout study](../peak-rollout-20261007/index.md)); available only for 2024-25 and 2025-26.

Evaluation: every Wednesday October-February of the held-out season, reports visible at
the FluSight deadline; 500 sampled season paths per seed. Season peak per path = max over
October-May of observed reports (finalized value where no report was archived, as in the
standard reported inputs) and the sampled future. Truth: finalized frozen panel.

Scores (mean over issuances; 52 locations): **peak-size WIS divided by the true peak,
averaged over locations (lower is better)**; 95% interval coverage of the true peak
(target 0.95); probability on the true peak week +-1 week (higher is better).

## Results with seed 44 only (one seed per trained arm)

Peak-size relative WIS (lower is better):

| Arm | 2022-23 | 2023-24 | 2024-25 | 2025-26 | Mean of 4 | Mean 2024-25/2025-26 |
|---|---:|---:|---:|---:|---:|---:|
| climatology | 0.224 | 0.474 | 0.278 | 0.157 | 0.283 | 0.217 |
| peak | 0.125 | 0.325 | 0.309 | 0.218 | 0.244 | 0.264 |
| season | 0.142 | 0.218 | 0.345 | 0.214 | 0.230 | 0.280 |
| season_peak | 0.137 | 0.395 | 0.356 | 0.226 | 0.279 | 0.291 |
| rollout (A, 5 seeds) | - | - | 0.231 | 0.177 | - | 0.204 |

95% coverage of the true peak: climatology 0.97/0.73/0.52/0.93; peak 0.94/0.72/0.43/0.81;
season 0.97/0.83/0.38/0.75; season_peak 0.95/0.62/0.32/0.79; rollout -/-/0.53/0.82.
Probability on peak week +-1: climatology 0.50/0.45/0.30/0.52; season 0.56/0.47/0.32/0.46;
season_peak 0.64/0.44/0.25/0.48; peak 0.50/0.36/0.23/0.44; rollout -/-/0.43/0.48.

Findings (single seed, provisional):
1. Whole-season paths ("season") have the best four-season mean peak-size WIS, but most of
   that comes from 2022-23/2023-24, folds trained on later seasons.
2. In the two recent seasons, none of the new arms beats climatology, and all are worse
   than the no-fitting rollout of production A. That rollout still benefits from things
   these arms lack (correction, 128 members, early stopping, five seeds).
3. Adding the peak loss to the weekly loss did not help (worse in 3 of 4 seasons); the
   peak-only arm is not better than whole-season paths and, as expected, has no timing skill
   in October-November.
4. 2024-25 is the hard season for every method: peak underestimated (median ratio 0.3-0.6
   before January) and coverage below 0.55.

So, on this evidence, more supervision from whole-season paths is at least as good as
predicting the peak directly, but neither is yet better than climatology or the rollout
in recent seasons. Next steps if pursued: more seeds; A's correction and 128 members;
early stopping; an equal mixture with climatology; ILINet pretraining for timing.

Update 17:58: seed 45 finished for 2022-23 only. With seeds 44-45 pooled, 2022-23 peak-size
relative WIS is season 0.129, peak 0.137, season_peak 0.146 (conclusions unchanged). The CSVs
in this folder include these pooled fits. The local run continues with the remaining
seed 45/46 fits (about 43 minutes per round of four; expected to finish around 21:30);
rerun `.venv/bin/python scripts/peak_study.py report --out output/peak-study` to update.
