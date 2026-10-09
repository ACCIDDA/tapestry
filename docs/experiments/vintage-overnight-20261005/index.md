# Artificial reporting vintages: overnight research, 5 October 2026

This is an iterative development comparison using the strongest original B2 C1,
C2 and C3 architectures. Each arm retrains a forecasting model. These seasons
have already been used for development; none is an untouched prospective test.

## Data and labels

When evaluating Wednesday reports in **2024–25**, forecast models train on
finalized trajectories and future labels from **2022–23, 2023–24 and 2025–26**.
Only archived/final reporting pairs from **2025–26** train the artificial
reporting generator. This is a retrospective split.

When evaluating Wednesday reports in **2025–26**, forecast models train on
finalized trajectories and future labels from **2022–23, 2023–24 and 2024–25**.
Only archived/final reporting pairs from **2024–25** train the artificial
reporting generator.

Every forecasting model learns the same finalized future values at t+1 through t+4. Artificial
reporting errors alter training inputs only. Reference normalization and target
scales are fitted on the permitted unperturbed historical episodes; stochastic
errors change minibatch inputs without refitting those reference scales. Generator
seasonal peaks use permitted training dates from the whole season, not only dates
before each historical origin; these offline generator features are not separately
supplied to the forecasting network. Evaluation always uses Wednesday
archived numerical values under the original B2 source-availability schedule.
Targets are available through the latest completed week; covariates keep their
source-specific reporting delays and the Vermont exception. If an archive value
is absent while the assumed schedule says it is available, the frozen final
value is an explicit numerical proxy. This assumption does not establish
operational availability. The generator's empirical training pairs never use
such proxies. The arm using actual reports for the recent training season does
use the same explicit final-value proxy policy as evaluation.

The full-availability assumption applies only where the panel has historical
final truth. It does not invent absent historical series: RSV admission inputs
are absent in 2022–23, so artificial RSV-admission errors can affect 2023–24
within the first-two-season treatment but cannot create 2022–23 observations.
Input availability and finalized-label support remain unchanged.

## Forecast architectures

C1 fits one multilayer perceptron per pathogen, shares spatial information with
distance weights and has no covariates. C2 fits one multilayer perceptron per
target without spatial sharing and uses inpatient, wastewater WVAL-like, Kinsa,
ILINet, clinical laboratory and FluSurv covariates. C3 fits one multilayer
perceptron per target with neighbor spatial sharing and Kinsa covariates. All use
12-week histories, fourth-root admission transforms, logit ED transforms and
training-fitted B0 normalization. Epoch selection has a cap of 300 and patience
30. The first comparison removes the original B2 finality channel and 20%
episode missingness regularizer.

## First comparison

The original plan assigned seeds 42, 43 and 44 to each arm on each architecture.
The final completed plan contains 78 runs: arms 3 and 5 below retain only seed 42
on each architecture; the other eight arms have all three seeds. The narrowed
arms are explicitly single-seed pilots and are excluded from confirmed three-seed
comparisons. Each run evaluates both seasonal folds.


1. Finalized inputs in all three training seasons, evaluated on archived reports.
2. Half-strength empirical reporting errors in all three training seasons.
3. Full-strength empirical reporting errors in all three training seasons.
4. Half-strength empirical errors in 2022–23 and 2023–24; finalized recent season.
5. Full-strength empirical errors in 2022–23 and 2023–24; finalized recent season.
6. Half-strength empirical errors in 2022–23 and 2023–24; actual recent-season reports.
7. Full-strength empirical errors in 2022–23 and 2023–24; actual recent-season reports.
8. Learned conditional reporting means plus centered empirical residual windows
   in 2022–23 and 2023–24; actual recent-season reports.
9. Learned conditional reporting means only in 2022–23 and 2023–24; actual
   recent-season reports.
10. Learned conditional reporting means plus centered empirical residual windows
    in all three training seasons.

Treatment scope follows the **episode origin season**. A context crossing a
season boundary receives its origin season's treatment. The fitted empirical
reporting-error library still excludes every reference week outside the named
donor season.

The empirical method transports a signed log ratio on the original floor of
5% of the location/signal seasonal peak, at least one admission or 0.0001 for
ED proportions. Half strength multiplies this log ratio by one half. Donor
windows match location, final epidemic phase and calendar position.

The learned generator fits a ridge regression separately for each report age
and signal. It predicts the signed log ratio from finalized normalized level,
preceding-week growth, origin growth, annual sine/cosine, proximity to Christmas
and a location indicator shrunk toward the shared regression. The ridge penalty
is 20; the intercept is effectively unpenalized. Fewer than 30 observed pairs
leave that age/signal's conditional mean at zero. Finalized epidemic features
are legitimate inputs to this training-data generator, but would not be
available to an operational nowcaster. The forecasting model never receives
these generator features separately.

For residual draws, the generator subtracts its fitted conditional mean from
observed reporting errors and transports a matched complete residual window.
Those residuals are centered in-sample: shrinkage limits overfit, but the residual
variance may be conservative. Predicted log means are capped at a factor of four;
ED inputs remain between zero and one. National covariates remain shared across
locations. Missing donor archive cells have zero empirical residual, while a
learned conditional mean may be predicted from other supported pairs. This is
an explicit modeling assumption.

Early stopping excludes hidden validation weeks from the generator's fits and
peak scales. No evaluation-season reporting error or label enters the fitting
library. Recent-season actual training inputs preserve the episode's available
cells. Finality indicators and artificial missingness regularization are omitted
for the first comparison to match the weekend architectures. This is a choice
of training regularization, distinct from the assumed evaluation availability.

## Score

Lower is better. Each location's summed forecast weighted interval score (WIS)
is divided by the hub ensemble's WIS on identical frozen tasks. States/DC share
80% of the weight equally and US receives 20%. Admission targets have twice the
weight of ED proportions. Seasons count equally; report means and variation
across three seeds. A value below one beats the hub ensemble under this objective.
Report the two evaluation seasons separately as well as their equal average.
**Frozen benchmark support differs:** 2024–25 scores flu and COVID hospital
admissions only; 2025–26 scores all six targets. Models still learn finalized
future labels for all six targets in both folds. A secondary diagnostic restricts
both seasons to flu and COVID admissions, equally weighted, using the same fitted
models and predictions. It does not replace or modify the official objective.

## Code and manager commands

Private local source: `/tmp/chromantis-vintage-overnight-20261005`.
Private remote project:
`/proj/jlessler/projects/tapestry-all/tapestry-vintage-overnight-20261005`.
The shared source tree is not edited. Inputs and the environment link to the
existing reporting-augmentation project; experiment names are unique.

The plan wrapper calls the common manager's `plan` command and saves the exact
expanded invocation in each experiment's `manager-plan.txt`.

```bash
cd /proj/jlessler/projects/tapestry-all/tapestry-vintage-overnight-20261005
PYTHONPATH=src .venv/bin/python scripts/plan_vintage_overnight.py -e vintage-overnight-smoke-20261005 --smoke
LANES=3 GPUS=1 sbatch --job-name=vintage-overnight-smoke-20261005 --array=0 --time=00:20:00 scripts/jlessler.sbatch vintage-overnight-smoke-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e vintage-overnight-smoke-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e vintage-overnight-smoke-20261005 --no-plots
```

The two-epoch execution pilot (job 3800115) checks the learned generator's path,
not forecasting efficacy. Research results will be added as the full comparisons
finish.

## Donor-model diagnostic before forecast results

The conditional ridge reporting generator was evaluated by withholding each of
four contiguous blocks of donor origins in turn. This predicts archived/final
log reporting errors within one donor season, not future disease forecasts.
Lower root mean squared log error is better. For current-week flu admissions
from 2024–25, the conditional model scored 0.1227, compared with 0.1373 for a
shared donor-training mean; using 2025–26 pairs, the scores were 0.1255 and
0.1379. This is evidence for structured reporting bias, not evidence that the
forecast model improves. The diagnostic ridge is retrained excluding each block.

Current-week ED has only about 17% archive support among the 2024–25 donor
windows, concentrated in one of the four blocks. The remaining blocks cannot
train its current-week conditional model. A ratio of one for this case reflects
that lack of training support, not demonstrated equivalence of the two models.
Age-three COVID ED log-error RMS is 0.0555 in the 2024–25 donor season but 0.1823
in 2025–26. This large shift limits confidence in transporting ED errors between
seasons. The `reporting-generator-blocked-validation.csv` table includes exact
support fractions, errors and season definitions.

First research launch: job 3800158, four L40 allocations, four processes each,
six-hour limit, 30 configurations and 90 seed runs (180 outer folds).

```bash
PYTHONPATH=src .venv/bin/python scripts/plan_vintage_overnight.py -e vintage-overnight-round1-20261005
LANES=4 GPUS=4 sbatch --job-name=vintage-overnight-round1-20261005 --array=0-3 --time=06:00:00 scripts/jlessler.sbatch vintage-overnight-round1-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e vintage-overnight-round1-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e vintage-overnight-round1-20261005 --allow-incomplete --no-plots
```

The outer empirical library contains 34 origins from 23 November 2024 to
19 July 2025 for the 2024–25 donor season, and 48 origins from 2 August 2025 to
25 July 2026 for the 2025–26 donor season. Calendar matching therefore cannot
supply observed early-autumn 2024 reporting patterns. The preexisting metadata
field `target_evidence_fraction` describes finalized-truth support, not actual
archived/final pair coverage; use `available_pairs_fraction` from the diagnostic
CSV for archival evidence.

## Admissions-only follow-up

The donor archive diagnostic and a separate H100 development run both implicated
ED revision transport. The H100 evidence is a design clue, not pooled with L40
results. The second L40 comparison contains four arms on each C1/C2/C3 model,
again with seeds 42, 43 and 44:

- Keep 2022–23 and 2023–24 inputs finalized; use actual Wednesday reports for the
  recent permitted training season. This isolates the effect of adding artificial
  vintages to the first two seasons.
- Add half-strength empirical reporting errors to admissions in 2022–23 and
  2023–24; retain actual reports for the recent permitted training season.
- Add half-strength learned conditional reporting means plus empirical residuals
  to admissions in 2022–23 and 2023–24; retain actual recent-season reports.
- Add half-strength empirical errors to admissions in all three training seasons;
  keep all ED targets and covariates finalized in training. This arm tests whether
  admission gains survive without introducing ED/covariate errors.

Here admissions-only means the three NHSN targets. Early-season ED targets and
all covariates remain finalized. For the first three arms, actual recent-season inputs still include every
signal on the documented source schedule. The fourth arm instead uses synthetic
admission inputs and finalized ED/covariate inputs in all three seasons. The recent season is 2025–26 when
forecasting the held-out 2024–25 season, and 2024–25 when forecasting 2025–26.
Prediction labels, evaluation reports, reporting lags and final-value archive
proxies remain exactly as specified above.

The second code snapshot also makes a missing-adjacent-week growth feature zero
rather than interpreting missingness as a large epidemic change, and caches
unchanging conditional means for speed. The first snapshot is retained
inside its experiment and is not modified. These are separately identified
research runs; any comparison between their learned generators includes the
stated growth-feature change.

Array 3804709 requests four L40 GPUs, four processes per GPU, with an allocation limit of 135 minutes and an 08:25 EDT scheduling deadline. It is queued behind the
first comparison. Slurm's status, not this protocol, determines actual starts.

```bash
PYTHONPATH=src .venv/bin/python scripts/plan_vintage_overnight.py -e vintage-overnight-admissions-focused-20261005 --admissions
LANES=4 GPUS=4 sbatch --job-name=vintage-overnight-admissions-focused-20261005 --array=0-3 --time=02:15:00 --deadline=2026-10-05T08:25:00 scripts/jlessler.sbatch vintage-overnight-admissions-focused-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e vintage-overnight-admissions-focused-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e vintage-overnight-admissions-focused-20261005 --allow-incomplete --no-plots
```

At 02:00 EDT the first completed three-seed arm was the C1 pathogen MLP with
learned reporting means plus empirical residuals in all three training seasons.
Trained on 2022–23, 2023–24 and 2025–26, with reporting errors learned only from
2025–26 and finalized future labels, it scored 0.88178 on 2024–25 Wednesday
reports. Trained on 2022–23, 2023–24 and 2024–25, with errors learned only from
2024–25, it scored 0.99093 on 2025–26 Wednesday reports. Lower relative WIS is
better. Its matched all-finalized-input training control was not yet complete,
so these early numbers do not establish a benefit from artificial vintaging.

### Budget adjustment at 02:20 EDT

Observed target-partition seed runs take roughly 60–70 minutes on four concurrent
processes per L40, while C1 takes roughly 30–40 minutes. To leave time for the
admissions follow-up before 08:30 EDT, twelve **unstarted** replications were
removed from the live dispatcher queue: seeds 43 and 44 for full-strength
empirical errors in all three seasons, and for full-strength early-two errors
with recent-season finalized inputs, on each of C1/C2/C3. Their seed-42 pilots
remain. All matched controls, half-strength arms, recent-actual-report arms and
learned-generator arms retain three seeds. This change uses the earlier weekend
finding that full-strength blanket perturbations hurt the forward season; it
is not hidden selection of favorable overnight seed results.

The active first-batch budget is now 78 seed runs. The initial manager plan recorded all 90. `deferred-seeds.json` records the
initial temporary deferral. The subsequent `vintage_overnight_narrow.py` helper
removed only these twelve unstarted entries from both `jobs.csv` and the
dispatch state under its lock. `narrowed-plan.json` records the exact change and
`design.csv` lists each arm's planned seeds. Ordinary manager resubmission now
**preserves the 78-run plan**. No running fit was cancelled. Partial rankings must not present these one-seed pilots as equivalent
to completed three-seed comparisons.


The originally queued follow-up array 3801336 was cancelled before any fit
started. Its half-clean learned arm was replaced by the all-three-season
admissions-only empirical arm, keeping the 36-run budget unchanged. The first
matched L40 C2 seed-42 results motivated this change: with training on 2022–23,
2023–24 and 2024–25, errors learned only from 2024–25 and finalized future labels,
the model evaluated on 2025–26 Wednesday reports scored 0.96577 after all-finalized
input training, 0.97333 with half-strength errors on all signals in all three
seasons, and 1.02152 with half-strength early-two errors plus actual recent-season
reports. Lower is better. For the all-three half-strength arm, flu admissions
improved from 0.93842 to 0.87188, while COVID ED worsened from 0.90394 to 0.98138
and RSV ED worsened from 0.82830 to 1.05088. These are single-seed development
clues, not three-seed confirmations.


To reproduce the narrowed scheduling after a new first-round plan, apply the
following helper before launching. It changes seed lists only; no configuration,
task ID, cost estimate, fitted result or pinned source is changed.

```bash
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_narrow.py -e vintage-overnight-round1-20261005
```

Existing-experiment resumes already use the narrowed plan automatically. An
exact *new* reproduction of the original generator must set
`PYTHONPATH=reproduction-source-v1/src` while planning under a fresh name.
That independent, checksum-verified source copy includes the notifier required
by the planner. Using an experiment's own snapshot as a re-planning source is
unsafe because planning replaces the destination snapshot; the bare snapshot
also lacks the expected notifier path. No such re-plan was executed. The private
working source now contains the documented second-version growth-feature change.
The normal Slurm launcher always uses each existing experiment's pinned source.
See `manager-commands.md` for the complete safe reproduction commands.


At 03:10 EDT, completed seed runtimes averaged 31.4 minutes for C1 (four runs),
58.1 minutes for C2 (ten runs) and 59.0 minutes for C3 (six runs). The pending
follow-up limit was increased from 120 to 135 minutes to allow for the last few
fits. With its unchanged 08:25 EDT scheduling deadline, it must start by 06:10
EDT. Actual scheduling and completeness will be reported, not inferred from
this estimate.

At approximately 03:20 EDT, the dispatcher briefly held unstarted noncontrol
entries until the remaining five matched-control seeds were claimed. It then
automatically restored all 35 held fits. No planned seed, scenario or active fit
was removed. `control-priority.json` records this scheduling change. All three
C1 controls and the remaining C2/C3 controls were running when the hold ended.

## First matched three-seed result

The C1 pathogen MLP, with distance-based spatial sharing and no covariates,
trained on finalized 2022–23, 2023–24 and 2024–25 inputs and finalized future
labels, scored **0.93984 ± 0.01230** on the six-target 2025–26 Wednesday-report
benchmark. Retraining that same architecture with learned conditional reporting
means plus empirical residuals from 2024–25 applied to all three training
seasons scored **0.99093 ± 0.08785**, a **5.44% worsening** in the mean. Lower is
better; the spread is standard deviation across seeds 42, 43 and 44. Two of the
three paired seeds worsened.

For the retrospective 2024–25 benchmark, which scores flu and COVID admissions
only, C1 trained on 2022–23, 2023–24 and 2025–26 improved from 0.93467 to 0.88178
when the learned errors came only from 2025–26. All future labels stayed
finalized, and both models received the same Wednesday reports, source lags and
explicit absent-archive final proxies at evaluation. The secondary forward
flu/COVID-admission score also worsened, from 0.95531 to 0.97765. Thus C1's
forward deterioration is not solely an artifact of adding ED/RSV targets to the
forward benchmark. Learned blanket vintaging has not solved the forward problem.

Per-seed and per-season results and the graph are in
`vintage-overnight-round1-20261005/`. Only comparisons with three completed seeds
in both the treatment and matching control receive a plotted percentage change.


The location-specific archive audit confirms that the national series itself
has only seven current-week ED reporting origins in 2024–25, versus 42–43 in
2025–26. National admissions have 33–34 versus 46–47 origins. Counts refer to
observed report/final pairs, not final-truth support. The full-season audit
contains 52 and 53 origins respectively; the fitted donor windows use the
subset documented above. Missouri has no current-week ED pairs in 2024–25, and
Wyoming has none for flu ED. These gaps are not zero-error evidence. The
conditional model can extrapolate a mean to missing archive cells, while
empirical residuals are zero there; this is a material modeling assumption,
especially for ED. See `archive-support-by-location.csv`.

## Effective perturbation audit

`transport-error-magnitudes.csv` and its graph measure the actual changes made to
available training inputs, after nonnegative and proportion clipping. The audit
uses the first-round pinned generator, seed 20261005, and five draws per origin.
It compares current-week and three-week-old inputs in each training season. The
reporting model is fitted only to the permitted donor season; this diagnostic
does not fit a forecasting model or score predictions. Future labels are never
changed. The plotted subset concerns 2022–23 inputs and errors learned from
2024–25, as used for the forward forecast fold. RSV admissions are absent in
2022–23 and omitted rather than filled with synthetic observations.

At full strength, empirical donor windows leave 98% of available current-week
2022–23 RSV ED cells unchanged (effective log-error RMS 0.019), whereas the
learned conditional mean plus residual generator leaves 5% unchanged (RMS
0.064). For flu ED, the unchanged fraction is 91% versus 1% (RMS 0.012 versus
0.022); for COVID ED, 86% versus approximately zero (RMS 0.057 versus 0.073).
Admissions perturbation RMS is similar: flu 0.122 versus 0.120 and COVID 0.203
versus 0.207. Thus the learned generator is also an extrapolation into absent
archive cells, with a much denser ED perturbation pattern. This is a plausible
mechanism to investigate, not proof that it caused any forecast score change.
The admissions-only follow-up avoids synthetic ED perturbations.

Reproduce the input audit with:

```bash
PYTHONPATH=data/experiments/vintage-overnight-round1-20261005/code/src .venv/bin/python scripts/vintage_overnight_transport_diagnostic.py
```

All three seeds of each matched finalized-input training comparison are now
complete on L40. The following models use finalized training inputs and
unchanged finalized t+1 through t+4 prediction labels. For evaluation on
2025–26 they train on 2022–23, 2023–24 and 2024–25. For retrospective evaluation
on 2024–25 they train on 2022–23, 2023–24 and 2025–26. Every evaluation uses
Wednesday archived values, B2 source lags and the documented final-value proxies.
Scores are mean relative WIS ± standard deviation over seeds 42, 43 and 44;
lower is better. The two columns have different target support and should not
be compared as a common six-target metric.

| Model trained on finalized inputs | Retrospective 2024–25: flu/COVID admissions | Forward 2025–26: six targets |
| --- | ---: | ---: |
| C1: pathogen MLP, distance sharing, no covariates | 0.93467 ± 0.03536 | 0.93984 ± 0.01230 |
| C2: target MLP, no spatial sharing, six covariate groups defined above | 0.93397 ± 0.04085 | 0.98733 ± 0.01868 |
| C3: target MLP, neighbor sharing, Kinsa | 0.97920 ± 0.06795 | 0.97501 ± 0.05111 |

The focused scientific checks in `scripts/vintage_overnight_scientific_checks.py`
passed for both held-out seasons and empirical/learned generators. They check
that donor origins belong only to the permitted reporting-error season, finalized
future labels are identical before and after augmentation, input episodes are
not mutated, and admissions-only draws leave ED values and covariates exactly
unchanged in 2022–23/2023–24. The recent-season actual-report treatment is
intentionally allowed to change all available input signals. Run these checks
with `PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_scientific_checks.py`.

## C2 matched three-seed empirical comparisons

C2 is the target MLP without spatial sharing, using inpatient, wastewater,
Kinsa, ILINet, clinical-lab and FluSurv covariates. For the forward benchmark it
was retrained on 2022–23, 2023–24 and 2024–25 with the following training-input
treatments, using only 2024–25 to construct artificial reporting errors. All
future prediction labels remained finalized. Evaluation inputs were identical
2025–26 Wednesday reports with B2 source lags and documented absent-archive
final-value proxies. Mean six-target relative WIS is lower when better.

| Training-input treatment | Mean forward score | Change versus finalized training | Seeds improving out of three |
| --- | ---: | ---: | ---: |
| Finalized inputs in all three seasons | 0.98733 | — | — |
| Half-strength empirical errors in all three seasons | 0.98778 | +0.05% | 1 |
| Half-strength errors in 2022–23/2023–24; 2024–25 finalized | 0.97395 | −1.36% | 1 |
| Half-strength errors in 2022–23/2023–24; actual 2024–25 reports | 0.97874 | −0.87% | 2 |

These small forward gains are variable across seeds. The secondary forward
flu/COVID-admission score worsens from 0.94281 after finalized-input training
to 0.94925, 0.95130 and 0.97991 for the three respective treatments. In the
all-three-season half-strength arm, mean COVID admissions improve from 0.9712
to 0.9343 and RSV admissions from 1.1863 to 1.0822, but flu admissions worsen
from 0.9144 to 0.9642 and all three ED targets worsen. Thus the admissions-only
follow-up remains a hypothesis, not a confirmed fix derived from the seed-42
result.

For retrospective 2024–25 evaluation, C2 instead trains on 2022–23, 2023–24 and
2025–26, constructs errors only from 2025–26 and learns the same finalized
future labels. The flu/COVID-admission score is 0.93397 for finalized-input
training, 1.00346 for all-three-season half errors, 0.99514 for early-two half
errors with recent finalized inputs, and 0.84579 for early-two half errors with
actual recent reports. The last improves all three seeds by an aggregate 9.44%.
The recent-actual-only control in the follow-up is needed to determine how much
of this comes from real recent-season inputs rather than artificial early-season
vintaging.

The admissions-focused report explicitly imports the matched finalized-input
training controls from the first round on the same L40 hardware:

```bash
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_report.py -e vintage-overnight-admissions-focused-20261005 --reference-experiment vintage-overnight-round1-20261005
```

Its primary percentage changes compare every arm to those finalized-input
controls. A separate column compares the early-two synthetic arms to the
follow-up's finalized-early-two/actual-recent-input control; this isolates the
additional effect of artificial early-season vintaging. That second contrast
is intentionally absent for the all-three-season synthetic arm. Seed-paired
absolute differences and the number of improving seeds accompany means; they
are only reported after three matched seeds complete. Three seeds quantify
training variability, not independent prospective evaluation seasons.

## Nonlinear reporting-generator screen

A second learned reporting model was screened before forecast retraining:
shallow histogram gradient boosting with 80 iterations, eight leaves, minimum
50 cells per leaf, learning rate 0.05 and L2 penalty 20. It predicts signed log
report/final error from normalized final level, two growth features, annual
sine/cosine and holiday proximity. It was compared with ridge and with ridge
plus boosted residuals, which retains ridge's location effects. Each model was
fitted to three contiguous-origin blocks within the permitted donor season and
predicted the fourth block; neither forecast evaluation season supplied fitting
pairs. These are reporting-error prediction results, not forecast scores.

For current-week admissions from donor 2024–25, ridge RMSE is 0.1227/0.1660/0.1762
for flu/COVID/RSV; standalone boosting gives 0.1325/0.1760/0.1884, and ridge plus
boosting gives 0.1256/0.1669/0.1780. For donor 2025–26, ridge gives
0.1256/0.1550/0.1781, standalone boosting 0.1321/0.1647/0.1822, and ridge plus
boosting 0.1351/0.1576/0.1783. Lower RMSE is better. Neither nonlinear candidate
improves any of the six current-week admissions comparisons, so no forecasting
GPU batch was assigned to this candidate. This does not rule out other nonlinear
models, nor does it contradict a report-to-final nowcaster: the direction and
prediction task here are final-to-report error generation.

The four-block comparison cannot evaluate temporal generalization of current-week
2024–25 ED errors because their archive pairs are concentrated in one block;
identical ED entries are not evidence of equivalence. Full results are in
`nonlinear-reporting-blocked-validation.csv`. Reproduce with:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_nonlinear_diagnostics.py
```

At 05:20 EDT, the first admissions-focused allocation started after idle
first-round owner `3800158_2` was released. A second idle first-round owner,
`3800158_3`, was released shortly afterward. No active fit was cancelled.
An attempted extension of the first follow-up owner's running time limit to
08:24 EDT was denied by Slurm (`Access/permission denied`); the original
135-minute limits remain. The contemplated balanced-pilot scheduling hold was
therefore abandoned before any seed state changed, to retain the common
manager's existing cost and load balancing. The 36-run plan is unchanged.

## Completed first-round outcome

All 78 planned seed runs finished by approximately 05:29 EDT with both folds
scored and no failures. Six deliberately narrowed full-strength comparisons
retain only their seed-42 pilots; all other configurations have seeds 42/43/44.
C1, the pathogen MLP with distance sharing and no covariates, remains the best
forward mean among these completed runs when trained on finalized inputs:
0.93984. Every three-seed C1 artificial-vintaging treatment worsens its mean;
the closest is half-strength empirical errors in 2022–23/2023–24 while retaining
finalized 2024–25 inputs, at 0.94604 (+0.66%). Training labels stay finalized;
evaluation is the same six-target 2025–26 Wednesday-report benchmark with source
lags and absent-archive final-value proxies.

The clearest within-architecture early-season candidate is C3, a target MLP
with neighbor sharing and Kinsa. Trained on 2022–23/2023–24/2024–25, with learned
conditional **mean-only** reporting errors from 2024–25 applied to the first two
seasons and actual reports in 2024–25, it scores 0.94960 versus 0.97501 after
all-finalized input training (−2.61%). Both use unchanged finalized future labels
and identical 2025–26 evaluation inputs. Two of three seeds improve; the paired
absolute score difference is −0.02541 with SD 0.03055 across seeds. This does not
beat the stronger C1 finalized-input model. Adding empirical residuals to that
C3 learned mean gives 0.95873 (−1.67%, two of three seeds improve).

For the retrospective fold, the corresponding C3 models train on
2022–23/2023–24/2025–26, learn reporting errors only from 2025–26, and receive
2024–25 Wednesday inputs. The flu/COVID-admission score improves from 0.97920
with finalized training inputs to 0.83555 with early-two learned means and
actual recent reports, or 0.82555 with learned means plus residuals; all three
seeds improve in both cases. Final labels are unchanged. The follow-up's
actual-recent-only control is still needed to attribute these benefits to
artificial early-season vintaging rather than real recent-season reports.

All four admissions-focused allocations started between 05:20 and 05:29 EDT.
Owners 0/1 retain 135-minute limits; pending owners 2/3 were extended successfully
to 150 minutes before starting. Thus the last allocations end around 07:58 EDT,
within the 08:30 research deadline. The complete common-manager ranking and
matched-comparison figures are in `vintage-overnight-round1-20261005/`.

## First admissions-focused three-seed results

For C1, trained on 2022–23/2023–24/2024–25 with finalized future labels and
reporting errors learned only from 2024–25, half-strength empirical errors
restricted to admissions in all three seasons score 0.94444 ± 0.03289 on the
six-target 2025–26 Wednesday-report benchmark. ED inputs remain finalized during
training. This is better than the matched all-signal half-strength treatment
(0.95561), but still 0.49% worse than finalized-input training (0.93984); only one
of three seeds improves against that finalized-input control. Evaluation inputs,
source lags and absent-archive final proxies are identical for every arm.

C1 with learned half-strength admission errors in 2022–23/2023–24 and actual
2024–25 reports scores 0.95857 ± 0.03448 (+1.99%; one of three seeds improves).
That treatment changes both strength and affected signals relative to the
first-round full-strength learned generator, so their difference cannot be
attributed solely to restricting perturbations to admissions. The
actual-recent-only control was not complete at this interim report.

For retrospective 2024–25 flu/COVID-admission scoring, training instead uses
2022–23/2023–24/2025–26 and errors only from 2025–26, with unchanged finalized
future labels. The two respective C1 treatments score 0.87064 and 0.86100 versus
0.93467 after finalized-input training. The latter improves all three seeds.
The forward and retrospective columns have different benchmark target support.

## Final small dilution comparison

After the first two C1 admissions-only treatments failed to beat finalized-input
training on the forward fold, one additional three-seed C1 comparison was queued:
half of training episodes remain clean and half receive half-strength empirical
reporting errors across all three training seasons. The same strong C1 architecture
learns unchanged finalized future labels. For 2025–26 evaluation it trains on
2022–23/2023–24/2024–25, with errors only from 2024–25; for retrospective 2024–25
evaluation it trains on 2022–23/2023–24/2025–26, with errors only from 2025–26.
Both use unchanged Wednesday-report evaluation inputs, B2 source lags and the
explicit final-value proxies. This tests the clean/noisy mixture that was most
consistent on other fully replicated weekend backbones, for which C1 had no
completed comparison. It is not an architecture or missingness-mask sweep.

Job `3818337` uses one L40 with three concurrent fits and a 45-minute limit.
Its 08:25 EDT deadline requires a start by 07:40. Admissions-focused completion
has priority; this queued job will be cancelled if that work needs the capacity.
Plan and manager commands, in the private remote project:

```bash
PYTHONPATH=src .venv/bin/python scripts/plan_vintage_overnight.py -e vintage-overnight-c1-diluted-20261005 --diluted
LANES=3 GPUS=1 sbatch --job-name=vintage-overnight-c1-diluted-20261005 --array=0 --time=00:45:00 --deadline=2026-10-05T08:25:00 scripts/jlessler.sbatch vintage-overnight-c1-diluted-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner status -e vintage-overnight-c1-diluted-20261005
PYTHONPATH=src .venv/bin/python -m chromantis.experiment.planner rank -e vintage-overnight-c1-diluted-20261005 --no-plots
PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_report.py -e vintage-overnight-c1-diluted-20261005 --reference-experiment vintage-overnight-round1-20261005
```

`reporting-metadata-clarification.json` distinguishes final-truth support from
observed archived/final-pair support for every target and the C2 covariates in
each permitted donor season. The older inherited `target_evidence_fraction`
field means final-truth support only. Its inherited `unsupported_cells=unchanged`
description is correct for empirical errors but not for learned conditional
means, which can extrapolate into cells without archived pairs. This sidecar
clarifies the completed pinned runs without altering their fitted inputs,
checkpoints or scores. The integration patch now labels those quantities and
the learned extrapolation explicitly in newly generated metadata. The metadata
addition changes no numerical computation. Reproduce the audit with
`PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_metadata.py`.

The nonlinear reporting-model diagnostic uses scikit-learn 1.7.2, already
available in the shared Longleaf virtual environment. The ridge generator uses
NumPy. The paired-forward plot can be regenerated after the first-round report
with `PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_paired_plot.py`.

## Allocation handoff at 06:52 EDT

One C3 learned half-strength admissions fit, seed 44, started at 06:37 on an
allocation ending at 07:35; its matched seed had taken 63.5 minutes. After eight
minutes the fit subprocess was deliberately stopped and its failed attempt
preserved. Two earlier-ending queue-manager processes were paused for about
seven minutes while their independent training children continued. The same
seed and unchanged numerical configuration restarted on owner `3804709_2` at
about 06:52, with an allocation ending at 07:58. Both managers resumed and
pending work was restored. The guarded controller had an eight-minute fail-safe
and a `finally` block to resume managers. Its record is
`data/experiments/vintage-overnight-admissions-focused-20261005/reroute-record.json`.
This was an allocation-timing intervention, not a selection based on fit quality.
The new attempt is included once, through the ordinary common scorer.

## Forward C3 score components before attribution control completes

The C3 target MLP with neighbor sharing and Kinsa, trained on
2022–23/2023–24/2024–25 with learned conditional mean errors in the first two
seasons and actual recent-season reports, has forward six-target mean WIS ratio
0.94960 versus 0.97501 after finalized-input training. Both learn unchanged
finalized future labels and evaluate the same 2025–26 Wednesday reports with
the stated schedules and proxies. Its flu-admission target ratio changes from
0.9430 to 0.8747 and RSV admissions from 1.0906 to 0.9758. Their normalized
underprediction penalties fall from 0.4144 to 0.3327 and 0.7442 to 0.5970.
Nominal 90% interval coverage rises from 68.4% to 78.5% for flu admissions and
67.7% to 75.3% for RSV admissions; neither reaches nominal coverage. COVID
admissions worsen from 0.9425 to 1.0139 as overprediction rises. These are
three-seed target diagnostics, not independent confirmation or proof that
artificial errors caused the gain. The actual-recent-only reference is needed
to isolate that contribution.

The generator threshold of 30 supported cells does not mean 30 independent
reporting dates: cells include locations. Donor-origin counts and archive-pair
support, including their seasonal concentration, are reported separately.
Three seed replicates quantify optimization variability; they do not account
for all epidemiological or reporting-process uncertainty.

For current-week ED errors in the 2024–25 donor season, all supported dates
fall into one of the four contiguous validation blocks. When that block is
held out, the fitting blocks have no observed current-week ED error pairs.
The shared-mean and ridge diagnostics therefore both fall back to zero and
their equal RMSE is not evidence that the reporting processes are equivalent
or that the learned model was meaningfully validated for those ED cells.
The admission diagnostics have evidence across blocks and are more informative.

## Numerical proxy fractions in scheduled inputs

`scheduled-input-proxy-audit.csv` counts the cells actually used by the scheduled
input protocol, before model fitting, by season, signal and report age. Unlike
the donor-error library audit, it includes all eligible origins: 51 in 2024–25
and 52 in 2025–26. At age zero, 2024–25 scheduled admission inputs use frozen-final
proxies in 33.1%–36.7% of available cells, and ED inputs in 88.5%–88.7%. For
2025–26 the respective fractions are 11.5%–13.5% and 20.5%–22.6%. Denominators
exclude absent historical final truth and cells hidden by source-specific lags.
The same source-season mixture applies to actual-recent training inputs and
evaluation inputs. Consequently, “actual recent reports” always means archived
numerical values where present plus the explicitly assumed frozen-final proxies;
it does not imply a fully observed archive. Labels are unchanged finalized
future values. Reproduce with `PYTHONPATH=src .venv/bin/python
scripts/vintage_overnight_proxy_audit.py`; this audits inputs and fits no model.

## C2 attribution with three reference seeds completed

The C2 target MLP without spatial sharing, using inpatient, wastewater, Kinsa,
ILINet, clinical labs and FluSurv, scores 0.96640 ± 0.02106 on forward 2025–26
when trained on finalized 2022–23/2023–24 inputs and actual-or-proxy 2024–25
reports. All labels remain finalized t+1 through t+4 and evaluation inputs are
identical Wednesday archived/proxy values. This improves the all-finalized-input
reference, 0.98733, by 2.12%; all three paired seeds improve.

Adding artificial errors to the first two seasons does not improve that
three-seed forward mean in the original comparison: half-strength empirical
errors worsen it by 1.28%, full-strength errors by 7.62%, learned conditional
means by 2.75%, and learned means plus residuals by 11.46%. The half-strength
arm improves two of three seeds but its worse seed reverses the mean gain.
For retrospective 2024–25 scoring, the actual-recent-only reference instead
uses 2025–26 reports and scores 0.90910 versus 0.93397 after finalized-input
training; adding half-strength empirical errors improves that retrospective
mean by a further 6.96%. The forward six-target and retrospective two-admission
objectives differ. These results support distinguishing use of actual recent
reports from the contribution of artificial first-two-season errors.

`actual-recent-attribution.csv`, `.md`, `.png` and `.pdf` explicitly compare
those matched inputs; three-seed references are required before an architecture
appears. Regenerate after the principal export with
`PYTHONPATH=src .venv/bin/python scripts/vintage_overnight_attribution.py`.

The admissions-only arms restrict which inputs are perturbed; they retain the
same joint donor-window eligibility rule as the original generator. That rule
requires some archived evidence within the latest four context weeks for every
target signal, excluding pre-archive/onboarding origins. Thus admission-only
perturbations still use the 34 eligible 2024–25 donor origins or 48 eligible
2025–26 origins, rather than all archived admission pairs from those seasons.
The restriction does not recover missing early-season donor dates. Calendar and
phase extrapolation beyond those supported dates remains a model assumption.

## C1 attribution with three reference seeds completed

For C1, the pathogen MLP with distance sharing and no covariates, finalized
2022–23/2023–24 inputs plus actual-or-proxy 2024–25 reports yield forward
2025–26 mean 0.98968 ± 0.07812, compared with 0.93984 ± 0.01230 after
finalized-input training in all three seasons. Actual recent inputs worsen
this mean by 5.30%. Adding learned half-strength admission errors in the first
two seasons improves the actual-recent reference by 3.14%, to 0.95857; two of
three paired seeds improve. Nevertheless, this remains 1.99% worse than
all-finalized-input training. Half-strength empirical errors in all early-season
signals likewise improve the actual-recent reference by 1.81%, but do not beat
all-finalized-input training. These are partial rescues of a worse C1 training
setup, not a better overall C1 result. Labels remain finalized t+1 through t+4
and every arm evaluates identical Wednesday archived/proxy inputs.

Retrospective C1 actual-recent-only training uses 2025–26 reporting inputs and
evaluates 2024–25 flu/COVID admissions. It scores 0.86245; adding learned
half-strength early-season admission errors scores 0.86100, a nearly unchanged
mean. The large retrospective improvement against all-finalized-input training
therefore comes mainly from actual recent reporting inputs in this comparison.

At 07:06 EDT, admissions owner `3804709_0` had zero active fits and no pending
work. It was released after a queue-lock check, allowing the final diluted C1
job `3818337_0` to start on its L40. No active fit was stopped in this release.
The dilution job has a 45-minute limit, ending about 07:51; all remaining
admissions fits continue on their existing allocations.

## Completed C2/C3 admissions-only all-season comparison

For C2, the target MLP without spatial sharing using six covariate groups,
training on 2022–23/2023–24/2024–25 with half-strength empirical admission
errors in every training season, while ED and covariate values stay finalized,
scores 0.96053 ± 0.03444 on 2025–26. Errors come only from 2024–25; finalized
future labels and Wednesday archived/proxy evaluation inputs are unchanged.
This improves all-finalized-input C2 training, 0.98733, by 2.72%; all three
paired seeds improve. It is an alternative to actual-recent training, not a
pure first-two-season attribution contrast: the recent season and affected
signals differ. Its retrospective score is 0.90603 when training instead on
2022–23/2023–24/2025–26 with errors only from 2025–26, versus 0.93397 after
all-finalized-input training.

The C2 learned half-strength admission generator in the first two seasons,
retaining actual recent reports, scores 0.97044 ± 0.01683 forward. Although
it improves all-finalized-input training by 1.71% in all three seeds, it is
0.42% worse than actual-recent-only training, 0.96640; only one of three seeds
improves that attribution reference. Its retrospective score is 0.84738.

C3 with neighbor sharing and Kinsa, using empirical half-strength admission
errors in all three training seasons and finalized ED/covariate inputs, scores
0.96745 ± 0.03656 forward versus 0.97501 after all-finalized-input training.
Only one of three paired seeds improves despite the 0.77% lower mean. Its
retrospective mean is 0.92468. These results use the same labels and evaluation
protocol above. The strongest completed C1 mean remains 0.93984 after
all-finalized-input training; no cross-architecture superiority claim is based
solely on one favorable seed.

Transporting reporting errors to 2022–23/2023–24 assumes that the permitted
donor season's reporting process is informative about those earlier seasons
after matching level, epidemic phase, calendar position and location. The
experiment does not establish that historical reporting systems were stable
across those years. The artificial-input goal is robustness to the specified
evaluation regime, not reconstruction of unobserved historical archives.

## C3 attribution with three reference seeds completed

The C3 target MLP with neighbor sharing and Kinsa, trained on finalized
2022–23/2023–24 inputs plus actual-or-proxy 2024–25 reports, scores
0.94373 ± 0.02378 on 2025–26. All future labels remain finalized and evaluation
inputs are identical archived/proxy Wednesday values. This is 3.21% better
than all-finalized-input C3 training, 0.97501; two of three paired seeds improve.
Its retrospective mean, using actual 2025–26 training inputs and evaluating
2024–25 flu/COVID admissions, is 0.86213.

Against that matched actual-recent-only reference, the original artificial
first-two-season treatments have worse forward means: learned mean only
+0.62%, learned mean plus residuals +1.59%, full-strength empirical +1.39%,
and half-strength empirical +5.32%. Hence their apparent improvements against
all-finalized-input training do not demonstrate added value from synthetic
first-two-season errors.

The admissions-only half-strength empirical treatment in the first two
seasons does score slightly better: 0.94058 ± 0.03156, 0.33% below the
actual-recent reference, with two of three seeds improving. It improves all
three seeds relative to all-finalized-input C3 training, by 3.53% in the mean.
Its retrospective score is 0.83249. The tiny forward incremental benefit over
actual recent reports alone should not be described as a robust breakthrough;
it is also essentially tied with the simpler all-finalized-input C1 model's
0.93984. The learned half-strength admissions-only C3 fit remains pending
at this interim comparison.

C2's empirical half-strength admission errors in the first two seasons,
with actual recent reports, finish at 1.00168 ± 0.06650 forward: 3.65% worse
than its actual-recent-only reference and 1.45% worse than all-finalized-input
training. Its retrospective mean is 0.86422.

Idle admissions owners 1 and 3 were released after locked checks confirmed
zero active fits and no pending work. At about 07:20 EDT, only the rerouted
C3 learned-admissions seed 44 remained on owner 2; the final C1 dilution
comparison continued on its separate allocation.

## Shared-target seasonal diagnostic

Without retraining, the secondary score restricts both evaluation seasons to
flu and COVID admissions with equal target weights. On forward 2025–26,
C2 trained on finalized 2022–23/2023–24 inputs and actual-or-proxy 2024–25
inputs scores 0.96370 on these shared targets versus 0.94281 after all-finalized
input training: 2.22% worse despite its better official six-target score. C3
with the analogous inputs scores 0.94502 versus 0.94275: 0.24% worse despite
its official six-target improvement. Both learn unchanged finalized future
labels and evaluate identical archived/proxy Wednesday inputs. Their
retrospective shared-target gains are 2.66% and 11.96%, respectively, when
training instead uses 2025–26 actual inputs and evaluation uses 2024–25.

The C2 model with half-strength empirical admission errors in all three
training seasons improves both forward objectives: 2.72% on the official
six-target score and 1.20% on shared flu/COVID admissions, to 0.93146.
In contrast, C1 with empirical half-strength admission errors in the first
two seasons plus actual recent reports improves forward shared admissions
by 3.11% but worsens the six-target score by 3.66%. Target support therefore
explains part, but not all, of the seasonal difference. The six-target forward
score remains primary. Complete values and unchanged-fit definitions are in
each experiment's `common-admission-season-results.csv`.

## Admissions follow-up completed

All 36 admissions-focused seed runs completed both seasonal folds by about
07:27 EDT. The last C3 learned half-strength admissions treatment, applied only
to 2022–23/2023–24 while retaining actual-or-proxy recent reports, scores
0.94701 ± 0.01996 forward and 0.84071 ± 0.02983 retrospectively. Forward
training uses 2022–23/2023–24/2024–25 with errors learned only from 2024–25;
retrospective training instead uses 2025–26 as the recent donor season. Both
learn unchanged finalized future labels and evaluate the same archived/proxy
Wednesday inputs as their references.

The learned treatment is 2.87% better than all-finalized-input C3 training,
but 0.35% worse than C3 trained with actual recent reports and no artificial
early-season errors. Only one of three seeds improves the latter reference;
the paired score difference is +0.00328 with SD 0.01701. Consequently, none
of the tested learned first-two-season generators improves the forward mean
beyond actual recent reports alone on C2 or C3. C1's learned admission
generator partly repairs its worse actual-recent setup but still does not beat
all-finalized-input C1 training.

The final common ranking for the completed follow-up uses the ordinary
`rank -e vintage-overnight-admissions-focused-20261005 --no-plots`, without
`--allow-incomplete`. Its owner 2 exited normally. The only intentional active
fit interruption was the documented timing-driven restart; that seed completed
on its second attempt. The final diluted C1 comparison is the only remaining
training work at this point.

## Target tradeoffs for the strongest C2 artificial-input treatment

C2's empirical half-strength admission errors in all three training seasons
reduce forward RSV-admission relative WIS from 1.1863 to 1.0153 and COVID
admissions from 0.9712 to 0.9476. Flu admissions are almost unchanged,
0.9144 to 0.9153. All three ED targets worsen: COVID 0.9635 to 0.9877, flu
0.9736 to 1.0094, and RSV 0.8052 to 0.8910. This comparison retrains the same
C2 target MLP with six covariate groups on 2022–23/2023–24/2024–25, uses
only 2024–25 reporting errors, preserves finalized future labels and evaluates
identical 2025–26 Wednesday archived/proxy inputs. The aggregate improvement
is driven mainly by RSV admissions under the specified admission-twice-ED
weighting; it is not an improvement for every target.

“Admissions-only” identifies which training input signals receive artificial
errors. Every target-specific MLP still uses the other target signals as
predictors, so perturbed admission inputs can change ED forecasts. Comparisons
with all-signal perturbations also restore finalized covariates, not only ED
values; C2/C3 cannot isolate an ED-only effect from that contrast. C1 has no
covariates.

`shared-target-season-comparison.csv`, `.png` and `.pdf` place the retrospective
flu/COVID-admission objective beside the forward shared-admission diagnostic
and the primary forward six-target objective. They use the same fitted
forecasts and three seeds in every cell. The completed C3 learned half-strength
admission treatment improves forward shared admissions by 4.02% relative to
all-finalized-input C3 training, despite failing to improve the primary
six-target score beyond actual recent reports alone. Target tradeoffs remain
material even when a learned reporting model helps selected admissions.
Regenerate with `PYTHONPATH=src .venv/bin/python
scripts/vintage_overnight_shared_targets.py`.

## Final diluted result and completion

The final C1 pathogen MLP with distance sharing and no covariates, trained with
half clean episodes and half half-strength empirical-error episodes across all
three permitted training seasons, scores 0.94765 ± 0.02118 on forward 2025–26.
Training uses 2022–23/2023–24/2024–25, with errors only from 2024–25; finalized
future labels and archived/proxy evaluation inputs are unchanged. This is 0.83%
worse than C1 trained on finalized inputs, 0.93984; only one of three paired
seeds improves. Retrospectively, training instead uses 2025–26 as the recent
error-donor season and evaluates 2024–25; its mean is 0.86518 ± 0.05055 versus
0.93467 after all-finalized-input training, a 7.43% improvement. The two score
columns have the previously documented different target support.

All 117 successful research seed runs completed both folds by 07:35 EDT:
78 first-round runs, 36 admissions-focused runs and three final dilution runs.
There are 37 fully replicated three-seed conditions and six single-seed pilots.
The primary Slurm arrays have exited; at 07:36 no research GPU allocation
remained. `final-job-state.json` records the scheduler and dispatcher audit.
Idle-owner cancellations are distinct from the single intentional timing-driven
fit restart, which completed successfully on attempt 002.

The compact final report is `results.md`. Full reproducible planning, launching,
status and ranking commands, including the narrowed first-round dispatch and
source-version distinction, are in `manager-commands.md`. All scores, paired
comparisons, graphs and scientific assumptions are retained in this directory.

## Final source and metadata integrity checks

Independent `reproduction-source-v1` and `reproduction-source-v2` directories
were prepared on Longleaf by copying the original pinned sources and their
notifiers. All 74 source/configuration files in each copy match the originals;
checksums are saved in `reproduction-source-checksums.json`. These copies avoid
self-replacement during a new plan and preserve the old generator versions.
No new experiment was planned or launched during this preparation.

The latest integration source also limits learned-model metadata fields to
`generator=ridge`. The older pinned empirical runs inherited a conditional-model
and centered-residual description even though they used raw empirical error
windows. `reporting-metadata-clarification.json` explicitly corrects this
description, alongside the truth-support/archive-pair distinction. These
descriptive changes do not alter fitted numerical inputs, checkpoints or scores.
The final integration patch applies cleanly to the saved initial private source.
