# Direct forecasting with covariates and geographic information sharing

This study tests the combined training recipe suggested by the completed B1-to-B0
comparison, then asks which available covariates help and how information should
move across targets and locations. It fits direct forecasters from scratch. There
is no separate nowcasting stage. In the old terminology this is a B2-style model
using a B0 forecasting backbone.

The combined recipe is a hypothesis. The completed audit changed choices
consecutively; it did not establish that their individually promising settings
work best together. The same three historical evaluation seasons have already
informed model selection, so this is retrospective development research, not a
new prospective confirmation.

## Common training and evaluation recipe

| Choice | Setting |
| --- | --- |
| Historical target and covariate inputs for fitting | Complete finalized histories from allowed training weeks, before artificial target masking |
| Inputs at validation and forecast issuance | Operational availability inferred from 2025–26 reporting and actual deadlines; reported vintages preferred, finalized historical proxies where availability is assumed but a vintage is absent |
| Forecast labels | Finalized target truth; no separate revision-correction stage |
| Target normalization | B0 normalization fitted within the training partition |
| Covariate normalization | Observed fitting data only; no held-out or inner-validation weeks enter fitted statistics |
| Artificial missingness | Applied to target inputs in 50% of fitting examples, mixed recent/gap/outage patterns in proportions 50/30/20; no artificial covariate masking |
| Finality indicator | Retained |
| Validation weeks | B0 fixed calendar, common to every arm; no peak-dependent date selection |
| Validation simulations | Current fixed draws, 256 members |
| Training-error multiplier | Current smaller scale, preserving target/geography/season weights |
| Evaluation simulations | **256 members**, common seeds and draw protocol across matched arms |
| Seeds | 42, 43, 44 |
| Evaluation outcomes | All six admissions/ED targets on identical frozen benchmark support |
| Scores | Equal season weight; within season admissions weight 1 and ED weight 0.5; states/DC weight 0.8 equally and native US weight 0.2 |

Complete training histories do not by themselves authorize unavailable forecast inputs. Data
held out for either evaluation or epoch selection remain excluded from training
histories and fitted statistics. The user explicitly confirmed finalized
covariates as well as targets for training. Source reporting delays restrict
validation and forecasting, not the complete training histories. Artificial
masking changes target inputs, not the definition of a source's reporting deadline.
Following the user's clarification, doubtful historical availability is assumed
available and documented. This is an operational-availability hypothesis rather
than a claim that the checked archives reconstruct every historical report.

Common architecture settings are 12 context weeks, width 64, latent dimension
16, fourth-root admission rates, logit ED, calendar/geography/dynamics features,
shared output heads, legacy decoder, global latent noise, batch size 8, 128
training members, Adam learning rate 0.001, no weight decay, a 300-epoch cap and
patience 30. Selected epochs are refitted on the full allowed training split.
These settings stay fixed within every paired comparison. Complete finalized
covariate fitting histories receive no additional artificial dropout.

## Three supported backbones

| Backbone | Fitted output groups | Epoch cap | Prior evidence |
| --- | --- | ---: | --- |
| Pathogen MLP | Three models, admissions and ED together for each pathogen | 300 | B0.1 rank 2, WIS ratio 0.895; B1 300-epoch final-flag/mixed-mask leader, 0.941 |
| Target MLP | Six separate target models | 300 | B0.1 rank 1 used cap 100, 0.883; its cap-300 version ranked 3, 0.895 |
| Target multiscale convolution | Six separate target models | 300 | B0.1 rank 5 used cap 100, 0.909; a distinct temporal inductive structure |

These historical results motivate the backbone families, not an assertion that
the new equal-budget variants are superior. All use cap 300 and patience 30 to
make training/selection budgets consistent; this extends the two cap-100 winning
families and early stopping may select beyond their old budget. The historical
scores have different input protocols and must not serve as the new study's
numerical controls. Every control is trained afresh.

Evidence: [B0.1 architecture ranking](../legacy-v0/results/b0-1-crosses/index.md),
[B1 experiment overview](../legacy-v0/results/b1-conclusions.md), and
[B1 300-epoch ranking](../legacy-v0/results/b1-overnight/epoch300/configuration-ranking.csv).

## Source attribution

The primary independent-location source panel has **16 covariate sets per
backbone**: no external covariates; each of seven source groups alone; all seven
together; and seven all-source arms each dropping one group. A source group can
contain more than one pathogen-specific channel.

| Source group | Panel inputs |
| --- | --- |
| Kinsa | National cough/cold/flu signal |
| Inpatient claims | Flu and COVID inpatient claims |
| Outpatient claims | Flu and COVID outpatient claims |
| Wastewater | Flu/COVID/RSV WVAL-like indices |
| ILINet | Unweighted influenza-like-illness proportion |
| Clinical laboratories | Flu percent positive |
| FluSurv | Flu hospitalization rate in covered catchments |

Compact summary representation is the primary choice, held fixed across source
arms. It uses the latest value, recent observed mean, observed-time slope,
standard deviation, coverage and latest-report age after fitting-only
standardization and the existing signed-log transform. Representation remains
fixed throughout this study, so its effect is not mixed into source attribution.

A source-alone contrast measures incremental value beyond the target histories.
An all-minus-one contrast measures contribution conditional on the other sources.
Their disagreement is informative about redundancy or interactions. This is not
the full factorial of all source combinations and cannot identify every possible
interaction. The all-source bundle includes Kinsa; it uses the WVAL-like
wastewater index, not the separate percent-rank index.

Every arm keeps all six target histories as inputs. Thus “Kinsa only” means
Kinsa is the only **external covariate**, not that target histories disappear.
The user clarified that “ED visits only” meant **ILI only**: that requested arm
is the ILINet singleton above. No own-target, admissions-only or ED-only
target-history ablations are included.

## Geographic and target information exchange

Every tested geographic method has its own no-external-covariate control and
matched source arms. The geographic panel crosses no covariates, Kinsa, ILINet
and the all-source bundle with all three backbones and the eight methods below.

| Method | Question |
| --- | --- |
| `none` | Are local histories plus directly supplied national covariates sufficient? |
| `pooled` | Does a shared mean of observed state contexts plus a separate US context help? |
| `attention` | Does learned attention over complete location contexts help? |
| `national_broadcast` | Does sharing the native-US learned context with every location help? |
| `gated_pool` | Can each location control how much shared pooled/national context it uses? |
| `pathogen_spatial` | Does location attention over paired admissions/ED representations for each pathogen help? |
| `target_spatial` | Does target-specific attention across locations help? |
| `joint_location_target` | Does attention across target-location tokens help transfer information across both dimensions? |

An additional matched geographic-encoding panel adds a learned 8-dimensional
location-ID embedding under `none` and `attention`, for the same four covariate
bundles and three backbones. Every arm already receives log population and a
native-US indicator; this comparison adds location identity to those features.

The equal-state mean excludes US and carries native US separately, avoiding
double counting. It pools learned representations rather than counts with
incompatible scales. These methods study national information exchange; they
do not establish distance, adjacency, travel or regional transmission effects.

**Kinsa national values and availability are broadcast directly to every location
in every selected Kinsa arm**, including `none`. This was already implemented in
the preceding geography study. The new `national_broadcast` method is a separate
comparison involving the learned US context, not the first opportunity for a
state forecast to use Kinsa. Broadcasting replicates information, not labels or
scoring weight, and does not create state Kinsa measurements.

`pathogen_spatial` encodes admissions and ED together for each pathogen before
attention across locations. `target_spatial` has a separate location-attention
stream for each input target.
`joint_location_target` attends across all target-location tokens. Target-token
encoders combine their scoped target histories, geographic features and a
dedicated projection of the selected covariate representation. Covariates travel
explicitly through the target, pathogen and joint remote-token streams, as well
as entering the local context. A token can send when its scoped target or a
selected covariate is observed. Missing senders are excluded from attention;
an episode without an observed sender contributes no attention update.

Separate target/pathogen fits still receive the allowed six-channel input tensor,
so these attention scopes remain meaningful: another target's history can
influence the fitted outcome. They do not pass fitted predictions between
independently trained models or jointly optimize those models. In particular,
same-target spatial attention is not equivalent to cross-target attention.

## Reporting availability and support

The preceding B1-to-B0 audit reconstructed 2025–26 **target** availability at
actual deadlines. It retained older Wednesday masks for prior seasons and left
covariate masks unchanged. Reusing that panel alone does not establish corrected
covariate availability. A strict evidence panel first checks each source's dated
reports against actual cutoffs, including holidays. The main study then derives
source-specific operational lag assumptions from 2025–26 evidence and follows
the user's instruction to assume availability in doubtful cases. Near-complete
admissions and ED availability provides no evidence about Kinsa, claims or
wastewater delays: those sources receive their own assumptions.

The main dataset is `data/processed/panel-b2-operational.npz`. During validation
and scoring it prefers an actual reported vintage; when its operational lag rule
assumes an input available but no vintage exists, it substitutes the finalized
historical observation. This avoids equating incomplete archive acquisition
with a real reporting outage, but introduces possible revision optimism. It is
not a fully reconstructed real-time backtest. Both assumptions and substituted
cell counts must accompany results. The preceding geography study used finalized
values wherever an archived report existed, so fresh controls remain essential.
A change from an older published score cannot be attributed solely to the new
training combination or covariates.

When one joint forecast serves several Hubs, use the earliest participating Hub
deadline for every input. A later deadline for one output must not leak later
reports into an earlier-deadline output. Calendar reference dates and context end
weeks stay unchanged when submissions receive a holiday extension.

The common-cutoff holiday audit used December 29, 2025 at 23:00 Eastern for the
December 27 reference date, and January 4, 2026 at 23:00 Eastern for the January 3
reference date. The extra FluSight day did not change audited target presence.
See [completed chain and deadline assumptions](../results/b0-reproduction/index.md)
and [common deadline policy](../results/b1-to-b0-chain/common_deadline_policy.json).

Kinsa's strict archive starts in April 2026, and some other archives are also
incomplete historically. Their operational arms can therefore contain finalized
historical proxies under inferred availability. Report finalized fitting
availability, actual vintage coverage, assumed availability and proxy use
separately by source, fold, season and lag. A Kinsa improvement supports its
usefulness under these stated reporting assumptions; it does not verify older
Kinsa publication dates or its unrevised historical values. No second strict
archive training sweep or separate retrospective Kinsa sensitivity is included.
The data builder's source evidence and operational policy files contain the
concrete lag choices and substitution counts.

## What the previous experiments support and what is excluded

The geography screen completed 189 runs: 63 formulations, three seeds, three
seasons. Independent mixed summaries improved against their matched control in
all three seeds (relative WIS 1.124 to 1.036). Kinsa summaries improved states/DC
by 5.1% across all three seeds, already using national broadcast. Smoothed new flu
sources improved overall by 6.6%. Those bundle gains did not identify which
individual source mattered.

Geographic complexity was not generally beneficial: attention won only 5 of 21
matched comparisons and pooling 4 of 21. Attention worsened mixed summaries
from 1.036 to 1.210. They remain explicit comparators under the new training
recipe, not presumed improvements. The new gated/national and target-specific
comparisons ask more specific information-sharing questions.

The selected top-half no-mask follow-up completed 96 runs. Removing artificial
masking worsened 29 of 32 formulation means, including all three seeds of the
leading mixed-summary model. This motivates retaining masking, with the caveat
that the models were selected under the masked condition.

The B1-to-B0 chain found its largest recent-season improvement when complete
training histories were restored. Normalization helped 2025–26 but not the
overall mean. Restoring B0 validation draws had negligible benefit. Restoring
the larger B0 training-error multiplier worsened 2025–26 in every repeated fit.
The new study fixes these protocol decisions rather than rerunning their entire
consecutive audit.

Excluded from this study: separate or auxiliary nowcasting, revision-correction
branches, a no-mask sweep, finality-flag removal, stochastic trend decoders, raw
count/rate-only admission encodings, direct quantile prediction, and a broad
raw/smooth/summary/shared representation grid. The previous shared bottleneck
did not offer a general improvement; summary-source attribution is more useful
than repeating that full screen. No calibration or winning seed mixture should
be fitted using final evaluation outcomes.

Evidence: [geography conclusions and matched comparisons](../results/forecast-geography-v2/conclusions.md),
[original design](forecast-covariates.md), [no-mask follow-up](forecast-no-mask.md),
and [consecutive B1-to-B0 audit](../results/b0-reproduction/index.md).

## Analysis and operational record

Report paired differences against the correct backbone/exchange
control, by seed, season, target and states/DC versus US. Report interval coverage
alongside WIS and source availability alongside source attribution. Three seeds
measure optimization variation; they are not three independent epidemic samples.
Do not select a source or exchange method solely by the smallest aggregate mean.
The benchmark seasons have different target support: flu admissions in 2023–24,
flu/COVID admissions in 2024–25, and all six targets in 2025–26.

The planned grid contains **156 configurations × three seeds = 468 runs**, each
with three held-out seasons: **1,404 outer folds**.

| Panel | Configurations | New seed runs |
| --- | ---: | ---: |
| Source attribution: 3 backbones × 16 source sets, `spatial=none` | 48 | 144 |
| Geographic/target exchange: 3 backbones × 4 bundles × 8 methods, subtract 12 overlapping independent controls | 84 additional | 252 |
| Location identity: 3 backbones × 4 bundles × 2 exchange methods, embedding dimension 8 | 24 additional | 72 |
| Total unique configurations | 156 | 468 |

One seed run contains three held-out-season folds; independent target/pathogen
fits and inner epoch-selection fits multiply the actual training jobs within
each run. All evaluation uses 256 samples.

## Completed results — September 28, 2026

All **468 runs and 1,404 outer folds** completed, with 256 evaluation draws.
The final ranking is `ranking-ac92a3c7dbcc`; the earlier partial ranking is excluded.
See the [analysis, source and spatial comparisons, and fan plots](../results/b2-direct-research-v1/index.md).
The leading configuration uses target multiscale encoders, all sources and pooling
(WIS ratio 1.059). Its matched covariate gain is concentrated in 2024–25;
coverage and forecast-time finalized proxies limit the interpretation.

## Launch record — September 27, 2026

Experiment **`b2-direct-research-v1`** was submitted on Longleaf as L40 array
**2699893** (four GPUs, six concurrent seed processes per GPU) and H100 array
**2699894** (two GPUs, ten concurrent seed processes per GPU). All workers share
one queue. Startup verification found all six GPU allocations running, all 44
seed processes advancing through fitting, and no error logs. A seed process handles all three folds. The scheduler allocations have
a 48-hour limit; incomplete work is resumed through the manager, not by re-planning.

The four-configuration smoke experiment `b2-direct-research-smoke-v1` completed
all 12 folds on L40 under job **2699171**, with 256 evaluation members and the
full width/batch/member dimensions. It used the prior panel solely to validate
execution, not to estimate research effects. Thirteen targeted leakage and
information-flow checks passed. The final operational panel was additionally
checked for unchanged truth, unchanged observed vintages, protected explicit
nulls, no future context weeks, and correct holiday cutoffs.

Main dataset: `data/processed/panel-b2-operational.npz`, SHA256
`84ca1a89f3da43cbb41a794e06b09c039aaf26d4f4c3443a42dbd1f4be8ebd81`.
The manager pins its code, dataset hash, population hash, benchmark support hash,
and 256 evaluation members. The frozen design and reporting script live inside
the experiment directory. Evaluation draws reset to seed + 1000 per fold;
validation retains the current fixed seed + 900000 per component.

From `/proj/jlessler/projects/tapestry-all/tapestry` on Longleaf:

```bash
# Plan once only: the recipe refuses to replace an existing experiment.
.venv/bin/python experiments/b2-direct-research.py --plan
LANES=6 GPUS=6 sbatch --job-name=b2-direct-research-v1 --array=0-3 scripts/jlessler.sbatch b2-direct-research-v1
LANES=10 GPUS=6 sbatch --job-name=b2-direct-research-v1-h100 --array=0-1 --nodelist=g1803jles02 scripts/jlessler.sbatch b2-direct-research-v1
.venv/bin/python -m tapestry.experiment.planner status -e b2-direct-research-v1
.venv/bin/python -m tapestry.experiment.planner rank -e b2-direct-research-v1
```

`status` prints the exact retry/resubmission command for unfinished work. For
in-progress ranking use `rank --allow-incomplete --no-plots` after at least one
seed run completes. Notifications remain enabled. The after-any analysis job **2699901**
runs the manager ranking and the frozen research reporter after both arrays end;
it explicitly labels any incomplete comparisons.

Assumed release delays after the reference Saturday are targets 4 days;
inpatient claims 3; outpatient flu/COVID 0/1; Kinsa 1; ILI, clinical labs and
FluSurv 6; wastewater flu/COVID 13 and RSV 9. These govern only archive-gap
substitutes: an actual observed release remains available even if earlier than
the typical delay. The substitute release must precede the actual cutoff.
[Exact policy and hashes](../data/availability/provenance/operational-policy.json),
[source lags and counts](../data/availability/provenance/operational-source-lags.csv),
and [context-level proxy usage](../data/availability/provenance/operational-proxy-coverage.csv)
make every assumption auditable. Fixed delays and finalized substitutes are an
optimistic operational hypothesis; effects do not establish prospective skill.

Smoke manager commands:

```bash
.venv/bin/python experiments/b2-direct-research.py --plan --smoke --dataset data/processed/panel.npz
LANES=2 GPUS=1 sbatch --job-name=b2-direct-research-smoke-v1 --array=0-0 --time=00:30:00 scripts/jlessler.sbatch b2-direct-research-smoke-v1
.venv/bin/python -m tapestry.experiment.planner status -e b2-direct-research-smoke-v1
.venv/bin/python -m tapestry.experiment.planner rank -e b2-direct-research-smoke-v1 --no-plots
```


The dedicated scientific report is separate from the planner's generated
`index.md`. It reads the experiment's frozen `research-design.json` and the
latest manager ranking. It writes paired source additions/removals, spatial and
embedding effects to CSV, including target/season/geography decompositions, plus
static figures and `docs/results/b2-direct-research-v1/research.md`.

```bash
.venv/bin/python -m tapestry.experiment.planner rank -e b2-direct-research-v1
.venv/bin/python analysis/b2-research/report.py --experiment b2-direct-research-v1
```

For a progress snapshot, add `--allow-incomplete` to the rank command. The research
report marks incomplete runs and pairs explicitly; only existing candidate/control
seeds enter a comparison. Error bars are across-seed standard deviations, not
confidence intervals. `--ranking PATH` selects a specific saved ranking.

### Decision log

- 2026-09-27: Selected three historical backbone recipes and a matched source
  singleton/leave-one-group-out design. Retained artificial missingness and final
  flags, adopted complete historical target and covariate training inputs, B0 normalization
  and fixed validation calendar, current fixed validation draws and smaller loss
  scale. All backbones use the same cap 300/patience 30, extending historical
  cap-100 winners as new equal-budget variants. Added explicit geographic/target
  exchange and location-ID embedding panels and the requested
  ILI-only covariate arm. All six target histories remain inputs. The combination
  remains untested; source availability must be audited
  separately from target availability.
- 2026-09-27: User clarified that expected reporting availability should be
  inferred from 2025–26 and doubtful cases assumed available. The main panel
  therefore uses actual vintages where possible and finalized historical proxies
  otherwise under explicit operational lag assumptions. Results must label the
  resulting potential revision optimism and report source-specific proxy counts.
