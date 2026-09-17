# B1 decisive execution

This implements `b1-decisive-experiment.md` at handoff 33412b2. The experiment
is retrospective engineering model selection on already-used development seasons.
It does not identify the historical B0/B1 input effect or operational performance.

## Fixed assumptions and implementation

- All candidates use the exact rank-1 B0 recipe (target components, h12, fourth-root
  admissions, logit ED, geography/dynamics/calendar, MLP width64, latent16,
  shared legacy head/global noise, 100-epoch cap/patience30, batch8, learning
  rate .001, 128 training and 256 validation members). Mask probability is .5,
  conditional recent/gap/outage probabilities .5/.3/.2.
- A retains its canonical v2 scenario identity and exact numerical path. B's
  `direct_finalflag` pipeline adds per-cell final status to the B0 context/focal
  inputs. C's `joint_aux025` adds independent recent horizon embedding, norm and
  probabilistic heads to B's encoder. Both heads receive the same member noise.
  The latest visible target value (or B0's missing-history prior) anchors both
  heads; recent predictions never enter future decoding. Marginal CRPS does not
  identify joint path dependence.
- All three use A's forecast-eligible origins, folds and component eligibility.
  Selection uses forecast validation loss only, under the existing fixed
  validation masking. C uses forecast + .25 recent CRPS during training, with
  each task's scientific weights and partition-only scales. Absent recent loss
  is zero. Final flags are intersected with visibility, hidden finals can be
  reconstructed, visible finals bypass recent predictions and are unscored.
- Ten seeds 42–51. Evaluation uses 2,048 draws per fit, matching A's completed
  seeds. Natural-input totals must reproduce the existing rankings before use.
  Shared scoring uses frozen final observations, q23 levels, task keys and
  ensemble denominators; location ratios, US .2/states .8, target weights
  admissions1/ED.5, and equal seasons are unchanged in every stress condition.
- Paired seed stress comparisons keep the archived seed-dependent masks,
  verified equal across A/B/C. Predictive mixtures require one input condition:
  their stress masks use seed42 for every fitted distribution (plus the existing
  origin index *101 and mask RNG +3000 offsets). Natural mixtures use the saved
  raw draws. Stress mixture draws are regenerated from checkpoints under the
  common masks. Each mixture pools 2,048 draws per seed, 20,480 total per cell,
  before quantiles and the existing admissions rounding. No quantile averaging.
- Seed uncertainty: percentile 95% bootstrap intervals, 10,000 paired seed
  resamples. Temporal uncertainty: percentile 95% intervals, 2,000 paired moving
  block resamples of weekly forecast origins within each season, 8 weeks primary,
  4/12 sensitivity. Non-circular blocks cannot cross season boundaries; concatenate
  blocks and truncate to original season length. All targets, locations and
  horizons of each origin stay together. Recompute location denominators and
  season-first weights. Missing frozen origins remain zero-contribution calendar weeks, so blocks keep
  their declared calendar length. Samples losing required location support are
  excluded and counted; abort if none retain support. This conditioning is an
  explicit assumption; exclusion fractions must accompany the intervals.
  Temporal reports cover mixtures and the mean of fitted-seed objectives;
  averaging per-cell WIS here is algebraically equivalent to averaging scores,
  and is distinct from the predictive mixture. Seeds are not epidemic replicates.
- Natural WIS is primary: require >=5% reduction versus A in both mean paired-seed
  and mixture objectives; no >5% increase in any individual stress objective
  (recent report loss, local gap, channel outage), in either comparison.
  Operationalize "supported" conservatively: paired-seed and primary 8-week
  mixture temporal intervals must both exclude no improvement. If both B and C qualify against A, advance C only if the paired-seed and
  primary mixture temporal C-minus-B intervals both exclude no improvement;
  otherwise advance the simpler B. Report all three paired contrasts. No
  automatic architecture grid. These cutoffs, interval
  convention, block lengths and .25 coefficient are engineering assumptions.
- C's recent revision, artificial reconstruction and naturally missing cells
  are reported separately in native diagnostics and scaled-loss contributions.
  The manager's natural persistence-relative nowcast score remains separate from
  ensemble-relative forecast WIS. Forecast calibration reports target/season and
  US/state groups at 50/80/90/95% coverage.
- Source is committed before manager snapshots/planning. The existing dispatcher
  now owns independent seeds individually, allowing multiple fits of one scenario
  on a GPU without duplicate ownership. Slurm's existing afterany notification
  mechanism remains enabled. A dependency job performs the final comparison.

## Validation and reuse

A compatibility audit checks dataset/population hashes, frozen-support input
hashes, exact scenario string, all three completed folds and the refit protocol.
Same-device predictions from the preserved source snapshot are compared to the
new implementation; CPU and CUDA random generators intentionally do not promise
identical draws. The report additionally verifies regenerated CUDA quantiles and
masks against archived files, and reproduces natural totals using the shared
scorer. No obsolete fromB0 experiment or matched-final control is recreated.

Focused checks cover hidden flags/values, exact visible final return, forecast
independence from recent heads, auxiliary loss mass, forecast-only validation,
scientific score ratios and paired temporal blocks. A small three-candidate
research run exercises select/refit/evaluation through the manager before full
launch. Execution results and job IDs are appended below.

2026-09-17: reuse checks passed for A seeds42/43/44; original/current CPU member
predictions are identical. Corrected natural objectives reproduce saved totals:
0.9893939742665779, 1.0553258464097028, 1.0518708781661001 respectively.
See the adjacent compatibility and natural-verification JSON records. Scientific
checks: 45 existing B0/B1 checks, 11 scorer/objective/fold checks and four focused
new checks passed. Raw members will be recovered using CUDA to reproduce the
original evaluation generator; CPU-vs-CUDA samples are not compared for equality.
