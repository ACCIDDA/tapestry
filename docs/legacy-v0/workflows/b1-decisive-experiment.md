# Cluster handoff: choose the next B1 model

2026-09-17. Implementation handoff requested after discussing a more ambitious
experiment. This supersedes the retired narrow matched-final control plan.
Candidates B and C have not been implemented and
no new experiment has been submitted by the author of this handoff.

## Decision and scope

Prioritize model selection on B1's actual tasks. Defer the matched-final control
and broad historical B0 reruns. The resulting experiment will not identify how
much of the historical B0/B1 gap is caused by vintage inputs. Preserve the masked
baseline for reuse wherever its configuration, data, and protocol match.

## Completed cleanup of the obsolete fromB0 experiments

The user explicitly requested deleting the fromB0 runs. Both experiment
directories below have been deleted from the Longleaf repository:

- `data/experiments/B1-fromB0`
- `data/experiments/B1-fromB0-refit`

The Longleaf job queue was empty before removal. Their run artifacts, rankings
and eight dedicated Slurm logs were removed, along with the unused matched-final
NPZ/JSON. Local comparison/refit audit directories were removed too. The legacy
launch suite, refit handoff, ladder-report script and matched-final builder/test
have been removed from the source tree. The short decision history remains in
the B1 design document. Do not recreate the retired experiments or their tools.

`B1-onlymask-refit` is preserved and supplies candidate A's reusable seeds. The
cleanup does not include `B0.1`, `b0-us-cross-4`, the other B1 experiments, the
canonical Wednesday dataset, or frozen evaluation references. The shared B0
architecture and B1 select-then-refit implementation remain in use.

Keep the current retrospective dataset, pinned final labels, three season folds,
select-then-refit protocol, rank-1 B0 backbone and scientific forecast objective.
Recent supplied finals remain legitimate inputs under this retrospective policy;
the experiment must not be described as a real-time Wednesday backtest.

## Three candidates

1. A: existing direct B0 model, artificial masking rate 0.5, values and
   availability as inputs. This is the existing `B1-onlymask-refit` control.
2. B: direct model with the same settings, additionally consuming per-cell
   supplied-final flags. Zero flags for every unavailable or artificially hidden
   cell. This tests whether identifying the input's status improves forecasting.
3. C: B's encoder with parallel recent and future probabilistic output heads.
   Predict two recent and four future weeks from the shared encoded observations;
   do not feed predicted recent values into the forecast head. Share sampled
   member noise between heads, but do not claim that marginal CRPS identifies
   the joint dependence of the six-week paths.

For C, use `forecast_loss + 0.25 * recent_loss`. Each loss uses the existing
scientific within-task weighting and training-only scales. The 0.25 coefficient
is a proposed fixed engineering choice to keep forecasting primary, not an
estimated optimum. Set an absent recent loss to zero. Visible supplied finals
are excluded from recent loss; hiding a supplied final removes its flag and
allows reconstruction supervision. Known recent finals are returned unchanged
in prediction output and remain excluded from nowcast scoring.

Select epochs using the primary forecast validation objective for all candidates,
then refit from scratch on the full permitted training partition. Candidate C's
auxiliary loss shapes training, not the model-selection criterion. Preserve A's
current behavior and implement B/C as explicit scenario options with distinct
configuration identities and metadata.

Candidate C tests whether auxiliary reconstruction helps the encoder while
removing the requirement to pass through an imputed recent trajectory. It does
not assume that the current two-stage model has been proved inferior.

## Replication and scoring

Use seeds 42 through 51 for each candidate, with identical fold definitions and
evaluation task keys. Ten seeds reduce uncertainty from fitting randomness; they
do not create ten independent epidemic histories. A's existing seeds 42/43/44
can be reused if experiment preparation confirms compatible inputs, code and
protocol. This is 30 configuration-seed runs: 27 new runs if those three are
reused, or 30 if compatibility checks require rerunning A. Each run evaluates
three held-out seasons.

Fix stress scoring before interpretation: use saved forecasts on identical
frozen support, final reference observations, quantile levels, ensemble
denominators, location ratios, target weights and season weights. The corrected
natural condition must reproduce existing ranked natural scores. Verify matched
stress masks across candidates. Do not use raw cross-target WIS averages as the
combined objective. This reporting correction cannot be skipped.

Report natural forecasts, recent-report loss, complete channel outages, and
local gaps separately. For C, report recent revision correction and artificial
missing-value reconstruction separately, excluding visible supplied finals.
Include target/season and US/state calibration breakdowns. Do not mix the
nowcast persistence-relative score with the forecast ensemble-relative score.

Report both paired seed differences and an equal mixture of each candidate's
ten fitted predictive distributions. Construct mixture quantiles from pooled
draws rather than averaging quantiles. Score each mixture on identical support.
The mixture score is a separate deployed-candidate comparison, not the mean of
the ten seed scores.

Use paired temporal-block resampling of forecast origins to assess variation
across the evaluated dates, keeping all locations, targets and horizons of an
origin together and preserving the season-first objective. Predeclare an
eight-week primary block length, with four- and twelve-week sensitivity checks;
these lengths are assumptions, not established optimal choices. Seed uncertainty
and temporal uncertainty must be reported distinctly. Three seasons and reused
development data limit interpretation of either interval.

## Decision rule

Natural-input forecast WIS remains primary. A proposed practical threshold is
at least a 5% relative reduction against A, supported by paired results and
without a material stress-condition regression. Define a material stress
regression in advance as more than a 5% relative increase on the corrected
stress objective. These thresholds are proposed project choices, not universal
statistical cutoffs. Report uncertainty rather than promoting a point estimate
at a threshold into proof. If evidence remains ambiguous, advance the simpler
candidate. Do not start another architecture grid automatically.

All three seasons have already informed development. This experiment supports
a stronger engineering decision, not an untouched generalization claim. After
selection, freeze the candidate and evaluate prospectively on future issuances.
An operational evaluation additionally requires authentic as-of inputs and
training labels available by each fit cutoff; ten seeds or chronological splits
cannot remove retrospective final-value information by themselves.

## Implementation boundary

Candidates B and C, the mixture comparison, and the corrected stress aggregation
need implementation. No current manager scenario should be presented as already
providing them. Implement these using the shared manager and scorer, with focused
checks for leakage, loss masks, units/alignment and score mathematics, followed
by a small research run. Do not build a separate experiment framework.

Use `B1-onlymask-refit` for compatible extensions of A, `B1-direct-finalflag` for
B, and `B1-joint-aux025` for C. Preserve the completed A attempts when extending
its seed list. If A must be rerun, give it a fresh experiment identity and record
why. Save or regenerate member draws needed for the predictive mixtures from
fitted checkpoints; stored quantiles alone must not be treated as raw samples.

Commit implementation changes before planning jobs so experiment code snapshots
are reproducible. Run on the `jlessler` partition, packing independent fits on
the patron GPUs through the existing dispatcher. Report exact manager plan,
Slurm launch, manager status and rank commands for every experiment, along with
job IDs. Use Slurm completion notifications rather than an in-session watcher
as the sole completion mechanism. Implement, validate, launch, and return the
comparison when complete; no further approval is needed for this handoff's scope.
No jobs were launched or stopped when writing this handoff.
