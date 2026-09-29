# Why these results differ from B0

The [complete B0 investigation](../b0-reproduction/index.md) is the authoritative
report, with terms explained in context, completed controls, graphs and evidence.
This page explains the connection to the geography screen.

**Holiday-cutoff correction (September 25):** the Wednesday availability policy
does not follow the Hubs' extended Christmas deadline. Missing inputs on
December 24 do not establish missing inputs at submission. The availability
percentages and WIS intervention are for fixed Wednesdays, not a fully
Hub-schedule-aligned backtest. [Schedule evidence and implications](../b0-reproduction/index.md).


**September 25 completed reproduction:** [All 18 B0 forecast folds reproduce
byte-for-byte](../b0-reproduction/index.md), including matching stopping epochs.
The original target/pathogen scores are **0.883377 / 0.894767**. No scoring
mistake was found; finalized target values are almost identical across panels.
The strict “unicorn” claim was overstated: no historical formulation wins every
supported target–season comparison.

On matched L40 hardware, old pathogen training scores **0.888892**; current
unscaled training on the same finalized panel scores **0.974340**; restoring B0
normalization improves this to **0.929660**; applying Wednesday availability
during fitting and forecasting worsens it to **1.198981**. The paired mean
changes are +0.085449, −0.044680 and +0.269321. Availability worsens all three
seeds and all nine target–season means; normalization improves two of three
seeds. The training-code change includes the normalization omission. After normalization, the gap is +0.040768. The completed fitting controls
reduce it to +0.021532 by restoring B0's validation calendar (score 0.910424);
validation randomness alone has no scored effect. The remaining cause is unresolved.
[Full fitting comparison](../b0-reproduction/index.md).

These conditional comparisons show that the task and training protocol changes
matter. They are not a factorial attribution of the latest 1.036/1.124 screen,
which also changes artificial masking, the finality flag and evaluation draws.
The separate normalization-only intervention on that screen gives 1.124 → 1.082
for the control and 1.036 → 1.048 for the covariate leader. The frozen benchmark,
finalized revisions, exploratory model selection and three seeds limit all
claims of generalization or real-time superiority. Missing archive reports do
not prove data were unpublished.

## Verified differences

| | Historical B0.1 | Current geography/covariate screen |
| --- | --- | --- |
| Target history | Finalized available observations, without historical Wednesday publication masking | Finalized values only where the Wednesday snapshot contains a report |
| Artificial masking | No added augmentation in the B0.1 design | Mixed target masking on 50% of training episodes |
| Model family | Best: separate target MLP, cap 100; pathogen MLP cap 300 also strong | Pathogen MLP, cap 300, supplied-final flag, from the later B1 design |
| Data | Earlier finalized panel | Rebuilt panel with corrected publication dates and source resolution |
| Best relative WIS | Target MLP 0.883; pathogen MLP 0.895 | Mixed summaries 1.036; independent no-covariate control 1.124 |

B0.1's [design](../../legacy-v0/design/b0.1.md#fixed-data-objective-and-training-contract)
explicitly uses finalized inputs with no augmentation or vintage reconstruction.
Its [results](../../legacy-v0/results/b0-1-crosses/index.md) document the historical
scores above. The current run borrowed a successful B1 design, not an exact B0
training recipe. Adding the new covariates was therefore not the only change.

The no-fallback Wednesday mask can remove the recent target levels that drive a
short-horizon forecast. Finalizing the reports that remain does not restore
missing recent observations. A separately trained nowcaster was not used in this
screen. This is a concrete reason the task is harder than B0; it is not a measured
attribution of the score difference. Additional artificial masking changes fitting
as well. Neither improved revisions nor a shared architecture guarantees that the
current model has the same information as B0 or more information than the ensemble.

The September 22 data work also corrected early release timestamps and source
priority. The earlier B1 results could use final-truth fallback when recent reports
were absent; the current screen cannot. These corrections are documented in the
[data decision log](../../design/restructure-2026-unified.md#hub-vintages-stop-shadowing-hub-ed-targets-wired-no-truth-fallback--2026-09-22-later).
They should not be equated with a change in the historical B0 architecture.

## Wednesday availability: wastewater example

Yes: a value for the preceding Saturday that had no report in the Wednesday
snapshot is unavailable to the model, even when that final value is now known.

For North Carolina at issuance **2024-10-02**, the COVID wastewater WVAL-like
value for **2024-09-28** is **5.64605** in final data. The Wednesday-masked episode
sets its availability to false and its stored numerical placeholder to zero.
The zero is not an observed wastewater reading. The same holds for flu and RSV
in that example. [Exact example records](availability_examples.json).

The rule applies to every context week and both targets and covariates:

- A report exists by Wednesday and final truth exists: use its finalized value.
- No report exists in the Wednesday archive: unavailable; do not substitute truth.
- Smoothing may compute a feature from older observed reports in its trailing
  three-week window. It never reads the unpublished week's final value.
- Kinsa is broadcast only with its national reporting availability preserved.

“Unavailable” here means absent in the assembled historical snapshot. An archive
gap and a genuinely unpublished report both yield a missing input; this audit
alone does not establish which caused every missing cell.

Across overlapping usable calendar origins (not restricted or weighted to the
Hub-scored support), **92–94%** of latest-week wastewater cells that exist in final
truth are masked. Latest-week target cells are also masked: about **25% for flu
admissions, 46% for COVID admissions, 38% for RSV admissions, and 66% for ED**.
These descriptive counts illustrate the input difference, not its contribution
to WIS. [Counts and denominators](availability_audit.csv).

## Correction: scored Wednesdays mostly do have the latest target

The full-calendar percentages above are not representative of scored dates.
On the frozen scored target/location/issuance support (horizons deduplicated),
latest-week availability is **96.7%** for 2023–24 flu admissions; **92.6% flu and
91.4% COVID** in 2024–25; and **100% for all three admissions targets in 2025–26**.
For 2025–26 ED it is **98.6% COVID, 94.6% flu and 95.1% RSV**.

The user was right to challenge an explanation based on broadly missing recent
targets. These are counts for each scored target's own latest input, not all six
cross-target histories or training episodes. They weaken any claim that natural
reporting gaps alone explain the score deterioration. Added artificial masking
was tested separately; the matched no-artificial-mask experiment is
now complete: removal worsened 29/32 selected formulations, including all three prior leaders. [Scored-support counts](scored_target_availability.csv).

## What the completed controls establish

The [reproduction report](../b0-reproduction/index.md) gives exact fold evidence,
paired-seed CSVs, all target–season comparisons, graphs and manager commands.
It establishes retrospective repeatability and a substantial score effect of
the assembled Wednesday-availability policy under normalized current training.
The policy changes both training and forecast inputs and can change fitted
normalization statistics. Separate training-only and forecast-only interventions
would be needed to attribute the policy effect to either stage. No additional
experiments were launched for this report.

The current covariate findings remain matched comparisons within their own
protocol. The [completed no-artificial-mask experiment](conclusions.md#follow-up-removing-artificial-masking-mostly-hurt)
worsened 29/32 selected formulations, so it does not support artificial masking
as the main cause of the B0 gap. These results do not automatically justify
promoting normalization or replacing unavailable observations with finalized truth.
