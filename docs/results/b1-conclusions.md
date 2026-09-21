# B1 overview: what we tried and what happened

B1 asks how to forecast from surveillance histories containing preliminary
reports, finalized observations and missing values. We first compared direct
forecasting with explicit final-status flags and auxiliary nowcasting, then
expanded the architectures and masking, increased the training budget, and
finally tested models designed to correct preliminary reports.

## What A, B, C and the other models mean

A **forecast** predicts the next four weeks. A **nowcast** estimates the finalized
value of either of the two most recent weeks. Correcting a visible preliminary
report and reconstructing a missing observation are different nowcasting tasks.

| Formulation | What we changed | How nowcasts affect forecasts |
|---|---|---|
| **A — direct forecast** | Predict four future weeks from values and observation masks, without an explicit supplied-final flag or recent-output head. | No nowcast is produced. |
| **B — direct forecast + final flags** | Add per-cell indicators distinguishing supplied finalized observations from preliminary reports. | No nowcast is produced; this tests whether knowing input finality helps forecasting. |
| **C — parallel auxiliary nowcast** | Add a recent-output head to B's shared representation. Initially train with forecast loss + 0.25 × recent loss; later test coefficients 0.10 and 0.20. | Recent predictions are not fed into the forecast. Their training loss changes the shared representation. |
| **Two-stage — nowcast then forecast** | Sample recent values, then use those estimates in the forecast stage. Visible finalized inputs pass through exactly. | Forecasts explicitly depend on sampled recent estimates. The initial recipe also differs in decoder/anchoring and loss/selection weighting. |
| **Gated revision branch — later experiment** | Start from B's direct forecast structure; add separate correction and reconstruction heads. Correct each visible report relative to its own observed value. | A learned gate controls an adjustment from recent estimates to B's future outputs. Trained jointly from scratch; it is not a frozen, previously fitted B model. |

We crossed these formulations with four **backbones**: target MLP (six separate
output fits), pathogen MLP (three fits, each covering admissions and ED visits),
target convolution (six fits with temporal convolution), and joint MLP (one fit
for all targets). **Every backbone receives all six input histories**, including
the independently fitted target models.

Artificial masking hides inputs during fitting: **recent** removes 1–2 recent
weeks of a channel/source family across locations; **gap** removes 1–3 weeks of
one channel at one location; **outage** removes an entire channel's history across
locations. **Mixed** chooses recent/gap/outage with probabilities 50/30/20 when
an episode is masked. A masking rate of 50% means half of episodes are corrupted,
not half of cells. **No mask** means no artificial hiding; natural gaps remain.
Later **revision augmentation** adds simulated reporting errors during fitting,
a different intervention from hiding values.

## Experiments and results, in the order tried

Forecast numbers below are **WIS relative to the Hub ensemble: lower is better,
1 means ensemble parity**. Multi-seed results are means of fitted-model scores,
not scores of a predictive ensemble. Comparisons belong within each experiment:
evaluation draws and selection rules changed between stages.

| Order | What was tried | Main forecast results | What this established |
|---|---|---|---|
| **1. Initial A/B/C comparison** — September 17 | Target MLP, mixed 50% masking; one completed seed, three season folds, 2,048 evaluation draws. The planned ten-seed comparison was deferred. | **A 0.989 → B 0.942; C 0.959.** | Final flags helped in this first screen. Adding a parallel nowcast did not improve on B. One seed did not establish a reliable winner. |
| **2. Architecture and masking screen** — September 17 | Four backbones; A/B/C/two-stage at mixed 50%, plus B mask rates/mechanisms and C outage-only. **40 configurations × 3 seeds**, 256 draws. | **Target B gap-only 0.939** led; pathogen B mixed 50% 0.952; target B no-mask 0.953. **15/40** configuration means beat the ensemble. See the matched architecture table below. | B was consistently competitive. Masking interacted with backbone. Two-stage lost on natural forecasts across every matched architecture and seed. |
| **3. Longer training** — September 17 | A/B/C/two-stage across all four backbones, mixed 50% only; **16 configurations × 3 seeds**, cap 300, 256 draws. Joint MLP already had cap 300 and served as a repeat control. | **Pathogen B 0.941, C 0.945** led. Target B worsened **0.980 → 1.029**; pathogen B improved **0.952 → 0.941**. Two-stage means **1.194–1.292**. | More training helped some recipes, not all. It did not rescue two-stage. The original gap-only leader was not repeated in this panel. |
| **4. Revision and nowcasting experiment** — completed September 18 | Target/pathogen MLPs; B, C, two-stage and new gated heads; recent-loss coefficients 0.10/0.20; revision augmentation on/off; B gap/no-mask controls. **32 configurations × 5 seeds**, cap 300, 1,024 draws; natural-input forecast-only selection throughout. | **Target B gap-only 0.938; pathogen B no-mask 0.943; pathogen C 20%, no augmentation 0.947; target gated 20%, no augmentation 0.948.** Two-stage **1.093–1.282**. | Simple B controls retained the best forecast means. Target gated improved matched mixed-mask B (**0.997 → 0.948**, all five seeds), but did not beat gap-only B. Nowcast quality could now be compared on common report support. |

Sources and full results: [initial comparison](first-good-b1.md),
[architecture/masking screen](b1-overnight/index.md#snapshot-and-experiment),
[300-epoch comparison](b1-overnight/index.md#300-epoch-results-and-comparison),
[revision experiment](b1-overnight/index.md#revision-experiment-separating-forecasting-nowcasting-and-reconstruction).
Navigation and report sections follow this same experimental sequence.

### Architecture comparison: hold masking fixed

These are the original three-seed screen means with **mixed 50% masking**.
Target/pathogen/convolution used a 100-epoch cap; joint MLP used 300, so this is
a comparison of the tested recipes, not an equal-budget architecture ablation.

| Backbone | A: direct | B: final flags | C: parallel nowcast | Two-stage |
|---|---:|---:|---:|---:|
| Target MLP | 1.029 | **0.980** | 0.985 | 1.179 |
| Pathogen MLP | 1.045 | **0.952** | 0.957 | 1.104 |
| Target convolution | 1.100 | 1.031 | **1.005** | 1.206 |
| Joint MLP | 1.012 | 0.990 | **0.972** | 1.211 |

**B improves the mean over A in all four backbones.** C's contribution depends
on backbone: close to B for target/pathogen MLPs, better for convolution and joint
MLP. The best tested forecast recipes use target or pathogen MLPs, but there is
no single architecture winner across masking and training budgets. The two-stage
penalty concerns the complete tested formulation; it does not isolate feedback
as the cause.

## Three findings worth examining

### 1. Masking trades natural-input skill against missing-input robustness

Gap-only and no-mask B controls remained strong in the five-seed revision panel.
Mixed masking was best for pathogen B in the original screen, and 25% masking
was not consistently intermediate between 0% and 50%. There is no evidence for
one universally best masking rate or mechanism.

The original target B gap-only winner moved from **0.939 naturally to 1.716 under
full-channel outage**. Some two-stage recipes handled outages better despite
weaker natural forecasts. A natural-input leaderboard therefore does not select
an outage-robust model. In the original screen masking changed validation as
well as training; the later revision panel held validation natural throughout.
Its cross-panel changes cannot isolate the benefit of that selection change.

### 2. Forecast gains and useful nowcasts are separate results

The revision experiment compared each recent prediction with **that week's own
preliminary report**, rather than with the forecast Hub ensemble.

| Recent-output family | Adjusted nowcast WIS ratio across configurations | Full-support scaled CRPS / report error, age 11 days / 4 days | Interpretation |
|---|---:|---:|---|
| C: parallel | 2.687–2.857 | 2.53 / 0.85 | Competitive forecasts can coexist with poor recent outputs, especially for the older week. |
| Two-stage | 1.004–1.129 | 0.83 / 0.77 | Whether it beats the report depends on the metric; natural forecasts remain weaker. |
| Gated revision | **0.890–0.978** | **0.78 / 0.73** | Strongest recent-head family under the displayed diagnostics, but not fully calibrated. |

The adjusted WIS excludes **two of 819 target/season/location groups** whose
baseline errors are near zero, using a **post-hoc** threshold. Thus the best
roughly 11% WIS gain is conditional on that exclusion. The registered unfiltered
ratios are numerically unstable. The scaled-CRPS diagnostic retains those groups,
uses training-only Q95 normalization and scientific weights, and averages all
eight configurations and five seeds per family separately by age. It is a
different metric, not a replacement WIS ranking. Pooled WIS also preserves the
family ordering, but changes geographic weighting and whether two-stage beats
the report.

Gated forecasts improved matched target B but generally worsened pathogen B
without augmentation. Revision augmentation improved mean adjusted nowcast WIS
in all recent-head families, while worsening gated forecast WIS by **0.041** on
average (only 5/20 paired forecast contrasts improved). **Gated plus gap-only
masking was not tested**; the gains of those two choices cannot be added together.

### 3. Reconstruction and calibration remain unresolved

For gated models, hiding the same genuine report increased scaled CRPS from
**0.014 to 0.045** at 11 days and **0.018 to 0.055** at four days. Correcting an
available report is easier than reconstructing it. Full-channel outage produced
near-zero admission nowcasts and poor coverage even though recent heads executed.
Anchoring is a candidate explanation requiring an ablation, not a demonstrated
scorer error.

The cap-300 forecast leader covered **43.3% / 84.6%** of observations with nominal
50% / 95% intervals. Gated natural nowcasts had about **86% / 84%** coverage at
the nominal 95% level for the two ages. Lower WIS has not solved undercoverage.

### Reporting regimes limit the nowcasting conclusion

The intended use is next-season forecasting: **judge NHSN nowcasting on the
reporting regime beginning November 2024**, assuming it continues next season.
Earlier-regime performance cannot decide whether nowcasting helps that task.

NHSN spans a previous regime before May 2024, voluntary reporting in May–October
2024, and a new mandate from November 2024, as specified by the user. Seasonal
cross-validation mixes these reporting processes and uneven genuine-report
coverage. Weak transfer does not establish that nowcasting is unhelpful under
the current regime, and regime mixing has not yet been shown to cause the failures.
See the [matched-age revision and coverage graphs](b1-reporting-regimes/index.md)
for NHSN and NSSP over time. These dates are recorded as user-provided premises
in the repository's `icare.md`.

Separate training was not tested in these retrospective panels. The later
[2025–26 forward benchmark](Forward-2025/analysis.md) favors the independently
trained pipeline over Direct B and Joint gated under a historical cutoff; that
one-season development result uses a different information regime. A joint model can
retain older forecast examples while restricting revision supervision to the
relevant reporting regime and available reports. The target gated result is
positive evidence for keeping that option. See the
[data exploration and proposed matched comparison](../data/reporting-availability.md#what-this-does-and-does-not-say-about-joint-models).

## Conclusion and scope

For natural forecasting, retain **target B gap-only and pathogen B no-mask** as
controls. For useful recent estimates, continue with **gated revision heads**,
including a matched test with gap-only masking. The current two-stage recipe
has not earned a place as the default natural forecaster. Calibration and
full-channel reconstruction need explicit attention.

The recommendation assumes natural forecasting is the primary selection task;
nowcasting and robustness remain separate objectives because no operational
trade-off weights were supplied. Forecast aggregation uses states/DC 80%, US 20%,
admissions/ED weights 1/0.5 and equal seasons. Support is flu admissions in
2023–24, flu/COVID admissions in 2024–25, and all six targets in 2025–26.

All evidence is retrospective development CV on seasons already used for model
choices, with finalized older history and supplied-final fallbacks for missing
recent reports. It is not a strict real-time backtest. Seeds measure fitting
variability, not independent epidemics; small mean gaps do not establish
superiority. The initial result did not meet the original promotion rule, and
later exploratory screens do not complete the deferred ten-seed test. This
page summarizes saved results without new training, scoring or image inspection.

## Log

- 2026-09-18: ordered B1 navigation and combined results by experimental sequence;
  report generators retain that order and preserve the revision section.
- 2026-09-18: restructured the overview to define A/B/C, two-stage, gated models,
  backbones and masking before presenting chronological experiment results,
  matched architecture scores, and focused interpretation.

- 2026-09-19: linked the completed forward benchmark and qualified the older separate-training conclusion to its original retrospective scope.
