# Multi-source, mask-native probabilistic forecasting for FluSight

Design note. Three candidate architectures (A, B, C), one shared data layer, one evaluation
protocol. Objective: minimize WIS on the FluSight primary target while supporting COVID-19 and
RSV, arbitrary missing covariates at train and inference time, and revision-aware nowcasting.

**Update, 2026-09-05:** see the [detailed Solution B plan](../solution-b-plan.md) for
the implementation specification, verified paper attribution, and data/validation protocol.
It supersedes this initial sketch's B-specific performance forecasts, assumptions about
automatic historical-task transfer and joint coherence, and interpretation of optional samples.

Date: 2026-09-03. Target: FluSight 2026-2027 (challenge period expected ~Nov 2026 to May 2027).

---

## 1. What the prior work establishes

### 1.1 Flusion (Ray, Wang, Wolfinger, Reich; *Epidemics* 50:100810, 2025; arXiv:2407.19054)

Top model in FluSight 2023/24 (rMWIS 0.610 vs 0.731 for the FluSight ensemble).

Structure: quantile average of GBQR (LightGBM quantile regression, 114 features, 100 bags),
GBQR-no-level, and a Bayesian ARX in NumPyro. Trained jointly on three signals (NHSN, FluSurv-NET,
ILI+) and all locations.

Ablation results, in order of importance:

| Finding | Evidence |
|---|---|
| Joint training on multiple signals is the main driver | GBQR-only-NHSN rMWIS 0.857 vs GBQR 0.625 |
| Joint training across locations is second | GBQR-by-location 0.780 vs 0.625 |
| Ensembling contributes little | GBQR alone 0.625 vs Flusion 0.622 |
| Reporting adjustments were counterproductive | no-reporting-adj 0.600, better than GBQR |
| Power transform helps | no-transform 0.642 vs 0.625 |

Preprocessing worth copying: rate per 100k, fourth-root transform, per-location-and-source
centering/scaling by the 95th percentile, target = change in transformed signal from the last
observed value, season week and weeks-from-Christmas as features (both in the top five by
importance).

Stated limitations, verbatim in substance:
- Predictions for a location were not informed by contemporaneous observations in other locations
  or by contemporaneous observations of other signals in the same location.
- Features are computed on finalized data during training but on provisional data at prediction
  time; no revision model.
- No spatial structure in the features.

### 1.2 InfluPaint (Lemaitre and Lessler; arXiv:2604.24913)

DDPM over a 52 x 51 season image (time x location, intensity = incident hospitalizations),
U-Net epsilon-network, unconditional training, conditioning at sampling time by inpainting
(RePaint, later CoPaint/O-DDIM).

Results: FluSight 2024-2025, rank 1/36 on absolute WIS (97.9) and MAE (128.3), 11th on relative
WIS (0.71), but 50% and 90% coverage of 14.8% and 52.6% against nominal 50/90. 2023-2024 was
24/27. Retrospective (non-real-time) totals beat the FluSight ensemble in both seasons.

Ablations, in order of effect size:

| Lever | Effect |
|---|---|
| Training data mix | 30% surveillance / 70% simulation best; 70/30 -8.8%, surveillance-only -9.2%, simulation-only -11.8% |
| Diffusion steps | T=500 > T=200 |
| Transform | sqrt > linear |
| Augmentation | Poisson resampling, temporal padding, intensity scaling all hurt |
| U-Net variant | negligible |
| Inpainting schedule | negligible |

Cost: 20-40 minutes for 512 conditioned trajectories; several hours to train on L40/A100;
batch 512 needs >=20 GB.

Two facts that drive the design below:
- Training loss (unconditional generative fit) correlates positively with forecast skill but does
  not order models. The authors' own conclusion: training directly for the forecast task may
  improve performance.
- The listed future work is exactly the specification of this project: train conditionally,
  include nowcasting, add auxiliary channels (climate, mobility, virological), extend to RSV,
  COVID-19, dengue.

### 1.3 GenCast (Price et al., *Nature* 637:84-90, 2025; arXiv:2312.15796)

Conditional diffusion over the next atmospheric state: p(X^{t+1} | X^t, X^{t-1}), trajectories
built autoregressively as a product of single-step transitions. 15-day 0.25-degree forecasts,
~8 min per trajectory on a TPUv5, ensembles generated in parallel. First ML model to beat ECMWF
ENS. Architecture: graph-transformer encoder-processor-decoder; conditioning by concatenating the
two previous states to the noisy candidate.

Transferable lessons: conditioning by concatenation of past states works; sampling one block at a
time and rolling out is viable for long horizons; the paper's own ablation (deterministic model of
the same architecture, ensemble by perturbation) scores worse, i.e. the generative head earns its
place at long lead times.

### 1.4 FGN (Alet et al., "Skillful joint probabilistic weather forecasting from marginals",
arXiv:2506.10772)

Same group, same backbone as GenCast, but: a 32-element noise vector conditions the layer-norm
layers, and training is CRPS on marginals. One noise draw per sample, shared across all outputs,
which yields coherent joint samples even though the loss only sees marginals. Beats GenCast at a
fraction of the sampling cost.

This is the single most relevant reference for the objective "best WIS", because WIS is a
discretized CRPS and FGN demonstrates that training on it directly, with a shared latent for
coherence, beats the diffusion model it was derived from.

Supporting references for the mask/conditioning machinery:
- CSDI, Tashiro et al., NeurIPS 2021 (arXiv:2107.03502): conditional score-based imputation with
  observation masks; the train-time-conditioning pattern used in A.
- RePaint, Lugmayr et al. (arXiv:2201.09865); CoPaint, Zhang et al. (arXiv:2304.03322);
  Rout et al. (arXiv:2302.01217) for inpainting generalization to unseen masks.
- Flow matching / rectified flow: Lipman et al. (arXiv:2210.02747), Liu et al. (arXiv:2209.03003).
- Multi-signal transfer alternatives contemporaneous with Flusion: Meyer et al.
  (medRxiv 2024.07.17.24310565), Benefield et al. Both imputed history rather than training
  jointly; both ranked below Flusion.

### 1.5 FluSight 2025-2026 target structure (cdcepi/FluSight-forecast-hub README)

- Primary: quantile forecasts of weekly lab-confirmed influenza hospital admissions, NHSN Weekly
  Hospital Respiratory Dataset, horizons -1 to 3, 53 locations (50 states, DC, PR, US), 23 quantile
  levels, integers required. Horizon -1 submitted but not scored.
- Secondary, in CDC's stated priority order: (1) proportion of ED visits due to influenza (NSSP,
  `percent_visits_influenza`/100), (2) rate-change categories, (3) peak week probabilities,
  (4) peak incidence quantiles.
- Sample submissions: exactly 100 samples per model task, required to be temporally connected
  across horizons, not resampled from quantiles.
- Reference date = Saturday after the Wednesday deadline. Data released Wednesday midday.
- Hub data mirrored on S3 (`cdcepi-flusight-forecast-hub`), readable with hubData or pyarrow.

Consequences: a sample-generating model satisfies every target from one draw. A quantile-regression
model satisfies the primary and ED targets only. The 100-sample requirement rules out independent
per-horizon quantile heads.

---

## 2. Shared data layer (identical for A, B, C)

All three consume the same tensor; build it once.

Implementation status (2026-09-11): the shape-independent raw acquisition repository is now
implemented under `src/influpaintx/data`, with the reproducible entry point
`scripts/pull_covariates.py`. It preserves direct CDC snapshots, Delphi V5 report-time
archives, Hubverse row and Git vintages, checksums, and native geographic support. Tensor
assembly and binary-mask policy remain deliberately deferred to the later `Dataset` class.

The active Delphi catalog is NHSN, NSSP, NWSS, and inpatient/outpatient claims. The historical ILINet/ILI+/FluSurv experiments below
remain research proposals; their old API acquisition code has been removed and
those feeds require a future V5 addition. CDC and Hubverse sources remain enabled.
See the [current catalog](../../data/sources.md) for supported inputs.

**Channels (C ~ 15).** NHSN admissions flu / COVID / RSV; NSSP ED visit proportion flu / COVID /
RSV; NWSS wastewater influenza A / SARS-CoV-2 / RSV; ILINet ILI; ILI+; NREVSS positivity per
pathogen; FluSurv-NET rate.

**Transform.** Counts to rate per 100k using `auxiliary-data/locations.csv` populations, then a
power transform (sqrt per InfluPaint, fourth root per Flusion; treat the exponent as a
hyperparameter), then per-(location, channel) centering and scaling.

**Mask and vintage.** Every value is accompanied by `m` in {0,1} and `a` = weeks since the value
was first reported, plus NHSN percent-of-hospitals-reporting where available. Values are never fed
without their mask. This is what makes "covariate absent" an ordinary input rather than an
exception, at train time and at inference.

**Vintage construction.** Every training example is assembled from data as of a reference date
(Delphi V5 `snapshot_date` or archived `report_time`, plus the hub's `target-data` git history), with finalized values as
targets. This is the mechanism that turns the horizon -1 and 0 outputs into a learned backfill and
nowcast model, and it is the documented gap in Flusion.

**Extra training material.**
- Long-history signals: ILINet from 1997, FluSurv-NET from 2010. Windows from these seasons are
  training examples with the NHSN channels masked. This is Flusion's largest single lever,
  reproduced.
- Simulated trajectories: Flu Scenario Modeling Hub rounds (and the COVID and RSV hubs), entering
  as examples with only the hospitalization channel unmasked. Mixing weight is a hyperparameter;
  InfluPaint found roughly 70% simulation optimal for A-type models. Expect a lower optimum for B.
- Excluded seasons: 2008/09, 2009/10 (pandemic H1N1), 2020/21, 2021/22 (suppressed flu), per
  Flusion.

**Signal dropout (train-time augmentation).** Randomly zero entire channels (p ~ 0.3) and random
(location, channel) blocks. Without this, the model becomes dependent on NWSS and NSSP, which have
roughly three usable seasons and can disappear operationally. This is the only augmentation
recommended; note that InfluPaint found Poisson resampling, padding, and intensity scaling all
degraded performance.

**Observation layer.** Negative binomial with a learned per-channel dispersion, fit by maximum
likelihood on residuals rather than inside the generative loss. Converts a latent rate to counts
and is the principal fix for InfluPaint's overconfidence in small states.

---

## 3. Solution A: full-season masked conditional generative model

### 3.1 Structure

Model p(unobserved cells | observed cells) over X in R^{C x L x T}, T = 52 season weeks, L = 53.

- Tokens: (location, week) cells. Channels live in the feature dimension, concatenated with their
  mask bits and vintage age.
- Network: transformer with factorized attention, alternating over the time axis within a location
  and the location axis within a week. d_model 96, 6 blocks, roughly 1M parameters. Sinusoidal
  embedding of the flow time; season-week, log population, and location embeddings as static
  features.
- Generative process: flow matching (rectified flow) rather than DDPM. Same objective structure,
  ODE sampling in 20-40 steps, and the T=200-vs-500 sensitivity InfluPaint documented disappears.
- Conditioning: at training time, by concatenation (CSDI). Sample a season, a reference week d, and
  a vintage; context = cells observed as of d; targets = all remaining cells, including the
  provisional last weeks whose targets are the finalized values. Loss on target cells only.
- Sampling: 100 trajectories by `vmap`, then the NB count layer.

### 3.2 Outputs

Every FluSight target from one draw: primary quantiles, ED proportion quantiles, 100 coherent
samples, rate-change categories, peak week, peak incidence. Plus arbitrary-mask reconstruction
(missing states, missing weeks, missing signals), which InfluPaint already demonstrated
qualitatively.

### 3.3 Pros

- Only design that produces peak targets natively and well; a sample is a season.
- Full multimodality early in the season, which is the documented strength of the InfluPaint
  approach and is difficult for mechanistic and autoregressive models.
- Consumes SMH simulations in their native shape, so the largest lever from the InfluPaint
  ablations transfers unchanged.
- Strongest publication story: closes every limitation listed in the InfluPaint discussion.

### 3.4 Cons and risks

- The loss is a surrogate for WIS. InfluPaint measured a positive but non-ordering relationship
  between training loss and forecast skill; there is no reason to expect this to vanish.
- Calibration. The realized 2024-2025 coverage was 14.8% (50%) and 52.6% (90%). Conditional
  training and the NB layer should improve this, but it must be verified, not assumed, and a
  per-horizon quantile-width correction fit on held-out seasons should be budgeted.
- Data scarcity on the 52-week canvas: NSSP and wastewater exist as full-season frames for about
  three seasons, i.e. ~3 x 53 examples for learning their lead relationships.
- Backtest cost measured in GPU-hours, so few ablations get run before November, and the ablations
  are where the wins are.
- Structural bias toward the seasonal template. This is a feature for flu and a defect for
  off-cycle or emerging pathogens (see section 6).

### 3.5 Probable performance and complexity

Primary target: top-10, high variance; top-5 achievable if calibration is solved. Peak targets:
best of the three. Complexity: high, 4-8 weeks to a tuned system, ~1M parameters, roughly a dozen
new hyperparameters with no epidemiological prior (flow-time sampling distribution, horizon loss
weighting, context dropout rate).

---

## 4. Solution B: windowed stochastic forecaster trained on CRPS

### 4.1 Structure

Model a function f(window, statics, z) -> Delta in R^{L x H x C_out}, with z ~ N(0, I_32).

- Input window: P = 8 to 12 past weeks x C channels x [value, mask, vintage age], per location.
- Statics per (location, channel): cumulative-to-date, weeks since season onset, season week,
  weeks-from-Christmas, log population, location embedding. These are load-bearing: an 8-week
  window alone cannot know whether the peak has passed.
- Per-location encoder: 2-3 residual MLP blocks on the flattened window plus statics -> h_l in
  R^128. Attention over time is unnecessary at P = 8.
- Cross-location block: 1-2 self-attention layers over the L = 53 tokens. This is the only spatial
  machinery, and it closes Flusion's stated gap.
- Stochastic head: z injected by FiLM / conditional layer norm into the decoder (FGN). One draw of
  z is shared across all locations, horizons, and pathogens, which is what makes the 100 samples
  jointly coherent as the hub requires.
- Output: Delta parameterized as change from the last observed transformed value (Flusion), H = 8
  horizons, C_out = hospitalizations x 3 pathogens and ED proportion x 3. Horizons -1 and 0 target
  finalized values, so backfill and nowcast come out of the same head.
- Loss: fair CRPS over M = 8-16 draws of z, on every observed target cell of every channel.
  Long-history ILI+ and FluSurv windows train the same weights with the NHSN channels masked.
- Parameters: roughly 0.3M.

### 4.2 Pros

- The training loss is the scored metric up to quantile discretization. This is the central
  argument for B over A.
- Backtest in minutes, not hours: 50-100 ablations feasible before the season. Given that both
  reference papers found their gains in ablation rather than architecture, this is worth more than
  it appears.
- Uses contemporaneous cross-location and cross-signal information, the exact thing Flusion says it
  did not do.
- Nowcast and backfill are in-model rather than a preprocessing hack.
- Smallest parameter count, consistent with the stated constraint.
- No season canvas, so emerging and off-cycle pathogens are handled (section 6).

### 4.3 Cons and risks

- CRPS constrains marginals only. z may be ignored (collapse toward a point forecast), or the joint
  structure across locations and horizons may be poor. FGN obtained coherent joints emergently, but
  from 40 years of ERA5; the available corpus here is roughly 30 seasons of ILI and 5 of NHSN.
  Monitoring required: predicted national aggregate versus the sum of state samples, rank
  histograms of multi-week differences, spread-skill ratio per horizon.
- Peak targets only by autoregressive rollout over ~30 steps, with compounding error and nothing in
  the loss constraining trajectory shape at that range.
- Scale mismatch: CRPS on the transformed scale weights all states equally; WIS on counts is
  dominated by large states. Population weighting of the loss is a knob to tune, not a detail.
- Weaker novelty claim: "Flusion's data insight, plus contemporaneous covariates, spatial
  attention, vintage awareness, and a CRPS-trained generative head."

### 4.4 Probable performance and complexity

Primary target: best expected WIS of the three; top-3 is a reasonable target. If it fails to beat a
Flusion-style GBQR fed the same covariate table in retrospective evaluation, suspect the
implementation, not the idea. Peak targets: middling. Complexity: low to medium, 2-3 weeks, with
most of the effort in the vintage data layer that A needs anyway.

---

## 5. Solution C: pretrained time-series foundation model

### 5.1 Structure

Treat each (location, target signal) as a univariate series with past covariates passed through the
model's covariate interface. Candidates: the Chronos, TimesFM, and Moirai families, all of which
accept covariates and missing values and emit quantiles. Zero-shot first, then LoRA fine-tuning on
the multi-season, multi-signal corpus. Verify current model versions and interfaces before
committing; these change every few months.

### 5.2 Pros

- One day to a first backtest; a genuine baseline rather than a strawman.
- A strong prior on trend continuation and turning points learned from millions of unrelated series.
- Errors are uncorrelated with anything trained only on epidemic data, which makes it a good
  ensemble member even when it is not the best single model.
- Satisfies the LLM branch of the original framing.

### 5.3 Cons and risks

- 100M+ parameters, against the stated preference.
- No vintage concept: provisional values are taken at face value, which is precisely the failure
  mode that hurts real-time WIS.
- Cross-location structure only if neighbours are hand-fed as covariates.
- Output quantile grids typically omit 0.01 and 0.99, so WIS tails come from extrapolation.
- Quantile heads give no coherent samples, so the 100-sample target is unreachable without an
  autoregressive sampling variant.
- Fine-tuning on a small corpus risks destroying the pretrained prior.

### 5.4 Probable performance and complexity

Top third of the leaderboard alone; contributes in a quantile-averaged ensemble. Complexity: low to
test, medium to make robust; no training infrastructure required.

---

## 6. Peak targets and non-seasonal pathogens

**Peaks.** A produces them natively: a sample is a full season, so peak week and peak intensity are
read off directly, and the predictive distribution over them is the empirical distribution across
samples. B produces them only by rolling the 8-week block forward autoregressively (the GenCast
pattern), redrawing z each step, for roughly 30 steps to the end of season. Error compounds and no
part of the training loss constrains behaviour at that range. Given that CDC ranks peak targets
third and fourth in priority and treats all secondary targets as optional, this costs little
competitively and a lot scientifically.

**Non-seasonal, emerging, off-cycle.** B is the better tool, for a structural reason rather than a
vague flexibility argument. A's inductive bias is a fixed 52-week canvas anchored to a season start;
what it learns is the marginal distribution of season shapes. For a pathogen with no annual
periodicity, an out-of-phase emergence, or a mid-season variant wave, that prior is wrong, and it
actively pulls samples toward the seasonal template. This is the same mechanism that makes A strong
on influenza. A also cannot form a frame at all for a pathogen with three months of history,
whereas B needs only P weeks plus whatever covariates exist. The cost is that B has no structural
notion of season position beyond the features supplied, which is why the season-summary statics
matter.

Corollary for the paper: A is the influenza and full-season instrument, B is the general
respiratory and outbreak instrument. They are complements, not competitors, and share a backbone.

---

## 7. Comparison

| | A: full-season generative | B: windowed CRPS | C: foundation model |
|---|---|---|---|
| Loss versus scored metric | surrogate | matches | surrogate, fixed quantile grid |
| Parameters | ~1M | ~0.3M | ~100M+ |
| Missing covariates | mask-native | mask-native | native, per-series only |
| Revisions and backfill | vintage in training | vintage in training | none |
| Coherent 100 samples | yes | yes, shared z | no |
| Peak week / intensity | strong | weak, rollout | not supported |
| Cross-location information | attention | attention | manual covariates |
| Non-seasonal pathogens | poor, season prior | good | good |
| Backtest wall clock | hours | minutes | minutes |
| Principal risk | calibration, memorization | z-collapse, weak joints | no vintage, tail extrapolation |
| Expected primary-target WIS | top 10, high variance | top 3 | top third |
| Novelty | high | medium | low |
| Build time | 4-8 weeks | 2-3 weeks | days |

---

## 8. Recommended plan

1. Build the shared vintage data layer. Required by all three; the largest single block of work.
2. Build B. It is the candidate for the primary target, and its backtest speed pays for the
   ablations that both reference papers show are where performance is actually found.
3. Build C in a day, as baseline and ensemble candidate.
4. Build A's head on B's backbone: replace the FiLM decoder with a flow-matching decoder over an
   H-week block, then extend the canvas to the full season. Use it for peak targets and for the
   multimodality argument.
5. Reproduce a Flusion-style GBQR on the same covariate table as a floor. If the neural models do
   not clear it retrospectively, submit the GBQR.
6. Submit the quantile average of whatever survives the three-season vintage backtest. Two
   structurally different good models generally outrank either alone; note however that Flusion's
   own ablation found ensembling contributed almost nothing over its best component, so verify
   rather than assume.

### 8.1 Evaluation protocol

Retrospective on 2023-2024, 2024-2025, 2025-2026 using as-of data only. Comparators: Flusion
(public code, reichlab/flusion), InfluPaint (public weights, ACCIDDA/Influpaint), FluSight-ensemble,
FluSight-baseline, and a trend baseline. Metrics: WIS and relative WIS via pairwise tournament
(Cramer et al., PNAS 2022), MAE, 50% and 95% interval coverage, one-sided quantile coverage
differentials. Scoring with `scoringutils`; hub interaction via hubverse. Report the sensitivity
analysis Flusion used: exclude location-date pairs whose latest available value was subsequently
revised by 10 or more admissions.

### 8.2 Ablation order (highest expected effect first)

1. B's CRPS head versus A's flow-matching head on the same backbone. Settles the diffusion question
   empirically rather than by argument.
2. Long-history multitask windows (ILI+, FluSurv) on/off. Flusion's largest lever.
3. Vintage-aware training versus finalized-data training.
4. Simulation mixing weight. InfluPaint's largest lever.
5. Signal dropout rate; behaviour under forced removal of NWSS and NSSP at inference.
6. Power transform exponent; population weighting of the loss.
7. Cross-location attention on/off.
8. Negative binomial observation layer on/off; post-hoc quantile-width correction.

### 8.3 Known risks to manage

- NSSP and NWSS have roughly three usable seasons. Signal dropout and weight decay are not optional.
- COVID and RSV simulation pools are thinner than the flu SMH rounds.
- NHSN reporting completeness varies by jurisdiction and has been interrupted; the
  percent-hospitals-reporting feature is the model's only defence.
- Puerto Rico has been excluded from some official evaluations for data instability; decide
  explicitly whether to submit for it.

---

## 9. References

- Ray E.L., Wang Y., Wolfinger R.D., Reich N.G. Flusion: Integrating multiple data sources for
  accurate influenza predictions. *Epidemics* 50:100810, 2025. arXiv:2407.19054.
- Lemaitre J., Lessler J. Generative diffusion models for spatiotemporal influenza forecasting.
  arXiv:2604.24913. Code: github.com/ACCIDDA/Influpaint.
- Price I., Sanchez-Gonzalez A., Alet F., et al. Probabilistic weather forecasting with machine
  learning (GenCast). *Nature* 637(8044):84-90, 2025. arXiv:2312.15796.
- Alet F., et al. Skillful joint probabilistic weather forecasting from marginals (FGN).
  arXiv:2506.10772.
- Tashiro Y., Song J., Song Y., Ermon S. CSDI: Conditional score-based diffusion models for
  probabilistic time series imputation. NeurIPS 2021. arXiv:2107.03502.
- Lugmayr A., et al. RePaint: Inpainting using denoising diffusion probabilistic models.
  arXiv:2201.09865.
- Zhang G., et al. Towards coherent image inpainting using denoising diffusion implicit models
  (CoPaint). arXiv:2304.03322.
- Rout L., et al. A theoretical justification for image inpainting using DDPMs. arXiv:2302.01217.
- Lipman Y., et al. Flow matching for generative modeling. arXiv:2210.02747.
- Liu X., Gong C., Liu Q. Flow straight and fast: rectified flow. arXiv:2209.03003.
- Ho J., Jain A., Abbeel P. Denoising diffusion probabilistic models. arXiv:2006.11239.
- Song J., Meng C., Ermon S. Denoising diffusion implicit models. arXiv:2010.02502.
- Bracher J., Ray E.L., Gneiting T., Reich N.G. Evaluating epidemic forecasts in an interval format.
  *PLOS Comput Biol* 17(2):e1008618, 2021.
- Cramer E.Y., et al. Evaluation of individual and ensemble probabilistic forecasts of COVID-19
  mortality in the United States. *PNAS* 119(15):e2113561119, 2022. (Pairwise tournament rWIS.)
- Mathis S.M., et al. Evaluation of FluSight influenza forecasting in the 2021-22 and 2022-23
  seasons. *Nature Communications* 15, 2024.
- Meyer A.G., et al. A prospective real-time transfer learning approach to estimate influenza
  hospitalizations with limited data. medRxiv 2024.07.17.24310565.
- Loo S.L., et al. The US COVID-19 and Influenza Scenario Modeling Hubs. *Epidemics* 46:100738, 2024.
- Lemaitre J.C., et al. flepiMoP: The evolution of a flexible infectious disease modeling pipeline.
  *Epidemics* 47:100753, 2024.
- Consortium of Infectious Disease Modeling Hubs. Coordinating collaborative infectious disease
  modeling projects with the hubverse. medRxiv 2025.10.03.25337284.
- Bosse N.I., et al. Evaluating forecasts with scoringutils in R. arXiv:2205.07090.
- Farrow D.C., et al. Delphi Epidata API. github.com/cmu-delphi/delphi-epidata.
- CDC FluSight Forecast Hub. github.com/cdcepi/FluSight-forecast-hub. S3 mirror:
  `cdcepi-flusight-forecast-hub`.
