# Tapestry

This is the project with all my wishlist for infectious disease modeling:
Flusion's idea of learning across surveillance sources, InfluPaint's modeling
ideas and infrastructure, CRPS fitting like that very smart DeepMind paper,
and a way to bring it all together. I have wanted to implement quite a bit of
this for years. It was made easier by InfluPaint already having the Slurm job
manager and simulation storage, and EpiBenchmark having the scoring.

The idea is fairly simple: give a model the recent history of several pathogens
and ask it to generate possible futures. Learn across signals, keep track of
what is actually observed, and fit the predictive distribution directly. For
now I am leaning on simple MLP encoders, convolutions, and transformer-style
attention for exchanging information across locations and targets. The
[architecture notes](design/architecture.md#2-what-is-borrowed-from-each-paper)
explain what comes from Flusion, InfluPaint, and DeepMind's Functional Generative
Networks paper, and what I adapted here.

## Good news: I found unicorns

By **unicorns**, I mean model formulations that can beat the ensemble across
three pathogens, on both admissions and ED visits — six targets — across the
past three seasons. I wasn't sure those would exist, especially given how
different the seasons are. They do in these retrospective experiments. I use
season cross-validation: train on two seasons and forecast the third, rotating
through 2023–24, 2024–25, and 2025–26.

There is an important catch: these models see finalized, more recently revised
data than the hub ensembles had at forecast time. The margins make me hopeful
this will carry over to vintaged data, but that is the next thing to establish.
Ensemble comparisons cover flu admissions in 2023–24, flu and COVID
admissions in 2024–25, and all six targets in 2025–26; this is not an
18-cell sweep of wins. These same three seasons also guided architecture selection,
so this is exploratory cross-validation, not an untouched final evaluation.

[**Results here — scroll straight to Fan plots.**](results/b0-1-crosses/index.md#fan-plots)
The write-up is AI-written; the plots are what I would start with. Green is the
new models, grey is the ensemble. I really like how the intervals look: they
feel very “flu” to me, especially the uptick. That is a visual impression;
measured coverage is still below nominal levels.

Across the full B0.1 experiment, **36 of 172 configurations beat the ensemble
on the combined score**, with a best score of **0.883** (1 is ensemble parity;
lower is better). That combined-score count is not a count of unicorns winning
every target/season comparison. The top five configurations fit separate
models by target or pathogen, while still receiving all six input histories.
The simpler encoders look particularly encouraging.

## B0: six channels, no revision nowcasting

B0 is the finalized-data experiment. The channels are:

| | Influenza | COVID-19 | RSV |
|---|---|---|---|
| NHSN hospital admissions | Channel 1 · counts | Channel 2 · counts | Channel 3 · counts |
| NSSP ED visits | Channel 4 · proportion | Channel 5 · proportion | Channel 6 · proportion |

Every channel has an observation mask. A missing value is different from an
observed zero. The panel contains 50 states, DC, and the native US series;
US is forecast directly rather than constructed by adding state predictions.

```mermaid
flowchart TD
    A["NHSN finalized admissions<br/>Flu · COVID-19 · RSV"] --> C["Shared source selection<br/>Weekly dates · native geography · units"]
    B["NSSP latest frozen ED snapshot<br/>Flu · COVID-19 · RSV"] --> C
    C --> D["Saved B0 panel<br/>week × 6 channels × value/mask × 52 locations"]
    D --> E["History window<br/>8, 12, or 26 weeks · all six channels"]
    D --> F["Future labels + separate supervision masks<br/>Four weeks after the context end"]
    E --> G["Fold-fitted transforms and scales<br/>Fit on training seasons only"]
    G --> H["Conditional sample generator"]
    F --> I["Fair CRPS in native units<br/>Training-side scaling and target/location weights"]
    H --> I
    H --> J["Sampled futures → 23 quantiles<br/>EpiBenchmark WIS vs hub ensembles"]
```

Admissions are stored as counts; ED percentages are converted to proportions.
Admission-rate transforms and normalization are fitted inside each training
partition. B0 treats the frozen latest NSSP values as retrospective truth;
they are not guaranteed immutable. The input mask handles existing gaps, but
this experiment does not establish performance with arbitrary missing sources
or historical reporting delays. See the [data contract](data/build-b-finalized.md).

## What architectures have I tried?

The common model is a conditional sample generator. A history encoder describes
what is happening; random noise modulates a decoder to produce a possible
future. Repeating this gives a forecast distribution. Fair CRPS rewards
samples that are both accurate and appropriately spread, in admission counts
or ED proportions. Forecast quantiles are then evaluated with WIS.

```mermaid
flowchart TD
    A["Local six-channel history + masks<br/>Calendar · population · optional recent dynamics"] --> B["Temporal encoder<br/>MLP / convolution / multiscale convolution"]
    B --> C["Context exchange<br/>None / spatial / pathogen / target / joint attention"]
    C --> D["Residual sample decoder<br/>Shared, pathogen-specific, or target-specific heads"]
    Z["Fresh random draw per member<br/>Global noise; optional local noise"] --> D
    D --> E["Four future weeks × requested targets × locations<br/>Inverse transforms → native-unit samples"]
    E --> F["Predictive intervals and quantiles"]
```

| Part | Tried in B0.1 | What I take from it so far |
|---|---|---|
| History encoder | Flattened-history MLP, temporal convolution, multiscale convolution | MLP and ordinary convolution improved the tested matched references. More complexity did not automatically help. |
| Information exchange | Local only; shared spatial, pathogen-specific, target-specific, or joint location–target attention | These are transformer-style attention blocks over encoded histories, not a separate temporal-transformer encoder. Joint attention can see 52 × 6 = 312 tokens; its benefit depends on the recipe. |
| Sharing | Shared heads, three pathogen heads, six target heads; also fully independent fits by pathogen or target | The five leading configurations use independent fits. Each component still sees all six local input channels. Sharing inputs and sharing model weights are different choices. |
| Uncertainty | Noise-modulated residual decoders, global/local noise, stochastic trend decoder | The trend decoder worsened all four matched comparisons and has been removed from the active grid. Historical results retain it. |

The completed experiment crossed **172 configurations and three random seeds**,
with three held-out-season folds per seed. The [B0.1 specification](design/b0.1.md)
has the exact recipes; the [results](results/b0-1-crosses/index.md) have the
comparisons. These findings are about the tested combinations, not a general
ranking of MLPs, convolutions, and transformers.

## From prediction to nowcasting

My next step is to make sure we can use whatever sources are available, with
masking, then add covariates, and then establish the nowcasting performance.
The [B1 implementation](design/b1.md) already has the Wednesday-vintage data
and correction/forecast path; the unicorn results above are still **B0 results**.
Nowcasting here means estimating the eventual values of recently completed
weeks whose reports are missing or provisional.

```mermaid
flowchart TD
    A["Historical Wednesday information state<br/>Only reports released by the cutoff"] --> B["Six-channel history + availability masks<br/>Keep provisional values and missing reports"]
    B --> C["B1 history encoder"]
    C --> D["Sample revision corrections<br/>Two recently completed weeks"]
    D --> E["Condition on each sampled correction<br/>Generate the next four weeks"]
    Z["Random member draws"] --> D
    Z --> E
    T["Separately pinned reference truth<br/>Recent and future labels + supervision masks"] -. "training only" .-> D
    T -. "training only" .-> E
    E --> F["Forecast samples → hub quantiles → evaluation"]
    G["Later: additional sources and covariates"] -.-> B
```

The key distinction is event time versus release time: what happened last week
is not necessarily what we knew on Wednesday. Later revised values are labels,
not replacements for the historical inputs. B1 currently uses an end-of-Wednesday
cutoff; an actual intraday submission deadline needs its own availability check.

I feel good about this. I would bet on ensemble-matching performance across all
six hub targets, but that is my expectation, not an established result. The
timeline is tight, and I will not be able to keep working at this pace once my
family joins me. In forecasting you really need to nail the details, much more
than just have a good model. I now understand the reporting and revision timing
much better. Perhaps the FluSight submission will be an ensemble of the three
cross-validated models; that mixture still needs to be evaluated.

## Working with Tapestry

Start with [Getting started](getting-started.md), the
[training workflow](workflows/training.md), or the [data explorer](explorer/overview.md).
The [architecture notes](design/architecture.md) explain the model and masks; the [B1 page](design/b1.md) documents the current vintage-aware work.

*Documentation note · September 16, 2026: this front page uses my research update
for the personal narrative and “unicorn” terminology, and the saved B0.1 report
for numerical summaries. No scores were recomputed for this update. “Across
three seasons” means the available target/season evaluation support, not a
complete six-by-three grid. Diagrams distinguish evaluated B0, implemented B1,
and later source/covariate extensions.*
