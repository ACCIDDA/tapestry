# B0 definition and implementation: finalized six-channel pilot

Implemented 13 September 2026. The existing intake and `SelectedData` were already
coded; the tensor materializer and window querier were added for this pilot.
B0, transforms, the stochastic decoder, losses, training, and canonical scoring
are implemented; see [training and prediction](../workflows/training.md) and the
[scored results](../results/b0-configuration-comparison.md). The
[architecture](../design/architecture.md) records the broader proposals.

Implemented and scored in B0: finalized retrospective NHSN/NSSP inputs, six channels,
configurable lookback windows, masked stochastic prediction, the configuration
comparison, three seeds, and hub/scoringutils evaluation.

## Contract and assumptions

User requirements: Build B, no wastewater, six channels, configurable history
length starting at 8 (12 is supported), values plus masks, locations, season
queries, and data beginning September 2023.

The pilot interprets the first dimension as **history weeks**, independent of
forecast horizon count. Stored axes are `[week, channel, value_or_mask, location]`.
A query returns `X[lookback, 6, 2, L]` and independently masked
`Y[number_of_horizons, 6, 2, L]`. Field index 0 is value, 1 is mask.
Channel order is NHSN admissions flu/COVID/RSV, then NSSP ED proportions
flu/COVID/RSV. Admissions remain counts; reported, unsmoothed weekly CDC ED
percentages are divided by 100. The dataset builder fits no scalers and applies no population conversion.
Training fits these inside each fitting fold.

Finality assumption: use only `cdc_nhsn_final` and the frozen latest
`cdc_nssp_trajectories` snapshot. NSSP does not expose a finality flag here, so
latest published values are treated as retrospective truth, not guaranteed
immutable values. No preliminary NHSN, provider substitution, daily aggregation,
smoothing, wastewater, or real-time vintage reconstruction is involved.
The manifest records source snapshot IDs and the original manifests, plus hashes
of those manifests (which themselves record raw artifact checksums).

Geography assumption: use the existing selector's 50 states + DC + native US,
ordered alphabetically by postal code with US last. No national/state broadcast
or sum is calculated. Territories are not supported by the current selector.
All six series can serve as focal targets for the shared decoder described in B;
select a channel slice of X for its focal history and retain X for context.
The dataset builder supplies tensors; B0 implements the shared decoder and source embeddings.

The first stored week is September 2, 2023. Earlier history is masked padding,
so initial September examples have fewer than eight observed history weeks.
For full eight-week calendar histories, begin origins on October 21, 2023.
Missing/suppressed/invalid cells have `(value, mask)=(0,0)`; a valid zero is `(0,1)`.
Duplicates with conflicting values cause a build error. Missing/invalid raw counts
and observed coverage by season/channel/location are recorded in the manifest.
Per-cell quality-reason categories and release-age features are deferred.

Season assignment follows the design's CDC epiweek 31 through week 30;
this groups respiratory seasons and does not assert official FluSight challenge
issuance dates. Training data begin in September, not at the August season boundary.
Windows may use prior-season context. The season filter applies to the context
end; explicit target bounds mask labels outside the permitted fitting period.
No train/development/holdout roles are silently assigned: the old plan's 2023–24
validation fold needs revisiting now that training starts in September 2023.

## Build and query

```bash
pip install -e '.[model-data]'
python -m tapestry.model_data --data-root data \
  --start 2023-09-01 --output data/processed/build_b_finalized.npz
```

```python
from tapestry.model_data import FinalizedDataset

ds = FinalizedDataset.load('data/processed/build_b_finalized.npz')
q = ds.query('2023-10-21', lookback=8)
assert q['X'].shape == (8, 6, 2, 52)
q12 = ds.query('2023-10-21', lookback=12, locations=['NY', 'NJ'])

# Illustrative fitting boundary, not a chosen evaluation protocol.
for episode in ds.windows(season_id='2023-2024', target_end='2024-07-27'):
    X, Y = episode['X'], episode['Y']
    target_values, target_mask = Y[:, :, 0, :], Y[:, :, 1, :]
```

`context_end` must be a Saturday and is included in X. Default future offsets
are `(1,2,3,4)` weeks after context end. These correspond to FluSight horizons
0–3 if the reference Saturday is one week after context end. For eight future
weeks, pass `horizons=tuple(range(1,9))`. This is explicitly different from the
old plan's revision-aware `[-1,0,...,6]` reference-date offsets: finalized inputs
would expose those recent finalized labels. Revision nowcasts are deferred.
`query` includes date/channel/location labels; unknown locations raise an error,
and absent dates are masked. `windows` omits episodes with no supervised labels.

## Local result and checks

The materialized artifact has shape `[157,6,2,52]`, covering September 2, 2023
through August 29, 2026. Of 8,164 possible cells per channel, observed counts are
8,128 each for flu/COVID admissions; 6,890 for RSV admissions; and 8,155 each for
the three ED proportions. Sparse RSV coverage remains masked, never imputed.
This is a corpus, not a declaration that all its seasons are training data.

22 focused tests passed across model data, source selection, and geography.
Checks include zero versus missing, percent-to-proportion conversion, invalid
values, conflicting duplicates, 8/12-week windows, date alignment, season
boundaries including a 53-week year, round-trip loading, and target-bound masks.
The dataset and model-data checks passed locally. Model fitting and evaluation are
reported in the canonical B0 results page linked above.
