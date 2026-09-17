"""Nowcast scoring for the two completed weeks B1 predicts (offsets -2 and -1).

No Hub ensemble forecasts a week that has already happened, so the ensemble
denominator B0 uses does not exist here. The denominator is instead the naive
nowcast "the value visible on Wednesday is already final": preliminary-value
persistence, falling back to the latest visible context week when this week's
preliminary report is missing. That is `season_cv.persistence`, the same rule B0
uses for its own persistence diagnostic.

Everything else is B0's: `totals.quantile_scores` computes the metrics and
`totals.season_scores`/`run_scores`/`configuration_ranking` do the aggregation,
so a nowcast rank means the same thing as a forecast rank (location-relative
ratio, states/DC 80% and US 20%, admissions 1 and ED .5 within equal seasons)
with a different denominator. The two scores share weights but not support and
are reported separately; they must never be averaged into one number.
"""
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.models.quantiles import LEVELS, select_quantiles
from tapestry.model_data.finalized import season
from tapestry.models.season_cv import SEASONS
from .hubs import HUBS
from .totals import METRICS, quantile_scores

BASELINE = 'preliminary-value persistence (latest Wednesday-visible value)'
SCORE_VERSION = 'nowcast-persistence-relative-season-first-us20-v1'
SCORE_DEFINITION = (
    'Per target/season/location: total native-unit model WIS / total '
    f'{BASELINE} WIS over identical nowcast cells at offsets -2 and -1. '
    'States/DC ratios average equally with 80% weight; US gets 20%. Within a '
    'season, available targets average with admissions 1 and ED .5. Seasons '
    'average equally. Cells with no visible history anywhere have no baseline '
    'and are excluded from both numerator and denominator, and counted in '
    'nowcast-support.json. This is skill against a naive no-revision nowcast, '
    'not against a competing nowcasting method.')
# `season_scores` and friends key off these names; the baseline takes the
# ensemble's place so B0's aggregation is reused verbatim, not reimplemented.
CHANNEL_TARGETS = {c: target for spec in HUBS.values() for target, c in spec['targets'].items()}


def nowcast_cells(run, manifest):
    """Model and baseline WIS sums per target/season/location/horizon for one run.

    Each fold contributes only its own held-out season, as the forecast scorer
    does; a fold's recent weeks are in the season it is evaluated on by
    construction, but the filter is applied explicitly rather than assumed.
    """
    prefix = f'{manifest["run_id"]}-s{manifest["seed"]}'
    parts, excluded, scored = [], 0, 0
    for held in SEASONS:
        path = Path(run) / f'eval_{held}' / f'forecasts-{prefix}-natural.npz'
        with np.load(path, allow_pickle=False) as data:
            horizons = data['horizons']
            recent = horizons < 0
            if not recent.any():
                return None, None  # a direct configuration produces no nowcasts
            quantiles = select_quantiles(data['quantiles'], data['quantile_levels'])
            q = quantiles[:, :, recent]                      # levels, origin, week, channel, location
            truth, valid = data['truth'][:, recent], data['mask'][:, recent]
            dates = data['target_dates'][:, recent]
            locations = [str(loc) for loc in data['locations']]
            # The baseline is one value per channel/location for the episode; both
            # recent weeks are nowcast from the same latest visible observation.
            base = np.broadcast_to(data['baseline'][:, None], truth.shape)
            base_valid = np.broadcast_to(data['baseline_mask'][:, None], valid.shape)
            in_season = np.array([[season(date.fromisoformat(str(d))) == held for d in row] for row in dates])
            usable = valid & base_valid & in_season[:, :, None, None]
            excluded += int((valid & in_season[:, :, None, None] & ~base_valid).sum())
            scored += int(usable.sum())
            for c, target in sorted(CHANNEL_TARGETS.items()):
                cells = usable[:, :, c]
                if not cells.any():
                    continue
                origin, week, location = np.nonzero(cells)
                y = truth[origin, week, c, location]
                model = quantile_scores(q[:, origin, week, c, location].T, y).add_prefix('model_')
                # A deterministic baseline is the same value at every quantile.
                flat = np.repeat(base[origin, week, c, location][:, None], len(LEVELS), axis=1)
                baseline = quantile_scores(flat, y).add_prefix('ensemble_')
                table = pd.concat([model, baseline], axis=1)
                table['location'] = [locations[i] for i in location]
                table['geography'] = np.where(table.location.eq('US'), 'US', 'states_dc')
                # Offsets -2/-1 keep their own identity, as horizons 0-3 do.
                table['horizon'] = horizons[recent][week]
                grouped = table.groupby(['geography', 'location', 'horizon'])
                summed = grouped.sum()
                summed.insert(0, 'n', grouped.size())
                parts.append(summed.reset_index().assign(target=target, season=held))
    if not parts:
        return None, None
    columns = ['target', 'season', 'geography', 'location', 'horizon', 'n',
               *[f'{who}_{m}' for who in ('model', 'ensemble') for m in METRICS]]
    audit = dict(baseline=BASELINE, scored_cells=scored, excluded_no_history=excluded,
                 definition=SCORE_DEFINITION, score_version=SCORE_VERSION)
    return pd.concat(parts, ignore_index=True)[columns], audit


def score_nowcasts(run):
    """Write `nowcast-totals.csv` beside `totals.csv`, or nothing for a direct fit."""
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    totals, audit = nowcast_cells(run, manifest)
    if totals is None:
        return None
    temporary = run / 'nowcast-totals.tmp'
    totals.to_csv(temporary, index=False)
    temporary.replace(run / 'nowcast-totals.csv')
    (run / 'nowcast-support.json').write_text(json.dumps(audit, indent=2) + '\n')
    return totals


def rank(runs, output):
    """Rank runs that have `nowcast-totals.csv`, reusing B0's aggregation exactly."""
    from .totals import configuration_ranking, run_scores, season_composites, season_scores
    from tapestry.models.objective import US_WEIGHT
    output = Path(output)
    available = [run for run in runs if (Path(run['path']) / 'nowcast-totals.csv').is_file()]
    if not available:
        return None
    output.mkdir(parents=True, exist_ok=True)
    totals = pd.concat([pd.read_csv(Path(run['path']) / 'nowcast-totals.csv')
                        .assign(config_id=run['config_id'], seed=run['seed']) for run in available],
                       ignore_index=True)
    support_keys = ['target', 'season', 'location', 'horizon', 'n']
    supports = [part[support_keys].sort_values(support_keys).reset_index(drop=True)
                for _, part in totals.groupby(['config_id', 'seed'])]
    if any(not part.equals(supports[0]) for part in supports[1:]):
        raise ValueError('Nowcast runs scored different cells; rescore on identical tasks')
    seasons = season_scores(totals)
    scores = run_scores(seasons)
    if not np.isfinite(scores.loc[scores.geography == 'all', 'combined']).all():
        raise ValueError('Incomplete or invalid combined nowcast scores')
    ranking = configuration_ranking(scores)
    seasons.to_csv(output / 'season_scores.csv', index=False)
    season_composites(seasons).to_csv(output / 'season_composite_scores.csv', index=False)
    scores.to_csv(output / 'run_scores.csv', index=False)
    ranking.to_csv(output / 'configuration_ranking.csv', index=False)
    (output / 'manifest.json').write_text(json.dumps(dict(
        runs=[dict(run, path=str(run['path'])) for run in available],
        quantile_levels=LEVELS.tolist(), us_weight=US_WEIGHT, baseline=BASELINE,
        definition=SCORE_DEFINITION, score_version=SCORE_VERSION), indent=2) + '\n')
    return ranking
