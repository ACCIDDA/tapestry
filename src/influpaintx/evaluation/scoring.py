"""Shared frozen-task validation and aggregation for B0 comparisons and experiments."""
import numpy as np
import pandas as pd

from .hubs import KEY, QCOLS

METRICS = ['wis', 'ae_median', 'interval_coverage_50', 'interval_coverage_95',
           'underprediction', 'dispersion', 'overprediction', 'bias']


def validate(frame, target):
    q = frame[QCOLS].to_numpy()
    if frame.duplicated(KEY).any() or not np.isfinite(q).all() or (q < 0).any() or (np.diff(q, axis=1) < 0).any():
        raise ValueError(f'Invalid forecast units/quantiles: {target}')
    if 'prop' in target and (q > 1).any():
        raise ValueError('ED forecasts must be proportions')
    delta = (pd.to_datetime(frame.target_end_date) - pd.to_datetime(frame.reference_date)).dt.days
    if not frame.horizon.isin(range(4)).all() or not delta.eq(frame.horizon * 7).all():
        raise ValueError('Expected hub horizons 0–3 with exact weekly target dates')


def match_forecasts(predictions, units, target):
    """Require one valid forecast for every frozen evaluation task."""
    wide = units[KEY + ['observed']].merge(predictions[KEY + QCOLS], on=KEY, validate='one_to_one')
    if len(wide) != len(units):
        raise ValueError(f'Missing frozen tasks: {target}')
    validate(wide, target)
    return wide


def matched_scores(scores, units, metrics=METRICS):
    """Never reward missing difficult tasks, or accept duplicate scoring units."""
    matched = units[KEY + ['observed']].merge(scores, on=KEY, validate='one_to_one', suffixes=('_truth', ''))
    if len(matched) != len(units) or not np.allclose(matched.observed_truth, matched.observed):
        raise ValueError('Scores must cover every frozen unit with identical truth')
    if not np.isfinite(matched[list(metrics)].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite scoring result')
    return matched.drop(columns='observed_truth')


def aggregate_scores(scores, metrics=METRICS):
    """Mean scores per target, season, model, geography, and horizon."""
    rows = []
    for geography in ('states_dc', 'US'):
        geo = scores[scores.location.eq('US') if geography == 'US' else scores.location.ne('US')]
        for horizon in ('all', 0, 1, 2, 3):
            part = geo if horizon == 'all' else geo[geo.horizon == horizon]
            for (target, season, model), group in part.groupby(['target', 'season', 'model']):
                rows.append(dict(target=target, season=season, model=model, geography=geography, horizon=str(horizon),
                                 n=len(group), **group[list(metrics)].mean().to_dict(),
                                 wis_ratio=group.wis.mean() / group.ensemble_wis.mean()))
                if 'rwis' in metrics:
                    rows[-1]['rwis_n'] = int(group.rwis.notna().sum())
    result = pd.DataFrame(rows)
    return result


def rank(scores):
    result = aggregate_scores(scores, METRICS + ['rwis'])
    result['rank'] = result.groupby(['target', 'season', 'geography', 'horizon']).wis.rank(method='min')
    return result


def objective(summary, multi=False):
    selected = summary[summary.horizon.astype(str).eq('all') & summary.target.str.endswith('hosp')]
    if not multi:
        selected = selected[selected.target.eq('wk inc flu hosp')]
    # Equal target weight, then equal available season/geography cells per target.
    return float(np.exp(selected.assign(log_ratio=np.log(selected.wis_ratio))
                        .groupby('target').log_ratio.mean().mean()))


