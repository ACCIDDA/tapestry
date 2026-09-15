"""Total-WIS scoring against the ensemble and the season-equal, admissions-weighted ranking."""
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')

from tapestry.evaluation.hubs import KEY, QCOLS, SEASONS, export_b0
from tapestry.evaluation.totals import (METRICS, configuration_ranking, quantile_scores, run_scores,
                                        score_run, season_scores)
from tapestry.models.quantiles import LEVELS


def test_wis_components_match_interval_and_pinball_definitions():
    rng = np.random.default_rng(0)
    q = np.sort(rng.gamma(2, 10, size=(50, len(LEVELS))), axis=1)
    y = rng.gamma(2, 10, size=50)
    scores = quantile_scores(q, y)
    error = y[:, None] - q
    np.testing.assert_allclose(scores.wis, 2 * np.maximum(LEVELS * error, (LEVELS - 1) * error).mean(1))
    k = len(LEVELS) // 2
    total = .5 * np.abs(y - q[:, k])
    for i in range(k):
        alpha, lower, upper = 2 * LEVELS[i], q[:, i], q[:, -1 - i]
        total = total + alpha / 2 * ((upper - lower) + 2 / alpha * (lower - y) * (y < lower)
                                     + 2 / alpha * (y - upper) * (y > upper))
    np.testing.assert_allclose(scores.wis, total / (k + .5))
    np.testing.assert_allclose(scores.covered_50, (q[:, 6] <= y) & (y <= q[:, 16]))
    np.testing.assert_allclose(scores.covered_95, (q[:, 1] <= y) & (y <= q[:, 21]))
    assert (scores[['dispersion', 'underprediction', 'overprediction']] >= 0).all().all()
    with pytest.raises(ValueError, match='symmetric'):
        quantile_scores(q[:, :-1], y, LEVELS[:-1])


def totals_row(config, seed, target, season, geography, model_wis, ensemble_wis):
    row = dict(config_id=config, seed=seed, target=target, season=season, geography=geography, horizon=0, n=10)
    for who, value in (('model', model_wis), ('ensemble', ensemble_wis)):
        row.update({f'{who}_{metric}': 0. for metric in METRICS})
        row[f'{who}_wis'] = value
    return row


def test_scores_are_total_wis_ratios_with_equal_seasons_and_double_admissions():
    rows = [
        # Season A: location ratios 2 and .5, but the target's ratio is total over total.
        totals_row('a', 42, 'wk inc flu hosp', 'A', 'states_dc', 10, 5),
        totals_row('a', 42, 'wk inc flu hosp', 'A', 'US', 1000, 2000),
        # Season B counts as much as season A despite far smaller totals.
        totals_row('a', 42, 'wk inc flu hosp', 'B', 'states_dc', 3, 2),
        totals_row('a', 42, 'wk inc flu hosp', 'B', 'US', 30, 30),
    ]
    others = {'wk inc covid hosp': .8, 'wk inc rsv hosp': 1.2, 'wk inc flu prop ed visits': 1.5,
              'wk inc covid prop ed visits': .9, 'wk inc rsv prop ed visits': 1.1}
    rows += [totals_row('a', 42, target, 'B', 'US', ratio, 1) for target, ratio in others.items()]
    totals = pd.DataFrame(rows)
    seasons = season_scores(totals)
    season_a = seasons[(seasons.geography == 'all') & (seasons.target == 'wk inc flu hosp') & (seasons.season == 'A')]
    assert season_a.wis_ratio.item() == pytest.approx(1010 / 2005)
    runs = run_scores(seasons).set_index('geography')
    flu = (1010 / 2005 + 33 / 32) / 2
    assert runs.loc['all', 'wk inc flu hosp'] == pytest.approx(flu)
    assert runs.loc['states_dc', 'wk inc flu hosp'] == pytest.approx((2 + 1.5) / 2)
    expected = (2 * (flu + .8 + 1.2) + 1.5 + .9 + 1.1) / 9
    assert runs.loc['all', 'combined'] == pytest.approx(expected)
    # States/DC has no rows for the other targets here, so no partial combined score.
    assert np.isnan(runs.loc['states_dc', 'combined'])

    worse_seed = totals.assign(seed=43, model_wis=totals.model_wis * 1.5)
    better = totals.assign(config_id='b', model_wis=totals.model_wis / 2)
    ranking = configuration_ranking(run_scores(season_scores(pd.concat([totals, worse_seed, better])))).set_index('config_id')
    assert ranking.index.tolist() == ['b', 'a']
    assert ranking.loc['a', 'seeds'] == 2
    assert ranking.loc['a', 'combined_mean'] == pytest.approx(expected * 1.25)
    assert ranking.loc['b', 'combined_mean'] == pytest.approx(expected / 2)


def write_run(folder):
    for year, held in zip((2023, 2024, 2025), SEASONS):
        fold = folder / f'eval_{held}'
        fold.mkdir(parents=True)
        context = date(year, 10, 7)
        q = np.broadcast_to(np.linspace(1, 23, len(LEVELS))[:, None, None, None, None], (len(LEVELS), 1, 4, 6, 2)).copy()
        q[:, :, :, 3:] /= 100
        np.savez(fold / 'forecasts.npz', quantile_levels=LEVELS, quantiles=q, context_end=[context.isoformat()],
                 target_dates=[[(context + timedelta(weeks=h)).isoformat() for h in range(1, 5)]],
                 locations=['NC', 'US'], truth=np.zeros((1, 4, 6, 2)), mask=np.ones((1, 4, 6, 2), dtype=bool))


def test_score_run_sums_model_and_ensemble_on_identical_frozen_tasks(tmp_path):
    run = tmp_path / 'run'
    write_run(run)
    forecast = export_b0(run)[('2023-2024', 'wk inc flu hosp')]
    observed = np.array([0., 5, 12, 30] * 2)
    units = forecast[KEY].assign(observed=observed)
    ensemble = forecast[KEY + QCOLS].copy()
    ensemble[QCOLS] = ensemble[QCOLS] + 2
    frozen = tmp_path / 'frozen'
    (frozen / 'flu').mkdir(parents=True)
    units.to_parquet(frozen / 'flu' / 'units.parquet', index=False)
    ensemble.assign(model='FluSight-ensemble', observed=observed).to_parquet(frozen / 'flu' / 'quantiles.parquet', index=False)
    case = dict(status='scored', directory='flu', target='wk inc flu hosp', season='2023-2024', ensemble='FluSight-ensemble')
    unscored = dict(status='no ensemble forecasts', target='wk inc rsv hosp', season='2023-2024')
    (frozen / 'manifest.json').write_text(json.dumps(dict(quantiles=QCOLS, cases=[case, unscored])))
    totals = score_run(run, frozen)
    assert (run / 'totals.csv').is_file()
    assert set(totals.geography) == {'states_dc', 'US'} and totals.n.sum() == 8
    model = quantile_scores(forecast[QCOLS].to_numpy(), observed)
    reference = quantile_scores(ensemble[QCOLS].to_numpy(), observed)
    assert totals.model_wis.sum() == pytest.approx(model.wis.sum())
    assert totals.ensemble_wis.sum() == pytest.approx(reference.wis.sum())
    us = forecast.location.eq('US').to_numpy()
    assert totals[totals.geography == 'US'].model_wis.sum() == pytest.approx(model.wis[us].sum())
    (frozen / 'manifest.json').write_text(json.dumps(dict(quantiles=QCOLS[::5], cases=[case])))
    with pytest.raises(ValueError, match='rebuild frozen support'):
        score_run(run, frozen)
