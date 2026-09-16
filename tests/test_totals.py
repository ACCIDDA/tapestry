"""Native WIS, equal-location relative skill, and season-first aggregation."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')

from tapestry.evaluation.totals import (METRICS, configuration_ranking, quantile_scores, rank, run_scores,
                                        season_scores)
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


def totals_row(config, seed, target, season, geography, model_wis, ensemble_wis):
    row = dict(config_id=config, seed=seed, target=target, season=season, geography=geography, location='US' if geography == 'US' else 'NC', horizon=0, n=10)
    for who, value in (('model', model_wis), ('ensemble', ensemble_wis)):
        row.update({f'{who}_{metric}': 0. for metric in METRICS})
        row[f'{who}_wis'] = value
    return row


def test_location_ratios_and_season_first_target_weighting():
    rows = [
        # Season A: 80% state ratio 2 + 20% US ratio .5 = 1.7, irrespective of size.
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
    assert season_a.wis_ratio.item() == pytest.approx(1.7)
    runs = run_scores(seasons).set_index('geography')
    flu = (1.7 + 1.4) / 2
    assert runs.loc['all', 'wk inc flu hosp'] == pytest.approx(flu)
    assert runs.loc['states_dc', 'wk inc flu hosp'] == pytest.approx((2 + 1.5) / 2)
    expected = (1.7 + (1.4 + .8 + 1.2 + .5 * (1.5 + .9 + 1.1)) / 4.5) / 2
    assert runs.loc['all', 'combined'] == pytest.approx(expected)
    # States/DC lacks target support present elsewhere; no silent partial score.
    assert np.isnan(runs.loc['states_dc', 'combined'])

    worse_seed = totals.assign(seed=43, model_wis=totals.model_wis * 1.5)
    better = totals.assign(config_id='b', model_wis=totals.model_wis / 2)
    ranking = configuration_ranking(run_scores(season_scores(pd.concat([totals, worse_seed, better])))).set_index('config_id')
    assert ranking.index.tolist() == ['b', 'a']
    assert ranking.loc['a', 'seeds'] == 2
    assert ranking.loc['a', 'combined_mean'] == pytest.approx(expected * 1.25)
    assert ranking.loc['b', 'combined_mean'] == pytest.approx(expected / 2)


def test_state_size_and_task_count_do_not_set_jurisdiction_weights():
    rows = [totals_row('a', 42, 'wk inc flu hosp', 'A', 'states_dc', 2, 1),
            totals_row('a', 42, 'wk inc flu hosp', 'A', 'states_dc', 100, 100),
            totals_row('a', 42, 'wk inc flu hosp', 'A', 'US', 50, 100)]
    rows[1].update(location='CA', n=100)
    scored = season_scores(pd.DataFrame(rows)).set_index('geography')
    assert scored.loc['all', 'wis_ratio'] == pytest.approx(.8 * 1.5 + .2 * .5)
    assert scored.loc['states_dc', 'wis_ratio'] == pytest.approx(1.5)
    assert scored.loc['all', 'us_weight_used'] == pytest.approx(.2)
    rows[0]['ensemble_wis'] = 0
    with pytest.raises(ValueError, match='positive finite ensemble'):
        season_scores(pd.DataFrame(rows))


def test_rank_refuses_missing_location_in_one_run(tmp_path):
    runs = []
    rows = [totals_row('a', 42, 'wk inc flu hosp', 'A', 'states_dc', 2, 1),
            totals_row('a', 42, 'wk inc flu hosp', 'A', 'US', 50, 100)]
    for i in range(2):
        path = tmp_path / str(i)
        path.mkdir()
        pd.DataFrame(rows[i:]).drop(columns=['config_id', 'seed']).to_csv(path / 'totals.csv', index=False)
        runs.append(dict(config_id=str(i), seed=42, path=path))
    with pytest.raises(ValueError, match='different frozen'):
        rank(runs, tmp_path / 'ranking')
