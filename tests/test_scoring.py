"""WIS mathematics, location-relative skill, the CDC pairwise method and raw-task checks."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')

from tapestry.evaluation.standard import METRICS, quantile_scores, season_scores
from tapestry.evaluation.quantiles import LEVELS


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


def test_state_size_and_task_count_do_not_set_jurisdiction_weights():
    """States/DC: equal mean of location ratios (not pooled totals); the US is scored alone."""
    rows = [totals_row('a', 42, 'wk inc flu hosp', 'A', 'states_dc', 2, 1),
            totals_row('a', 42, 'wk inc flu hosp', 'A', 'states_dc', 100, 100),
            totals_row('a', 42, 'wk inc flu hosp', 'A', 'US', 50, 100)]
    rows[1].update(location='CA', n=100)
    scored = season_scores(pd.DataFrame(rows)).set_index('geography')
    assert scored.loc['states_dc', 'wis_ratio'] == pytest.approx(1.5)
    assert scored.loc['US', 'wis_ratio'] == pytest.approx(.5)
    assert 'all' not in scored.index
    rows[0]['ensemble_wis'] = 0
    with pytest.raises(ValueError, match='positive finite ensemble'):
        season_scores(pd.DataFrame(rows))


def test_pairwise_relative_wis_uses_shared_tasks_and_baseline():
    """CDC method: mean WIS ratios on shared tasks, geometric mean over all models, over the baseline."""
    import pandas as pd
    from tapestry.evaluation.standard import relative_wis
    scores = pd.DataFrame({'A': [1., 2., 3.], 'B': [2., 4., np.nan], 'base': [4., 4., 4.]})
    got = relative_wis(scores, 'base')
    ab = 1.5 / 3  # A vs B on the two shared tasks
    theta = {'A': (1 * ab * (2 / 4)) ** (1 / 3), 'B': ((1 / ab) * 1 * (3 / 4)) ** (1 / 3),
             'base': ((4 / 2) * (4 / 3) * 1) ** (1 / 3)}
    for model in theta:
        np.testing.assert_allclose(got[model], theta[model] / theta['base'])
    assert got['base'] == 1


def test_raw_wis_rejects_invalid_forecasts_instead_of_averaging_them_away():
    import pandas as pd
    from tapestry.evaluation.standard import check_raw_tasks, transform
    from tapestry.evaluation.hubs import QCOLS, KEY
    frame = pd.DataFrame([dict(zip(KEY, ('2025-11-22', '2025-11-22', '37', 0)), **{q: float(i) for i, q in enumerate(QCOLS)},
                               model_original_truth=5.)])
    check_raw_tasks(frame, 'wk inc flu hosp', 'ok')
    for broken in (dict(model_original_truth=np.nan), {QCOLS[3]: np.nan}, {QCOLS[3]: 100.}):
        with pytest.raises(ValueError):
            check_raw_tasks(frame.assign(**broken), 'wk inc flu hosp', 'broken')
    with pytest.raises(ValueError):
        check_raw_tasks(frame, 'wk inc flu prop ed visits', 'ED above one')
    with pytest.raises(ValueError):
        transform([-1.], 'log')
