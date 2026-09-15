import json

import numpy as np
import pandas as pd
import pytest

from tapestry.evaluation.configurations import identify
from tapestry.evaluation.hubs import KEY, QCOLS, wide_quantiles
from tapestry.evaluation.sweep import hubverse, matched_scores, METRICS, validate


def test_fan_selection_weights_targets_and_selects_local_winner():
    from tapestry.evaluation.sweep import fan_selection

    rows = []
    # Three admission seasons must not outweigh the single ED season.
    for model, admission, ed in [('a', .5, 4), ('b', 1, 1), ('c', 1.1, 1.1), ('d', 1.2, 1.2)]:
        for target, seasons, ratio in [('hosp', ['s1', 's2', 's3'], admission), ('ed', ['s3'], ed)]:
            for season in seasons:
                for geography in ['US', 'states_dc']:
                    rows.append(dict(model=model, target=target, season=season,
                                     geography=geography, horizon='all', wis_ratio=ratio))
    frame = pd.DataFrame(rows)
    # Ensemble and individual horizon rows do not enter model selection.
    frame = pd.concat([frame, frame.assign(model='ensemble', wis_ratio=.01),
                       frame.assign(horizon='0', wis_ratio=.01)])
    runs = pd.DataFrame(dict(model=list('abcd'), config_id=list('abcd'), label=list('abcd'), seed=42))
    top, best = fan_selection(frame, runs, dict(target='hosp', season='s3'))
    assert top == ['b', 'c', 'd']
    assert best == 'a'


def test_fan_ranking_averages_seed_scores_and_uses_middle_performance():
    from tapestry.evaluation.sweep import fan_ranking, fan_selection, ranking_tables

    rows, runs = [], []
    # A lucky seed must not put configuration a in the overall top three.
    for config, scores in [('a', [.01, 5, 6]), ('b', [.9, .7, .8]),
                           ('c', [1, 1.1, 1.2]), ('d', [1.3, 1.4, 1.5])]:
        for seed, score in zip([42, 43, 44], scores):
            model = f'{config}-{seed}'
            runs.append(dict(model=model, config_id=config, label=config, seed=seed))
            for target, value in [('hosp', score), ('ed', score)]:
                rows.append(dict(model=model, target=target, season='s1', horizon='all', wis_ratio=value))
    frame, runs = pd.DataFrame(rows), pd.DataFrame(runs)
    ranking = fan_ranking(frame, runs)
    assert ranking.config_id.tolist() == ['b', 'c', 'd', 'a']
    assert ranking.iloc[0]['mean'] == pytest.approx(.8)
    assert ranking.iloc[-1]['mean'] == pytest.approx(11.01 / 3)
    top, best = fan_selection(frame, runs, dict(target='hosp', season='s1'))
    assert top == ['b-44', 'c-43', 'd-43']
    assert best == 'b-44'
    exported_runs, exported_configs = ranking_tables(frame, runs,
        pd.DataFrame(index=pd.Index(list('abcd'), name='config_id')))
    assert exported_configs.index.tolist() == ranking.config_id.tolist()
    assert exported_configs.loc['b', 'all_target_mean'] == pytest.approx(.8)
    assert exported_configs.loc['a', 'all_target_mean'] == pytest.approx(11.01 / 3)
    assert exported_configs.loc['b', 'middle_model'] == top[0]
    assert exported_runs.iloc[0].model == 'a-42'  # Best single seed is not the winning configuration.
    # The seasonal middle seed can differ from the overall representative.
    frame.loc[frame.target.eq('hosp') & frame.model.str.startswith('b-'), 'wis_ratio'] = [.6, .7, .8]
    top, best = fan_selection(frame, runs, dict(target='hosp', season='s1'))
    assert top[0] == 'b-42'
    assert best == 'b-43'


def test_identity_is_the_scenario_string_not_code_or_paths(tmp_path):
    from dataclasses import asdict
    from tapestry.models.scenarios import ANCHOR
    manifest = dict(config=dict(asdict(ANCHOR), seed=42, output='old', device='cpu'),
                    dataset_sha256='abc', code_sha256={'/old/model.py': 'def'})
    def write():
        (tmp_path / 'run').mkdir(exist_ok=True)
        (tmp_path / 'run' / 'manifest.json').write_text(json.dumps(manifest))
        return identify(tmp_path / 'run', tmp_path / 'comparison')
    a = write()
    assert a['config_id'] == ANCHOR.scenario_string
    assert a['model_id'] == ANCHOR.scenario_string + ':s42'
    assert a['run'] == '../run'
    manifest['config'].update(output='new', device='cuda', seed=43)
    manifest['code_sha256'] = {'/new/model.py': 'changed'}
    b = write()
    assert b['config_id'] == a['config_id']
    assert b['model_id'] == ANCHOR.scenario_string + ':s43'
    assert b['provenance']['code_sha256'] != a['provenance']['code_sha256']
    manifest['config']['width'] = 32
    assert write()['config_id'] != a['config_id']


def test_hubverse_roundtrip_and_bad_dates(tmp_path):
    frame = pd.DataFrame(dict(reference_date=['2025-01-04'] * 4,
        target_end_date=['2025-01-04', '2025-01-11', '2025-01-18', '2025-01-25'],
        location=['01'] * 4, horizon=range(4)))
    frame[QCOLS] = np.arange(len(QCOLS))
    # Supply all hubs because the submission writer creates a complete repository.
    frames = {('2024-2025', f'wk inc {d} hosp'): frame for d in ('flu', 'covid', 'rsv')}
    assert hubverse(frames, 'B0-test-s42', tmp_path, csv=True) == 3 * 4 * len(QCOLS)
    path = next((tmp_path / 'hubverse/flusight').rglob('*.parquet'))
    long = pd.read_parquet(path)
    csv = pd.read_csv(path.with_suffix('.csv'), dtype={'location': str, 'output_type_id': str})
    pd.testing.assert_frame_equal(long, csv, check_dtype=False)
    assert set(long.columns) == set(KEY + ['target', 'output_type', 'output_type_id', 'value'])
    assert set(long.location) == {'01'}
    wide, audit = wide_quantiles(long, 'wk inc flu hosp')
    np.testing.assert_allclose(wide[QCOLS], frame[QCOLS])
    frame.loc[0, 'target_end_date'] = '2025-01-05'
    with pytest.raises(ValueError, match='exact weekly'):
        validate(frame, 'wk inc flu hosp')


def test_missing_units_and_changed_truth_rejected():
    units = pd.DataFrame(dict(reference_date=['2025-01-04'] * 2,
        target_end_date=['2025-01-04'] * 2, location=['01', 'US'], horizon=[0, 0], observed=[1., 2.]))
    scores = units.copy()
    scores[METRICS] = 0.
    assert len(matched_scores(scores, units)) == 2
    with pytest.raises(ValueError, match='every frozen unit'):
        matched_scores(scores.iloc[:1], units)
    scores.loc[0, 'observed'] = 2
    with pytest.raises(ValueError, match='identical truth'):
        matched_scores(scores, units)


def test_end_to_end_export_score_rank_and_plot(tmp_path, monkeypatch):
    from datetime import date, timedelta
    import importlib.util
    import shutil
    from tapestry.evaluation.hubs import export_b0, SEASONS
    from tapestry.evaluation.compare import score_with_r
    from tapestry.evaluation.sweep import main
    from tapestry.models.season_cv import LEVELS
    if not shutil.which('Rscript') or importlib.util.find_spec('epibench') is None:
        pytest.skip('Integration requires local R and installed EpiBenchmark')
    run = tmp_path / 'run'
    run.mkdir()
    from dataclasses import asdict
    from tapestry.models.scenarios import ANCHOR
    (run / 'manifest.json').write_text(json.dumps(dict(config=dict(asdict(ANCHOR), seed=42), dataset_sha256='test', code_sha256={})))
    for year, held in zip((2023, 2024, 2025), SEASONS):
        folder = run / f'eval_{held}'
        folder.mkdir()
        context = date(year, 10, 7)
        q = np.broadcast_to(np.linspace(.01, .23, len(LEVELS))[:, None, None, None, None], (len(LEVELS), 1, 4, 6, 2))
        np.savez(folder / 'forecasts.npz', quantile_levels=LEVELS, quantiles=q,
                 context_end=[context.isoformat()],
                 target_dates=[[(context + timedelta(weeks=h)).isoformat() for h in range(1, 5)]],
                 locations=['NC', 'US'], truth=np.full((1, 4, 6, 2), .1), mask=np.ones((1, 4, 6, 2), dtype=bool))
    frozen = tmp_path / 'frozen'
    folder = frozen / 'flu-test'
    folder.mkdir(parents=True)
    case = dict(hub='flusight', status='scored', directory='flu-test', target='wk inc flu hosp', season='2023-2024', ensemble='FluSight-ensemble')
    (frozen / 'manifest.json').write_text(json.dumps(dict(cases=[case], hubs={'flusight': {'commit':'HEAD', 'truth_vintages': {'wk inc flu hosp':'2023-12-01'}}})))
    forecast = export_b0(run)[('2023-2024', 'wk inc flu hosp')]
    units = forecast[KEY].assign(observed=.1)
    units.to_parquet(folder / 'units.parquet', index=False)
    wide = forecast[KEY + QCOLS].assign(observed=.1, model=case['ensemble'])
    wide.to_parquet(folder / 'quantiles.parquet', index=False)
    score_with_r(wide, folder)
    output = tmp_path / 'result'
    monkeypatch.setattr('sys.argv', ['sweep', '--runs', str(run), '--frozen', str(frozen), '--output', str(output)])
    main()
    scores = pd.read_parquet(output / 'scores.parquet')
    assert len(scores) == 16
    np.testing.assert_allclose(scores.rwis, 1)
    ranking = pd.read_csv(output / 'configuration_ranking.csv')
    np.testing.assert_allclose(ranking.flu_mean, 1)
    assert len(list((output / 'plots').rglob('*.svg'))) == 8
    assert (output / 'REPORT.md').exists()
    assert (output / 'manifest.json').exists()
    assert (output / 'epibench/flu-test/output/EpiBenchmark_scores.csv').exists()
    assert (output / 'epibench/flu-test/output/summary.md').exists()
    assert json.loads((output / 'manifest.json').read_text())['scoring_engine'] == 'epibench score --config-path'


def test_shared_forecast_matching_rejects_missing_and_duplicate_tasks():
    from tapestry.evaluation.scoring import match_forecasts
    units = pd.DataFrame(dict(reference_date=['2025-01-04'] * 2,
        target_end_date=['2025-01-04'] * 2, location=['01', 'US'], horizon=[0, 0], observed=[1., 2.]))
    predictions = units[KEY].copy()
    predictions[QCOLS] = np.arange(len(QCOLS))
    matched = match_forecasts(predictions.iloc[::-1], units, 'wk inc flu hosp')
    assert matched.location.tolist() == ['01', 'US']
    with pytest.raises(ValueError, match='Missing frozen tasks'):
        match_forecasts(predictions.iloc[:1], units, 'wk inc flu hosp')
    with pytest.raises(pd.errors.MergeError):
        match_forecasts(pd.concat([predictions, predictions.iloc[:1]]), units, 'wk inc flu hosp')


def test_objective_weights_geographies_equally():
    """Unequal state/US task counts must not change equal-geography weighting."""
    from tapestry.evaluation.scoring import aggregate_scores, objective

    metrics = ['wis', 'ae_median', 'interval_coverage_50', 'interval_coverage_95']
    units = pd.DataFrame(dict(reference_date=['2025-01-04'] * 3,
        target_end_date=['2025-01-04'] * 3, location=['01', '02', 'US'], horizon=[0] * 3))
    scores = units.assign(model='candidate', target='wk inc flu hosp', season='2024-2025', ensemble_wis=1.)
    scores[metrics] = 1.
    scores['wis'] = [2., 4., 8.]
    summary = aggregate_scores(scores, metrics)
    pooled = summary[summary.horizon.eq('all')].set_index('geography')
    assert pooled.loc['states_dc', 'n'] == 2
    assert pooled.loc['US', 'n'] == 1
    assert pooled.loc['states_dc', 'wis_ratio'] == 3.
    assert pooled.loc['US', 'wis_ratio'] == 8.
    assert objective(summary) == pytest.approx(np.sqrt(3 * 8))
