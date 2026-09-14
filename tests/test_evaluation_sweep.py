import json

import numpy as np
import pandas as pd
import pytest

from tapestry.evaluation.configurations import identify
from tapestry.evaluation.hubs import KEY, QCOLS, wide_quantiles
from tapestry.evaluation.sweep import hubverse, matched_scores, METRICS, validate


def test_identity_stable_and_future_fields_change_id(tmp_path):
    manifest = dict(config=dict(seed=42, output='old', device='cpu', width=64),
                    dataset_sha256='abc', code_sha256={'/old/model.py': 'def'})
    def write():
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
        return identify(tmp_path)
    a = write()
    manifest['config'].update(output='new', device='cuda', seed=43)
    manifest['code_sha256'] = {'/new/model.py': 'def'}
    b = write()
    assert a['config_id'] == b['config_id']
    assert a['model_id'] != b['model_id']
    manifest['config']['future_option'] = True
    assert write()['config_id'] != a['config_id']


def test_hubverse_roundtrip_and_bad_dates(tmp_path):
    frame = pd.DataFrame(dict(reference_date=['2025-01-04'] * 4,
        target_end_date=['2025-01-04', '2025-01-11', '2025-01-18', '2025-01-25'],
        location=['01'] * 4, horizon=range(4)))
    frame[QCOLS] = np.arange(5)
    # Supply all hubs because the submission writer creates a complete repository.
    frames = {('2024-2025', f'wk inc {d} hosp'): frame for d in ('flu', 'covid', 'rsv')}
    assert hubverse(frames, 'B0-test-s42', tmp_path, csv=True) == 3 * 4 * 5
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
    from pathlib import Path
    import shutil
    from tapestry.evaluation.hubs import export_b0, SEASONS
    from tapestry.evaluation.compare import score_with_r
    from tapestry.evaluation.sweep import main
    from tapestry.models.season_cv import LEVELS
    if not shutil.which('Rscript') or not Path('../epibench/src/epibench/build_plots.py').exists():
        pytest.skip('Integration requires local R and sibling epibench')
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'manifest.json').write_text(json.dumps(dict(config=dict(seed=42), dataset_sha256='test', code_sha256={})))
    for year, held in zip((2023, 2024, 2025), SEASONS):
        folder = run / f'eval_{held}'
        folder.mkdir()
        context = date(year, 10, 7)
        q = np.broadcast_to(np.linspace(.01, .23, 5)[:, None, None, None, None], (5, 1, 4, 6, 2))
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
    predictions[QCOLS] = np.arange(5)
    matched = match_forecasts(predictions.iloc[::-1], units, 'wk inc flu hosp')
    assert matched.location.tolist() == ['01', 'US']
    with pytest.raises(ValueError, match='Missing frozen tasks'):
        match_forecasts(predictions.iloc[:1], units, 'wk inc flu hosp')
    with pytest.raises(pd.errors.MergeError):
        match_forecasts(pd.concat([predictions, predictions.iloc[:1]]), units, 'wk inc flu hosp')


def test_staged_runner_and_sweep_share_geographic_objective(tmp_path, monkeypatch):
    """Unequal state/US task counts must not change equal-geography weighting."""
    import importlib.util
    from pathlib import Path
    from tapestry.evaluation.scoring import aggregate_scores, objective

    path = Path(__file__).resolve().parents[1] / 'experiments/b0/run_b0_experiments.py'
    spec = importlib.util.spec_from_file_location('b0_recipe', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    frozen = tmp_path / 'frozen'
    case = dict(hub='flusight', status='scored', directory='flu', target='wk inc flu hosp',
                season='2024-2025', ensemble='official', n_units=3)
    folder = frozen / 'flu'
    folder.mkdir(parents=True)
    (frozen / 'manifest.json').write_text(json.dumps(dict(cases=[case], hubs={'flusight': {'commit':'HEAD', 'truth_vintages': {'wk inc flu hosp':'2023-12-01'}}})))
    units = pd.DataFrame(dict(reference_date=['2025-01-04'] * 3,
        target_end_date=['2025-01-04'] * 3, location=['01', '02', 'US'], horizon=[0] * 3,
        observed=[1., 2., 3.]))
    units.to_parquet(folder / 'units.parquet', index=False)
    predictions = units[KEY].copy()
    predictions[QCOLS] = np.arange(5)
    ensemble = units[KEY].assign(model='official')
    ensemble[runner.METRICS] = 1.
    ensemble.to_csv(folder / 'scores.csv', index=False)
    ensemble[QCOLS] = np.arange(5)
    ensemble.to_parquet(folder / 'quantiles.parquet', index=False)
    scores = units[KEY].assign(model='flu')
    scores[runner.METRICS] = 1.
    scores['wis'] = [2., 4., 8.]
    monkeypatch.setattr(runner, 'FROZEN', frozen)
    monkeypatch.setattr(runner, 'export_b0', lambda _: {(case['season'], case['target']): predictions})
    monkeypatch.setattr(runner, 'score_case', lambda *_, **__: pd.concat([
        scores.assign(model='candidate', observed=units.observed),
        scores.assign(model='official', wis=1., observed=units.observed)], ignore_index=True).assign(ensemble_wis=1.))
    run = tmp_path / 'run'
    run.mkdir()
    summary = runner.score_run(run)
    pooled = summary[summary.horizon.eq('all')].set_index('geography')
    assert pooled.loc['states_dc', 'n'] == 2
    assert pooled.loc['US', 'n'] == 1
    assert pooled.loc['states_dc', 'wis_ratio'] == 3.
    assert pooled.loc['US', 'wis_ratio'] == 8.
    assert pooled.loc['US', 'ensemble_interval_coverage_95'] == 1.
    assert objective(summary) == pytest.approx(np.sqrt(3 * 8))
    comparable = aggregate_scores(scores.assign(target=case['target'], season=case['season'],
        ensemble_wis=1.), runner.METRICS)
    assert objective(comparable) == pytest.approx(objective(summary))
    assert (run / 'hub_scores/support.json').exists()
