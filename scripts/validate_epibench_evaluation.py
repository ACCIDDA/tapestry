"""Audit five-quantile EpiBench scores, saved exports, and frozen evaluation support."""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tapestry.evaluation.hubs import KEY
from tapestry.evaluation.scoring import matched_scores
from tapestry.models.quantiles import LEVELS


def validate(comparison, previous):
    manifest = json.loads((comparison / 'manifest.json').read_text())
    assert manifest['scoring_engine'] == 'epibench score --config-path'
    assert manifest['quantile_levels'] == LEVELS.tolist()
    scores = pd.read_parquet(comparison / 'scores.parquet')
    keys = ['target', 'season', 'model', *KEY]
    assert not scores.duplicated(keys).any()
    models = {r['model_id'] for r in manifest['runs']}
    cases, provenance, max_error = [], [], 0.
    for case in manifest['cases']:
        units = pd.read_parquet(Path(manifest['frozen']) / case['directory'] / 'units.parquet')
        part = scores[(scores.target == case['target']) & (scores.season == case['season'])]
        assert set(part.model) == models | {case['ensemble']}
        output = comparison / 'epibench' / case['directory']
        config = yaml.safe_load((output / 'score.yaml').read_text())
        truth = pd.read_parquet(Path(config['hub_path']) / 'target-data/time-series.parquet')
        truth_check = units.merge(truth, on=['target_end_date', 'location'], validate='many_to_one')
        assert len(truth_check) == len(units)
        np.testing.assert_array_equal(truth_check.observed, truth_check.observation)
        for model, group in part.groupby('model'):
            assert len(group) == len(units)
            matched_scores(group, units)
            source = (Path(config['hub_path']) / 'model-output' / model if model == case['ensemble']
                      else Path(config['models'][model]))
            quantiles = pd.concat([pd.read_csv(p, dtype={'location': str}) for p in source.glob('*.csv')])
            assert set(quantiles.output_type_id) == set(LEVELS)
            assert set(quantiles.output_type) == {'quantile'}
            assert not quantiles.duplicated(KEY + ['output_type_id']).any()
            assert len(quantiles) == 5 * len(units)
            q = quantiles.merge(units[KEY + ['observed']], on=KEY, validate='many_to_one')
            assert len(q) == len(quantiles)
            error = q.observed - q.value
            q['pinball'] = 2 * np.maximum(q.output_type_id * error, (q.output_type_id - 1) * error)
            independent = q.groupby(KEY).pinball.mean().sort_index()
            actual = group.set_index(KEY).wis.sort_index()
            pd.testing.assert_index_equal(independent.index, actual.index)
            np.testing.assert_allclose(actual, independent, rtol=1e-10, atol=1e-10)
            max_error = max(max_error, float(np.abs(actual - independent).max()))
        expected = part.wis.div(part.ensemble_wis).where(part.ensemble_wis.ne(0))
        np.testing.assert_allclose(part.rwis, expected, equal_nan=True)
        for name in ('epibench.log', 'provenance.json', 'output/EpiBenchmark_scores.csv', 'output/summary.md'):
            assert (output / name).is_file()
        provenance.append(json.loads((output / 'provenance.json').read_text()))
        cases.append(dict(case=case['directory'], n_units=len(units), n_scores=len(part),
                          undefined_rwis=int(part.rwis.isna().sum())))
    before = pd.read_parquet(previous / 'scores.parquet').set_index(keys).sort_index()
    after = scores.set_index(keys).sort_index()
    pd.testing.assert_index_equal(before.index, after.index)
    # These use the same retained quantiles, so they must survive the grid change.
    stable = ['ae_median', 'interval_coverage_50', 'interval_coverage_95']
    np.testing.assert_allclose(before[stable].astype(float), after[stable].astype(float), rtol=1e-10, atol=1e-10)
    a = pd.read_csv(previous / 'configuration_ranking.csv').set_index('config_id')
    b = pd.read_csv(comparison / 'configuration_ranking.csv').set_index('config_id')
    changed = (a.flu_rank.sort_index() != b.flu_rank.sort_index())
    archive = comparison / 'hubverse'
    n_rows, n_files = 0, 0
    for path in archive.rglob('*.parquet'):
        grid = pd.read_parquet(path, columns=['output_type_id']).output_type_id.astype(float)
        assert set(grid) == set(LEVELS)
        n_rows += len(grid)
        n_files += 1
    assert n_rows == sum(r['hubverse_rows'] for r in manifest['runs'])
    assert all(p['quantile_levels'] == LEVELS.tolist() for p in provenance)
    assert all(p['epibench_code_sha256'] == provenance[0]['epibench_code_sha256'] for p in provenance)
    assert all(p['r_versions'] == provenance[0]['r_versions'] for p in provenance)
    figures = list((comparison / 'plots').rglob('*.svg'))
    for figure in figures:
        ET.parse(figure)
    assert len(figures) == len(manifest['cases']) * 8
    report = dict(epibench_code_sha256=provenance[0]['epibench_code_sha256'],
                  r_versions=provenance[0]['r_versions'], quantile_levels=LEVELS.tolist(),
                  scoring_commands=[p['command'] for p in provenance],
                  scoring_engine=manifest['scoring_engine'], n_scores=len(scores), n_runs=len(models),
                  cases=cases, previous=str(previous), max_abs_wis_error=max_error,
                  previous_wis_difference=float(np.abs(before.wis - after.wis).max()),
                  changed_flu_rank_configurations=changed[changed].index.tolist(),
                  n_figures=len(figures), n_exported_quantile_rows=n_rows,
                  n_parquet_exports=n_files, n_csv_exports=sum(1 for _ in archive.rglob('*.csv')),
                  checks=['Exact frozen support and finite metrics for every model',
                          'Full EpiBench outputs exist for every case',
                          'Every score matches an independent five-quantile pinball calculation',
                          'EpiBench relative WIS matches paired ensemble scores',
                          'Median AE and 50/95 coverage unchanged from the 23-quantile evaluation',
                          'Every current Parquet export contains only the five requested levels',
                          'All generated SVG figures parse successfully'])
    (comparison / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ('n_scores', 'max_abs_wis_error', 'n_exported_quantile_rows',
                                           'n_figures', 'changed_flu_rank_configurations')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=Path('data/evaluation/b0_epibench_five_quantiles'))
    parser.add_argument('--previous', type=Path, default=Path('data/evaluation/b0_epibench_comparison'))
    args = parser.parse_args()
    validate(args.comparison, args.previous)
