"""Reuse B0 frozen-Hub scoring on B1's matched future tasks, separately from nowcasts."""
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.data.geography import STATE_FIPS
from tapestry.model_data.finalized import season
from tapestry.evaluation.hubs import HUBS, KEY, QCOLS
from tapestry.evaluation.scoring import match_forecasts
from tapestry.evaluation.totals import case_totals, frozen_cases, quantile_scores
from .quantiles import select_quantiles


def export_forecasts(output, metadata, config):
    """Wednesday issuance maps to the following Saturday reference; horizons 0–3."""
    prefix = f'{metadata["run_id"]}-s{metadata["seed"]}'
    postal = {v: k for k, v in STATE_FIPS.items()} | {'US': 'US'}
    frames = {}
    with np.load(output / f'forecasts-{prefix}-natural.npz', allow_pickle=False) as a:
        quantiles = select_quantiles(a['quantiles'], a['quantile_levels'])
        future = a['horizons'] >= 0
        dates = a['target_dates'][:, future]
        n, h = dates.shape
        locations = [postal[str(loc)] for loc in a['locations']]
        for spec in HUBS.values():
            for target, c in spec['targets'].items():
                frame = pd.DataFrame(dict(reference_date=np.repeat(dates[:, 0], h * len(locations)),
                    target_end_date=np.repeat(dates.reshape(-1), len(locations)),
                    location=np.tile(locations, n * h), horizon=np.tile(np.repeat(np.arange(h), len(locations)), n)))
                frame[QCOLS] = quantiles[:, :, future, c, :].reshape(len(QCOLS), -1).T
                keep = frame.target_end_date.between(config['evaluation_start'], config['evaluation_end'])
                frame = frame[keep].copy()
                for label, part in frame.groupby(frame.target_end_date.map(lambda d: season(date.fromisoformat(d)))):
                    frames[(label, target)] = part.copy()
    return frames


def score_hubs(output, frozen, metadata, config):
    """Frozen observations govern this benchmark, even if B1's reference differs."""
    from .manager import save
    frames = export_forecasts(output, metadata, config)
    model_id = f'B1-{metadata["run_id"].rsplit("-", 1)[-1]}-s{metadata["seed"]}'
    audits, totals, scores = [], [], []
    frozen = Path(frozen)
    for case in frozen_cases(frozen):
        source = frozen / case['directory']
        all_units = pd.read_parquet(source / 'units.parquet')
        prediction = frames.get((case['season'], case['target']))
        if prediction is None:
            audits.append(dict(case=case['directory'], frozen=len(all_units), matched=0))
            continue
        units = all_units.merge(prediction[KEY], on=KEY, validate='one_to_one')
        audits.append(dict(case=case['directory'], frozen=len(all_units), matched=len(units)))
        if units.empty:
            continue
        q = pd.read_parquet(source / 'quantiles.parquet')
        model = match_forecasts(prediction, units, case['target'])
        ensemble = match_forecasts(q[q.model == case['ensemble']], units, case['target'])
        folder = output / 'hub' / case['directory']
        folder.mkdir(parents=True, exist_ok=True)
        units.to_parquet(folder / 'units.parquet', index=False)
        model.assign(model=model_id).to_parquet(folder / 'model.parquet', index=False)
        ensemble.assign(model=case['ensemble']).to_parquet(folder / 'ensemble.parquet', index=False)
        totals.append(case_totals(model, ensemble, case))
        for name, frame in ((model_id, model), (case['ensemble'], ensemble)):
            metrics = quantile_scores(frame[QCOLS].to_numpy(), frame.observed.to_numpy())
            scores.append(pd.concat([frame[KEY + ['observed']].reset_index(drop=True), metrics], axis=1)
                          .assign(model=name, target=case['target'], season=case['season']))
    (output / 'hub').mkdir(exist_ok=True)
    if totals:
        pd.concat(totals, ignore_index=True).to_csv(output / 'hub/totals.csv', index=False)
    scored = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    scored.to_parquet(output / 'hub-scores.parquet', index=False)
    save(output / 'hub-support.json', dict(cases=audits, model_id=model_id, frozen=str(frozen),
         definition='Natural inputs, future horizons 0–3 only. Exact B1/frozen-task intersection. '
                    'Candidate and ensemble both scored against the frozen Hub observations; '
                    'the native six-target report uses B1 pinned reference finals separately. '
                    'Saturday reference identifies the week; B1 used Wednesday inputs, not a reconstructed Hub submission cutoff.'))
    return audits


def rank_hubs(outputs, done, destination):
    from tapestry.evaluation.totals import rank
    # Exact dates, truths and ensemble quantiles must match, not merely row counts.
    baseline = None
    for output in outputs:
        current = {}
        for file in sorted((output / 'hub').glob('*/ensemble.parquet')):
            frame = pd.read_parquet(file).sort_values(KEY).reset_index(drop=True)
            current[file.parent.name] = frame
        if baseline is not None and (current.keys() != baseline.keys() or
                any(not current[key].equals(baseline[key]) for key in current)):
            raise ValueError('B1 runs differ in exact frozen-Hub support, truth or reference forecasts')
        baseline = current
    if not baseline:
        (destination / 'hub-unavailable.txt').write_text('No matched frozen-Hub tasks in this evaluation period. Native scoring remains available.\n')
        return False
    rank([dict(config_id=r['scenario'], name=r['name'], seed=r['seed'], path=p / 'hub')
          for p, r in zip(outputs, done)], destination / 'hub-ranking')
    return True


def compare_hubs(outputs, done, destination, config, workers):
    """The same full EpiBench scorer, diagnostic plots and fans used by B0."""
    from concurrent.futures import ThreadPoolExecutor
    import importlib.util
    import matplotlib.pyplot as plt
    from tapestry.evaluation.epibench import package_source, score_case
    from tapestry.evaluation.scoring import rank
    from tapestry.evaluation.sweep import hubverse, fans, fan_selection, ranking_tables
    frozen = Path(config['frozen'])
    manifest = json.loads((frozen / 'manifest.json').read_text())
    cases = [c for c in frozen_cases(frozen) if (outputs[0] / 'hub' / c['directory']).exists()]
    if not cases:
        return
    records = []
    for output, run in zip(outputs, done):
        metadata = json.loads((output / 'manifest.json').read_text())
        audit = json.loads((output / 'hub-support.json').read_text())
        model = audit['model_id']
        records.append(dict(model=model, config_id=run['scenario'], seed=run['seed'], label=run['name']))
        hubverse(export_forecasts(output, metadata, config), model, destination, csv=True)
    run_table = pd.DataFrame(records)
    by_case, units_by_case = {}, {}
    for case in cases:
        name = case['directory']
        units_by_case[name] = pd.read_parquet(outputs[0] / 'hub' / name / 'units.parquet')
        by_case[name] = pd.concat([*[pd.read_parquet(p / 'hub' / name / 'model.parquet') for p in outputs],
                                  pd.read_parquet(outputs[0] / 'hub' / name / 'ensemble.parquet')], ignore_index=True)

    def evaluate(case):
        from datetime import datetime, timezone
        hub = manifest['hubs'][case['hub']]
        name = case['directory']
        folder = destination / 'epibench' / name
        # Keep interrupted outputs for inspection, and rerun scoring without fitting.
        if (folder / 'output/EpiBenchmark_scores.csv').exists() and not (folder / 'provenance.json').exists():
            archive = destination / 'interrupted-scoring' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            archive.mkdir(parents=True, exist_ok=True)
            folder.rename(archive / name)
        scored = score_case(by_case[name], units_by_case[name],
            dict(case, truth_release=hub['truth_vintages'][case['target']]),
            folder, commit=hub['commit'])
        return scored.assign(target=case['target'], season=case['season'])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = pd.concat(list(pool.map(evaluate, cases)), ignore_index=True)
    scores.to_parquet(destination / 'hub-epibench-scores.parquet', index=False)
    leaderboard = rank(scores)
    leaderboard.to_csv(destination / 'hub-leaderboard.csv', index=False)
    configs = run_table.groupby('config_id').seed.count().rename('seeds').to_frame()
    ranked_runs, ranked_configs = ranking_tables(leaderboard, run_table, configs)
    ranked_runs.to_csv(destination / 'hub-run-ranking.csv', index=False)
    ranked_configs.to_csv(destination / 'hub-configuration-ranking.csv')
    module = package_source() / 'build_plots.py'
    spec = importlib.util.spec_from_file_location('epibench_plots', module)
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)
    for case in cases:
        folder = destination / 'hub-plots' / case['directory']
        folder.mkdir(parents=True, exist_ok=True)
        subset = scores[(scores.target == case['target']) & (scores.season == case['season'])]
        for geography in ('US', 'states_dc'):
            part = subset[subset.location.eq('US') == (geography == 'US')].copy()
            if part.empty:
                continue
            for col in ('reference_date', 'target_end_date'):
                part[col] = pd.to_datetime(part[col])
            for name, fig in zip(('components', 'relative-wis', 'timeseries'), plotting.build_summary_figures(part)):
                fig.savefig(folder / f'{name}-{geography}.svg', bbox_inches='tight')
                plt.close(fig)
        top, best = fan_selection(leaderboard, run_table, case)
        fans(by_case[case['directory']], units_by_case[case['directory']], case, folder, ['US', '37'],
             top_models=top, season_best=best,
             model_names={r['model']: f'{r["label"]} · seed {r["seed"]}' for r in records})
