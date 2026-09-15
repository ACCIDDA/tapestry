"""Export saved runs, score frozen tasks, and rank/plot configurations without refitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .configurations import identify, digest
from .hubs import HUBS, KEY, QCOLS, export_b0
from .epibench import package_source, score_case
from .scoring import METRICS, validate, match_forecasts, matched_scores, rank, objective as score_objective


def hubverse(frames, model, output, csv=False):
    """Hub-native submissions; optional CSV companions support EpiBench's CLI."""
    count = 0
    for hub, spec in HUBS.items():
        chunks = []
        for (season, target), frame in frames.items():
            if target not in spec['targets']:
                continue
            validate(frame, target)
            long = frame[KEY + QCOLS].melt(id_vars=KEY, var_name='output_type_id', value_name='value')
            long['output_type_id'] = long.output_type_id.str.removeprefix('q')
            long['output_type'] = 'quantile'
            long['target'] = target
            chunks.append(long)
        folder = output / 'hubverse' / hub / 'model-output' / model
        folder.mkdir(parents=True, exist_ok=True)
        for reference, frame in pd.concat(chunks).groupby('reference_date', sort=True):
            columns = ['reference_date', 'target', 'horizon', 'target_end_date', 'location', 'output_type', 'output_type_id', 'value']
            frame[columns].to_parquet(folder / f'{reference}-{model}.parquet', index=False)
            if csv:
                frame[columns].to_csv(folder / f'{reference}-{model}.csv', index=False)
            count += len(frame)
    return count


def seed_objectives(leaderboard, runs, case=None):
    """Equal-target geometric WIS ratios for each seed, including ED targets."""
    cells = leaderboard[leaderboard.horizon.astype(str).eq('all') & leaderboard.model.isin(runs.model)].copy()
    if case is not None:
        cells = cells[cells.target.eq(case['target']) & cells.season.eq(case['season'])]
    cells['log_ratio'] = np.log(cells.wis_ratio.clip(lower=1e-12))
    return np.exp(cells.groupby(['model', 'target']).log_ratio.mean().groupby('model').mean())


def fan_ranking(leaderboard, runs, case=None):
    """Average seed objectives arithmetically; illustrate the median-scoring seed."""
    scores = seed_objectives(leaderboard, runs, case)
    seeds = runs[['model', 'config_id', 'seed', 'label']].merge(scores.rename('score'), on='model')
    rows = []
    for config, part in seeds.groupby('config_id'):
        # Break score ties by seed number; for even counts use the upper middle.
        middle = part.sort_values(['score', 'seed', 'model']).iloc[len(part) // 2]
        rows.append(dict(config_id=config, label=middle.label, mean=part.score.mean(),
                         sd=part.score.std(), seeds=len(part), model=middle.model, seed=middle.seed))
    return pd.DataFrame(rows).sort_values(['mean', 'config_id']).reset_index(drop=True)


def ranking_tables(leaderboard, runs, configs):
    """Use the same all-target objective for exports, report selection, and fans."""
    runs = runs.copy()
    runs['all_target_objective'] = runs.model.map(seed_objectives(leaderboard, runs))
    runs['all_target_rank'] = runs.all_target_objective.rank(method='min')
    overall = fan_ranking(leaderboard, runs).set_index('config_id')
    configs = configs.copy()
    for column, source in [('all_target_mean', 'mean'), ('all_target_sd', 'sd'),
                           ('middle_model', 'model'), ('middle_seed', 'seed')]:
        configs[column] = overall[source]
    configs['all_target_rank'] = configs.all_target_mean.rank(method='min')
    return (runs.sort_values(['all_target_rank', 'model']),
            configs.sort_values(['all_target_rank', 'config_id']))


def fan_selection(leaderboard, runs, case):
    overall = fan_ranking(leaderboard, runs)
    seasonal = fan_ranking(leaderboard, runs, case)
    return overall.head(3).model.tolist(), seasonal.iloc[0].model


def fans(wide, units, case, output, locations, *, top_models, season_best, model_names=None):
    # Each fan is a SINGLE origin connected across its four future weeks.
    for location in locations:
        data = wide[wide.location == location]
        models = list(dict.fromkeys([*top_models, case['ensemble'], season_best]))
        models = [model for model in models if model in set(data.model)]
        if not models:
            continue
        fig, axes = plt.subplots(len(models), 1, figsize=(14, 2.2 * len(models)), sharex=True, sharey=True, squeeze=False)
        truth = units[units.location == location][['target_end_date', 'observed']].drop_duplicates().sort_values('target_end_date')
        for ax, model in zip(axes[:, 0], models):
            color = '#ef9a9a' if model == season_best else '#90caf9' if model == case['ensemble'] else '#3879a8'
            role = 'season best' if model == season_best else 'ensemble' if model == case['ensemble'] else 'overall top 3'
            ax.plot(pd.to_datetime(truth.target_end_date), truth.observed, color='black', lw=1, label='Frozen truth')
            part = data[data.model == model]
            # Prespecified every fourth reference week keeps overlapping fans legible.
            refs = sorted(part.reference_date.unique())[::4]
            for i, ref in enumerate(refs):
                f = part[part.reference_date == ref].sort_values('horizon')
                # Reindex missing horizons so lines cannot jump across absent weeks.
                f = f.set_index('horizon').reindex(range(4))
                x = pd.date_range(ref, periods=4, freq='7D')
                ax.fill_between(x, f['q0.025'], f['q0.975'], color=color, alpha=.2, label='95%' if i == 0 else None)
                ax.fill_between(x, f['q0.25'], f['q0.75'], color=color, alpha=.45, label='50%' if i == 0 else None)
                ax.plot(x, f['q0.5'], color=color, lw=1)
            ax.set_ylabel(f'{(model_names or {}).get(model, model)}\n{role}', fontsize=9)
            ax.set_ylim(bottom=0)
        axes[0, 0].legend(loc='upper right', ncol=3)
        axes[0, 0].set_title(f"{case['target']} · {case['season']} · {location} · four-week projection fans")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output / f'fans-{location}.svg', bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, type=Path)
    parser.add_argument('--frozen', type=Path, default=Path('data/evaluation/b0_hub_comparison'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv', action='store_true', help='Also save uncompressed Hubverse CSV files for the EpiBench CLI')
    parser.add_argument('--epibench', type=Path, help='Optional development checkout; defaults to installed EpiBenchmark')
    parser.add_argument('--score-workers', type=int, default=2,
                        help='Concurrent run loads and EpiBench cases; the default two suits a 32 GiB machine')
    parser.add_argument('--mirrors', type=Path, default=Path('data/mirrors'))
    parser.add_argument('--locations', nargs='+', default=['US', '37'], help='Hub FIPS codes; default US and NC')
    args = parser.parse_args()
    if args.score_workers < 1:
        parser.error('--score-workers must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    records = [identify(run, args.output) for run in args.runs]
    if len({r['model_id'] for r in records}) != len(records):
        raise ValueError('Duplicate model identities: pass one copy of each configuration/seed')
    registry = args.output / 'configurations.json'
    if registry.exists():
        previous = json.loads(registry.read_text())
        if {r['model_id'] for r in previous} != {r['model_id'] for r in records}:
            raise ValueError('Use a new output directory when changing the set of compared runs')
    (args.output / 'configurations.json').write_text(json.dumps(records, indent=2) + '\n')
    pd.DataFrame([{**{k: v for k, v in r.items() if k not in ('scenario', 'provenance')}, **r['scenario']}
                  for r in records]).to_csv(args.output / 'configurations.csv', index=False)
    frozen = json.loads((args.frozen / 'manifest.json').read_text())
    cases = [c for c in frozen['cases'] if c['status'] == 'scored']
    by_case = {c['directory']: [] for c in cases}
    all_scores = []
    print(f'Loading {len(args.runs)} runs with {args.score_workers} workers', flush=True)
    with ThreadPoolExecutor(max_workers=args.score_workers) as pool:
        exports = list(pool.map(export_b0, args.runs))
    for record, frames in zip(records, exports):
        model = record['model_id']
        print(f'Exporting {model}', flush=True)
        record['hubverse_rows'] = hubverse(frames, model, args.output, csv=args.csv)
        for case in cases:
            units = pd.read_parquet(args.frozen / case['directory'] / 'units.parquet')
            wide = match_forecasts(frames[(case['season'], case['target'])], units, case['target'])
            by_case[case['directory']].append(wide.assign(model=model))
    # Two independent subprocesses keep memory bounded on the 32 GiB research machine.
    # EpiBench freshly scores the ensemble together with every candidate.
    def evaluate_case(case):
        print(f"EpiBench scoring {case['directory']}", flush=True)
        source = args.frozen / case['directory']
        units = pd.read_parquet(source / 'units.parquet')
        q = pd.read_parquet(source / 'quantiles.parquet')
        by_case[case['directory']].append(q.loc[q.model == case['ensemble'], ['model', *KEY, 'observed', *QCOLS]])
        hub_info = frozen['hubs'][case['hub']]
        scoring_case = dict(case, truth_release=hub_info['truth_vintages'][case['target']])
        scored = score_case(pd.concat(by_case[case['directory']], ignore_index=True), units,
                            scoring_case, args.output / 'epibench' / case['directory'],
                            epibench=args.epibench, mirrors=args.mirrors, commit=hub_info['commit'])
        print(f"EpiBench complete {case['directory']} ({len(scored):,} scores)", flush=True)
        return scored.assign(target=case['target'], season=case['season'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for case in cases:
        # A killed EpiBench case can leave scores without provenance; set it aside and rescore.
        folder = args.output / 'epibench' / case['directory']
        if folder.is_dir() and not ((folder / 'provenance.json').is_file() and
                                    (folder / 'output/EpiBenchmark_scores.csv').is_file()):
            archive = args.output / 'interrupted-scoring' / stamp
            archive.mkdir(parents=True, exist_ok=True)
            folder.rename(archive / case['directory'])
    with ThreadPoolExecutor(max_workers=args.score_workers) as pool:
        futures = [pool.submit(evaluate_case, case) for case in cases]
        for future in as_completed(futures):
            all_scores.append(future.result())
    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_parquet(args.output / 'scores.parquet', index=False)
    leaderboard = rank(scores)
    leaderboard.to_csv(args.output / 'leaderboard.csv', index=False)
    # Retain admission-focused secondary diagnostics; all-target scores set the ranking below.
    selected = leaderboard[leaderboard.horizon.eq('all') & leaderboard.target.str.endswith('hosp') & leaderboard.model.isin([r['model_id'] for r in records])]
    objectives = []
    for model, part in selected.groupby('model'):
        objectives.append(dict(model=model,
            flu_objective=score_objective(part), admissions_objective=score_objective(part, multi=True)))
    objective = pd.DataFrame(objectives).merge(pd.DataFrame(records)[['model_id', 'config_id', 'seed', 'label']], left_on='model', right_on='model_id')
    objective['flu_rank'] = objective.flu_objective.rank(method='min')
    objective['admissions_rank'] = objective.admissions_objective.rank(method='min')
    configs = objective.groupby('config_id').agg(seeds=('seed','count'), flu_mean=('flu_objective','mean'), flu_sd=('flu_objective','std'), admissions_mean=('admissions_objective','mean'), admissions_sd=('admissions_objective','std'))
    configs['flu_rank'] = configs.flu_mean.rank(method='min')
    configs['admissions_rank'] = configs.admissions_mean.rank(method='min')
    objective, configs = ranking_tables(leaderboard, objective, configs)
    objective.to_csv(args.output / 'run_ranking.csv', index=False)
    configs.to_csv(args.output / 'configuration_ranking.csv')
    module_path = package_source(args.epibench) / 'build_plots.py'
    spec = importlib.util.spec_from_file_location('epibench_plots', module_path)
    plots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plots)
    for case in cases:
        print(f'Plotting {case["directory"]}', flush=True)
        folder = args.output / 'plots' / case['directory']
        folder.mkdir(parents=True, exist_ok=True)
        s = scores[(scores.target == case['target']) & (scores.season == case['season'])].copy()
        for geography in ('US', 'states_dc'):
            g = s[s.location.eq('US') if geography == 'US' else s.location.ne('US')].copy()
            g.to_csv(folder / f'epibench-scores-{geography}.csv', index=False)
            for col in ('reference_date', 'target_end_date'):
                g[col] = pd.to_datetime(g[col])
            for name, fig in zip(('components', 'relative-wis', 'timeseries'), plots.build_summary_figures(g)):
                fig.suptitle(f"{case['target']} · {case['season']} · {geography}\nRelative WIS reference: official ensemble", fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, .94))
                fig.savefig(folder / f'{name}-{geography}.svg', bbox_inches='tight')
                plt.close(fig)
        units = pd.read_parquet(args.frozen / case['directory'] / 'units.parquet')
        top_models, season_best = fan_selection(leaderboard, objective, case)
        fans(pd.concat(by_case[case['directory']], ignore_index=True), units, case, folder, args.locations,
             top_models=top_models, season_best=season_best,
             model_names={r['model_id']: f"{r['label']} · seed {r['seed']}" for r in records})
    manifest = dict(quantile_levels=[float(q[1:]) for q in QCOLS], scoring_engine='epibench score --config-path', runs=records, frozen=str(args.frozen), frozen_manifest_sha256=digest(frozen),
                    epibench_plot_sha256=digest(module_path.read_text()), cases=cases,
                    evaluation_code_sha256={p.name: digest(p.read_text()) for p in Path(__file__).parent.glob('*') if p.suffix in {'.py', '.R'}},
                    assumptions=['Finalized retrospective CV; exploratory ranking, not prospective validation.',
                                 'Fixed ensemble-supported units; all configurations must cover every unit.',
                                 'Hub horizons 0–3 equal internal leads 1–4; ED stays in proportions.',
                                 'Relative WIS uses official ensemble; mean per-unit ratios differs from ratio of means.',
                                 'Seed counts differ; configuration means and sample SD are descriptive.',
                                 'Fans show every fourth origin; all origins exported and scored.'])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_report(args.output, records, cases, objective, configs)
    print(f'Complete: {args.output}', flush=True)


def write_report(output, records, cases, objective, configs):
    """A navigable artifact index with the assumptions alongside the rankings."""
    def link(label, path):
        return f'[{label}]({Path(path).as_posix()})'
    lines = ['# B0 configuration evaluation', '',
             f'{len(records)} saved runs; {len(configs)} configurations; {len(cases)} target/season comparisons.', '',
             'All runs use the same frozen ensemble-supported forecast tasks and truth. '
             'Scores come from the full EpiBench config pipeline (including R scoringutils and relative WIS); '
             'the diagnostic plots call EpiBench plotting code. '
             'These are finalized retrospective CV results and exploratory selections, not prospective rankings.', '',
             'Configurations rank by the arithmetic mean of all-target seed objectives, including ED. '
             'Each seed objective is the geometric WIS ratio with equal target weight, then equal season/geography weight. '
             'Fans use the middle-performing seed. Lower is better. The secondary influenza objective is the geometric mean of WIS/ensemble-WIS '
             'ratios, equally weighting each season and US versus states/DC. The admissions objective '
             'first gives each admission target equal weight. Configuration results average the run objectives; '
             'different seed counts and selection on these folds limit comparisons. '
             'The plotted relative WIS is the mean of per-task ratios, a different statistic.', '',
             '| Configuration | Variant | Seeds | All-target mean ± SD | Middle seed | Flu objective | Admissions objective |',
             '|---|---|---:|---:|---:|---:|---:|']
    for config_id, row in configs.sort_values('all_target_rank').iterrows():
        label = next(r['label'] for r in records if r['config_id'] == config_id)
        lines.append(f'| {config_id} | {label} | {int(row.seeds)} | {row.all_target_mean:.4f} ± {row.all_target_sd:.4f} | {int(row.middle_seed)} | {row.flu_mean:.4f} | {row.admissions_mean:.4f} |')
    lines += ['', link('Configuration ranking (means and seed SD)', Path('configuration_ranking.csv')),
              '', link('Individual run ranking and identifiers', Path('run_ranking.csv')),
              '', link('Detailed target/season/geography/horizon leaderboard', Path('leaderboard.csv')),
              '', link('Configuration definitions and provenance', Path('configurations.json')),
              '', 'Hubverse forecasts are stored under `hubverse/<hub>/model-output/<model_id>/`, '
              'one Parquet file per reference date with the saved quantile grid and horizons 0–3. '
              'Use `--csv` for CSV companions accepted directly by the EpiBench CLI. '
              'ED values are proportions. Every available origin is exported; fans display every fourth origin '
              'for readability, with median, 50% and 95% intervals. NC uses FIPS 37.', '',
              '| Target / season | US fans | NC fans | State WIS components | State relative WIS | State WIS over time |',
              '|---|---|---|---|---|---|']
    for case in cases:
        base = Path('plots') / case['directory']
        labels = [('US', 'fans-US.svg'), ('NC', 'fans-37.svg'), ('Components', 'components-states_dc.svg'),
                  ('Relative WIS', 'relative-wis-states_dc.svg'), ('Time series', 'timeseries-states_dc.svg')]
        links = [link(label, base / filename) if (output / base / filename).exists() else 'Not plotted' for label, filename in labels]
        lines.append(f'| {case["target"]} / {case["season"]} | ' + ' | '.join(links) + ' |')
    lines += ['', 'Each plot directory also contains US diagnostics and EpiBench-compatible score CSVs. '
              'Zero ensemble WIS yields undefined relative WIS, with the valid ratio count retained in the leaderboard.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
