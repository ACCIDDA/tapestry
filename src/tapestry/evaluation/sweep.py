"""Export saved runs, score frozen tasks, and rank/plot configurations without refitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def fans(wide, units, case, output, locations):
    # Each fan is a SINGLE origin connected across its four future weeks.
    for location in locations:
        data = wide[wide.location == location]
        models = sorted(data.model.unique())
        if not models:
            continue
        fig, axes = plt.subplots(len(models), 1, figsize=(14, 2.2 * len(models)), sharex=True, sharey=True, squeeze=False)
        truth = units[units.location == location][['target_end_date', 'observed']].drop_duplicates().sort_values('target_end_date')
        for ax, model in zip(axes[:, 0], models):
            ax.plot(pd.to_datetime(truth.target_end_date), truth.observed, color='black', lw=1, label='Frozen truth')
            part = data[data.model == model]
            # Prespecified every fourth reference week keeps overlapping fans legible.
            refs = sorted(part.reference_date.unique())[::4]
            for i, ref in enumerate(refs):
                f = part[part.reference_date == ref].sort_values('horizon')
                # Reindex missing horizons so lines cannot jump across absent weeks.
                f = f.set_index('horizon').reindex(range(4))
                x = pd.date_range(ref, periods=4, freq='7D')
                ax.fill_between(x, f['q0.025'], f['q0.975'], color='#3879a8', alpha=.13, label='95%' if i == 0 else None)
                ax.fill_between(x, f['q0.25'], f['q0.75'], color='#3879a8', alpha=.3, label='50%' if i == 0 else None)
                ax.plot(x, f['q0.5'], color='#20638f', lw=1)
            ax.set_ylabel(model, fontsize=7)
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
    parser.add_argument('--family', default='B0')
    parser.add_argument('--csv', action='store_true', help='Also save uncompressed Hubverse CSV files for the EpiBench CLI')
    parser.add_argument('--epibench', type=Path, help='Optional development checkout; defaults to installed EpiBenchmark')
    parser.add_argument('--score-workers', type=int, default=2, choices=(1, 2), help='Concurrent independent EpiBench cases')
    parser.add_argument('--mirrors', type=Path, default=Path('data/mirrors'))
    parser.add_argument('--locations', nargs='+', default=['US', '37'], help='Hub FIPS codes; default US and NC')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [identify(run, args.family) for run in args.runs]
    if len({r['model_id'] for r in records}) != len(records):
        raise ValueError('Duplicate model identities: pass one copy of each configuration/seed')
    registry = args.output / 'configurations.json'
    if registry.exists():
        previous = json.loads(registry.read_text())
        if {r['model_id'] for r in previous} != {r['model_id'] for r in records}:
            raise ValueError('Use a new output directory when changing the set of compared runs')
    (args.output / 'configurations.json').write_text(json.dumps(records, indent=2) + '\n')
    pd.DataFrame([{k:v for k,v in r.items() if k != 'identity'} for r in records]).to_csv(args.output / 'configurations.csv', index=False)
    frozen = json.loads((args.frozen / 'manifest.json').read_text())
    cases = [c for c in frozen['cases'] if c['status'] == 'scored']
    by_case = {c['directory']: [] for c in cases}
    all_scores = []
    for record, run in zip(records, args.runs):
        model = record['model_id']
        print(f'Exporting and scoring {run.name} as {model}', flush=True)
        frames = export_b0(run)
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
    with ThreadPoolExecutor(max_workers=args.score_workers) as pool:
        futures = [pool.submit(evaluate_case, case) for case in cases]
        for future in as_completed(futures):
            all_scores.append(future.result())
    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_parquet(args.output / 'scores.parquet', index=False)
    leaderboard = rank(scores)
    leaderboard.to_csv(args.output / 'leaderboard.csv', index=False)
    # Same exploratory objective as the existing sweep. No raw-score pooling across units.
    selected = leaderboard[leaderboard.horizon.eq('all') & leaderboard.target.str.endswith('hosp') & leaderboard.model.isin([r['model_id'] for r in records])]
    objectives = []
    for model, part in selected.groupby('model'):
        objectives.append(dict(model=model,
            flu_objective=score_objective(part), admissions_objective=score_objective(part, multi=True)))
    objective = pd.DataFrame(objectives).merge(pd.DataFrame(records)[['model_id', 'config_id', 'seed', 'label']], left_on='model', right_on='model_id')
    objective['flu_rank'] = objective.flu_objective.rank(method='min')
    objective['admissions_rank'] = objective.admissions_objective.rank(method='min')
    objective.sort_values('flu_rank').to_csv(args.output / 'run_ranking.csv', index=False)
    configs = objective.groupby('config_id').agg(seeds=('seed','count'), flu_mean=('flu_objective','mean'), flu_sd=('flu_objective','std'), admissions_mean=('admissions_objective','mean'), admissions_sd=('admissions_objective','std'))
    configs['flu_rank'] = configs.flu_mean.rank(method='min')
    configs['admissions_rank'] = configs.admissions_mean.rank(method='min')
    configs.sort_values('flu_rank').to_csv(args.output / 'configuration_ranking.csv')
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
        fans(pd.concat(by_case[case['directory']], ignore_index=True), units, case, folder, args.locations)
    manifest = dict(quantile_levels=[float(q[1:]) for q in QCOLS], scoring_engine='epibench score --config-path', runs=records, frozen=str(args.frozen.resolve()), frozen_manifest_sha256=digest(frozen),
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
        return f'[{label}]({(output / path).resolve()})'
    lines = ['# B0 configuration evaluation', '',
             f'{len(records)} saved runs; {len(configs)} configurations; {len(cases)} target/season comparisons.', '',
             'All runs use the same frozen ensemble-supported forecast tasks and truth. '
             'Scores come from the full EpiBench config pipeline (including R scoringutils and relative WIS); '
             'the diagnostic plots call EpiBench plotting code. '
             'These are finalized retrospective CV results and exploratory selections, not prospective rankings.', '',
             'Lower is better. The influenza objective is the geometric mean of WIS/ensemble-WIS '
             'ratios, equally weighting each season and US versus states/DC. The admissions objective '
             'first gives each admission target equal weight. Configuration results average the run objectives; '
             'different seed counts and selection on these folds limit comparisons. '
             'The plotted relative WIS is the mean of per-task ratios, a different statistic.', '',
             '| Configuration | Example run | Seeds | Flu objective | Admissions objective |',
             '|---|---|---:|---:|---:|']
    for config_id, row in configs.sort_values('flu_rank').iterrows():
        label = next(r['label'] for r in records if r['config_id'] == config_id)
        lines.append(f'| {config_id} | {label} | {int(row.seeds)} | {row.flu_mean:.4f} | {row.admissions_mean:.4f} |')
    lines += ['', link('Configuration ranking (means and seed SD)', Path('configuration_ranking.csv')),
              '', link('Individual run ranking and identifiers', Path('run_ranking.csv')),
              '', link('Detailed target/season/geography/horizon leaderboard', Path('leaderboard.csv')),
              '', link('Configuration definitions and provenance', Path('configurations.json')),
              '', 'Hubverse forecasts are stored under `hubverse/<hub>/model-output/<model_id>/`, '
              'one Parquet file per reference date with five quantiles and horizons 0–3. '
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
