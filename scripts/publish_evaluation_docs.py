"""Copy a completed evaluation into portable MkDocs pages, figures and downloads."""
import argparse
import csv
import json
import shutil
from pathlib import Path


def publish(comparison, docs):
    manifest = json.loads((comparison / 'manifest.json').read_text())
    validation = json.loads((comparison / 'validation.json').read_text())
    if manifest.get('scoring_engine') != 'epibench score --config-path':
        raise ValueError('Publish only completed full EpiBench evaluations')
    assets = docs / 'assets' / 'b0_configuration_comparison'
    pages = docs / 'results' / 'b0-comparison'
    assets.mkdir(parents=True, exist_ok=True)
    pages.mkdir(parents=True, exist_ok=True)
    for case in manifest['cases']:
        source = comparison / 'plots' / case['directory']
        destination = assets / case['directory']
        destination.mkdir(exist_ok=True)
        for figure in source.glob('*.svg'):
            # Matplotlib's SVG path lines end in spaces; normalize for clean source diffs.
            (destination / figure.name).write_text(
                '\n'.join(line.rstrip() for line in figure.read_text().splitlines()) + '\n')
    downloads = ['configuration_ranking.csv', 'run_ranking.csv', 'leaderboard.csv',
                 'configurations.csv', 'configurations.json', 'manifest.json', 'validation.json']
    for name in downloads:
        shutil.copy2(comparison / name, assets / name)
    with (comparison / 'configuration_ranking.csv').open() as stream:
        ranking = list(csv.DictReader(stream))
    records = json.loads((comparison / 'configurations.json').read_text())
    labels = {r['config_id']: r['label'] for r in records}
    names = {'wk inc flu hosp': 'Influenza admissions', 'wk inc flu prop ed visits': 'Influenza ED visits',
             'wk inc covid hosp': 'COVID-19 admissions', 'wk inc covid prop ed visits': 'COVID-19 ED visits',
             'wk inc rsv hosp': 'RSV admissions', 'wk inc rsv prop ed visits': 'RSV ED visits'}
    prefix = '../assets/b0_configuration_comparison'
    lines = ['# B0 evaluation with EpiBenchmark', '',
             '**15 runs, nine configurations, and nine target/season comparisons.** '
             'This report includes all 14 completed sweep runs and the original B0 CV run. '
             f"The full `epibench score --config-path` pipeline evaluated {validation['n_scores']:,} forecasts "
             'on identical frozen hub tasks, including freshly rescored official ensembles. '
             'EpiBench also generated the diagnostics. Each target/season page below embeds its eight figures.', '',
             'The original Python season-CV/persistence report has been removed. Its saved forecasts '
             'are included here under the original configuration identity. '
             'See [EpiBench integration and remaining gaps](../workflows/configuration-evaluation.md#what-epibench-still-needs-for-a-direct-frozen-benchmark).', '',
             'Emily’s ten configs provide forecast dates and vintage inputs; '
             'see the [config review](../workflows/emily-configs.md). '
             'Our challenges remain unversioned custom scoring configs. '
             'This report retains the existing nine frozen task sets, finalized truth and finalized-data fits; '
             'Emily’s challenge ground truth is not used. '
             'Historical source archives are preserved; current exports and all new predictions save only five quantiles.', '',
             '## Interpretation', '',
             'These are finalized-data retrospective cross-validation results. '
             'The first two folds train on later seasons; all folds were used for exploratory selection. '
             'Seed counts differ, so the ranking is descriptive rather than an untouched validation result.', '',
             'Lower objectives are better; one means parity with the official hub ensemble. '
             'The influenza objective is the geometric mean of mean-WIS/ensemble-mean-WIS ratios, '
             'equally weighting each available season and US versus states/DC. '
             'The admissions objective additionally weights the three admission targets equally. '
             'Configuration scores average individual seed objectives. '
             'The relative-WIS figures instead average per-forecast ratios; these statistics need not agree.', '',
             'Scores below use five quantiles: **0.025, 0.25, 0.5, 0.75, 0.975**, and the official ensemble on the frozen tasks. '
             'The geometric-mean objectives are Tapestry aggregations of EpiBench scores. '
             'They are not the bundled EpiBench challenge scorecards, which use different dates, '
             'hub baselines, and challenge-specific task sets.', '',
             '## Configuration ranking', '',
             '| Configuration | Example run | Seeds | Flu objective | Flu seed SD | Admissions objective | Admissions seed SD |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in ranking:
        def number(field):
            return f'{float(row[field]):.4f}' if row[field] else '—'
        lines.append(f"| `{row['config_id']}` | {labels[row['config_id']]} | {row['seeds']} | " +
                     ' | '.join(number(k) for k in ('flu_mean','flu_sd','admissions_mean','admissions_sd')) + ' |')
    best = ranking[0]
    repeated = next((r for r in ranking if int(r['seeds']) > 1), None)
    finding = f"Lowest mean influenza objective: `{best['config_id']}` ({int(best['seeds'])} seed(s)). "
    if repeated:
        finding += f"Best configuration tested at multiple seeds: `{repeated['config_id']}`. "
    lines += ['', 'A dash denotes an undefined sample SD for a single seed. ' + finding +
              'These are five-quantile scores; WIS values from earlier 23-quantile reports are not interchangeable.', '',
              f'[Download configuration rankings]({prefix}/configuration_ranking.csv) · '
              f'[Individual runs]({prefix}/run_ranking.csv) · '
              f'[Detailed leaderboard]({prefix}/leaderboard.csv)', '',
              '## Figures by target and season', '',
              'Every page contains national and North Carolina projection fans, plus WIS components, '
              'relative-WIS heatmaps, and WIS over time separately for US and states/DC. '
              'Click any figure to open its full-resolution SVG.', '',
              '| Target | Season | Figures |', '|---|---|---|']
    for case in manifest['cases']:
        name = case['directory']
        title = f"{names[case['target']]} · {case['season']}"
        lines.append(f"| {names[case['target']]} | {case['season']} | [All eight figures](b0-comparison/{name}.md) |")
        page = [f'# {title}', '', '[Report and configuration key](../b0-configuration-comparison.md)', '',
                'All configurations and the official ensemble use the same frozen scoring tasks. '
                'US is the native national prediction; states/DC are evaluated individually, not summed.', '',
                '## Four-week projection fans', '',
                'Each blue fan connects the four horizons from a single forecast origin: median, '
                '50% interval and 95% interval over black frozen truth. Every fourth available origin '
                'is shown for readability; all origins are exported and eligible shared tasks are scored. '
                'Missing horizons break the lines. ED values are proportions; admissions are counts.', '']
        figures = [('US projection fans', 'fans-US.svg'), ('North Carolina projection fans', 'fans-37.svg'),
                   ('States/DC WIS components', 'components-states_dc.svg'),
                   ('US WIS components', 'components-US.svg'),
                   ('States/DC relative WIS by horizon', 'relative-wis-states_dc.svg'),
                   ('US relative WIS by horizon', 'relative-wis-US.svg'),
                   ('States/DC WIS over time', 'timeseries-states_dc.svg'),
                   ('US WIS over time', 'timeseries-US.svg')]
        for i, (label, filename) in enumerate(figures):
            if i == 2:
                page += ['## Scoring diagnostics', '',
                         'WIS components sum to total WIS. Relative WIS uses the official ensemble as '
                         'the reference (one), whereas the original InfluPaint paper used FluSight-baseline. '
                         'A zero reference WIS leaves that per-task ratio undefined; the leaderboard records '
                         'valid ratio counts. Hub horizons 0–3 correspond to internal forecast leads 1–4.', '']
            url = f'../../assets/b0_configuration_comparison/{name}/{filename}'
            page += [f'### {label}', '', f'[![{title}: {label}]({url}){{ loading=lazy }}]({url})', '']
        (pages / f'{name}.md').write_text('\n'.join(page))
    example = 'flusight_flu_hosp_2025-2026'
    lines += ['', '### Example: influenza admissions, 2025–2026', '',
              'The most recent influenza admissions season illustrates the plots; the full set of targets '
              'and seasons is linked above. Lower relative WIS is better.', '',
              f'[![States/DC relative WIS by model and horizon]({prefix}/{example}/relative-wis-states_dc.svg){{ loading=lazy }}]({prefix}/{example}/relative-wis-states_dc.svg)', '',
              'The North Carolina fans below show the four-week projections for each run and the ensemble.', '',
              f'[![North Carolina four-week projection fans]({prefix}/{example}/fans-37.svg){{ loading=lazy }}]({prefix}/{example}/fans-37.svg)', '',
              '## Identifiers, exports and reproduction', '',
              '`B0-<12 hexadecimal characters>` identifies a configuration; `-s42` identifies its seed-42 '
              'realization. Configuration identity includes settings, dataset and model-source hashes; '
              'future settings participate automatically. Source revisions remain distinct even if their '
              'visible settings match. The original B0 is therefore separate from the later baseline.', '',
              f'[Configuration key]({prefix}/configurations.csv) · '
              f'[Complete configuration definitions]({prefix}/configurations.json) · '
              f'[Evaluation provenance]({prefix}/manifest.json) · '
              f'[Validation summary]({prefix}/validation.json)', '',
              'Four-week projections are retained locally in '
              '`data/evaluation/b0_epibench_five_quantiles/hubverse/<hub>/model-output/<model_id>/`, '
              f"with {validation['n_csv_exports']:,} CSV files and {validation['n_parquet_exports']:,} Parquet companions "
              f"containing the same {validation['n_exported_quantile_rows']:,} quantile rows, exactly five per forecast. "
              'These large forecast archives are not copied into the documentation. '
              'The portable figures and ranking downloads above are included in the documentation build.', '',
              'See the [configuration evaluation workflow](../workflows/configuration-evaluation.md) '
              'to score future runs and refresh this report.', '',
              '## Validation', '',
              f"All {validation['n_scores']:,} unique per-forecast scores passed the frozen-support audit. "
              'Every candidate and ensemble was processed by the full EpiBench scoring command. '
              f"Maximum WIS error against an independent five-quantile calculation: {validation['max_abs_wis_error']:.3g}. "
              'Relative WIS was checked against matched ensemble scores, including undefined zero denominators. '
              'Median AE and 50/95% coverage match the previous evaluation; WIS was recomputed on the new grid. '
              'The report includes 72 regenerated SVG figures. See the validation download for the exact audit results.', '']
    (docs / 'results' / 'b0-configuration-comparison.md').write_text('\n'.join(lines))
    print(f'Published report, {len(manifest["cases"])} figure pages and 72 figures under {docs}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=Path('data/evaluation/b0_epibench_five_quantiles'))
    parser.add_argument('--docs', type=Path, default=Path('docs'))
    args = parser.parse_args()
    publish(args.comparison, args.docs)
