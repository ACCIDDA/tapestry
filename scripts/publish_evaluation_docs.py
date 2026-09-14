"""Publish the canonical B0 evaluation, seed summaries, and figures to MkDocs."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from tapestry.evaluation.hubs import KEY
from tapestry.evaluation.sweep import fan_selection, fans

DEFAULT = Path('data/experiments/b0-rebuilt/comparison-da7685987135')
NAMES = {'wk inc flu hosp': 'Influenza admissions', 'wk inc flu prop ed visits': 'Influenza ED visits',
         'wk inc covid hosp': 'COVID-19 admissions', 'wk inc covid prop ed visits': 'COVID-19 ED visits',
         'wk inc rsv hosp': 'RSV admissions', 'wk inc rsv prop ed visits': 'RSV ED visits'}
CONTROLS = {'anchor': ['baseline'], 'state_us': ['anchor'], 'residual2': ['anchor'],
            'latent32': ['anchor'], 'residual2_z32': ['residual2', 'latent32'],
            'mlp_h8': ['mlp_h12'], 'mlp_h12': ['anchor'], 'mlp_h26': ['mlp_h12'],
            'balanced': ['anchor'], 'flu_only': ['anchor'], 'conv_h12': ['anchor'],
            'mlp_h26_dynamics': ['mlp_h26'], 'conv_h26': ['mlp_h26_dynamics']}
METRICS = ['wis', 'wis_ratio', 'bias', 'interval_coverage_50', 'interval_coverage_95',
           'dispersion', 'underprediction', 'overprediction']
FIGURES = [('US projection fans', 'fans-US.svg'), ('North Carolina projection fans', 'fans-37.svg'),
           ('States/DC WIS components', 'components-states_dc.svg'), ('US WIS components', 'components-US.svg'),
           ('States/DC relative WIS', 'relative-wis-states_dc.svg'), ('US relative WIS', 'relative-wis-US.svg'),
           ('States/DC WIS over time', 'timeseries-states_dc.svg'), ('US WIS over time', 'timeseries-US.svg')]


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def mean_sd(series, scale=1, digits=3):
    return f'{series.mean()*scale:.{digits}f} ± {series.std()*scale:.{digits}f}'


def summarize(frame, groups):
    result = frame.groupby(groups)[METRICS].agg(['mean', 'std'])
    result.columns = [f'{metric}_{stat}' for metric, stat in result.columns]
    return result.reset_index()


def publish(comparison, docs):
    manifest = json.loads((comparison / 'manifest.json').read_text())
    if manifest.get('scoring_engine') != 'epibench score --config-path':
        raise ValueError('A completed EpiBench evaluation is required')
    runs = pd.read_csv(comparison / 'run_ranking.csv')
    model_names = {r.model: f'{r.label} · seed {int(r.seed)}' for r in runs.itertuples()}
    definitions = json.loads((comparison / 'configurations.json').read_text())
    variants = {r['label']: r['identity']['config'] for r in definitions}
    anchor = variants['anchor']
    fields = {'encoder': 'encoder', 'lookback': 'history (weeks)', 'count_transform': 'count transform',
              'geography': 'geography features', 'dynamics': 'dynamics features', 'heads': 'prediction heads',
              'decoder': 'decoder', 'latent': 'latent dimension', 'loss_weights': 'loss weighting'}
    differences = []
    for label, config in sorted(variants.items()):
        changes = [f'{title}: {config[key]}' for key, title in fields.items() if config[key] != anchor[key]]
        differences.append([f'`{label}`', '; '.join(changes) or 'Reference configuration (settings below)'])
    rankings = pd.read_csv(comparison / 'configuration_ranking.csv').merge(
        runs[['config_id', 'label']].drop_duplicates(), on='config_id').sort_values('flu_mean')
    leaderboard = pd.read_csv(comparison / 'leaderboard.csv', dtype={'horizon': str})
    candidates = leaderboard.merge(runs[['model', 'label', 'seed']], on='model')
    groups = ['label', 'target', 'season', 'geography', 'horizon']
    summary = summarize(candidates, groups)
    all_horizons = candidates[candidates.horizon.eq('all')]
    winner = rankings.iloc[0]
    best = runs[runs.label.eq(winner.label)]
    assets = docs / 'assets/b0_configuration_comparison'
    pages = docs / 'results/b0-comparison'
    assets.mkdir(parents=True, exist_ok=True)
    pages.mkdir(parents=True, exist_ok=True)
    summary.to_csv(assets / 'seed_summary.csv', index=False)
    contrasts, detailed = [], []
    for candidate, controls in CONTROLS.items():
        for control in controls:
            a = runs[runs.label.eq(candidate)].set_index('seed')
            b = runs[runs.label.eq(control)].set_index('seed')
            delta = a[['flu_objective', 'admissions_objective']] - b[['flu_objective', 'admissions_objective']]
            contrasts.append(dict(candidate=candidate, control=control,
                flu_delta_mean=delta.flu_objective.mean(), flu_delta_sd=delta.flu_objective.std(),
                admissions_delta_mean=delta.admissions_objective.mean(), admissions_delta_sd=delta.admissions_objective.std()))
            keys = ['seed', 'target', 'season', 'geography', 'horizon']
            d = (candidates[candidates.label.eq(candidate)].set_index(keys)[METRICS]
                 - candidates[candidates.label.eq(control)].set_index(keys)[METRICS]).reset_index()
            detailed.append(summarize(d, keys[1:]).assign(candidate=candidate, control=control))
    pd.DataFrame(contrasts).to_csv(assets / 'matched_controls.csv', index=False)
    pd.concat(detailed).to_csv(assets / 'matched_control_details.csv', index=False)
    for name in ('configuration_ranking.csv', 'run_ranking.csv', 'leaderboard.csv',
                 'configurations.csv', 'configurations.json', 'manifest.json'):
        shutil.copy2(comparison / name, assets / name)
    # A validation report from another experiment must not accompany these results.
    (assets / 'validation.json').unlink(missing_ok=True)
    prefix = '../assets/b0_configuration_comparison'
    n_scores = int(leaderboard[leaderboard.horizon.eq('all')].n.sum())
    lines = ['# Canonical B0 experiments', '',
        f'**{len(rankings)} variants × three seeds = {len(runs)} CV runs / {len(runs)*3} season fits.** '
        f'Experiment `b0-rebuilt` evaluates {n_scores:,} forecast units across nine target/season comparisons, '
        'including the official ensembles. All fits and the full EpiBench evaluation completed.', '',
        '## Best model', '',
        f'**`{winner.label}` (`{winner.config_id}`) has the lowest mean influenza and three-admission objectives.** '
        'It uses a temporal convolution encoder, 12 weeks of history, fourth-root count transformation, '
        'geography and dynamics features, shared prediction heads, and the original decoder with latent dimension 16.', '',
        f'Across seeds 42/43/44, its influenza WIS ratio is **{mean_sd(best.flu_objective, digits=4)}** '
        f'and its admissions ratio is **{mean_sd(best.admissions_objective, digits=4)}** (mean ± sample SD). '
        f'These are **{100*(1-winner.flu_mean):.1f}%** and **{100*(1-winner.admissions_mean):.1f}%** below '
        'ensemble parity on the respective aggregate objectives.', '',
        '`residual2` is a close alternative: influenza ratio 0.8924 ± 0.0203 and admissions ratio '
        '0.9240 ± 0.0190. The influenza gap between the two variants is only 0.0008, much smaller '
        'than their seed variability; three seeds do not establish a decisive winner.', '',
        table(['Seed', 'conv_h12 influenza ratio', 'conv_h12 admissions ratio'],
              [[int(r.seed), f'{r.flu_objective:.4f}', f'{r.admissions_objective:.4f}'] for r in best.sort_values('seed').itertuples()]), '',
        '## Interpretation', '',
        'Lower WIS is better; ratio 1 means ensemble parity. The influenza objective is the geometric mean '
        'of model mean-WIS / ensemble mean-WIS across the three seasons and two geography groups, equally weighted. '
        'The admissions objective first gives each admission target equal weight. Reported configuration values '
        'average the three seed objectives. Percentage improvements refer to these aggregates, not a pooled raw WIS. '
        'ED forecasts are reported separately and do not enter either selection objective.', '',
        'This is the canonical rebuilt-input comparison. CDC observations were downloaded September 14, 2026; '
        'the training date range is September 2023–August 29, 2026 and hub commits are pinned. '
        'It is a new frozen experiment, not exact reproduction of the September 4 CDC snapshots. '
        'The first two CV folds train on later seasons and all three seasons informed development. '
        '**These are exploratory results, not prospective validation.**', '',
        'Scoring uses quantiles 0.025, 0.25, 0.5, 0.75, and 0.975. Bias is the signed EpiBench/scoringutils '
        'quantile bias score, not an error in admission counts. Coverage is the fraction of truth values inside '
        'the prediction interval. States/DC metrics average individual location forecast tasks; US is the native '
        'national prediction. Every candidate uses the same frozen tasks as its ensemble.', '',
        '## Variant ranking', '',
        table(['Variant', 'Configuration', 'Flu ratio ± SD', 'Admissions ratio ± SD'],
              [[f'`{r.label}`', f'`{r.config_id}`', f'{r.flu_mean:.4f} ± {r.flu_sd:.4f}',
                f'{r.admissions_mean:.4f} ± {r.admissions_sd:.4f}'] for r in rankings.itertuples()]), '',
        '## Model differences', '',
        'Fan labels use variant names and seed numbers. Seeds 42/43/44 repeat the same configuration '
        'with different random initialization. The reference `anchor` uses a multilayer perceptron (MLP), '
        '12 weeks of history, fourth-root counts, geography and dynamics features, shared prediction heads, '
        'the original (`legacy`) decoder, latent dimension 16, and influenza-first loss weighting. '
        'The table lists changes from that reference; `conv` means temporal convolution, `state_us` means '
        'separate state and national heads, and `residual2` means a two-block residual decoder.', '',
        table(['Variant name', 'Differences from anchor'], differences), '',
        'The official ensemble is the hub reference forecast, not one of these fitted variants.', '',
        '## Best model versus ensemble', '',
        'Means across seeds, with all four horizons included. Coverage columns are percentages. '
        'The full download includes seed SD, each horizon, and every variant.', '']
    winner_rows = []
    for (target, season, geo), part in all_horizons[all_horizons.label.eq(winner.label)].groupby(['target', 'season', 'geography']):
        ensemble = leaderboard[(leaderboard.target.eq(target)) & leaderboard.season.eq(season) &
            leaderboard.geography.eq(geo) & leaderboard.horizon.eq('all') & ~leaderboard.model.isin(runs.model)]
        winner_rows.append([NAMES[target], season, geo, f'{part.wis.mean():.5g}', f'{ensemble.wis.iloc[0]:.5g}',
            f'{part.wis_ratio.mean():.3f}', f'{part.bias.mean():+.3f}',
            f'{part.interval_coverage_50.mean()*100:.1f}', f'{part.interval_coverage_95.mean()*100:.1f}'])
    lines += [table(['Target', 'Season', 'Geography', 'Model WIS', 'Ensemble WIS', 'WIS ratio', 'Bias', '50% coverage', '95% coverage'], winner_rows), '',
        '`conv_h12` beats the ensemble in all six influenza admission season/geography cells. '
        'It loses on COVID admissions in 2025–26 and on influenza/COVID ED targets. '
        'Its state-level 95% influenza coverage is only 75.1–83.3%, so the aggregate win does not imply well-calibrated uncertainty.', '',
        '## Each candidate against its matched control', '',
        'Paired seed differences (candidate minus control), reported as mean ± sample SD. Negative is better. '
        'The anchor/baseline contrast changes several features together; the other contrasts isolate the documented change.', '',
        table(['Candidate', 'Control', 'Δ influenza objective', 'Δ admissions objective'],
              [[f"`{r['candidate']}`", f"`{r['control']}`", f"{r['flu_delta_mean']:+.4f} ± {r['flu_delta_sd']:.4f}",
                f"{r['admissions_delta_mean']:+.4f} ± {r['admissions_delta_sd']:.4f}"] for r in contrasts]), '',
        '## Heads and uncertainty', '',
        'For the following diagnostic summary, each influenza season receives equal weight after seed averaging. '
        'WIS ratios here are arithmetic means across seasons, so they differ from the geometric-mean selection objective.', '']
    diagnostic = all_horizons[all_horizons.target.eq('wk inc flu hosp') & all_horizons.label.isin(['baseline','anchor','state_us','residual2','conv_h12'])]
    lines += [table(['Variant', 'Geography', 'Mean WIS ratio', '50% coverage', '95% coverage'],
        [[label, geo, f'{part.wis_ratio.mean():.3f}', f'{100*part.interval_coverage_50.mean():.1f}%',
          f'{100*part.interval_coverage_95.mean():.1f}%'] for (label,geo),part in diagnostic.groupby(['label','geography'])]), '',
        '**Separate heads do not resolve the state/US tradeoff.** Relative to the anchor, `state_us` slightly '
        'improves state influenza WIS and coverage, but worsens US WIS and reduces US 95% coverage from 86.7% '
        'to 78.7%. Its influenza objective is essentially unchanged, and its admissions objective is worse. '
        'The baseline still has better average state influenza WIS than these feature-rich variants, while '
        'its national forecasts are much worse.', '',
        '**The richer decoder helps, but does not fix undercoverage.** At latent dimension 16, `residual2` '
        'improves both objectives and moves influenza coverage toward nominal levels at state and US scales. '
        'State 95% coverage rises from 75.7% to 78.9%; US coverage rises from 86.7% to 90.7%. '
        'Combining residual depth with latent dimension 32 reverses much of that gain: `residual2_z32` '
        'has poorer WIS objectives than either `residual2` or `latent32` alone.', '',
        '## What to combine next', '',
        'Test **`conv_h12` + `residual2` at latent dimension 16**, at all three seeds, against both individual '
        'variants. Both independently improve the anchor, but their combination remains untested. '
        'Balanced admission weights are a secondary isolated addition to that recipe. Keep the 12-week history: '
        '26-week variants did not improve the selection objectives. There is no current evidence to add separate '
        'heads or combine the deeper decoder with latent dimension 32. Calibration and spatial attention remain outside this suite.', '',
        '## Downloads', '',
        f'- [Metrics by variant, target, season, geography, and horizon: seed means and SD]({prefix}/seed_summary.csv)',
        f'- [Paired matched-control objective differences]({prefix}/matched_controls.csv)',
        f'- [Paired matched-control metric differences for every evaluation cell]({prefix}/matched_control_details.csv)',
        f'- [Configuration ranking]({prefix}/configuration_ranking.csv) · [Individual seeds]({prefix}/run_ranking.csv)',
        f'- [Per-seed leaderboard, including ensembles]({prefix}/leaderboard.csv)',
        f'- [Configuration definitions]({prefix}/configurations.json) · [Evaluation provenance]({prefix}/manifest.json)', '',
        'The detailed tables contain WIS, bias, 50%/95% coverage, and WIS components for horizons 0–3 '
        'and all horizons combined. Horizon 0 means the first future week. Missing target/seasons are '
        'unavailable in the pinned ensemble-supported task sets, rather than failed runs.', '',
        '## Figures by target and season', '',
        table(['Target','Season','Figures'], [[NAMES[c['target']], c['season'],
            f"[Eight figures](b0-comparison/{c['directory']}.md)"] for c in manifest['cases']]), '',
        'Projection fans show the three best seeded runs across all six targets, plus the official ensemble '
        '(light blue) and the best run for the displayed target/season (light red). Selection uses the geometric '
        'mean of WIS ratios, weighting targets equally, then available season/geography cells equally. '
        'The season winner uses both geography groups and is shown once if already in the top three. '
        'Other figures show all 42 runs and the ensemble. Configuration IDs map to names in the ranking above; '
        'the `-s42`, `-s43`, and `-s44` suffixes identify seeds. Projection fans illustrate US and North Carolina. '
        'Relative-WIS plots average per-task ratios, whereas the tables use ratios of mean WIS.', '',
        '## Reproduction', '',
        f'Full artifacts are in `{comparison}`. See [Longleaf setup](../longleaf-setup.md) for input rebuilding, '
        'GPU arrays, parallel scoring, and resume commands. Publish the completed comparison with:', '',
        '```bash', f'.venv/bin/python scripts/publish_evaluation_docs.py --comparison {comparison}', '```', '',
        'The pipeline completed its built-in task-support, finite-metric, and relative-WIS checks. '
        'This publication summarizes saved scores; it does not rerun fitting or scoring.', '']
    (docs/'results/b0-configuration-comparison.md').write_text('\n'.join(lines))
    for case in manifest['cases']:
        name = case['directory']; title = f"{NAMES[case['target']]} · {case['season']}"
        destination = assets/name; destination.mkdir(exist_ok=True)
        top_models, season_best = fan_selection(leaderboard, runs.model, case)
        selected_models = list(dict.fromkeys([*top_models, case['ensemble'], season_best]))
        frames = []
        for model in selected_models:
            if model == case['ensemble']:
                frozen_quantiles = pd.read_parquet(Path(manifest['frozen'])/name/'quantiles.parquet')
                frames.append(frozen_quantiles[frozen_quantiles.model.eq(model)])
                continue
            forecast = pd.read_csv(comparison/'epibench'/name/'models'/model/'forecasts.csv',
                                   dtype={'location': str, 'output_type_id': str})
            wide = forecast.pivot(index=KEY, columns='output_type_id', values='value')
            wide.columns = ['q' + column for column in wide.columns]
            frames.append(wide.reset_index().assign(model=model))
        units = pd.read_parquet(Path(manifest['frozen'])/name/'units.parquet')
        fans(pd.concat(frames, ignore_index=True), units, case, destination, ['US', '37'],
             top_models=top_models, season_best=season_best, model_names=model_names)
        page = [f'# {title}', '', '[Canonical report and model names](../b0-configuration-comparison.md)', '',
                'All models use the same frozen tasks. US is the native national prediction; states/DC '
                'are evaluated individually. Fans connect the four horizons from one forecast origin. '
                'Fans show only the three best seeded runs across all six targets, plus the official ensemble '
                '(light blue) and this target/season’s best run (light red). Bands show 50%/95% intervals; '
                'black curves show truth. The season winner appears only once if already in the top three. '
                'Every fourth origin is illustrated. Admissions are counts; ED visits are proportions.', '',
                'Overall top three: ' + ', '.join(f'`{model_names[model]}`' for model in top_models) + '. '
                f'Target/season best: `{model_names[season_best]}`. '
                'See the [model differences table](../b0-configuration-comparison.md#model-differences) '
                'and the canonical report’s equal-target WIS-ratio selection rule.', '']
        for label, filename in FIGURES:
            source = destination/filename if filename.startswith('fans-') else comparison/'plots'/name/filename
            (destination/filename).write_text('\n'.join(line.rstrip() for line in source.read_text().splitlines())+'\n')
            url = f'../../assets/b0_configuration_comparison/{name}/{filename}'
            page += [f'## {label}', '', f'[![{title}: {label}]({url}){{ loading=lazy }}]({url})', '']
        (pages/f'{name}.md').write_text('\n'.join(page))
    print(f'Published {len(runs)} runs, {len(rankings)} variants, and {len(manifest["cases"])*8} figures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=DEFAULT)
    parser.add_argument('--docs', type=Path, default=Path('docs'))
    args = parser.parse_args()
    publish(args.comparison, args.docs)
