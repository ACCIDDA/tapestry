"""Build the B1 overnight report from saved scores and forecasts, without refitting.

Run: .venv/bin/python scripts/plot_b1_overnight.py
The report is an explicitly dated snapshot; incomplete seed sets stay labelled.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tapestry.evaluation.hubs import export
from tapestry.evaluation.totals import rank, frozen_cases, season_scores, run_scores
from tapestry.models.manager import read_jobs, attempts
from plot_b01_crosses import NAMES, SHORT, case_order

ARCH = {'target_mlp': 'Target MLP', 'pathogen_mlp': 'Pathogen MLP',
        'target_multiscale': 'Target convolution', 'joint_mlp': 'Joint MLP'}
PIPE = {'direct': 'A', 'direct_finalflag': 'B', 'joint_aux025': 'C', 'two_stage': 'Two-stage'}
COLORS = {'A': '#64748b', 'B': '#16856c', 'C': '#387ac1', 'Two-stage': '#bc6037'}


def label(name):
    a, p, mask = name.split('__')
    mask = {'mask0.5': 'mixed 50%', 'mask0.25': 'mixed 25%', 'mask0': 'no masking'}.get(mask, mask.replace('_', ' '))
    return f'{ARCH[a]} · {PIPE[p]} · {mask}'


def table(frame):
    cols = list(frame.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join('—' if pd.isna(v) else f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def interpretation(configs, contrasts, stress_table, coverage):
    """Describe this screen; never infer causal mechanisms from the ranking."""
    by_name = configs.set_index('name')
    b = contrasts[contrasts.Contrast.eq('B − A')]
    two = contrasts[contrasts.Contrast.eq('Two-stage − B')]
    stress_rows = []
    for arch in ARCH:
        row = {'Architecture': ARCH[arch]}
        for pipe, short in PIPE.items():
            row[short] = stress_table.loc[f'{arch}__{pipe}__mask0.5', 'outage']
        stress_rows.append(row)
    winner = configs.iloc[0]
    cov = coverage[(coverage.geography == 'all') & (coverage.who == 'model')]
    intervals = cov.groupby(['name', 'level']).combined.mean()
    return f'''## What works, what does not

**Supplied-final flags are the most consistent improvement over direct forecasting.**
B lowers mean WIS versus A in {int((b['Mean Δ WIS'] < 0).sum())}/4 architectures,
by {abs(b['Mean Δ WIS'].max()):.3f}–{abs(b['Mean Δ WIS'].min()):.3f}.
The target MLP, pathogen MLP and target convolution improve on all three matched seeds;
the joint MLP improves on only one seed despite its better mean, so that contrast is less stable.

**Auxiliary nowcasting is competitive, not a general failure.** At mixed masking 50%,
C scores {by_name.loc['pathogen_mlp__joint_aux025__mask0.5', 'combined_mean']:.3f} versus
B's {by_name.loc['pathogen_mlp__direct_finalflag__mask0.5', 'combined_mean']:.3f} for pathogen MLP.
The target MLP also changes little. C improves mean WIS for the target convolution and joint MLP;
the joint comparison improves on all three seeds. These results do not establish that producing
a nowcast is itself the cause: C changes the training objective as well as the outputs.

**Two-stage is worse on natural inputs in every matched architecture and seed.**
Its mean penalty relative to B is {two['Mean Δ WIS'].min():.3f}–{two['Mean Δ WIS'].max():.3f}.
This is evidence against the current complete two-stage formulation for natural-input forecasting.
It does not identify whether the problem is training duration, flags, objective weighting,
or feeding recent estimates into the forecast. Those remain separate hypotheses.

**There is no universally best masking recipe.** Gap-only B wins overall for target MLP;
mixed 50% B wins within pathogen MLP. Training without artificial masking is competitive in both families.
The 25% setting is not an intermediate step in a monotonic improvement curve. Masking changes
validation as well as training, so these are recipe comparisons, not isolated regularization effects.

**Natural-input winners are not necessarily the most robust to outages.**
The overall winner moves from {winner.combined_mean:.3f} naturally to
{stress_table.loc[winner['name'], 'outage']:.3f} under a full-channel outage.
Two-stage target MLP and target convolution have lower outage scores than their B counterparts,
despite losing on natural inputs. The following matched panel makes that tradeoff explicit;
all values are forecast WIS relative to the original, uncorrupted ensemble baseline.

{table(pd.DataFrame(stress_rows))}

**Calibration still needs work.** The winning model's weighted 50% and 95% coverage are
{100*intervals.loc[(winner['name'], 50)]:.1f}% and {100*intervals.loc[(winner['name'], 95)]:.1f}%.
Winning on WIS does not mean its uncertainty intervals are calibrated.

For the next decision, retain the best natural-input B recipes, C as a competitive formulation,
and the two-stage outage tradeoff. Use the 300-epoch experiment to test the training-budget
explanation before changing the architecture or loss. Do not select a universal winner from
small mean differences over only three seeds.

'''


def snapshot(root):
    runs, states = [], []
    for job in read_jobs(root):
        for seed in job['seeds']:
            candidates = attempts(root, job['scenario'], seed)
            chosen = None
            for a in reversed(candidates):
                record = json.loads((a / 'run.json').read_text())
                if record['status'] == 'complete':
                    chosen = a
                    break
            state = 'complete' if chosen else json.loads((candidates[-1] / 'run.json').read_text())['status'] if candidates else 'planned'
            states.append(dict(name=job['name'], seed=seed, status=state))
            if chosen:
                path = chosen / 'b1'
                # Do not reload checkpoints: ranking validates common scored support.
                for filename in ['manifest.json', 'totals.csv']:
                    if not (path / filename).is_file():
                        raise ValueError(f'Completed run lacks {filename}: {path}')
                runs.append(dict(config_id=job['scenario'], name=job['name'], seed=seed, path=path))
    return runs, pd.DataFrame(states)


def fans(chosen, paths, frozen, output, prefix):
    frames = {name: export(paths[(name, seed)]) for name, seed in chosen}
    files = []
    for case in sorted(frozen_cases(frozen), key=case_order):
        target, season = case['target'], case['season']
        folder = frozen / case['directory']
        units = pd.read_parquet(folder / 'units.parquet')
        qs = pd.read_parquet(folder / 'quantiles.parquet')
        ensemble = qs[qs.model.eq(case['ensemble'])]
        series = [(f'{label(name)}\nseed {seed}', frames[name][(season, target)], COLORS[PIPE[name.split('__')[1]]]) for name, seed in chosen]
        series.append(('Hub ensemble', ensemble, '#757575'))
        fig, axes = plt.subplots(len(series), 2, figsize=(14, 2.15 * len(series)), sharex=True, sharey='col', squeeze=False)
        for column, loc in enumerate(['US', '37']):
            truth = units[units.location.eq(loc)][['target_end_date', 'observed']].drop_duplicates().sort_values('target_end_date')
            for row, (name, frame, color) in enumerate(series):
                ax = axes[row, column]
                # Match B0.1's full saved seasonal calendar and every-third-origin rule.
                # Only the ranking, not these illustrative fans, is restricted to Hub support.
                data = frame[frame.location.eq(loc)]
                origins = sorted(data.reference_date.unique())[::3]
                ax.plot(pd.to_datetime(truth.target_end_date), truth.observed, color='black', lw=1, label='Frozen truth')
                for i, origin in enumerate(origins):
                    f = (data[data.reference_date.eq(origin)].sort_values('horizon')
                         .set_index('horizon').reindex(range(4)))
                    x = pd.date_range(origin, periods=4, freq='7D')
                    ax.fill_between(x, f['q0.025'], f['q0.975'], color=color, alpha=.18, label='95%' if i == 0 else None)
                    ax.fill_between(x, f['q0.25'], f['q0.75'], color=color, alpha=.42, label='50%' if i == 0 else None)
                    ax.plot(x, f['q0.5'], color=color, lw=1)
                ax.set_ylim(bottom=0)
                if column == 0:
                    ax.set_ylabel(name, fontsize=7, rotation=0, ha='right', va='center', labelpad=8)
                if row == 0:
                    ax.set_title('United States' if loc == 'US' else 'North Carolina')
            axes[-1, column].set_xlabel('Target week ending')
        axes[0, 0].legend(fontsize=7, ncol=3)
        fig.suptitle(f'{NAMES[target]} · {season}\nNatural-input four-week forecasts; B0.1 calendar, every third saved origin', fontsize=12)
        fig.autofmt_xdate()
        name = f'{prefix}-{SHORT[target]}-{season}.png'
        save(fig, output, name)
        files.append((f'{NAMES[target]}, {season}', name))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--experiment', default='B1-screen-256')
    parser.add_argument('--output', type=Path, default=Path('docs/results/b1-overnight'))
    parser.add_argument('--skip-fans', action='store_true')
    args = parser.parse_args()
    root = Path('data/experiments') / args.experiment
    out = args.output
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    settings = json.loads((root / 'experiment.json').read_text())
    runs, states = snapshot(root)
    states.to_csv(out / 'run-status.csv', index=False)
    print(f'Snapshot: {states.status.value_counts().to_dict()}', flush=True)
    configs = rank(runs, out / 'ranking')
    names = {r['config_id']: r['name'] for r in runs}
    configs['name'] = configs.config_id.map(names)
    scores = pd.read_csv(out / 'ranking/run_scores.csv')
    scores['name'] = scores.config_id.map(names)
    natural = scores[scores.geography.eq('all')]
    seasons = pd.read_csv(out / 'ranking/season_scores.csv')
    seasons['name'] = seasons.config_id.map(names)
    configs['label'] = configs.name.map(label)
    configs.to_csv(out / 'configuration-ranking.csv', index=False)
    paths = {(r['name'], r['seed']): r['path'] for r in runs}
    complete = configs[configs.seeds.eq(3)].sort_values('combined_mean')
    leaders = complete.head(3).name.tolist()
    print('Forecast ranking saved', flush=True)

    fig, ax = plt.subplots(figsize=(11, 12))
    for i, r in enumerate(configs.itertuples()):
        color = COLORS[PIPE[r.name.split('__')[1]]]
        pts = natural[natural.name.eq(r.name)].combined
        ax.scatter(pts, [i] * len(pts), facecolors='none', edgecolors=color, s=23)
        ax.scatter(r.combined_mean, i, color=color, s=42, marker='o' if r.seeds == 3 else 'D')
    ax.set_yticks(range(len(configs)), [f'{r.label} ({r.seeds}/3)' for r in configs.itertuples()], fontsize=8)
    ax.invert_yaxis(); ax.axvline(1, color='black', ls='--', lw=1)
    ax.set_xlabel('Relative WIS; lower is better, ensemble = 1')
    ax.set_title('All configurations · filled mean, open seeds; diamonds = incomplete')
    save(fig, figures, 'ranking.png')

    # Paired contrasts always use the intersection of completed seeds.
    contrasts = []
    pairs = [('B − A', 'direct', 'direct_finalflag'), ('C − B', 'direct_finalflag', 'joint_aux025'), ('Two-stage − B', 'direct_finalflag', 'two_stage')]
    for arch in ARCH:
        for title, base, alt in pairs:
            a = natural[natural.name.eq(f'{arch}__{base}__mask0.5')].set_index('seed').combined
            b = natural[natural.name.eq(f'{arch}__{alt}__mask0.5')].set_index('seed').combined
            delta = b.subtract(a).dropna()
            contrasts.append({'Architecture': ARCH[arch], 'Contrast': title, 'Paired seeds': len(delta), 'Mean Δ WIS': delta.mean(), 'Seed Δ SD': delta.std(), 'Seeds improved': int((delta < 0).sum())})
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(out / 'paired-contrasts.csv', index=False)
    formulation = []
    for arch in ARCH:
        row = {'Architecture': ARCH[arch]}
        for p, short in PIPE.items():
            r = configs[configs.name.eq(f'{arch}__{p}__mask0.5')]
            row[short] = f'{r.iloc[0].combined_mean:.3f} ({int(r.iloc[0].seeds)}/3)' if len(r) else 'pending'
        formulation.append(row)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (title, _, _) in zip(axes, pairs):
        d = contrasts[contrasts.Contrast.eq(title)]
        ax.barh(d.Architecture, d['Mean Δ WIS'], color=['#16856c' if v < 0 else '#bc6037' for v in d['Mean Δ WIS']])
        ax.axvline(0, color='black', lw=1); ax.set_title(title); ax.set_xlabel('Paired mean Δ relative WIS')
    save(fig, figures, 'formulations.png')

    mask_rows = []
    for arch in ARCH:
        row = {'Architecture': ARCH[arch]}
        for mask in ['mask0', 'mask0.25', 'mask0.5', 'recent_only', 'gap_only', 'outage_only']:
            r = configs[configs.name.eq(f'{arch}__direct_finalflag__{mask}')]
            row[mask] = f'{r.iloc[0].combined_mean:.3f} ({int(r.iloc[0].seeds)}/3)' if len(r) else 'pending'
        mask_rows.append(row)

    # Stress diagnostics retain the scientific score, not pooled native-unit ratios.
    stress_parts = []
    for r in runs:
        t = pd.read_csv(r['path'] / 'stress-totals.csv').assign(config_id=r['config_id'], seed=r['seed'])
        stress_parts.append(t)
    stresses = pd.concat(stress_parts, ignore_index=True)
    support_keys = ['target', 'season', 'location', 'horizon', 'n']
    support = None
    for _, part in stresses.groupby(['config_id', 'seed', 'stress']):
        current = part[support_keys].sort_values(support_keys).reset_index(drop=True)
        if support is None: support = current
        elif not current.equals(support): raise ValueError('Stress scores differ in evaluation support')
    stress_scores = pd.concat([run_scores(season_scores(g)).assign(stress=k) for k, g in stresses.groupby('stress')], ignore_index=True)
    stress_scores['name'] = stress_scores.config_id.map(names)
    stress_scores.to_csv(out / 'stress-run-scores.csv', index=False)
    stress_table = stress_scores[stress_scores.geography.eq('all')].groupby(['name', 'stress']).combined.mean().unstack()
    stress_table.to_csv(out / 'stress-configuration-scores.csv')
    selected = list(dict.fromkeys(leaders + [f'pathogen_mlp__{p}__mask0.5' for p in PIPE]))
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in selected:
        if name not in stress_table.index: continue
        ax.plot(range(4), stress_table.loc[name, ['natural', 'recent', 'gap', 'outage']], marker='o', label=label(name))
    ax.set_xticks(range(4), ['Natural', 'Recent hidden', 'Gap', 'Outage'])
    ax.axhline(1, color='black', ls='--', lw=1); ax.set_ylabel('Relative forecast WIS')
    ax.set_title('Held-out stress diagnostics · unchanged Hub ensemble denominator')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    save(fig, figures, 'stress.png')
    print('Formulation and stress comparisons saved', flush=True)

    # Coverage uses the same target-within-season and geography weights as WIS.
    coverage_parts = []
    for who in ['model', 'ensemble']:
        for level in [50, 95]:
            s = seasons.copy(); s['wis_ratio'] = s[f'{who}_coverage_{level}']
            c = run_scores(s); c['who'] = who; c['level'] = level
            coverage_parts.append(c)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    coverage['name'] = coverage.config_id.map(names)
    coverage.to_csv(out / 'coverage.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, level in zip(axes, [50, 95]):
        c = coverage[(coverage.level == level) & coverage.geography.eq('all')]
        m = c[c.who.eq('model')].groupby('name').combined.mean()
        ax.scatter(configs.combined_mean, configs.name.map(m) * 100, color='#16856c')
        ax.axhline(level, color='black', ls=':', label='Nominal')
        ax.axhline(c[c.who.eq('ensemble')].combined.mean() * 100, color='#777777', ls='--', label='Hub ensemble')
        ax.set_xlabel('Relative WIS'); ax.set_ylabel(f'{level}% interval coverage (%)'); ax.legend(fontsize=8)
    save(fig, figures, 'coverage.png')

    target_table = seasons[seasons.geography.eq('all')].groupby(['name', 'season', 'target']).wis_ratio.mean().unstack(['season', 'target'])
    columns = sorted(target_table.columns, key=lambda pair: case_order({'season': pair[0], 'target': pair[1]}))
    target_table = target_table.reindex(columns=columns)
    target_table.to_csv(out / 'target-season-scores.csv')
    fig, ax = plt.subplots(figsize=(12, 5))
    data = target_table.reindex(selected)
    im = ax.imshow(data, cmap='RdYlGn_r', vmin=.7, vmax=1.3, aspect='auto')
    ax.set_yticks(range(len(data)), [label(n) for n in data.index], fontsize=8)
    ax.set_xticks(range(len(data.columns)), [f'{NAMES[t]}\n{s}' for s,t in data.columns], rotation=60, ha='right', fontsize=8)
    for i in range(len(data)):
        for j in range(len(data.columns)):
            ax.text(j, i, f'{data.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label='Relative WIS (colors clipped at 0.7 / 1.3)')
    save(fig, figures, 'target-season.png')

    # Epoch counts use actual history lengths, not the configured budget field.
    training = []
    for r in runs:
        for path in sorted(r['path'].glob('eval_*/manifest.json')):
            m = json.loads(path.read_text())
            for record in m['records']:
                if record['phase'] == 'select':
                    training.append(dict(name=r['name'], seed=r['seed'], season=m['held_out_season'], targets=str(record['targets']), cap=m['configuration']['epochs'], epochs=len(record['history']), selected=record['selected_epoch']))
    training = pd.DataFrame(training)
    training.to_csv(out / 'epoch-selection.csv', index=False)
    epoch_rows = []
    for p, title in PIPE.items():
        t = training[training.name.eq(f'pathogen_mlp__{p}__mask0.5')]
        epoch_rows.append({'Formulation': title, 'Selections': len(t), 'Hit cap': int(t.epochs.eq(t.cap).sum()), 'Best epoch ≥90': int(t.selected.ge(90).sum())})

    leader_fans, formulation_fans = [], []
    chosen = []
    for name in leaders:
        r = natural[natural.name.eq(name)].sort_values(['combined', 'seed'])
        chosen.append((name, int(r.iloc[len(r)//2].seed)))
    if not args.skip_fans:
        leader_fans = fans(chosen, paths, Path(settings['frozen']), figures, 'leaders')
        # A fixed common seed avoids selecting a favorable seed for each formulation.
        matched = [(f'pathogen_mlp__{p}__mask0.5', 42) for p in PIPE]
        if all(pair in paths for pair in matched):
            formulation_fans = fans(matched, paths, Path(settings['frozen']), figures, 'formulations')
        print('Fan plots saved', flush=True)

    stamp = datetime.now().astimezone().isoformat(timespec='minutes')
    meta = dict(experiment=args.experiment, generated=stamp, source_snapshot=settings['source_snapshot_sha256'],
                states=states.status.value_counts().to_dict(), planned=len(states), included=len(runs),
                leader_fan_seeds=chosen, formulation_fan_seed=42, score='location-relative-season-first-us20-v1')
    (out / 'snapshot.json').write_text(json.dumps(meta, indent=2)+'\n')
    top = configs[['label','combined_mean','combined_sd','seeds','states_dc_combined_mean','US_combined_mean']].copy()
    top.columns = ['Configuration','WIS ratio','Seed SD','Seeds / 3','States/DC','US']
    top.insert(0, 'Rank', range(1, len(top) + 1))
    winner = configs.iloc[0]
    target_details = target_table.loc[[winner['name'], 'pathogen_mlp__direct_finalflag__mask0.5',
                                     'pathogen_mlp__joint_aux025__mask0.5', 'pathogen_mlp__two_stage__mask0.5']].T
    target_details.columns = ['Overall leader', 'Pathogen B', 'Pathogen C', 'Pathogen two-stage']
    target_details = target_details.reset_index()
    target_details['target'] = target_details.target.map(NAMES)
    target_details = target_details.rename(columns={'season': 'Season', 'target': 'Target'})
    body = f'''# B1 overnight — Formulations and masking

**{int((configs.combined_mean < 1).sum())} of {len(configs)} configurations beat the Hub ensemble** on the combined forecast score.
The leading configuration is **{winner.label}**, at **{winner.combined_mean:.3f}** ({100*(1-winner.combined_mean):.1f}% lower relative WIS).
This page reports the original overnight screen, **not the new 300-epoch experiment**.

## Snapshot and experiment

Generated {stamp}. **{len(runs)}/{len(states)} seed runs complete**; status: {dict(Counter(states.status))}.
The page is a snapshot, not a live dashboard. Incomplete averages show their seed count;
paired contrasts use only seeds completed on both sides. No missing run receives an imputed score.

Four architectures × ten formulation/masking settings × seeds 42/43/44 = 120 runs,
with three held-out season folds per run. Target MLP, pathogen MLP and target convolution
use a 100-epoch cap; joint MLP uses 300. All use patience 30 and select then refit.
Evaluation uses 256 trajectories and the frozen 23 Hub quantiles. See the
[screen specification](../../workflows/b1-overnight.md) and [300-epoch follow-up](../../workflows/b1-300.md).

The score divides model WIS by ensemble WIS within target/season/location,
weights states/DC 80% and US 20%, then admissions 1 and ED 0.5 within each season,
and averages seasons equally. **Lower is better; 1 is ensemble parity.**
Target support is flu admissions in 2023–24; flu/COVID admissions in 2024–25;
and all six targets in 2025–26. Combined skill is not a claim of winning every target or season.

## What the formulations mean

| Formulation | Outputs and training |
|---|---|
| A — direct | Forecasts four future weeks; no explicit nowcast output. |
| B — supplied-final flags | Direct forecasts with indicators identifying supplied finalized inputs; no nowcast output. |
| C — auxiliary nowcast | Shared representation, separate recent/future heads, supplied-final flags; forecast loss + 0.25 × recent loss; forecast-only checkpoint selection. |
| Two-stage | Sample recent values and feed them into forecasting; no explicit supplied-final feature; equally weighted recent/future selection loss. |

Independent fits still receive all six input histories. Target models have six components,
pathogen models three, joint models one. Each component selects its epoch count separately,
then a fresh model refits on all permitted training weeks. Held-out seasons are excluded.

## Ranking

{table(top)}

Seed SD measures variation across the available seeds, not a confidence interval or significance threshold.
[All 40 configurations](configuration-ranking.csv) · [Per-seed scores](ranking/run_scores.csv).

![All configurations and seed scores](figures/ranking.png)

## Formulations at matched mixed masking

Every entry uses a 50% probability of corrupting a training episode, with a conditional
recent/gap/outage mixture of 50/30/20. Parentheses give completed seeds out of three.

{table(pd.DataFrame(formulation))}

![Paired formulation changes](figures/formulations.png)

Negative differences favor the first model named in the contrast (for example, C in C − B). These compare the complete
formulations: the two-stage/B contrast changes flags and loss weighting as well as feedback.

{table(contrasts)}

## Masking around B

Mask rates are episode probabilities, not percentages of cells. Natural missingness remains
in every configuration. A recent mask hides recent observations across locations; a gap
hides a short channel/location block; an outage removes a channel across the full context.
Validation masks follow the candidate mixture, so these comparisons change both training
and checkpoint-selection conditions.

{table(pd.DataFrame(mask_rows).rename(columns={'mask0':'None','mask0.25':'Mixed 25%','mask0.5':'Mixed 50%','recent_only':'Recent only','gap_only':'Gap only','outage_only':'Outage only'}))}

## Held-out stress diagnostics

These scores use the same frozen target cells and unchanged Hub ensemble denominator,
but artificially hide model inputs. They are robustness diagnostics, not additional independent
evaluations or a comparison against an ensemble subjected to the same corruption.

![Stress comparisons](figures/stress.png)

{table(stress_table.reindex(selected).rename(index=label).reset_index().rename(columns={'name':'Configuration'}))}

[All stress scores](stress-configuration-scores.csv).

## Target, season, geography, and calibration

The combined score can hide different behavior by pathogen or season. The panels below use
only available frozen Hub support; an absent target/season combination is not filled in.

![Relative WIS by target and held-out season](figures/target-season.png)

### Where the gains and failures occur

The overall leader uses target MLP, B, gap-only masking. The other columns fix
pathogen MLP and mixed 50% masking; values average all three seeds. These are
the nine available target/season cases, not nine equally weighted contributions
to the combined score: seasons receive equal weight and targets are weighted within seasons.

{table(target_details)}

The overall leader still loses to the ensemble on RSV admissions and COVID ED in
2025–26. Its strongest case-level gains are RSV ED in 2025–26 and flu admissions
in 2024–25. Pathogen C improves on B for flu admissions in 2025–26 while losing
ground on COVID admissions and RSV ED that season; the similar combined scores
therefore conceal meaningful differences across targets.

The pathogen two-stage model's failures are concentrated in flu admissions in
2023–24 and COVID admissions/ED and flu admissions in 2025–26. It remains
competitive on flu admissions in 2024–25 and RSV in 2025–26. These score-based
observations do not identify whether bias, spread, or timing causes those failures.

![Interval coverage against nominal and ensemble levels](figures/coverage.png)

Coverage here uses the same geography, target and season weighting as the combined score,
then averages seeds. It is a diagnostic, not an alternative ranking objective. Coverage below
nominal can reflect bias, narrow intervals, or both; these plots alone do not isolate the cause.
[Coverage data](coverage.csv) · [Target/season scores](target-season-scores.csv).

## Training budget and the 300-epoch question

For the matched pathogen-MLP formulation panel:

{table(pd.DataFrame(epoch_rows))}

Hitting a cap, or selecting a late epoch, suggests that a longer budget merits testing;
it does not establish undertraining as the cause of poor held-out WIS. Selection loss differs
between formulations and cannot be read as the final Hub WIS. The separate follow-up raises
the cap to 300 while retaining patience 30, three seeds, four formulations, and all four
architectures. Joint MLP already had cap 300, so its repeats are controls rather than a cap contrast.

## Fan plots

Natural-input four-week forecasts at every third saved origin, using the same full seasonal
calendar and plotting rule as [B0.1](../b0-1-crosses/index.md#fan-plots), for the United States
and North Carolina. Black is frozen truth; colored bands are
50% and 95% intervals, with median lines. Y scales match across models within each location.
These examples illustrate forecasts and do not replace the all-location scoring.
Forecasts extend beyond the scored Hub window; truth and ensemble remain limited to their
available frozen dates. No missing forecast or truth value is filled in. The ranking still
uses only identical frozen Hub support. Each row samples its own available origins, as in B0.1.

The standard order throughout the target/season panels is **influenza → COVID-19 → RSV**,
then **admissions → ED visits**, then **oldest → newest season**. Saved model dates span
September 9, 2023–July 27, 2024; August 10, 2024–July 26, 2025; and
August 9, 2025–August 1, 2026, respectively. The final fan can have fewer than four
available horizons at the held-out season boundary.

### Leading configurations

The top three configurations with all three seeds complete, each at its median-scoring seed:
{'; '.join(f'{label(n)} — seed {s}' for n,s in chosen)}.

'''
    for title, filename in leader_fans:
        body += f'![{title} — leading models](figures/{filename})\n\n'
    body += '### Matched A/B/C/two-stage comparison\n\nPathogen MLP, mixed masking 50%, fixed seed 42 for every formulation, followed by the Hub ensemble. This seed was fixed for comparability, not chosen for its score.\n\n'
    for title, filename in formulation_fans:
        body += f'![{title} — matched formulations](figures/{filename})\n\n'
    body += '''## Assumptions and limits

- Retrospective development CV: older inputs are finalized and missing recent vintage reports can receive supplied finals. This is not prospective deployment performance.
- The same seasons informed architecture selection and this ranking. Three seeds do not measure all sources of uncertainty; small gaps remain unresolved.
- Fan examples use US and North Carolina and a stated seed selection rule. No result is inferred from visual inspection of the plots.
- Completed attempts are resolved newest-complete first from manager records; saved score tables must have identical frozen support. Models are not refitted or rescored to create this page.
- Standalone nowcast ranking is omitted: the existing cross-formulation scorer rejects mismatched nowcast cell support. Forecast skill does not establish nowcast accuracy; those scores need a separate common-support comparison against persistence, not the Hub ensemble.
- B0.1 is context, not a controlled comparison: input histories and evaluation draw budgets differ. This page does not attribute the B0/B1 score gap to one change.

## Reproducing

```bash
.venv/bin/python scripts/plot_b1_overnight.py
.venv/bin/python -m mkdocs build --strict
```

This regenerates the snapshot and figures from completed runs; it launches no training or scoring jobs.
[Snapshot provenance](snapshot.json) · [Run status at generation](run-status.csv) · [Paired contrasts](paired-contrasts.csv).

## Log

- 2026-09-17: added the original overnight screen report while the separate 300-epoch experiment was queued. Reused saved forecast totals, the shared scientific aggregation and B1 forecast export. Explicitly separated forecast performance from nowcast accuracy and marked incomplete seed sets.
- 2026-09-17: expanded the page to all 40 ranked configurations and a numeric target/season breakdown of the leader and matched pathogen formulations; clarified that training without artificial masking remains competitive.
- 2026-09-17: matched fan dates to B0.1's full saved seasonal calendar; retained frozen support for scores and truth. Standardized disease, target, and chronological season order in fans, target/season tables, and the heatmap.
'''
    body = body.replace('## Masking around B', interpretation(configs, contrasts, stress_table, coverage) + '## Masking around B')
    (out / 'index.md').write_text(body)
    print(f'Wrote {out / "index.md"}', flush=True)


if __name__ == '__main__':
    main()
