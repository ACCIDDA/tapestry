"""Add the saved B1 300-epoch comparison to the original screen's results page.

Run after plot_b1_overnight.py. Aggregates existing scores; no fits or rescoring.
"""
from datetime import datetime
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_b1_overnight import snapshot, label, table, save, fans, ARCH, PIPE
from plot_b01_crosses import NAMES, SHORT, case_order
from tapestry.evaluation.totals import rank, run_scores, frozen_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-fans', action='store_true', help='Keep existing fixed-seed fan images when refreshing text and tables')
    args = parser.parse_args()
    page = Path('docs/results/b1-overnight')
    out = page / 'epoch300'
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    root = Path('data/experiments/B1-formulations-300')
    settings = json.loads((root / 'experiment.json').read_text())
    original = json.loads(Path('data/experiments/B1-screen-256/experiment.json').read_text())
    if settings['input_sha256'] != original['input_sha256']:
        raise ValueError('Experiments have different frozen inputs')
    runs, states = snapshot(root)
    states.to_csv(out / 'run-status.csv', index=False)
    print(f'300-epoch snapshot: {states.status.value_counts().to_dict()}', flush=True)
    configs = rank(runs, out / 'ranking')
    names = {r['config_id']: r['name'] for r in runs}
    configs['name'] = configs.config_id.map(names)
    configs['label'] = configs.name.map(label)
    configs.to_csv(out / 'configuration-ranking.csv', index=False)
    old_runs = json.loads((page / 'ranking/manifest.json').read_text())['runs']
    old_names = {r['config_id']: r['name'] for r in old_runs}
    old_ids = {r['name']: r['config_id'] for r in old_runs}
    for r in runs:
        if old_ids[r['name']].replace(':ep100:', ':ep300:') != r['config_id']:
            raise ValueError(f'Non-budget recipe difference: {r["name"]}')
    # rank validates support within the new experiment; check against the old one too.
    keys = ['target', 'season', 'location', 'horizon', 'n']
    supports = [pd.read_csv(Path(r['path']) / 'totals.csv')[keys].sort_values(keys).reset_index(drop=True)
                for r in (runs[0], old_runs[0])]
    if not supports[0].equals(supports[1]):
        raise ValueError('Original and 300-epoch scores have different support')
    new = pd.read_csv(out / 'ranking/run_scores.csv')
    new['name'] = new.config_id.map(names)
    old = pd.read_csv(page / 'ranking/run_scores.csv')
    old['name'] = old.config_id.map(old_names)
    paired = new[new.geography.eq('all')][['name', 'seed', 'combined']].merge(
        old[old.geography.eq('all')][['name', 'seed', 'combined']],
        on=['name', 'seed'], suffixes=('_300', '_screen'), validate='one_to_one')
    paired['delta'] = paired.combined_300 - paired.combined_screen
    paired.to_csv(out / 'paired-seeds.csv', index=False)
    summary = paired.groupby('name').agg(seeds=('seed', 'count'),
        screen=('combined_screen', 'mean'), cap300=('combined_300', 'mean'),
        delta=('delta', 'mean'), delta_sd=('delta', 'std'),
        improved=('delta', lambda x: int((x < 0).sum())))
    summary.to_csv(out / 'paired-summary.csv')
    print(summary.to_string(), flush=True)

    ordered = [f'{a}__{p}__mask0.5' for a in ARCH for p in PIPE if f'{a}__{p}__mask0.5' in summary.index]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    for i, name in enumerate(ordered):
        r = summary.loc[name]
        axes[0].plot([r.screen, r.cap300], [i, i], color='#aaaaaa')
        axes[0].scatter(r.screen, i, color='#64748b', marker='s', label='Original screen' if i == 0 else None)
        axes[0].scatter(r.cap300, i, color='#16856c', label='Cap 300' if i == 0 else None)
        delta = paired[paired.name.eq(name)].delta
        axes[1].scatter(delta, [i] * len(delta), facecolors='none', edgecolors='#64748b')
        axes[1].scatter(r.delta, i, color='#16856c' if r.delta < 0 else '#bc6037', marker='D')
    axes[0].set_yticks(range(len(ordered)), [f'{label(n)} ({int(summary.loc[n,"seeds"])}/3)' for n in ordered], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(1, color='black', ls='--', lw=1)
    axes[0].set_xlabel('Relative WIS; lower is better'); axes[0].legend()
    axes[1].axvline(0, color='black', lw=1)
    axes[1].set_xlabel('Cap 300 minus original; open seeds, diamond mean')
    fig.suptitle('Matched seed comparisons · joint MLP rows are same-budget repeat controls')
    save(fig, figures, 'comparison.png')

    new_seasons = pd.read_csv(out / 'ranking/season_scores.csv')
    new_seasons['name'] = new_seasons.config_id.map(names)
    old_seasons = pd.read_csv(page / 'ranking/season_scores.csv')
    old_seasons['name'] = old_seasons.config_id.map(old_names)
    target_pairs = new_seasons[new_seasons.geography.eq('all')].merge(
        old_seasons[old_seasons.geography.eq('all')], on=['name','seed','target','season'],
        suffixes=('_300','_screen'), validate='one_to_one')
    target_pairs['delta'] = target_pairs.wis_ratio_300 - target_pairs.wis_ratio_screen
    target_pairs.to_csv(out / 'target-season-pairs.csv', index=False)
    changes = target_pairs.groupby(['name','season','target']).delta.mean().unstack(['season','target'])
    columns = sorted(changes.columns, key=lambda k: case_order({'season': k[0], 'target': k[1]}))
    changes = changes.reindex(index=ordered, columns=columns)
    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(changes, cmap='RdYlGn_r', vmin=-.2, vmax=.2, aspect='auto')
    ax.set_yticks(range(len(changes)), [label(n) for n in changes.index], fontsize=8)
    ax.set_xticks(range(len(columns)), [f'{NAMES[t]}\n{s}' for s,t in columns], rotation=60, ha='right', fontsize=8)
    for i in range(len(changes)):
        for j in range(len(columns)):
            ax.text(j, i, f'{changes.iloc[i,j]:+.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label='Δ relative WIS (colors clipped at ±0.2)')
    ax.set_title('Matched changes by disease, target, and year; negative is better')
    save(fig, figures, 'target-season-change.png')

    coverage = []
    for budget, seasons in [('screen',old_seasons),('300',new_seasons)]:
        for level in [50,95]:
            s = seasons.copy(); s['wis_ratio'] = s[f'model_coverage_{level}']
            c = run_scores(s); c['name'] = c.config_id.map(old_names if budget == 'screen' else names)
            c = c[c.geography.eq('all')].merge(paired[['name','seed']],on=['name','seed'],validate='one_to_one')
            coverage.append(c.assign(budget=budget, level=level))
    coverage = pd.concat(coverage, ignore_index=True)
    coverage.to_csv(out / 'coverage.csv', index=False)
    cov = coverage.groupby(['name','budget','level']).combined.mean()

    # Keep the same fixed seed as the original formulation fans, at both budgets.
    paths, chosen = {}, []
    for pipe in ['direct_finalflag','joint_aux025','two_stage']:
        name = f'pathogen_mlp__{pipe}__mask0.5'
        for cap, source in [(100,old_runs),(300,runs)]:
            match = [r for r in source if r['name'] == name and r['seed'] == 42]
            if match:
                alias = f'pathogen_mlp__{pipe}__cap{cap}'
                paths[(alias,42)] = Path(match[0]['path']); chosen.append((alias,42))
    if args.reuse_fans:
        fan_files = [(f'{NAMES[c["target"]]}, {c["season"]}', f'budget-{SHORT[c["target"]]}-{c["season"]}.png')
                     for c in sorted(frozen_cases(settings['frozen']), key=case_order)]
        if any(not (figures / filename).exists() for _, filename in fan_files):
            raise ValueError('Generate the fan images before using --reuse-fans')
    else:
        fan_files = fans(chosen, paths, Path(settings['frozen']), figures, 'budget')
    print('Comparison figures and fans saved', flush=True)
    stamp = datetime.now().astimezone().isoformat(timespec='minutes')
    meta = dict(generated=stamp, experiment=root.name, included=len(runs), planned=len(states),
                status=states.status.value_counts().to_dict(), source_snapshot=settings['source_snapshot_sha256'],
                original_source_snapshot=original['source_snapshot_sha256'], fan_seed=42)
    (out / 'snapshot.json').write_text(json.dumps(meta, indent=2)+'\n')
    ranked = configs[['label','seeds','combined_mean','combined_sd','states_dc_combined_mean','US_combined_mean']].copy()
    ranked.columns = ['Configuration','Seeds / 3','WIS ratio','Seed SD','States/DC','US']
    comparison = summary.reindex(ordered).reset_index()
    comparison['name'] = comparison.name.map(label)
    comparison.columns = ['Configuration','Paired seeds','Original screen','Cap 300','Δ WIS','Seed Δ SD','Seeds improved']
    b = summary.loc['pathogen_mlp__direct_finalflag__mask0.5']
    c = summary.loc['pathogen_mlp__joint_aux025__mask0.5']
    two = summary.loc['pathogen_mlp__two_stage__mask0.5']
    target_b = summary.loc['target_mlp__direct_finalflag__mask0.5']
    target_c = summary.loc['target_mlp__joint_aux025__mask0.5']
    winner = configs.iloc[0]['name']
    section = f'''<!-- epoch300:start -->
## 300-epoch results and comparison

**{len(runs)}/{len(states)} runs complete**, generated {stamp}. This is a dated snapshot.
The follow-up contains 16 configurations (four architectures × four formulations),
all at mixed 50% masking, with seeds 42/43/44. Caps are maxima with patience 30,
not fixed epoch counts. The original screen below also includes masking variants
that were not repeated here, including its gap-only winner.

### Ranking at cap 300

**{int((configs.combined_mean < 1).sum())}/{len(configs)} available configuration means beat the ensemble.**
Lower relative WIS is better; 1 is ensemble parity. Each seed is a complete three-season run.
Incomplete configurations retain their seed counts; missing runs are not imputed.

{table(ranked)}

[Ranking CSV](epoch300/configuration-ranking.csv) · [Per-seed scores](epoch300/ranking/run_scores.csv).

### What changed from the first screen

Every row below uses the same architecture, formulation, masking, and completed seed
on both sides. Input hashes and scored target/season/location/horizon support match.
The target MLP, pathogen MLP, and target convolution change cap 100 → 300.
**Joint MLP was already capped at 300: its rows are repeat controls.**
Negative changes favor the follow-up; seed SD is descriptive, not a confidence interval.

{table(comparison)}

![Matched budget comparisons](epoch300/figures/comparison.png)

**Pathogen B and C remain the strongest candidates in this follow-up.** B changes
from {b.screen:.3f} to {b.cap300:.3f}, improving in {int(b.improved)}/{int(b.seeds)} seeds;
C changes from {c.screen:.3f} to {c.cap300:.3f}, improving in {int(c.improved)}/{int(c.seeds)}.
Their small gap does not establish a reliable preference between flags alone and auxiliary nowcasting.

**Longer training does not rescue the current two-stage formulation.** Its follow-up
configuration means remain above ensemble parity in every architecture. Pathogen two-stage
changes from {two.screen:.3f} to {two.cap300:.3f}; the paired seed table shows a large
deterioration for seed 42. This weakens the explanation that the initial poor results
were simply due to the 100-epoch cap. It does not identify which part of the formulation fails.

**More training is not a universal gain.** Use the paired rows, not the best score from
each differently sized suite, to judge a recipe. Target B changes from {target_b.screen:.3f}
to {target_b.cap300:.3f}, and target C from {target_c.screen:.3f} to {target_c.cap300:.3f};
neither improves in any of the three matched seeds. The original gap-only leader is not
included in this budget contrast. Repeat variability also matters: joint-direct seed 44
changes from 1.144 to 1.363 despite retaining cap 300. The cause is not established here;
the saved fitting code is unchanged between source snapshots, whose changes add suite routing.

**Calibration remains a limitation.** The cap-300 leader's weighted 50% and 95%
coverage are {100*cov.loc[(winner,'300',50)]:.1f}% and {100*cov.loc[(winner,'300',95)]:.1f}%.
These use the same geography/target/season weights as the ranking.

### Where performance changes

Disease → target → chronological season, matching the fan order. Values are paired
seed-mean changes, not changes between averages with different seeds. Joint rows
remain repeat controls. Color saturation does not truncate the printed values.

![Changes by disease, target, and year](epoch300/figures/target-season-change.png)

### Matched 100-vs-300 fan plots

Pathogen MLP B, C, and two-stage at both caps, followed by the Hub ensemble.
Every model uses seed 42, the same fixed seed as the original formulation fans.
It is illustrative, not an average: pathogen two-stage seed 42 is also the largest
observed deterioration, so judge the overall result using all three seeds above.
Dates match B0.1's full saved-season calendar; truth and ensemble retain their frozen
availability. Missing values are not filled in. Display dates extend beyond scoring support.

'''
    for title, filename in fan_files:
        section += f'![{title} — matched training budgets](epoch300/figures/{filename})\n\n'
    section += '''### Scope, provenance, and reproduction

This remains retrospective development CV on seasons used for model development,
not prospective evidence. Three seeds do not resolve small differences. Changing the
cap permits longer training and can change checkpoint selection; it does not mean
every component trained for 300 epochs. Forecast skill is not a standalone nowcast score.

[Matched seed data](epoch300/paired-seeds.csv) · [Target/season comparisons](epoch300/target-season-pairs.csv) ·
[Coverage](epoch300/coverage.csv) · [Snapshot](epoch300/snapshot.json) · [Run status](epoch300/run-status.csv).

Regenerate from saved scores and forecasts with `.venv/bin/python scripts/plot_b1_300.py`.
No training or scoring jobs are launched. Original-screen results follow below.

<!-- epoch300:end -->
'''
    (out / 'section.txt').write_text(section)
    body = (page / 'index.md').read_text()
    body = body.replace('# B1 overnight — Formulations and masking', '# B1 — First screen and 300-epoch comparison', 1)
    start, end = '<!-- epoch300:start -->', '<!-- epoch300:end -->'
    if start in body:
        body = body[:body.index(start)] + body[body.index(end)+len(end):]
    body = body.replace('This page reports the original overnight screen, **not the new 300-epoch experiment**.',
                        'These headline numbers describe the original overnight screen. The separate\n[300-epoch comparison](#300-epoch-results-and-comparison) is reported below.')
    body = body.replace('## Snapshot and experiment', section+'\n## Snapshot and experiment', 1)
    log = '- 2026-09-17: added the completed 300-epoch follow-up, paired seed comparisons, target/season changes, coverage, and matched-budget fans. Joint MLP repeats are explicitly treated as same-budget controls.\n'
    if log not in body:
        body = body.replace('## Log\n', '## Log\n\n' + log, 1)
    (page / 'index.md').write_text(body)
    print(f'Updated {page / "index.md"}', flush=True)


if __name__ == '__main__':
    main()
