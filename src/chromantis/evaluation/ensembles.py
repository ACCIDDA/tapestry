"""Combine saved forecasts of fitted models, without refitting: one rule for research and production.

Each recipe (one configuration) contributes its seeds. Every recipe gets equal weight,
and every seed equal weight within its recipe, for the targets the recipe was trained
to predict (`loss_weights` > 0). Two rules:

- `vincent`: quantile averaging, level by level (mean over recipes of the seed means);
- `mixture`: equal-weight mixture of the members' predictive distributions. Each
  member's quantile function is linearly interpolated between the 23 Hub levels and
  constant beyond the 1%/99% levels, evaluated at 256 equally spaced probabilities;
  the pooled values give the mixture's quantiles. Equal recipe weight requires equal
  seed counts per recipe, which is checked.

Admission quantiles are rounded to integers afterwards. A marginal mixture of
quantiles does not create joint trajectories.

`build` writes a run-shaped folder (`manifest.json`, `eval_<season>/<view>/forecasts.npz`)
that `evaluation.ranking` scores like any fitted run. The command line takes a groups
file and ranks every group:

    python -m chromantis.evaluation.ensembles GROUPS.json --out data/experiments/NAME/ensembles

GROUPS.json:
    {"recipes": {"A": {"experiment": "b7-folds-20261007", "scenario": "...", "seeds": [44, 45]},
                 "B": {"runs": ["path/to/run", ...]}},
     "groups": {"A+B": ["A", "B"], "A alone": ["A"]},
     "rules": ["mixture"], "views": ["corrected"], "forecast_view": "corrected"}

`experiment` + `scenario` + `seeds` resolve to completed planner runs; add
`"inputs": "reported"` to use their `planner replay` copies instead. `runs` lists
run folders directly. `views` defaults to every view all members share; `forecast_view`
(default `corrected`) is the view the ensemble is ranked in, like a recipe's own
`forecast_view`; the other views are diagnostics.

History: replaces scripts/ensemble_b4.py, ensemble_b6.py, ensemble_b6_candidates.py,
score_b7_ensembles.py and score_b7_vs_system2.py (2026-10-08); the mixture
computation is unchanged, so the submitted B7 file is reproduced exactly.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from chromantis.model.objective import LOSS_WEIGHTS
from chromantis.model.scenario import Scenario
from .quantiles import LEVELS

U = (np.arange(256) + .5) / 256
W = np.stack([np.interp(U, LEVELS, np.eye(len(LEVELS))[k]) for k in range(len(LEVELS))], 1)  # [u, levels]
RULES = ('vincent', 'mixture')


def trained_channels(scenario):
    """Flu admission (0) and ED (3) channels a scenario was trained to predict."""
    weights = LOSS_WEIGHTS[Scenario.from_string(scenario).loss_weights]
    return [c for c in (0, 3) if weights[c] > 0]


def combine(recipes, channel, rule):
    """Recipes [{'channels', 'values' [seed, quantile, ..., channel, location]}] -> [quantile, ...] for `channel`."""
    eligible = [r for r in recipes if channel in r['channels']]
    if not eligible:
        raise ValueError(f'No trained contributor for channel {channel}')
    values = [r['values'][..., channel, :] for r in eligible]
    if rule == 'vincent':
        if len({len(v) for v in values}) == 1:
            return np.concatenate(values, axis=0).mean(axis=0, dtype=np.float64)
        return np.mean([v.mean(0, dtype=np.float64) for v in values], 0)
    if rule != 'mixture':
        raise ValueError(f'Unknown ensemble rule: {rule}')
    if len({len(v) for v in values}) != 1:
        raise ValueError('Mixture requires equal seed counts per recipe')
    draws = np.concatenate([np.tensordot(W, q, axes=(1, 0)) for v in values for q in v], 0)
    return np.quantile(draws, LEVELS, axis=0)


def view_file(run, season, view):
    """`raw` = the run's own forecasts; `drawK_<view>` = artificial evaluation draw K."""
    folder = Path(run) / f'eval_{season}'
    if view.startswith('draw'):
        draw, view = view.split('_', 1)
        folder = folder / f'draw-{draw[4:]}'
    return folder / ('' if view == 'raw' else view) / 'forecasts.npz'


def shared_views(runs, seasons):
    from .ranking import VIEWS
    names = ['raw', *VIEWS] + [f'draw{d}_{v}' for d in range(1, 5) for v in ('raw', 'half', 'corrected')]
    return [v for v in names if all(view_file(r, s, v).exists() for r in runs for s in seasons)]


def build(name, recipes, rule, out, views=None, forecast_view='corrected'):
    """Write ensemble run `out/<name>-<rule>` from recipes {label: {'runs', 'scenario'}}; returns a ranking row."""
    run = Path(out) / f'{name}-{rule}'
    if run.exists():
        raise ValueError(f'{run} already exists; scores are cached per folder, so use a fresh --out')
    members = [r for recipe in recipes.values() for r in recipe['runs']]
    seasons = json.loads((Path(members[0]) / 'manifest.json').read_text())['folds']
    if any(json.loads((Path(m) / 'manifest.json').read_text())['folds'] != seasons for m in members):
        raise ValueError(f'{name}: members evaluate different seasons')
    views = views or shared_views(members, seasons)
    if forecast_view not in views:
        raise ValueError(f'{name}: the ranked view {forecast_view!r} is not among the built views {views}')
    channels = sorted({c for recipe in recipes.values() for c in trained_channels(recipe['scenario'])})
    loss = 'flu_hosp_ed' if channels == [0, 3] else 'flu_only' if channels == [0] else 'flu_ed'
    run.mkdir(parents=True)
    for season in seasons:
        for view in views:
            loaded = []
            for recipe in recipes.values():
                data = [dict(np.load(view_file(m, season, view))) for m in recipe['runs']]
                loaded.append(dict(channels=trained_channels(recipe['scenario']), data=data))
            every = [d for r in loaded for d in r['data']]
            # Members can differ by an issuance at the season edge (reconstruction training
            # changes the first usable origin); keep the shared issuances. The scorers still
            # require every frozen Hub task, so a dropped scored task fails loudly there.
            common = sorted(set.intersection(*(set(d['context_end'].tolist()) for d in every)))
            for d in every:
                keep = np.isin(d['context_end'], common)
                for key, value in list(d.items()):
                    if key not in ('quantiles', 'quantile_levels', 'locations') and value.ndim and len(value) == len(keep):
                        d[key] = value[keep]
                d['quantiles'] = d['quantiles'][:, keep]
            for key in ('truth', 'mask', 'context_end', 'target_dates', 'locations', 'quantile_levels'):
                if not all(np.array_equal(d[key], every[0][key]) for d in every):
                    raise ValueError(f'{name}/{season}/{view}: misaligned {key}')
            for r in loaded:
                r['values'] = np.stack([d['quantiles'] for d in r.pop('data')])
            merged = {k: v for k, v in every[0].items() if k != 'flu_admission_sum_quantiles'}
            merged['quantiles'] = np.zeros_like(every[0]['quantiles'], dtype=np.float64)
            for channel in channels:
                q = combine(loaded, channel, rule)
                merged['quantiles'][:, :, :, channel, :] = np.floor(q + .5) if channel == 0 else q
            target = view_file(run, season, view)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **merged)
    manifest = dict(scenario=f'forecast_targets=flu,loss_weights={loss}', folds=seasons, rule=rule, views=views,
                    forecast_view=forecast_view,
                    ensemble={label: dict(scenario=r['scenario'], runs=[str(m) for m in r['runs']])
                              for label, r in recipes.items()}, definition=__doc__)
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return dict(path=run, config_id=f'ensemble={name},rule={rule}', name=f'{name} ({rule})', seed=0)


def resolve(recipe):
    """Run folders and scenario of one recipe entry of a groups file."""
    if 'runs' in recipe:
        runs = [Path(r) for r in recipe['runs']]
        scenario = recipe.get('scenario') or json.loads((runs[0] / 'manifest.json').read_text())['scenario']
        return dict(runs=runs, scenario=scenario)
    from chromantis.experiment.planner import seed_state, replay_folder, read_jobs
    folder = Path(recipe.get('root', 'data/experiments')) / recipe['experiment']
    # Match by settings, not spelling: the experiment may store the string with older field names.
    wanted = Scenario.from_string(recipe['scenario'])
    stored = [job['scenario'] for job in read_jobs(folder) if Scenario.from_string(job['scenario']) == wanted]
    if len(stored) != 1:
        raise ValueError(f'{recipe["experiment"]}: {len(stored)} planned jobs match {recipe["scenario"]}')
    runs = []
    for seed in recipe['seeds']:
        attempt, _, done = seed_state(folder, stored[0], seed)
        if not done:
            raise ValueError(f'{recipe["experiment"]}: seed {seed} of {recipe["scenario"]} is not complete')
        runs.append(replay_folder(folder, attempt, recipe['inputs']) if recipe.get('inputs') else attempt)
    return dict(runs=runs, scenario=recipe['scenario'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('groups', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frozen', default='data/evaluation/b0_hub_comparison_q23')
    args = parser.parse_args(argv)
    spec = json.loads(args.groups.read_text())
    recipes = {label: resolve(recipe) for label, recipe in spec['recipes'].items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'groups.json').write_text(json.dumps(dict(spec, resolved={
        label: dict(scenario=r['scenario'], runs=[str(p) for p in r['runs']]) for label, r in recipes.items()}), indent=2) + '\n')
    built = [build(name, {label: recipes[label] for label in labels}, rule, args.out, spec.get('views'),
                   spec.get('forecast_view', 'corrected'))
             for name, labels in spec['groups'].items() for rule in spec.get('rules', ['mixture'])
             if rule == 'vincent' or len(labels) > 1 or len(recipes[labels[0]]['runs']) > 1]
    from .ranking import rank
    rank(built, args.frozen, args.out / 'ranking')


if __name__ == '__main__':
    main()
