"""B5 exploration: random neighbourhood search around the B4 leaders with the new options.

Anchors (exact B4 scenarios): A (sampled, real-tree-corrected synthetic training histories, Kinsa),
B (direct quantiles, artificial errors + reconstruction), Bs (B with the sampled head and 0.25 x
four-week-total WIS). Each configuration starts from one anchor and independently redraws every
factor below; the first level listed is the anchor's own value and is drawn most often, so
configurations stay near a leader and main effects can be estimated by regression. Four blocks:
training vintages, nowcasting, formulation, untested knobs. Labels stay finalized; folds and
evaluation are the B4 protocol. Each run also writes the corrected and (if nowcast_noise > 0)
sampled-history views, so stochastic nowcasting is compared within the same fitted model."""
import argparse, json, subprocess, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from tapestry.model.scenario import Scenario

p = argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true'); p.add_argument('--plan', action='store_true')
p.add_argument('--configurations', type=int, default=1000); args = p.parse_args()
name = 'b5-explore-check-20261006' if args.smoke else 'b5-explore-20261006'
heads = Path('docs/experiments/b4-flu-heads-20261006/scenarios.txt').read_text().split('\n')
ANCHORS = {'A': Scenario.from_string(heads[0]), 'B': Scenario.from_string(heads[6]), 'Bs': Scenario.from_string(heads[5])}
rng = np.random.default_rng(20261007)


def pick(levels, weights=None):
    w = np.ones(len(levels)) if weights is None else np.asarray(weights, float)
    return levels[rng.choice(len(levels), p=w / w.sum())]


def draw(anchor):
    s = ANCHORS[anchor]
    sampled = s.decoder != 'quantile'
    corrected = s.pilot_method in ('corrected', 'two_stage')
    kw = {}
    # Training vintages.
    kw['vintage_seasons'] = pick(['latest', 'all'])
    kw['actual_share'] = pick([0., .25, .5, 1.], [3, 1, 1, 1])
    kw['reporting_strength'] = pick([1., .5, 1.5, 2.], [3, 1, 2, 2])
    kw['reporting_random_strength'] = pick([False, True])
    kw['reporting_method'] = pick(['local_log', 'synchronous_log'], [3, 1])
    kw['reporting_missingness'] = pick([True, False], [3, 1])
    kw['revision_signals'] = pick(['admissions', 'all'], [3, 1])
    if corrected:
        kw['correction_realizations'] = pick([1, 2, 4], [2, 1, 1])
        kw['uncorrected_share'] = pick([0., .1, .25], [3, 1, 1])
    # Nowcasting.
    kw['nowcast_noise'] = pick([0., .5, 1.], [2, 1, 1])
    if corrected and kw['nowcast_noise']:
        kw['nowcast_noise_train'] = pick([False, True])
    kw['correction_penalty'] = pick([s.correction_penalty, 30., 100., 300.], [2, 1, 1, 1])
    kw['correction_weeks'] = pick([2, 1, 3], [4, 1, 1])
    kw['correction_strength'] = pick([1., .75], [3, 1])
    if not corrected:
        kw['pilot_nowcaster'] = pick([s.pilot_nowcaster, 'real_tree'], [2, 1])
    # Formulation.
    kw['growth_anchor'] = pick([False, True])
    kw['log_loss_weight'] = pick([0., .25, .5, 1.], [3, 1, 1, 1])
    if s.covariate_set:
        kw['covariate_dropout'] = pick([0., .1, .25], [2, 1, 1])
    if pick([False, True], [3, 1]):
        kw.update(ili_training='pretrain', ili_units='flu_scaled', ili_steps=1200)
    if sampled:
        kw['sum_wis_weight'] = pick([s.sum_wis_weight, .1, .5] if s.sum_wis_weight else [0., .1], [3, 1, 1] if s.sum_wis_weight else [4, 1])
    # Untested or edge knobs.
    kw['latent'] = pick([16, 8, 32], [3, 1, 1])
    kw['batch_size'] = pick([8, 4, 16], [3, 1, 1])
    kw['lr'] = float(s.lr * pick([1., .5, 2.], [2, 1, 1]))
    kw['weight_decay'] = pick([0., 3e-5, 1e-4], [3, 1, 1])
    kw['signal_features'] = pick(['none', 'multiscale'], [3, 1])
    kw['spatial'] = pick([s.spatial, 'distance'], [2, 1])
    kw['lookback'] = pick([12, 10, 14], [3, 1, 1])
    kw['width'] = pick([96, 64, 128], [3, 1, 1])
    if sampled:
        kw['noise'] = pick(['global', 'local'], [3, 1])
        kw['us_error'] = pick(['none', 'shared_factor'], [3, 1])
        kw['members'] = pick([128, 64, 256], [3, 1, 1])
        kw['decoder'] = pick(['legacy', 'residual2'], [4, 1])
    return replace(s, **{k: (v.item() if hasattr(v, 'item') else v) for k, v in kw.items()})


rows, seen = [], set()
for anchor, s in ANCHORS.items():  # the unmodified leaders, refitted under the same code snapshot
    rows.append(dict(anchor=anchor, kind='anchor', scenario=s.scenario_string)); seen.add(s.scenario_string)
while len(rows) < args.configurations:
    anchor = pick(['A', 'B', 'Bs'], [2, 2, 1])
    try:
        s = draw(anchor)
    except ValueError:
        continue
    if s.scenario_string not in seen:
        seen.add(s.scenario_string); rows.append(dict(anchor=anchor, kind='random', scenario=s.scenario_string))
if args.smoke:
    chosen = [0, 1, 2] + list(range(3, len(rows), max(1, len(rows) // 21)))[:21]
    rows = [dict(rows[i], scenario=replace(Scenario.from_string(rows[i]['scenario']), epochs=2, patience=1,
                                            ili_steps=3).scenario_string) for i in chosen]
seeds = ['42'] if args.smoke else ['42', '43']
folder = Path('docs/experiments') / name; folder.mkdir(parents=True, exist_ok=True)
(folder / 'design.json').write_text(json.dumps(dict(configurations=len(rows), seeds=[int(x) for x in seeds],
                                                    design_seed=20261007, rows=rows), indent=2) + '\n')
(folder / 'scenarios.txt').write_text('\n'.join(r['scenario'] for r in rows) + '\n')
print(f'{len(rows)} configurations x {len(seeds)} seeds = {len(rows) * len(seeds)} runs')
if args.plan:
    subprocess.run([sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', name, '-s', *[r['scenario'] for r in rows],
                    '--seeds', *seeds, '--device', 'cuda'], check=True)
