"""B4.refineTop2: training-history treatments and four-week-total loss weights on the two B4 leaders.

Item 4: each leader's architecture/optimizer/head is crossed with the six training-history
treatments of the 600-configuration sweep. Item 5: the sampled leaders are retrained with
several weights on the four-week flu-admission total WIS loss. Every run also produces the
training-season-calibrated view (item 3, `pilot.calibrate`)."""
import argparse, json, subprocess, sys
from dataclasses import replace
from pathlib import Path
from tapestry.model.scenario import Scenario

p = argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true'); p.add_argument('--plan', action='store_true')
args = p.parse_args()
name = 'b4-refinetop2-check-20261006' if args.smoke else 'b4-refinetop2-20261006'
heads = Path('docs/experiments/b4-flu-heads-20261006/scenarios.txt').read_text().split('\n')
A = Scenario.from_string(heads[0])          # samples, real-tree-corrected synthetic training histories, Kinsa
B = Scenario.from_string(heads[6])          # direct quantiles, artificial errors + reconstruction, neighbors
B_samples = Scenario.from_string(heads[4])  # B with the sampled head
assert A.pilot_method == 'two_stage' and A.decoder == 'legacy' and A.sum_wis_weight == 0
assert B.pilot_method == 'joint' and B.decoder == 'quantile' and B_samples.decoder == 'legacy' and B_samples.sum_wis_weight == 0

TREATMENTS = {  # the six training-history treatments of b4-flu-600-20261005
    'synthetic_corrected': dict(pilot_method='corrected', pilot_nowcaster='synthetic_tree'),
    'real_corrected': dict(pilot_method='two_stage', pilot_nowcaster='real_tree'),
    'old_errors_recent_reports': dict(pilot_method='errors', revision_scope='early_actual'),
    'reported': dict(pilot_method='reported'),
    'errors_reconstruction': dict(pilot_method='joint', joint_weight=B.joint_weight, nowcast_weeks=B.nowcast_weeks),
    'finalized': dict(pilot_method='finalized'),
}
rows = []
for recipe, base in (('A', A), ('B', B)):
    for treatment, change in TREATMENTS.items():
        rows.append(dict(item='4_treatment', recipe=recipe, treatment=treatment, sum_wis_weight=0.,
                         scenario=replace(base, **change).scenario_string))
for recipe, base, weights in (('A', A, [0., .05, .1, .25]), ('B_samples', B_samples, [0., .1, .25, .5, .75, 1.])):
    for w in weights:
        rows.append(dict(item='5_sum_wis', recipe=recipe, treatment='original', sum_wis_weight=w,
                         scenario=replace(base, sum_wis_weight=w).scenario_string))
# A with weight 0 is also A x real_corrected: keep one row per scenario, record both roles.
unique = {}
for r in rows:
    unique.setdefault(r['scenario'], dict(r, roles=[]))['roles'].append(r['item'])
rows = list(unique.values())
assert len(rows) == 21, len(rows)
assert A.scenario_string in unique and B.scenario_string in unique and B_samples.scenario_string in unique
if args.smoke:
    rows = [dict(r, scenario=replace(Scenario.from_string(r['scenario']), epochs=2, patience=1).scenario_string) for r in rows]
seeds = ['42'] if args.smoke else ['42', '43']
folder = Path('docs/experiments') / name; folder.mkdir(parents=True, exist_ok=True)
(folder / 'design.json').write_text(json.dumps(dict(configurations=len(rows), seeds=[int(s) for s in seeds], rows=rows), indent=2) + '\n')
(folder / 'scenarios.txt').write_text('\n'.join(r['scenario'] for r in rows) + '\n')
print(f'{len(rows)} configurations x {len(seeds)} seeds = {len(rows) * len(seeds)} runs')
if args.plan:
    subprocess.run([sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', name, '-s', *[r['scenario'] for r in rows],
                    '--seeds', *seeds, '--device', 'cuda'], check=True)
