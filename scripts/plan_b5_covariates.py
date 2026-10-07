"""B5.covariates: matched comparison of covariate representations on the B4 leaders.

Recipes: A (Kinsa), A with Kinsa + ILINet, B with Kinsa added, all otherwise exactly the B4
scenarios. Representations: covariate_encoder summary (current) / smooth (3-week trailing mean of
the raw history) / shared (learned 4-dim encoder) / growth (new: level, 1/2-week log growth,
acceleration at the newest report), each with and without smoothed multiscale 3/6/12-week
level/slope/curvature features (signal_features; these also apply to target histories). B without
covariates, with and without the multiscale features, is the reference. Labels finalized; B4 folds."""
import argparse, json, subprocess, sys
from dataclasses import replace
from pathlib import Path
from tapestry.model.scenario import Scenario

p = argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true'); p.add_argument('--plan', action='store_true')
args = p.parse_args()
name = 'b5-covariates-check-20261006' if args.smoke else 'b5-covariates-20261006'
heads = Path('docs/experiments/b4-flu-heads-20261006/scenarios.txt').read_text().split('\n')
A, B = Scenario.from_string(heads[0]), Scenario.from_string(heads[6])
recipes = {'A_kinsa': A, 'A_kinsa_ilinet': replace(A, covariate_set='kinsa+ilinet'), 'B_kinsa': replace(B, covariate_set='kinsa')}
rows = []
for recipe, base in recipes.items():
    for encoder in ('summary', 'smooth', 'shared', 'growth'):
        for signal in ('none', 'smooth_multiscale'):
            rows.append(dict(recipe=recipe, covariate_encoder=encoder, signal_features=signal,
                             scenario=replace(base, covariate_encoder=encoder, signal_features=signal).scenario_string))
for signal in ('none', 'smooth_multiscale'):
    rows.append(dict(recipe='B_no_covariates', covariate_encoder='none', signal_features=signal,
                     scenario=replace(B, signal_features=signal).scenario_string))
assert len({r['scenario'] for r in rows}) == len(rows) == 26
if args.smoke:
    rows = [dict(r, scenario=replace(Scenario.from_string(r['scenario']), epochs=2, patience=1).scenario_string) for r in rows]
seeds = ['42'] if args.smoke else ['42', '43', '44']
folder = Path('docs/experiments') / name; folder.mkdir(parents=True, exist_ok=True)
(folder / 'design.json').write_text(json.dumps(dict(configurations=len(rows), seeds=[int(s) for s in seeds], rows=rows), indent=2) + '\n')
(folder / 'scenarios.txt').write_text('\n'.join(r['scenario'] for r in rows) + '\n')
print(f'{len(rows)} configurations x {len(seeds)} seeds = {len(rows) * len(seeds)} runs')
if args.plan:
    subprocess.run([sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', name, '-s', *[r['scenario'] for r in rows],
                    '--seeds', *seeds, '--device', 'cuda'], check=True)
