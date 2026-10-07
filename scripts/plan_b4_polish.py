"""B4.polish: confirm the submission candidates on new seeds and test shorter training windows.

Recipes (exact B4 scenarios): A, B (direct quantiles), and B's architecture trained on unchanged
finalized histories ("B-finalized"). Each is fitted with seeds 44-48 under
  training_window=all     fold 2025-26: trained on 2022-23, 2023-24, 2024-25
                          fold 2024-25: trained on 2022-23, 2023-24, 2025-26
  training_window=recent2 fold 2025-26: trained on 2023-24, 2024-25 only
                          fold 2024-25: trained on 2022-23, 2023-24 only (no later season)
Every run also writes stress views of the same fitted model (`stress_views`): newest flu-admission
week missing at the deadline, and (A) every covariate missing. Labels stay finalized."""
import argparse, json, subprocess, sys
from dataclasses import replace
from pathlib import Path
from tapestry.model.scenario import Scenario

p = argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true'); p.add_argument('--plan', action='store_true')
args = p.parse_args()
name = 'b4-polish-check-20261006' if args.smoke else 'b4-polish-20261006'
heads = Path('docs/experiments/b4-flu-heads-20261006/scenarios.txt').read_text().split('\n')
A, B = Scenario.from_string(heads[0]), Scenario.from_string(heads[6])
recipes = {'A': A, 'B': B, 'B_finalized': replace(B, pilot_method='finalized')}
rows = [dict(recipe=r, training_window=w, scenario=replace(s, training_window=w, stress_views=True).scenario_string)
        for r, s in recipes.items() for w in ('all', 'recent2')]
if args.smoke:
    rows = [dict(r, scenario=replace(Scenario.from_string(r['scenario']), epochs=2, patience=1).scenario_string) for r in rows]
seeds = ['44'] if args.smoke else ['44', '45', '46', '47', '48']
folder = Path('docs/experiments') / name; folder.mkdir(parents=True, exist_ok=True)
(folder / 'design.json').write_text(json.dumps(dict(configurations=len(rows), seeds=[int(s) for s in seeds], rows=rows), indent=2) + '\n')
(folder / 'scenarios.txt').write_text('\n'.join(r['scenario'] for r in rows) + '\n')
print(f'{len(rows)} configurations x {len(seeds)} seeds = {len(rows) * len(seeds)} runs')
if args.plan:
    subprocess.run([sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', name, '-s', *[r['scenario'] for r in rows],
                    '--seeds', *seeds, '--device', 'cuda'], check=True)
