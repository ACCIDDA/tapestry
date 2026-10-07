"""Production fits for the 2026-27 submission: A and B (exact B4 recipes), seeds 44-48, trained on all
four seasons 2022-23 through 2025-26 (epochs chosen on internal validation weeks, then refitted), with
finalized next-four-week labels. The 2026-27 forecasts written by the fit are an unscored sanity check;
operational forecasts reuse the saved checkpoints and correction trees. Submission ensemble: equal
weight per recipe, quantile averaging over all ten runs (B4.polish)."""
import argparse, json, subprocess, sys
from dataclasses import replace
from pathlib import Path
from tapestry.model.scenario import Scenario

p = argparse.ArgumentParser(); p.add_argument('--plan', action='store_true'); args = p.parse_args()
name = 'b4-production-20261007'
heads = Path('docs/experiments/b4-flu-heads-20261006/scenarios.txt').read_text().split('\n')
rows = [dict(recipe=r, scenario=replace(Scenario.from_string(heads[i]), evaluation_seasons='production').scenario_string)
        for r, i in (('A', 0), ('B', 6))]
seeds = ['44', '45', '46', '47', '48']
folder = Path('docs/experiments') / name; folder.mkdir(parents=True, exist_ok=True)
(folder / 'design.json').write_text(json.dumps(dict(configurations=len(rows), seeds=[int(s) for s in seeds], rows=rows), indent=2) + '\n')
(folder / 'scenarios.txt').write_text('\n'.join(r['scenario'] for r in rows) + '\n')
print(f'{len(rows)} configurations x {len(seeds)} seeds = {len(rows) * len(seeds)} runs')
if args.plan:
    subprocess.run([sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', name, '-s', *[r['scenario'] for r in rows],
                    '--seeds', *seeds, '--device', 'cuda'], check=True)
