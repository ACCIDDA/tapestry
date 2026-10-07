"""B5 confirmation: the anchors, anchors plus the B5 options with clear helpful main effects, and the
two best B5 configurations per anchor, retrained on new seeds 44-46.

Constructed variants come from the B5 regression (scripts/analyze_b5.py, 398 configurations, seeds
42/43): training on actual archived reports (`actual_share`), the extra log-scale admissions loss
(`log_loss_weight`, helpful for A), batch size 16; for Bs also tree penalty 300 and random error
strength. All options were selected on the same two evaluation seasons, so new seeds test seed luck,
not new seasons. Three seeds (not five) to finish before the 7 October 9:00 deadline."""
import argparse, json, subprocess, sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
from tapestry.model.scenario import Scenario

p = argparse.ArgumentParser(); p.add_argument('--plan', action='store_true'); args = p.parse_args()
name = 'b5-confirm-top-20261007'
heads = Path('docs/experiments/b4-flu-heads-20261006/scenarios.txt').read_text().split('\n')
A, B, Bs = (Scenario.from_string(heads[i]) for i in (0, 6, 5))
built = {
    'A': A, 'A+actual0.5': replace(A, actual_share=.5), 'A+log0.5': replace(A, log_loss_weight=.5),
    'A+actual0.5+log0.5+batch16': replace(A, actual_share=.5, log_loss_weight=.5, batch_size=16),
    'B': B, 'B+actual1': replace(B, actual_share=1.), 'B+actual1+batch16': replace(B, actual_share=1., batch_size=16),
    'Bs': Bs, 'Bs+actual0.5+log0.25+random+penalty300': replace(Bs, actual_share=.5, log_loss_weight=.25,
                                                               reporting_random_strength=True, correction_penalty=300.),
}
rows = [dict(anchor=k.split('+')[0], role='anchor' if '+' not in k else 'constructed', label=k, scenario=s.scenario_string)
        for k, s in built.items()]
t = pd.read_csv('docs/experiments/b5-explore-20261006/analysis/configurations.csv')
for anchor in ('A', 'B', 'Bs'):
    for r in t[(t.anchor == anchor) & (t.kind == 'random')].nsmallest(2, 'flu_native').itertuples():
        if r.config_id not in {x['scenario'] for x in rows}:
            rows.append(dict(anchor=anchor, role='best_b5', label=f'{anchor} B5 best {r.flu_native:.4f}', scenario=r.config_id))
seeds = ['44', '45', '46']
folder = Path('docs/experiments') / name; folder.mkdir(parents=True, exist_ok=True)
(folder / 'design.json').write_text(json.dumps(dict(configurations=len(rows), seeds=[int(s) for s in seeds], rows=rows), indent=2) + '\n')
(folder / 'scenarios.txt').write_text('\n'.join(r['scenario'] for r in rows) + '\n')
print(pd.DataFrame(rows)[['anchor', 'role', 'label']].to_string(index=False))
print(f'{len(rows)} configurations x {len(seeds)} seeds = {len(rows) * len(seeds)} runs')
if args.plan:
    subprocess.run([sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', name, '-s', *[r['scenario'] for r in rows],
                    '--seeds', *seeds, '--device', 'cuda'], check=True)
