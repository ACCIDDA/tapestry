"""Matched output-head comparison on the native and log leaders of the full flu sweep."""
import argparse,json,subprocess,sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
from tapestry.model.scenario import Scenario
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--plan',action='store_true');args=p.parse_args()
name='b4-flu-heads-check-20261006' if args.smoke else 'b4-flu-heads-20261006'
d=pd.read_csv('docs/experiments/b4-flu-600-20261005/final-analysis/pilot-rankings.csv')
d=d[(d['count']==2)&(d.season=='equal_season_mean')]
rows=[]
for metric in ['flu_native','flu_admissions_log']:
 s=Scenario.from_string(d[d.metric==metric].sort_values('mean').iloc[0].config_id)
 for label,decoder,weight in [('samples','legacy',0.),('samples_sum_wis','legacy',.25),('quantiles','quantile',0.),('compact_quantiles','quantile_small',0.)]:
  scenario=replace(s,decoder=decoder,sum_wis_weight=weight)
  if args.smoke:scenario=replace(scenario,epochs=2,patience=1)
  rows.append(dict(anchor=metric,variant=label,scenario=scenario.scenario_string))
folder=Path('docs/experiments')/name;folder.mkdir(exist_ok=True)
(folder/'design.json').write_text(json.dumps(dict(configurations=8,seeds=[42] if args.smoke else [42,43],rows=rows),indent=2)+'\n')
(folder/'scenarios.txt').write_text('\n'.join(r['scenario'] for r in rows)+'\n')
if args.plan:subprocess.run([sys.executable,'-m','tapestry.experiment.planner','plan','-e',name,'-s',*[r['scenario'] for r in rows],'--seeds',*(['42'] if args.smoke else ['42','43']),'--device','cuda'],check=True)
