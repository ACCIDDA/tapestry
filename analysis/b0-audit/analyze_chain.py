"""Consecutive matched B1-to-B0 effects; no comparison baseline branches."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from tapestry.experiment.planner import read_jobs,seed_state
from tapestry.evaluation.totals import rank

out=Path('docs/results/b1-to-b0-chain');out.mkdir(parents=True,exist_ok=True)
labels={'00':'B1-derived baseline','01':'Restore input normalization','02':'Remove artificial masking','03':'Remove finality indicator','03b':'Restore forecast inputs only','04':'Restore training histories','05':'Restore B0 validation weeks','06':'Restore B0 validation randomness','07':'Restore B0 training-error weight','09':'Restore B0 loss representation'}
experiments={stage:f'b1-b0-score-{stage}' for stage in labels}
runs=[];progress=[];endpoints={}
for stage,name in experiments.items():
 root=Path('data/experiments')/name
 job=next(j for j in read_jobs(root) if stage!='04' or 'input_mode=finalized_available' not in j['scenario'])
 for seed in [42,43,44]:
  attempt,record,complete=seed_state(root,job['scenario'],seed)
  progress.append(dict(stage=stage,label=labels[stage],experiment=name,seed=seed,status=record['status'],complete=complete,folds=len(list(attempt.glob('eval_*/forecasts.npz'))) if attempt else 0,path=str(attempt)))
  if complete:runs.append(dict(config_id=f'stage{stage}',seed=seed,path=attempt))
  if stage=='09' and attempt:endpoints[seed]=attempt
pd.DataFrame(progress).to_csv(out/'progress.csv',index=False)
print(pd.DataFrame(progress).groupby(['stage','label','status']).size().to_string(),flush=True)
# Compare completed endpoint folds without waiting for the other stages.
equality=[]
for seed,attempt in endpoints.items():
 root=Path('data/experiments')/('b0-exact-reproduction-l40' if seed==44 else 'b0-training-code-reference')
 job=next(j for j in read_jobs(root) if 'fit_partition=pathogen' in j['scenario']);reference,_,_=seed_state(root,job['scenario'],seed)
 for season in ['2023-2024','2024-2025','2025-2026']:
  f=attempt/f'eval_{season}/forecasts.npz';g=reference/f'eval_{season}/forecasts.npz'
  if not f.exists():continue
  a,b=np.load(f),np.load(g)
  for key in ['truth','mask','context_end','target_dates','locations','quantile_levels']:np.testing.assert_equal(a[key],b[key])
  q0,q1=a['quantiles'],b['quantiles']
  equality.append(dict(seed=seed,season=season,equal_forecasts=bool(np.array_equal(q0,q1)),max_abs_difference=float(abs(q0-q1).max()),reference=str(reference),endpoint=str(attempt)))
  a.close();b.close()
if equality:
 pd.DataFrame(equality).to_csv(out/'endpoint_verification.csv',index=False);print(pd.DataFrame(equality)[['seed','season','equal_forecasts','max_abs_difference']].to_string(index=False),flush=True)
if not all(r['complete'] for r in progress):raise SystemExit(0)
ranking=rank(runs,out)
scores=pd.read_csv(out/'run_scores.csv');seasons=pd.read_csv(out/'season_composite_scores.csv');rows=[];paired=[]
for stage,label in labels.items():
 s=scores.loc[(scores.config_id==f'stage{stage}')&(scores.geography=='all')].set_index('seed').combined
 recent=seasons.loc[(seasons.config_id==f'stage{stage}')&(seasons.geography=='all')&(seasons.season=='2025-2026')].combined
 assert len(s)==3 and len(recent)==3
 row=dict(stage=stage,change=label,mean=s.mean(),sd=s.std(),season2025_mean=recent.mean(),delta=np.nan,improved_seeds=0)
 if stage!='00':
  preceding=list(labels)[list(labels).index(stage)-1]
  previous=scores.loc[(scores.config_id==f'stage{preceding}')&(scores.geography=='all')].set_index('seed').combined
  delta=s-previous;row.update(delta=delta.mean(),improved_seeds=int((delta<0).sum()))
  paired.extend(dict(stage=stage,seed=int(seed),before=float(previous[seed]),after=float(s[seed]),delta=float(delta[seed])) for seed in s.index)
 rows.append(row)
pd.DataFrame(rows).to_csv(out/'chain_table.csv',index=False);pd.DataFrame(paired).to_csv(out/'chain_paired_effects.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
(out/'completion.json').write_text(json.dumps(dict(complete_runs=len(runs),complete_folds=len(runs)*3,endpoint_verified=len(equality)==9 and all(r['equal_forecasts'] for r in equality)),indent=2)+'\n')
