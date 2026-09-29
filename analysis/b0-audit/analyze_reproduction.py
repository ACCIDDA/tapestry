"""Verify fresh B0 forecasts against historical forecasts, then compare controlled arms."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from tapestry.experiment.planner import read_jobs,seed_state
from tapestry.evaluation.totals import rank

out=Path('docs/results/b0-reproduction');out.mkdir(parents=True,exist_ok=True)
selected=json.loads(Path('data/audits/b0/selected.json').read_text())
rows=[];runs=[];missing=[]
for item in selected:
    label,seed=item['label'],item['seed']
    original=Path(item['path'])
    original_run=json.loads((original.parent/'run.json').read_text())
    l40='jles01' in original_run['host']
    experiment=Path('data/experiments')/('b0-exact-reproduction-l40' if l40 else 'b0-exact-reproduction')
    partition='target' if label=='b0-target100' else 'pathogen'
    job=next(j for j in read_jobs(experiment) if f'fit_partition={partition}' in j['scenario'])
    attempt,record,complete=seed_state(experiment,job['scenario'],seed)
    if not complete:missing.append(f'{label}/s{seed}')
    if attempt is None:continue
    original_manifest=json.loads((original/'manifest.json').read_text())
    for held in ('2023-2024','2024-2025','2025-2026'):
        f=attempt/'legacy'/f'eval_{held}/forecasts.npz'
        training=attempt/'legacy'/f'eval_{held}/training.json'
        if not f.exists() or not training.exists():continue
        before=np.load(original/f'eval_{held}/forecasts.npz');after=np.load(f)
        for key in ('quantile_levels','truth','mask','context_end','target_dates','locations'):
            assert np.array_equal(before[key],after[key]),(label,seed,held,key)
        q0,q1=before['quantiles'],after['quantiles']
        oldfold=next(v for v in original_manifest['folds'] if v['eval_season']==held)
        newfold=json.loads(training.read_text())
        rows.append(dict(config_id=label,seed=seed,season=held,gpu='L40' if l40 else 'H100',
                         exact_quantiles=bool(np.array_equal(q0,q1)),byte_identical_quantiles=bool(q0.dtype==q1.dtype and q0.tobytes()==q1.tobytes()),max_abs_difference=float(np.max(np.abs(q0-q1))),
                         quantile_values=int(q0.size),old_epochs=json.dumps(oldfold['epochs']),new_epochs=json.dumps(newfold['epochs']),
                         original_path=str(original),reproduction_path=str(attempt)))
        before.close();after.close()
    if complete:
        runs.append(dict(config_id='reproduced-'+label,seed=seed,path=attempt))
        runs.append(dict(config_id='historical-'+label,seed=seed,path=original))
        if label=='b0-pathogen300' and seed==44:
            runs.append(dict(config_id='old-training-l40',seed=seed,path=attempt))
reference=Path('data/experiments/b0-training-code-reference')
for job in read_jobs(reference):
    for seed in job['seeds']:
        attempt,record,complete=seed_state(reference,job['scenario'],seed)
        if not complete:missing.append(f'old-training-l40/s{seed}')
        else:runs.append(dict(config_id='old-training-l40',seed=seed,path=attempt))
for name in ('b0-current-training-control','b0-current-training-normalized'):
    experiment=Path('data/experiments')/name
    for job in read_jobs(experiment):
        label=('current-unscaled-finalized' if name.endswith('-control') else
               'current-normalized-wednesday' if 'input_mode=finalized_available' in job['scenario'] else
               'current-normalized-finalized')
        for seed in job['seeds']:
            attempt,record,complete=seed_state(experiment,job['scenario'],seed)
            if not complete:missing.append(f'{label}/s{seed}')
            else:runs.append(dict(config_id=label,seed=seed,path=attempt))
pd.DataFrame(rows).to_csv(out/'forecast_equality.csv',index=False)
if rows:
    print(pd.DataFrame(rows).groupby('config_id').agg(folds=('season','size'), exact=('exact_quantiles','sum'), max_error=('max_abs_difference','max')).to_string(),flush=True)
if missing:
    print('Pending runs:',len(missing),flush=True)
else:
    equality=pd.DataFrame(rows)
    assert len(equality)==18 and not equality.duplicated(['config_id','seed','season']).any()
    assert equality.exact_quantiles.all() and equality.byte_identical_quantiles.all()
    assert equality.max_abs_difference.eq(0).all()
    assert all(json.loads(row.old_epochs)==json.loads(row.new_epochs) for row in equality.itertuples())
    r=rank(runs,out);print(r[['config_id','combined_mean','combined_sd']].to_string(index=False))
    scores=pd.read_csv(out/'run_scores.csv')
    pairs=[]
    for effect,before,after in (
        ('training-code','old-training-l40','current-unscaled-finalized'),
        ('normalization','current-unscaled-finalized','current-normalized-finalized'),
        ('availability','current-normalized-finalized','current-normalized-wednesday')):
        paired=scores[scores.config_id.eq(before)].merge(
            scores[scores.config_id.eq(after)],on=['seed','geography'],suffixes=('_before','_after'),validate='one_to_one')
        assert len(paired)==9 and set(paired.seed)=={42,43,44}
        paired['effect']=effect
        paired['delta']=paired.combined_after-paired.combined_before
        pairs.append(paired)
    paired=pd.concat(pairs,ignore_index=True)
    paired.to_csv(out/'paired_seed_deltas.csv',index=False)
    summary=paired.groupby(['effect','geography'],sort=False).agg(
        before=('combined_before','mean'),after=('combined_after','mean'),
        delta_mean=('delta','mean'),delta_sd=('delta','std'),
        seeds_improved=('delta',lambda x:int((x<0).sum())))
    summary.to_csv(out/'paired_effect_summary.csv')
    print(summary.to_string())
