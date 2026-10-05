"""Paired week-block uncertainty with the shared location-relative WIS hierarchy."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from tapestry.dataset.build import load
from tapestry.data.geography import STATE_FIPS
from tapestry.evaluation.hubs import CHANNEL,KEY
from tapestry.evaluation.totals import forecast_cells,TARGET_WEIGHTS
from tapestry.experiment.finalization import reporting_support
from tapestry.model.scenario import Scenario
p=argparse.ArgumentParser();p.add_argument('-e','--experiment',required=True)
p.add_argument('--strength',type=float,required=True);p.add_argument('--uncertainty',type=float,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=Path('data/experiments')/a.experiment
rank=max(root.glob('ranking-*'),key=lambda x:x.stat().st_mtime)
runs=json.loads((rank/'manifest.json').read_text())['runs']
settings=json.loads((root/'experiment.json').read_text());panel=load(settings['dataset'])
frames={};configs={};keys=None
for r in runs:
    s=Scenario.from_string(r['config_id'])
    baseline=s.replay_inputs=='nowcast' and s.replay_nowcaster=='selected' and s.replay_uncertainty==0
    candidate=s.replay_inputs=='nowcast' and s.replay_nowcaster=='context_residual' and s.replay_strength==a.strength and s.replay_uncertainty==a.uncertainty
    if not (baseline or candidate):continue
    f=forecast_cells(r['path'],settings['frozen']);f=f[f.season.eq('2025-2026')].reset_index(drop=True)
    if keys is None:
        keys=f[['target',*KEY]]
        support=reporting_support(panel,pd.DataFrame(dict(signal=f.target.map({n:str(panel['target_names'][k]) for n,k in CHANNEL.items()}),
            location=f.location.map(STATE_FIPS|{'US':'US'}),
            issuance=(pd.to_datetime(f.reference_date)-pd.Timedelta(days=3)).dt.strftime('%Y-%m-%d'),
            boundary=(pd.to_datetime(f.reference_date)-pd.Timedelta(days=7)).dt.strftime('%Y-%m-%d'))))
    elif not keys.equals(f[['target',*KEY]]):raise ValueError('Bootstrap forecast support differs')
    key=('candidate' if candidate else 'control',r['seed'])
    frames[key]=f;configs[key]=r['config_id']
    print('Loaded',s.replay_strength,s.replay_uncertainty,r['seed'],flush=True)
expected=pd.read_csv(a.output/'forecast-seed-scores.csv');records=[]
for stratum,keep in [('complete_history_12',support.complete_history_12),('complete_and_uninterrupted',support.complete_history_12&support.uninterrupted_8)]:
    sample=next(iter(frames.values()))[keep]
    calendar=sorted(next(iter(frames.values())).reference_date.unique())
    groups=pd.MultiIndex.from_frame(sample[['target','location']].drop_duplicates())
    weights=np.zeros(len(groups))
    for target in groups.get_level_values('target').unique():
        t=groups.get_level_values('target')==target;us=groups.get_level_values('location')=='US'
        if (t&~us).any():weights[t&~us]=.8/(t&~us).sum()
        if (t&us).any():weights[t&us]=.2/(t&us).sum()
        weights[t]*=TARGET_WEIGHTS[target]/weights[t].sum()
    weights/=weights.sum()
    def matrix(f,column):
        return f[keep].groupby(['reference_date','target','location'])[column].sum().unstack(['target','location']).reindex(index=calendar,columns=groups).fillna(0).to_numpy()
    ensemble=matrix(next(iter(frames.values())),'ensemble_wis')
    n=len(calendar);rng=np.random.default_rng(42)
    draws=((rng.integers(0,n,(1000,int(np.ceil(n/4))))[:,:,None]+np.arange(4))%n).reshape(1000,-1)[:,:n]
    denominator=ensemble[draws].sum(1);valid=(denominator>0).all(1);denominator=denominator[valid]
    scores={};observed={}
    for key,f in frames.items():
        numerator=matrix(f,'model_wis')
        scores[key]=(numerator[draws[valid]].sum(1)/denominator)@weights
        observed[key]=float((numerator.sum(0)/ensemble.sum(0))@weights)
        match=expected[expected.config_id.eq(configs[key]) & expected.seed.eq(key[1]) & expected.season.eq('2025-2026') & expected.geography.eq('all') & expected.stratum.eq(stratum)]
        if len(match)!=1 or not np.isclose(observed[key],match.combined.iloc[0],atol=1e-10,rtol=1e-10):
            raise ValueError('Bootstrap hierarchy differs from shared scorer')
    seeds=sorted(s for label,s in frames if label=='control')
    delta=np.mean([100*(scores['candidate',s]/scores['control',s]-1) for s in seeds],axis=0)
    records.append(dict(stratum=stratum,strength=a.strength,uncertainty=a.uncertainty,
        change_pct=np.mean([100*(observed['candidate',s]/observed['control',s]-1) for s in seeds]),
        low=np.quantile(delta,.025),high=np.quantile(delta,.975),valid_draws=int(valid.sum()),block_weeks=4))
r=pd.DataFrame(records);r.to_csv(a.output/'forecast-bootstrap.csv',index=False);print(r.to_string(index=False))
