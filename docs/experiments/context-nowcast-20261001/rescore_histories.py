"""Re-score saved nowcaster history distributions with candidate-independent scales.

Read cached point forecasts and the original diagnostic support. Write a separate
diagnostic table; never modify original forecasts. Preserve the original draw count.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tapestry.dataset.build import load
from tapestry.model.scenario import Scenario
from tapestry.model.revision_uncertainty import RevisionUncertainty,trajectory_distribution_scores
from tapestry.experiment.provenance import sha256
import tapestry.model.revision_uncertainty as implementation
p=argparse.ArgumentParser();p.add_argument('-e','--experiment',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=Path('data/experiments')/a.experiment;a.output.mkdir(parents=True,exist_ok=True)
settings=json.loads((root/'experiment.json').read_text());panel=load(settings['dataset'])
rank=max(root.glob('ranking-*'),key=lambda x:x.stat().st_mtime)
runs=json.loads((rank/'manifest.json').read_text())['runs']
reference=Scenario.from_string(json.loads((root/'replay-source.json').read_text())['nowcaster_reference']['scenario'])
frames=[];sources=[]
for run in runs:
    s=Scenario.from_string(run['config_id'])
    if run['seed']!=42 or not s.replay_uncertainty:continue
    matches=[]
    for cache in root.glob('replay-nowcasts*.npz'):
        with np.load(cache) as saved:
            metadata=json.loads(str(saved['metadata']))
            expected=dict(mode='adaptive_chain' if s.replay_nowcaster=='selected' else s.replay_nowcaster,
                strength=s.replay_strength,penalty=s.replay_penalty,features_mode=s.replay_features,gate=s.replay_gate)
            if all(metadata.get(k,1. if k=='strength' else None)==v for k,v in expected.items()):matches.append(cache)
    if len(matches)!=1:raise ValueError('Need exact original point-cache match')
    with np.load(matches[0]) as saved:pred=saved['predictions']
    bank=RevisionUncertainty(panel,pred)
    arm='Seasonal control' if s.replay_nowcaster=='selected' else 'Context residual'
    if s.replay_nowcaster=='context_residual' and s.replay_strength!=reference.finalization_strength:arm+=f' ×{s.replay_strength:g}'
    arm+=f' + uncertainty {s.replay_uncertainty:g}'
    for file in Path(run['path']).glob('eval_*/trajectory-distributions.csv.gz'):
        original=pd.read_csv(file);records=[]
        members=json.loads((file.parent/'revision-uncertainty.json').read_text())['empirical_members']
        for issue,g in original.groupby('issuance',sort=True):
            wi=int(np.searchsorted(panel['issuance_dates'].astype(str),issue))
            end=np.datetime64(issue)-np.timedelta64(4,'D')
            ti=np.searchsorted(panel['dates'].astype('datetime64[D]'),end)-np.arange(s.lookback-1,-1,-1)
            raw=panel['asof_targets'][wi,ti];center=raw.copy();center[-8:]=pred[wi,::-1]
            if s.replay_schedule:center=np.where(np.isfinite(raw),center,panel['targets'][ti])
            center=np.nan_to_num(np.moveaxis(center,-1,-2)).astype(np.float32)
            draws,scale,_=bank.draw(issue,center,members,s.replay_uncertainty)
            truth=np.moveaxis(panel['targets'][ti],-1,-2)
            result=trajectory_distribution_scores(draws,truth,center,scale)
            k=g.signal.map({n:i for i,n in enumerate(panel['target_names'])}).to_numpy()
            l=g.location.map({n:i for i,n in enumerate(panel['locations'])}).to_numpy()
            updated=g.copy()
            for name,array in result.items():
                values=array[k,l]
                if name.endswith('coverage80') and not np.array_equal(values,g[name].to_numpy(),equal_nan=True):
                    raise ValueError('History draws or diagnostic support changed')
                updated[name]=values
            records.append(updated)
        f=pd.concat(records);f['season']=file.parent.name.removeprefix('eval_');f['arm']=arm
        f['geography']=np.where(f.location.eq('US'),'US','states');frames.append(f)
        sources.append(dict(diagnostics=str(file),diagnostics_sha256=sha256(file),cache=str(matches[0]),cache_sha256=sha256(matches[0]),members=members))
    print('Rescored',arm,flush=True)
if not frames:
    print('No history distributions to re-score')
    raise SystemExit(0)
d=pd.concat(frames,ignore_index=True)
cells_folder=root/'history-diagnostics';cells_folder.mkdir(exist_ok=True)
cells_path=cells_folder/'distribution-cells.csv.gz';d.to_csv(cells_path,index=False)
shutil.copyfile(__file__,cells_folder/'rescore_histories.py')
shutil.copyfile(implementation.__file__,cells_folder/'revision_uncertainty.py')
metrics=[c for c in d if c.endswith(('_crps','_point_error','_coverage80'))];records=[]
for stratum,keep in [('complete_history_12',d.complete_history_12),('complete_and_uninterrupted',d.complete_history_12&d.uninterrupted_8)]:
    f=d[keep].groupby(['arm','season','geography','signal','location'])[metrics].mean()
    records.append(f.groupby(['arm','season','geography','signal']).mean().reset_index().assign(stratum=stratum))
f=pd.concat(records);f.to_csv(a.output/'distribution-signal-scores.csv',index=False)
r=f.groupby(['arm','season','geography','stratum'])[metrics].mean().reset_index()
r['trajectory_crps_skill_pct']=100*(1-r.trajectory_crps/r.trajectory_point_error)
r.to_csv(a.output/'distribution-scores.csv',index=False)
(a.output/'distribution-score-provenance.json').write_text(json.dumps(dict(experiment=a.experiment,
    model_sha256=sha256(implementation.__file__),scorer_sha256=sha256(__file__),sources=sources,
    normalization='causal donor-maturity Q95; raw observed-history Q95 fallback; native resolution floor',
    cells_path=str(cells_path),candidate_independent_scale=True,forecast_artifacts_unchanged=True),indent=2)+'\n')
