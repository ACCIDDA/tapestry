import json
from pathlib import Path
from datetime import date
from types import SimpleNamespace
import numpy as np
from tapestry.model_data import FinalizedDataset
from tapestry.model_data.finalized import season
from tapestry.models.season_cv import fold_data, SEASONS
from tapestry.models.experiments import model_options

root=Path('data/experiments/b0_full_20260914')
ds=FinalizedDataset.load('data/processed/build_b_finalized.npz')
records=[]
for held in SEASONS:
    changed=ds.panel.copy()
    selected=np.array([season(date.fromisoformat(d))==held for d in ds.dates])
    changed[selected,:,0,:]=987654
    other=FinalizedDataset(changed,ds.dates,ds.locations,ds.metadata)
    for lookback in (8,12,26):
        a, ev, scales=fold_data(ds,held,lookback)
        b, _, other_scales=fold_data(other,held,lookback)
        assert scales==other_scales
        assert len(a)==len(b)
        for e,f in zip(a,b):
            assert np.array_equal(e['X'],f['X'])
            assert np.array_equal(e['Y'],f['Y'])
            for key,dates in [('X','context_dates'),('Y','target_dates')]:
                for i,d in enumerate(e[dates]):
                    if season(date.fromisoformat(d))==held:
                        assert not e[key][i].any()
        for transform in ('sqrt','fourth_root'):
            args=SimpleNamespace(count_transform=transform,geography=True,dynamics=True,population_file='data/metadata/b0_locations.csv')
            assert model_options(a,args)==model_options(b,args)
        records.append({'held_out':held,'lookback':lookback,'training_episodes':len(a),'passed':True})
for path in root.glob('*/manifest.json'):
    m=json.loads(path.read_text())
    for f in m['folds']:
        assert f['eval_season'] not in f['train_seasons']
        assert all(season(date.fromisoformat(d)) in f['train_seasons'] for d in f['training_context_ends'])
        assert all(season(date.fromisoformat(d))==f['eval_season'] for d in f['evaluation_context_ends'])
(root/'holdout_audit.json').write_text(json.dumps({'method':'Replace every held-out-season value with 987654 and assert unchanged training X/Y, native loss scales and transformed input scales. Check saved run origin partitions.', 'checks':records,'passed':True,'caveat':'Forecast context includes observed past evaluation-season data; stage/model selection uses exploratory evaluation scores.'},indent=2)+'\n')
print(json.dumps({'passed':True,'fold_lookback_checks':len(records),'transforms_checked':['sqrt','fourth_root']}))

# Completed full-run artifacts: inversion units, quantiles, and fold membership.
import torch
artifacts=[]
reference_scales={}
for run in sorted(root.glob('*/scores.csv')):
    folder=run.parent
    manifest=json.loads((folder/'manifest.json').read_text())
    assert len(manifest['folds'])==3
    for fold in manifest['folds']:
        held=fold['eval_season']
        assert len(fold['history'])==50 and np.isfinite(fold['history']).all()
        if held in reference_scales:
            assert reference_scales[held]==fold['scale']
        reference_scales[held]=fold['scale']
        checkpoint=torch.load(folder/f'eval_{held}'/'model.pt',weights_only=True,map_location='cpu')
        assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
        with np.load(folder/f'eval_{held}'/'forecasts.npz') as data:
            q=data['quantiles']
            assert np.isfinite(q).all() and (q>=0).all()
            assert (np.diff(q,axis=0)>=0).all() and (q[:,:,:,3:]<=1).all()
            for i,dates in enumerate(data['target_dates']):
                for j,d in enumerate(dates):
                    if season(date.fromisoformat(d))!=held:
                        assert not data['mask'][i,j].any()
            assert json.loads(str(data['metadata']))['eval_members']==2048
        artifacts.append({'run':folder.name,'season':held,'passed':True})
(root/'artifact_audit.json').write_text(json.dumps({'passed':True,'checks':artifacts,'loss_scales_identical_across_variants':True},indent=2)+'\n')
print(json.dumps({'artifact_folds_passed':len(artifacts)}))
