"""Matched native-unit target diagnostics; no fitting or model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('roots', nargs='+', type=Path)
parser.add_argument('--output', type=Path, default=Path(__file__).parent)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
records, weeks, curves = [], [], []
support = {}
for root in args.roots:
    runs = pd.read_csv(root/'runs.csv')
    for run in runs[runs.status.eq('complete')].itertuples():
        for path in (root/run.attempt).glob('eval_*/finalizations.csv.gz'):
            meta = json.loads((path.parent/'manifest.json').read_text())
            model = next((t.split('=')[1] for t in meta['scenario'].split(',') if t.startswith('finalization_model=')), 'triangle')
            gap = next((t.split('=')[1] for t in meta['scenario'].split(',') if t.startswith('finalization_gap=')), 'ridge')
            if gap != 'ridge':
                model += '+' + gap
            for key in ('finalization_halflife', 'finalization_pool', 'finalization_statistic', 'finalization_quantize'):
                value = next((t.split('=')[1] for t in meta['scenario'].split(',') if t.startswith(key+'=')), None)
                if value is not None:
                    model += '+' + key.removeprefix('finalization_') + '=' + value
            fold = meta['fold']
            f = pd.read_csv(path)
            f = f[f.signal.str.startswith(('nhsn_', 'nssp_'))].copy()
            keys = set(map(tuple, f[['signal','issuance','boundary','age','location']].values))
            if fold in support and keys != support[fold]:
                raise ValueError(f'Unmatched target cells for {fold}: {root}')
            support[fold] = keys
            f['geography'] = np.where(f.location.eq('US'), 'US', 'states')
            for window, win in [('newest', f[f.age.eq(0)]), ('eight_weeks', f)]:
                for kind, group in [('all', win), ('reported', win[win.kind.eq('reported')]), ('missing', win[~win.kind.eq('reported')])]:
                    for (signal, geo), g in group.groupby(['signal','geography']):
                        for method in ['prediction','persistence','baselinenowcast']:
                            err = (g[method]-g.truth).abs()
                            z = g[['location','truth','scale']].copy()
                            z['ae'], z['nae'], z['signed'] = err, err/g.scale, g[method]-g.truth
                            z['within5'] = np.where(g.truth>0, err <= .05*g.truth, np.nan)
                            loc = z.groupby('location').agg(ae=('ae','sum'), truth=('truth','sum'), nae=('nae','mean'), within5=('within5','mean'))
                            records.append(dict(model=model, fold=fold, window=window, kind=kind, signal=signal, geography=geo, method=method,
                                cells=len(g), normalized_mae=loc.nae.mean(), wape=err.sum()/g.truth.sum() if g.truth.sum()>0 else np.nan,
                                location_wape=(loc.ae/loc.truth.replace(0,np.nan)).mean(), within5=loc.within5.mean(),
                                zero_cells=int(g.truth.eq(0).sum()), correct_zero=int((g.truth.eq(0)&err.eq(0)).sum()), bias=(g[method]-g.truth).sum()/g.truth.sum() if g.truth.sum()>0 else np.nan))
            latest = f[f.age.eq(0)].copy()
            for method in ['prediction','persistence']:
                t=latest.assign(ae=(latest[method]-latest.truth).abs())
                t=t.groupby(['signal','issuance','geography'],as_index=False).agg(ae=('ae','sum'),truth_sum=('truth','sum'))
                t['wape']=t.ae/t.truth_sum.replace(0,np.nan)
                t['model'],t['fold'],t['method']=model,fold,method
                weeks.extend(t.to_dict('records'))
            t=f[f.kind.eq('reported')].groupby(['signal','issuance','age','geography'],as_index=False).agg(
                factor=('development_factor','median'),shared=('shared_development_factor','median'))
            t['model'],t['fold']=model,fold
            curves.extend(t.to_dict('records'))
d = pd.DataFrame(records)
d.to_csv(args.output/'target-scores.csv', index=False)
w = pd.DataFrame(weeks); w.to_csv(args.output/'weekly-scores.csv.gz', index=False)
c = pd.DataFrame(curves); c.to_csv(args.output/'learned-curves.csv.gz', index=False)
summary = d[d.kind.eq('all')].groupby(['model','fold','window','geography','method'])[['wape','location_wape','normalized_mae','within5']].mean()
summary.to_csv(args.output/'summary.csv')
print(summary.loc[(slice(None),'2025-2026','newest',slice(None),'prediction'),:].to_string())
signals = sorted(d.signal.unique())
# Four-week moving-block bootstrap, resampling whole weeks jointly across targets.
# Conditional uncertainty within this one reused development season, not across seasons.
intervals=[]
rng=np.random.default_rng(42)
for geo in ('states','US'):
    z=w[w.fold.eq('2025-2026')&w.geography.eq(geo)&w.method.eq('prediction')]
    reference=z[z.model.eq('triangle')]
    if reference.empty:
        continue
    idx=sorted(reference.issuance.unique())
    def matrix(frame, column):
        return frame.pivot(index='issuance',columns='signal',values=column).reindex(index=idx,columns=signals).to_numpy()
    ref=matrix(reference,'ae');truth=matrix(reference,'truth_sum');n=len(idx)
    starts=rng.integers(0,n,(2000,int(np.ceil(n/4))))
    draws=((starts[:,:,None]+np.arange(4))%n).reshape(2000,-1)[:,:n]
    ref_loss=np.mean(ref[draws].sum(1)/truth[draws].sum(1),axis=1)
    for model,g in z.groupby('model'):
        err=matrix(g,'ae')
        point=np.mean(err.sum(0)/truth.sum(0))
        loss=np.mean(err[draws].sum(1)/truth[draws].sum(1),axis=1)
        improvement=1-loss/ref_loss
        intervals.append(dict(model=model,geography=geo,wape=point,
            reduction_vs_triangle=1-point/np.mean(ref.sum(0)/truth.sum(0)),
            reduction_low=np.quantile(improvement,.025),reduction_high=np.quantile(improvement,.975),
            wape_low=np.quantile(loss,.025),wape_high=np.quantile(loss,.975),block_weeks=4,draws=2000))
pd.DataFrame(intervals).to_csv(args.output/'block-bootstrap.csv',index=False)
