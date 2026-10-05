"""Native recent-level/growth intervals for the balanced reconstruction candidate."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.dataset.build import load
from tapestry.dataset.cv import season
from tapestry.model.revision_uncertainty import RevisionUncertainty
p=argparse.ArgumentParser();p.add_argument('-e','--experiment',default='context-replay-v4-20261001')
p.add_argument('--output',type=Path,default=Path(__file__).parent/'quarter-correction');a=p.parse_args()
root=Path('data/experiments')/a.experiment
settings=json.loads((root/'experiment.json').read_text());panel=load(settings['dataset'])
cache=list(root.glob('replay-nowcasts-context_residual-*-s0.25.npz'))
if len(cache)!=1:raise ValueError('Expected exact quarter-strength cache')
with np.load(cache[0]) as saved:pred=saved['predictions']
u=RevisionUncertainty(panel,pred);records=[]
for wi,issue in enumerate(panel['issuance_dates'].astype(str)):
    end=np.datetime64(issue)-np.timedelta64(4,'D')
    if season(str(end))!='2025-2026':continue
    ti=np.searchsorted(panel['dates'].astype('datetime64[D]'),end)-np.arange(11,-1,-1)
    if ti.min()<0:continue
    raw=panel['asof_targets'][wi,ti]
    center=raw.copy();center[-8:]=pred[wi,::-1]
    center=np.where(np.isfinite(raw),center,panel['targets'][ti])
    center=np.nan_to_num(np.moveaxis(center,-1,-2)).astype(np.float32)
    draws,_,_=u.draw(issue,center,members=2048,strength=.5)
    truth=np.moveaxis(panel['targets'][ti],-1,-2)
    funcs=[lambda x:x[...,-4:,:,:].mean(-3),lambda x:x[...,-2:,:,:].mean(-3)-x[...,-4:-2,:,:].mean(-3)]
    for metric,fun in zip(['level','growth'],funcs):
        low,high=np.quantile(fun(draws),[.1,.9],axis=0);point=fun(center);y=fun(truth)
        for loc in ['US','NC']:
            li=list(panel['locations']).index(loc)
            for k,name in enumerate(panel['target_names']):
                complete=np.isfinite(raw[:,li,k]).all()
                records.append(dict(issuance=issue,location=loc,signal=name,metric=metric,
                    truth=y[k,li] if complete else np.nan,point=point[k,li] if complete else np.nan,
                    low=low[k,li] if complete else np.nan,high=high[k,li] if complete else np.nan))
f=pd.DataFrame(records);a.output.mkdir(parents=True,exist_ok=True);f.to_csv(a.output/'history-intervals.csv',index=False)
for loc in ['US','NC']:
    fig,axes=plt.subplots(6,2,figsize=(14,16))
    for i,name in enumerate(panel['target_names']):
        scale=100 if str(name).startswith('nssp_') else 1
        for j,metric in enumerate(['level','growth']):
            ax=axes[i,j];g=f[f.location.eq(loc)&f.signal.eq(name)&f.metric.eq(metric)]
            days=pd.to_datetime(g.issuance)
            ax.fill_between(days,g.low*scale,g.high*scale,color='#16877f',alpha=.2,label='Central 80% of draws')
            ax.plot(days,g.point*scale,color='#16877f',lw=1.4,label='Quarter correction')
            ax.plot(days,g.truth*scale,color='#222',lw=1.1,label='Later reference')
            ax.set_title(str(name).replace('nhsn_','').replace('nssp_','').replace('_',' ')+(' · four-week level' if j==0 else ' · two-week change'),fontsize=10)
            ax.set_ylabel(('ED visits (%)' if j==0 else 'Percentage points') if scale==100 else 'Admissions')
            ax.tick_params(axis='x',rotation=25);ax.grid(alpha=.15)
            if j:ax.axhline(0,color='#777',lw=.5)
    axes[0,1].legend(fontsize=8);fig.suptitle(f'{loc} · Recent trajectory with joint reporting uncertainty · 2025–26\nComplete target histories only; empirical intervals are not fully calibrated')
    fig.tight_layout();fig.savefig(a.output/f'history-intervals-{loc}.png',dpi=155);plt.close(fig)
