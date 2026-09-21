"""Descriptive revision audit, not a model fit or forecast score. Run from repo root."""
from pathlib import Path
import hashlib, json
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.model_data.wednesday import WednesdayDataset
from tapestry.model_data.finalized import CHANNELS, season

OUT = Path('analysis/b1-revision-seasons')
PATH = Path('data/processed/build_b1_wednesday_calendar.npz')
ds = WednesdayDataset.load(PATH)
a = ds.arrays
rows = []
for i, issuance in enumerate(a['issuance_dates']):
    for h in range(2):
        day = str(a['target_dates'][i,h]); dt = date.fromisoformat(day)
        start_year=int(season(dt)[:4]); jan4=date(start_year,1,4)
        boundary=jan4-timedelta(days=(jan4.weekday()+1)%7)+timedelta(weeks=30)
        season_week=(dt-boundary).days//7
        for c, target in enumerate(CHANNELS):
            for l, location in enumerate(ds.locations):
                if not a['Y_recent_valid'][i,h,c,l]: continue
                available = bool(a['X_available'][i,-2+h,c,l])
                final = bool(a['X_final'][i,-2+h,c,l])
                p = ds.metadata['provenance'][int(a['X_provenance'][i,-2+h,c,l])]
                rows.append(dict(issuance=str(issuance), date=day, season=season(dt), week=season_week,
                    month=dt.month, target=target, lag_days=11-7*h, location=location,
                    geography='US' if location=='US' else 'states', available=available, supplied_final=final,
                    source=p['source'], preliminary=float(a['X_values'][i,-2+h,c,l]),
                    final=float(a['Y_recent'][i,h,c,l])))
f=pd.DataFrame(rows)
f['reported']=f.available & ~f.supplied_final
f['delta']=f.final-f.preliminary
f.to_csv(OUT/'cells.csv',index=False)

def stats(g):
    denom=g.final.sum()
    positive=g.final>0
    ratio=g.loc[positive,'preliminary']/g.loc[positive,'final']
    return dict(n=len(g),weeks=g.date.nunique(),locations=g.location.nunique(),
        final_total=denom, net_revision_pct=100*g.delta.sum()/denom if denom else np.nan,
        absolute_revision_pct=100*g.delta.abs().sum()/denom if denom else np.nan,
        median_reported_pct=100*ratio.median(),p10_reported_pct=100*ratio.quantile(.1),
        p90_reported_pct=100*ratio.quantile(.9),downward_pct=100*(g.delta < -1e-8).mean())

def summarize(frame,keys,path):
    result=pd.DataFrame([{**dict(zip(keys,k if isinstance(k,tuple) else (k,))),**stats(g)} for k,g in frame.groupby(keys)])
    result.to_csv(OUT/path,index=False)
    return result

coverage=f.groupby(['season','target','lag_days','geography']).agg(labels=('reported','size'),reports=('reported','sum'),supplied_finals=('supplied_final','sum')).reset_index()
coverage.to_csv(OUT/'coverage.csv',index=False)
r=f[f.reported]
s=summarize(r,['season','target','lag_days','geography'],'season-summary.csv')
summarize(r,['season','target','lag_days','geography','source'],'source-summary.csv')
summarize(r[r.month.isin([10,11,12,1,2,3])],['season','target','lag_days','geography'],'winter-summary.csv')
w=summarize(r,['date','season','target','lag_days','geography'],'weekly-summary.csv')
# Pair the same week within CDC season and location between seasons, separately by target/lag.
# Source-matched version additionally requires exactly the same selected provider.
paired=[]
for (target,lag,geo),g in r.groupby(['target','lag_days','geography']):
    labels=sorted(g.season.unique())
    for ia,sa in enumerate(labels):
        for sb in labels[ia+1:]:
            for match_source in [False,True]:
                keys=['week','location']+(['source'] if match_source else [])
                ga=g[g.season==sa];gb=g[g.season==sb]
                assert not ga.duplicated(keys).any() and not gb.duplicated(keys).any()
                common=ga[keys].merge(gb[keys],on=keys)
                for label,part in [(sa,ga),(sb,gb)]:
                    matched=part.merge(common,on=keys)
                    if len(matched):paired.append(dict(target=target,lag_days=lag,geography=geo,pair=sa+' vs '+sb,source_matched=match_source,season=label,**stats(matched)))
pd.DataFrame(paired).to_csv(OUT/'matched-season-summary.csv',index=False)
colors={'2023-2024':'#4477aa','2024-2025':'#ee7733','2025-2026':'#228833'}
fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
for c,ax in enumerate(axes.flat):
    for label,color in colors.items():
        for lag,ls in [(4,'-'),(11,'--')]:
            q=w[(w.target==CHANNELS[c])&(w.season==label)&(w.lag_days==lag)&(w.geography=='states')].copy()
            q['season_week']=((pd.to_datetime(q.date)-pd.Timestamp(label[:4]+'-08-01')).dt.days/7)
            ax.plot(q.season_week,100-q.net_revision_pct,color=color,ls=ls,lw=1.4,label=f'{label}, {lag} days')
    ax.axhline(100,color='black',lw=.7);ax.set_title(CHANNELS[c]);ax.set_ylabel('Reported / final total (%)');ax.set_xlabel('Weeks since August 1')
axes.flat[0].legend(fontsize=7,ncol=2)
fig.suptitle('Genuine Wednesday reports only; states/DC pooled within week\nSolid: latest completed week (4 days); dashed: preceding week (11 days). Supplied finals excluded.')
fig.tight_layout();fig.savefig(OUT/'revision-curves.png',dpi=160);plt.close(fig)
manifest=dict(dataset=str(PATH),sha256=hashlib.sha256(PATH.read_bytes()).hexdigest(),truth_cutoff=ds.metadata['truth_cutoff'],n_report_cells=len(r),assumptions=['Season assigned by observation week, not issuance.','Only genuine available reports with valid final labels enter revision metrics; supplied finals excluded.','States/DC pooled by final-value mass; US separate. These are descriptive metrics, not scientific forecast score weights.','Net revision = 100 sum(final-report)/sum(final); absolute revision = 100 sum(abs(final-report))/sum(final). Negative net means downward revisions.','Median reported/final excludes zero finals; sum metrics retain zeros.','Matched comparisons intersect week within CDC season and location, optionally exact provider; cannot eliminate within-season epidemic intensity differences.','Cells and adjacent weeks are dependent; no significance test or causal attribution.','Reference finals use pinned truth cutoff and may still revise.','Full available seasonal calendar includes summer; winter sensitivity is October-March.'])
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(s[(s.geography=='states')&(s.lag_days==4)].round(2).to_string(index=False))
print('\nSOURCE-MATCHED 2024 vs 2025')
p=pd.DataFrame(paired); print(p[(p.geography=='states')&(p.lag_days==4)&p.source_matched&(p['pair']=='2024-2025 vs 2025-2026')][['target','season','n','weeks','net_revision_pct','absolute_revision_pct']].round(2).to_string(index=False))
