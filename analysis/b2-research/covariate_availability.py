"""Audit archived deadline covariates against frozen final values; no proxies."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from tapestry.dataset.build import load
from tapestry.dataset.cv import season

ROOT=Path('docs/data/availability')
ROOT.mkdir(parents=True,exist_ok=True)
path=Path('data/processed/panel-b2-deadline.npz')
p=load(path)
dates=pd.to_datetime(p['dates']);issuances=pd.to_datetime(p['issuance_dates'])
wi=[w for w,d in enumerate(issuances) if season((d-pd.Timedelta(days=4)).date())=='2025-2026']
lookup={d:i for i,d in enumerate(dates)}
rows=[];supports=[]
for key,nameskey,locs in [('covariates','covariate_names',p['locations'].astype(str)),('covariates_national','covariate_national_names',np.array(['US']))]:
 final=p[key];asof=p['asof_'+key]
 if final.ndim==2:final=final[:,None,:];asof=asof[:,:,None,:]
 for c,name in enumerate(p[nameskey].astype(str)):
  # Fixed native support, not changing with current report availability.
  ti=[lookup[issuances[w]-pd.Timedelta(days=4)] for w in wi]
  active=np.isfinite(final[ti,:,c]).any(axis=0)|np.isfinite(asof[wi,:,:,c]).any(axis=(0,1))
  supports.append(dict(covariate=name,locations=','.join(locs[active]),n_locations=int(active.sum())))
  for w in wi:
   end=issuances[w]-pd.Timedelta(days=4)
   for lag in range(12):
    t=lookup[end-pd.Timedelta(weeks=lag)]
    for l in np.where(active)[0]:
     a=float(asof[w,t,l,c]);f=float(final[t,l,c]);ok=np.isfinite(a) and np.isfinite(f)
     rows.append(dict(covariate=name,issuance=issuances[w].date().isoformat(),cutoff=str(p['forecast_cutoff_utc'][w]),context_end=end.date().isoformat(),lag=lag,observation_week=dates[t].date().isoformat(),location=locs[l],available=np.isfinite(a),final_available=np.isfinite(f),reported=a,final=f,revision=f-a if ok else np.nan,paired=ok))
cells=pd.DataFrame(rows);cells.to_csv(ROOT/'cells.csv.gz',index=False)
pd.DataFrame(supports).to_csv(ROOT/'native-support.csv',index=False)
summary=[]
for (name,lag),g in cells.groupby(['covariate','lag'],sort=False):
 q=g[g.paired];absrev=q.revision.abs();den=q.final.abs().sum()
 changed=~np.isclose(q.reported,q.final,rtol=1e-5,atol=1e-6)
 summary.append(dict(covariate=name,lag=lag,opportunities=len(g),available=int(g.available.sum()),availability_pct=100*g.available.mean(),weeks_any=int(g.groupby('issuance').available.any().sum()),weeks_all=int(g.groupby('issuance').available.all().sum()),weeks=g.issuance.nunique(),locations=g.location.nunique(),paired=len(q),available_without_final=int((g.available & ~g.final_available).sum()),changed=int(changed.sum()),changed_pct=100*changed.mean() if len(q) else np.nan,mean_revision=q.revision.mean(),mean_absolute_revision=absrev.mean(),p90_absolute_revision=absrev.quantile(.9),relative_absolute_revision_pct=100*absrev.sum()/den if den>0 else np.nan))
s=pd.DataFrame(summary);s.to_csv(ROOT/'summary.csv',index=False)
# Per-location table separates native national availability from state coverage.
weekly=cells.groupby(['covariate','issuance','lag']).agg(opportunities=('available','size'),available=('available','sum'),final_available=('final_available','sum'),paired=('paired','sum')).reset_index();weekly.to_csv(ROOT/'by-week.csv',index=False)
loc=cells.groupby(['covariate','location','lag']).agg(opportunities=('available','size'),available=('available','sum'),paired=('paired','sum'),mean_revision=('revision','mean'),mean_absolute_revision=('revision',lambda a:a.abs().mean())).reset_index();loc.to_csv(ROOT/'by-location.csv',index=False)
labels={'inpatient_flu':'Inpatient claims — flu','inpatient_covid':'Inpatient claims — COVID','outpatient_flu':'Outpatient claims — flu','outpatient_covid':'Outpatient claims — COVID','ilinet_ili':'ILINet ILI','clinical_lab_flu_pct_positive':'Clinical lab flu positivity','flusurv_flu_rate':'FluSurv flu','kinsa_ili':'Kinsa national'}
for pathogen in ['flu','covid','rsv']:
 labels['nwss_'+pathogen+'_wval_like']='Wastewater '+pathogen.upper()+' — WVAL-like'
 labels['nwss_'+pathogen+'_pct_rank']='Wastewater '+pathogen.upper()+' — percentile'
order=[*labels]
def frequency(name):
 if name.startswith(('inpatient','outpatient')):return 'Daily; Saturday trailing-7-day value'
 if name=='kinsa_ili':return 'Daily; complete 7-day mean'
 if name.startswith('nwss'):return 'Weekly index from irregular samples'
 return 'Weekly'

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
av=[];rev=[]
for name in order:
 g=s[s.covariate==name].set_index('lag');r=g.loc[0]
 av.append([labels[name],frequency(name),int(r.locations),int(r.opportunities)]+[f'{int(g.loc[k,"available"])} ({g.loc[k,"availability_pct"]:.1f}%)' for k in range(6)])
 rev.append([labels[name]]+[('—' if not g.loc[k,'paired'] else f'{g.loc[k,"relative_absolute_revision_pct"]:.1f}% (n={int(g.loc[k,"paired"])})') for k in range(6)])
units=[]
for name in order:
 unit='percentage points' if not name.startswith(('nwss','flusurv')) else ('per 100,000 catchment residents' if name.startswith('flusurv') else 'index units')
 g=s[(s.covariate==name)&s.paired.gt(0)].sort_values('lag')
 if len(g):
  r=g.iloc[0];units.append([labels[name],int(r.lag),int(r.paired),f'{r.changed_pct:.1f}%',f'{r.mean_revision:+.4g}',f'{r.mean_absolute_revision:.4g}',f'{r.p90_absolute_revision:.4g}',unit])
meta=json.loads(str(p['metadata']));source_details=meta['deadline_build']['source_details']
missingfinal=[]
for name,g in cells[cells.lag.eq(0)].groupby('covariate',sort=False):
 n=int((g.available & ~g.final_available).sum())
 if n:missingfinal.append([name,int(g.available.sum()),int(g.paired.sum()),n])
first=[]
for source,detail in source_details.items():
 for name,d in detail['first_release'].items():first.append(dict(covariate=name,first_archived_release=d['first_release']))
pd.DataFrame(first).to_csv(ROOT/'archive-starts.csv',index=False)
text=f'''# Covariate availability and revisions, 2025–26

This checks **actual archived values at the Hub deadline**, not the assumed-available finalized proxies used to fill gaps in the B2 operational panel. The B0 reproduction audited targets; the existing B2 covariate coverage table combined 12 history weeks and did not show this lag/revision breakdown.

[Full tables for all 14 covariates and 12 lags, including unique missing states and weeks](full-availability-summary.csv) · [All-lag revisions](summary.csv).

## What the counts mean

The season is defined by the latest observation week: **{cells.context_end.min()} through {cells.context_end.max()}**, with **{len(wi)} weekly issuance dates** ({cells.issuance.min()} through {cells.issuance.max()}). This is the complete modeled season, not only weeks with a scored Hub submission. **Lag 0** is the Saturday four days before the nominal Wednesday; lag 1 is the preceding Saturday, and so on. Each count is a **location × issuance** opportunity. The old label “possible pairs” meant the number of location-weeks checked, **not** the number of finite reported/final pairs. For a source covering 50 states, DC and US this is 52 locations × 53 deadlines = **2,756**. For national Kinsa it is 1 × 53 = **53**. A revision pair exists only when both reported and final values are finite. Kinsa is counted once nationally, not broadcast and counted 52 times. Geography is the fixed native support observed anywhere in this season's final data or eligible archived histories; structurally unsupported states are excluded. Full location lists are downloadable.

Cutoffs are Wednesday **23:00 America/New_York**, except the study's documented joint-Hub holiday deadlines: the nominal December 24 and December 31, 2025 rounds use December 29 and January 4. The context Saturday stays fixed. Thus this is availability **when submission is due**, including extensions, rather than a literal-Wednesday-only audit. Date-only release labels are conservatively placed at the end of their UTC day, following the frozen panel policy. No observations, lags or availability have been imputed here.

## Source frequency and what “latest” means

Claims are daily trailing-seven-day percentages; the current model takes the Saturday value, rather than whichever daily observation was most recently published. Kinsa is daily and is converted to a complete Sunday–Saturday mean. ILINet, clinical labs and FluSurv are weekly. Wastewater measurements have irregular sampling dates and are aggregated into weekly indices. This audit checks those **model input representations**; it does not count a Friday daily value as an available Saturday value.

## Availability by age of observation

These are the frozen pipeline counts, **after conflict rejection**. See the [raw archive conflict audit](conflict-audit.md) for finite reported values discarded by that rule. Each cell is **available count (percent of opportunities)**. The denominator is the same for every lag in a row. Wastewater percentiles are included for completeness but were not used by the seven-source B2 grid, which used WVAL-like indices.

{table(['Covariate','Native frequency / model representation','Locations','Location-weeks checked','Latest (0)','1 week earlier','2 earlier','3 earlier','4 earlier','5 earlier'],av)}

All **12 context lags**, weeks with any/all native locations available, and every location's counts are in the [summary](summary.csv) and [location table](by-location.csv).

## How far the deadline value is from final

**These are conditional on the old pipeline retaining an unambiguous deadline value and final value.** Same-report-date conflicts in claims were discarded; the [conflict audit](conflict-audit.md) explains why these revision estimates are incomplete. They must not be interpreted as revision estimates for all reported claims values.

For the **same observation week and location**, revision = frozen final value minus deadline value. Only finite pairs enter. The table reports **100 × sum(abs(final − reported)) / sum(abs(final))**, with the paired sample count. This is absolute revision relative to aggregate final magnitude, not an average of cellwise percentages; it remains meaningful when some final values are zero. Missing deadline values do not count as zero revision. Samples differ by lag, so this is not a fixed cohort estimate of revision decay.

{table(['Covariate','Latest (0)','1 week earlier','2 earlier','3 earlier','4 earlier','5 earlier'],rev)}

For an original-unit view, the following uses **the earliest lag with any paired observations for each source**, not a common lag. Positive bias means the final value was higher. “Changed” ignores float noise with rtol=1e-5 and atol=1e-6.

{table(['Covariate','Lag','Pairs','Changed','Mean revision','Mean absolute revision','90th percentile absolute','Units'],units)}

## Limitations and provenance

**Absent from our archive does not prove absent upstream.** Kinsa's archived releases start in April 2026; wastewater vintages also start in 2026, and FluSurv has archive gaps. Whole-season percentages therefore combine reporting timeliness and archive coverage. The [first archived releases](archive-starts.csv) identify the boundary for each source; they are not inferred historical launch dates. Revision estimates cover only observed pairs, not the missing months.

**Claims have an additional comparison gap:** some finite deadline values have no finite counterpart in the frozen final training panel. The table below counts these at lag 0. They contribute to availability but are excluded from revision calculations. This can make the revision subset unrepresentative; these figures do not establish the cause of the final-panel gaps or prove that upstream final values were unavailable.

{table(['Covariate','Reported at lag 0','Finite final pairs','Reported but final missing'],missingfinal)}

The [weekly counts](by-week.csv) separate temporal gaps from geographic coverage. All-season counts should not be interpreted as a source's operating-period reliability.

“Final” means the frozen finalized arrays used by this experiment, not an assertion that the source can never revise again. The deadline and final values are aligned in the panel's native units. Wastewater WVAL-like/percentile changes may include changes to index construction or historical baseline as well as revisions to underlying measurements; they should not be interpreted as raw-concentration revisions. Kinsa values are complete seven-day averages of its national daily signal. Claims are the source's trailing-seven-day percentage sampled at Saturday.

Source: `data/processed/panel-b2-deadline.npz`, SHA256 **{hashlib.sha256(path.read_bytes()).hexdigest()}**. Original [deadline policy and pinned source snapshots](provenance/deadline-policy.json). The operational proxy-filled panel is deliberately not used.

[All paired and missing cells](cells.csv.gz) · [Native geographic support](native-support.csv) · [Full summary](summary.csv) · [Main experiment report](../../results/b2-direct-research-v1/index.md).

Reproduce from the repository root:

```bash
.venv/bin/python analysis/b2-research/covariate_availability.py
```
'''
(ROOT/'index.md').write_text(text)
print(table(['Covariate','Frequency','Locations','N','lag0','lag1','lag2','lag3','lag4','lag5'],av))
print('\nREVISIONS\n'+table(['Covariate','lag0','lag1','lag2','lag3','lag4','lag5'],rev))
