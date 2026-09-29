"""All panel covariates: corrected availability and unique missing states/weeks."""
from pathlib import Path
import pandas as pd
from tapestry.dataset.build import load
p=Path('docs/results/b2-direct-research-v1/availability')
x=pd.read_csv(p/'cells.csv.gz');keys=['covariate','issuance','lag','location']
x['reported_available']=x.available
for file in p.glob('*-statements.csv.gz'):
 z=pd.read_csv(file);m=z.set_index(keys).raw_finite
 ix=pd.MultiIndex.from_frame(x[keys]);v=m.reindex(ix);keep=v.notna().to_numpy();x.loc[keep,'reported_available']=v[keep].to_numpy(dtype=bool)
x['missing']=~x.reported_available.astype(bool)
panel=load('data/processed/panel-b2-deadline.npz');names=[*map(str,panel['covariate_names']),*map(str,panel['covariate_national_names'])];assert set(names)==set(x.covariate)
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
def freq(n):return 'Daily; Saturday trailing-7-day value' if n.startswith(('inpatient','outpatient')) else 'Daily; complete weekly mean' if n=='kinsa_ili' else 'Weekly derived index; irregular samples' if n.startswith('nwss') else 'Weekly'
summary=[];byloc=[]
for (n,k),g in x.groupby(['covariate','lag'],sort=False):
 missing=g[g.missing];states=missing[~missing.location.isin(['US','DC'])];dc=missing[missing.location.eq('DC')];us=missing[missing.location.eq('US')]
 summary.append(dict(covariate=n,lag=k,frequency=freq(n),locations=g.location.nunique(),native_states=g.loc[~g.location.isin(['US','DC']),'location'].nunique(),dc_supported=bool(g.location.eq('DC').any()),us_supported=bool(g.location.eq('US').any()),opportunities=len(g),reported_available=int(g.reported_available.sum()),availability_pct=100*g.reported_available.mean(),missing_location_weeks=len(missing),unique_missing_states=states.location.nunique(),dc_missing_weeks=dc.issuance.nunique(),us_missing_weeks=us.issuance.nunique(),unique_missing_state_weeks=states.issuance.nunique(),unique_missing_state_dc_weeks=missing[~missing.location.eq('US')].issuance.nunique(),unique_missing_all_weeks=missing.issuance.nunique()))
 for loc,h in g.groupby('location'):
  m=h[h.missing];byloc.append(dict(covariate=n,lag=k,location=loc,deadlines=len(h),missing_weeks=m.issuance.nunique(),missing_issuance_dates=';'.join(sorted(m.issuance.unique())),missing_observation_dates=';'.join(sorted(m.observation_week.unique()))))
s=pd.DataFrame(summary);s.to_csv(p/'full-availability-summary.csv',index=False)
b=pd.DataFrame(byloc);b.to_csv(p/'missing-by-state-lag.csv',index=False)
x[x.missing][['covariate','lag','location','issuance','cutoff','observation_week']].to_csv(p/'missing-cells.csv.gz',index=False)
intro='''# Every modeled covariate: availability and missing states/weeks

All **14 covariate series** in the frozen panel are included: four claims signals, six wastewater indices (WVAL-like and percentile for each pathogen), ILI, clinical-lab positivity, FluSurv, and national Kinsa. These are the panel's full covariate set, not every signal in the broader data catalog. The B2 grid used WVAL-like wastewater, not its percentile alternatives.

Availability means at least one finite value in the latest eligible source statement, including finite claims conflicts discarded by the model pipeline. No finalized proxies are used. The 53 deadlines and holiday adjustments are unchanged. Counts use each source's fixed native geography. Unsupported states are listed on each source's detail page, excluded from the denominator, and **not silently counted as complete**.

- **—** means the source has no native support for that geography, not zero missing weeks.
- **Missing states:** unique members of the 50 states with at least one missing value at this lag. DC and the native US series are separate columns.
- **Distinct weeks, states:** unique submission dates with at least one missing state. The same week missing in 20 states counts once here but 20 times in missing location-weeks.
- **Per-state missing weeks:** each source link opens a state-by-lag table. Each count is the number of distinct missing submission dates for that state at that lag. Exact submission and observation dates are downloadable. At a fixed lag these date counts are identical; **do not sum across lags to obtain unique weeks**.
- **Location-weeks checked:** native locations × 53 deadlines; this is not the number of finite revision pairs. All-season archive gaps remain in the denominator.

[Full summary CSV](full-availability-summary.csv) · [Every state's counts and exact dates](missing-by-state-lag.csv) · [All missing cells](missing-cells.csv.gz) · [All-lag conditional revisions](full-revisions.md) · [Conflict diagnosis](conflict-audit.md) · [Main availability report](index.md).

'''
text=intro
for k in range(12):
 rows=[]
 for n in names:
  r=s[(s.covariate==n)&(s.lag==k)].iloc[0]
  rows.append([f'[{n}]({n}.md)',freq(n),f'{r.reported_available}/{r.opportunities} ({r.availability_pct:.1f}%)',r.unique_missing_states if r.native_states else '—',r.unique_missing_state_weeks if r.native_states else '—',r.dc_missing_weeks if r.dc_supported else '—',r.us_missing_weeks if r.us_supported else '—',r.missing_location_weeks])
 text+=f'## Lag {k}: '+('latest observation week' if k==0 else f'{k} observation weeks earlier')+'\n\n'+table(['Covariate','Frequency / representation','Reported / checked','Missing states (of 50)','Distinct weeks, states','DC missing weeks','US missing weeks','Missing location-weeks'],rows)+'\n\n'
(p/'full-tables.md').write_text(text)
allstates=set(map(str,panel['locations']))-{'US','DC'}
for n in names:
 g=b[b.covariate==n];native=set(g.location);unsupported=sorted(allstates-native)
 rows=[]
 for loc,h in g.groupby('location',sort=True):
  values=h.set_index('lag').missing_weeks
  rows.append([loc]+[int(values.loc[k]) for k in range(12)])
 (p/(n+'.md')).write_text(f'''# {n}: missing weeks per state and lag

Frequency/representation: **{freq(n)}**. Every cell is **distinct missing submission weeks out of 53**, using corrected finite-report availability. Zero means all 53 observations at that lag were reported for that native location. US is the native national series; DC is separate from the 50 states. Kinsa is counted nationally once.

Native locations ({len(native)}): {', '.join(sorted(native))}.

States outside this source's native season support ({len(unsupported)}): {', '.join(unsupported) or 'none'}. DC supported: {'yes' if 'DC' in native else 'no'}. Native US supported: {'yes' if 'US' in native else 'no'}. Unsupported locations are excluded rather than reported as zero missing.

{table(['Location']+[f'Lag {k}' for k in range(12)],rows)}

[Exact missing submission and observation dates for every state/lag](missing-by-state-lag.csv) · [All covariates and missing-state totals](full-tables.md).
''')
r=pd.read_csv(p/'summary.csv');rev='''# All covariates: revisions at every lag

These are **conditional on the old pipeline retaining both an unambiguous deadline value and final value**. The conflict correction improves availability counts but does not pick an arbitrary value to recompute revisions. Percentages are sum of absolute revisions divided by sum of absolute final values, ×100; n is the number of finite pairs. Missing pairs do not count as zero change. Sample composition varies by lag. Frozen final values can themselves have missing cells from conflict rejection.

[Original-unit bias, mean and 90th percentile absolute changes](summary.csv) · [Availability and missing states](full-tables.md) · [Conflict diagnosis](conflict-audit.md).

'''
for start in [0,6]:
 rows=[]
 for n in names:
  g=r[r.covariate.eq(n)].set_index('lag');rows.append([n,freq(n)]+[f'{g.loc[k,"relative_absolute_revision_pct"]:.1f}% (n={int(g.loc[k,"paired"])})' if g.loc[k,'paired'] else '— (n=0)' for k in range(start,start+6)])
 rev+=table(['Covariate','Frequency / representation']+[f'Lag {k}' for k in range(start,start+6)],rows)+'\n\n'
(p/'full-revisions.md').write_text(rev)
assert len(s)==14*12
assert (s.reported_available+s.missing_location_weeks==s.opportunities).all()
assert b.missing_weeks.between(0,53).all()
print(s[s.lag.eq(1)][['covariate','unique_missing_states','unique_missing_state_weeks','dc_missing_weeks','us_missing_weeks']].to_string(index=False))
