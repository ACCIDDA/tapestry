"""Validate expanded calendar, finalized NHSN alignment, overlap and CV roles."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from tapestry.dataset.build import load,finalized_admissions
from tapestry.dataset.cv import season,week_roles,SEASONS
from tapestry.model.scenario import Scenario
new=load('data/processed/panel-expanded-building.npz');old=load('data/processed/panel.npz')
dates=new['dates'].astype(str);issu=new['issuance_dates'].astype(str)
ti=np.array([list(dates).index(str(d)) for d in old['dates']]);wi=np.array([list(issu).index(str(d)) for d in old['issuance_dates']])
# Extending the calendar must not alter existing vintages or covariates.
for k in ['covariates','covariates_national']:
 np.testing.assert_equal(new[k][ti],old[k])
for k in ['asof_targets','asof_covariates','asof_covariates_national']:
 np.testing.assert_equal(new[k][np.ix_(wi,ti)],old[k])
np.testing.assert_equal(new['targets'][ti,:,3:],old['targets'][:,:,3:])
changes=[]
for c,n in enumerate(new['target_names'][:3]):
 expected=finalized_admissions(str(n),old['targets'][:,:,c],old['dates'].astype(str).tolist(),'data').astype(np.float32)
 np.testing.assert_equal(new['targets'][ti,:,c],expected)
 a,b=old['targets'][:,:,c],new['targets'][ti,:,c];changes.append(dict(target=str(n),changed_existing_cells=int((~((a==b)|(np.isnan(a)&np.isnan(b)))).sum())))
future=new['dates'][None,:]>new['issuance_dates'][:,None]-np.timedelta64(4,'D')
for k in ['asof_targets','asof_covariates','asof_covariates_national']:assert not np.isfinite(new[k][future]).any(),k
labels=np.array([season(d) for d in dates]);sel=labels=='2022-2023';rows=[]
for key,nameskey in [('targets','target_names'),('covariates','covariate_names'),('covariates_national','covariate_national_names')]:
 for c,n in enumerate(new[nameskey]):
  a=new[key][sel,...,c];finite=np.isfinite(a);byweek=finite if a.ndim==1 else finite.any(axis=1)
  rows.append(dict(series=str(n),kind=key,weeks_with_values=int(byweek.sum()),season_weeks=int(sel.sum()),finite_cells=int(finite.sum()),total_cells=a.size,first_week=dates[sel][byweek][0] if byweek.any() else None,last_week=dates[sel][byweek][-1] if byweek.any() else None))
for held in SEASONS:
 roles=week_roles(dates,Scenario(),held)
 assert np.isin(roles[sel],['fit','validation']).all()
 assert not np.isin(roles[labels==held],['fit','validation']).any()
out=Path('docs/data/dataset');out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/'coverage.csv',index=False);pd.DataFrame(changes).to_csv(out/'finalized-nhsn-changes.csv',index=False)
record=dict(start=dates[0],end=dates[-1],weeks=len(dates),issuances=len(issu),training_added='2022-2023',evaluation_seasons=list(SEASONS),future_vintages_empty=True,overlap_vintages_covariates_ed_unchanged=True,nhsn_final_precedence_changes=changes,sha256=hashlib.sha256(Path('data/processed/panel-expanded-building.npz').read_bytes()).hexdigest())
(out/'verification.json').write_text(json.dumps(record,indent=2));print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(record,indent=2))
