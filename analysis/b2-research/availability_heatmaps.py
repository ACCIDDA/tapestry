"""Season-wide publication triangles; raw finite statements, never inferred proxies."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from tapestry.dataset.build import load
from tapestry.dataset import extract as ex
p=Path('docs/results/b2-direct-research-v1/availability');out=p/'timeline';out.mkdir(exist_ok=True)
a=load('data/processed/panel-b2-deadline.npz');dates=pd.to_datetime(a['dates']);issuance=pd.to_datetime(a['issuance_dates']);ti=np.where((dates>='2025-08-02')&(dates<='2026-08-01'))[0];wi=np.where(issuance>='2025-08-06')[0];obs=dates[ti];issued=issuance[wi];cutoffs=pd.to_datetime(a['forecast_cutoff_utc'][wi],utc=True);locs=a['locations'].astype(str)
colors=['#f5f5f5','#18864b','#b5d96c','#f1c84b','#e77a42','#a93446'];labels=['Future week','Complete','Missing 1–2 states','Missing 3–10 states','Missing >10 states','No archived values'];cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,6.5),6)
records=[];payload=[]
def cats(v,n):
 m=n-v;z=np.where(m==0,1,np.where(m<3,2,np.where(m<=10,3,4)));return np.where(v==0,5,z)
series=[]
for key,names in [('targets','target_names'),('covariates','covariate_names'),('covariates_national','covariate_national_names')]:
 for c,n in enumerate(a[names].astype(str)):
  v=a['asof_'+key][np.ix_(wi,ti)][...,c];f=a[key][ti,...,c]
  ll=locs if key!='covariates_national' else np.array(['US'])
  if v.ndim==2:v=v[...,None];f=f[:,None]
  valid=np.isfinite(v)
  if n in ex.DELPHI_COVARIATES:
   cache=out/(n+'-raw-statements.csv.gz')
   if not cache.exists():pd.read_pickle('/tmp/tapestry-'+n+'-statements.pkl').to_csv(cache,index=False)
   g=pd.read_csv(cache);release=pd.to_datetime(g.release,utc=True);midnight=release.eq(release.dt.normalize());g['release']=release+pd.to_timedelta(midnight.astype(int),unit='D')-pd.to_timedelta(midnight.astype(int),unit='ns')
   valid[:]=False
   for j,cut in enumerate(cutoffs):
    rows=g[g.release<=cut].sort_values('release').drop_duplicates(['day','location'],keep='last');vals=rows.set_index(['day','location'])['count'];ix=pd.MultiIndex.from_product([obs.strftime('%Y-%m-%d'),ll]);valid[j]=vals.reindex(ix).fillna(0).gt(0).to_numpy().reshape(len(ti),len(ll))
  native=np.isfinite(f).any(axis=0)|valid.any(axis=(0,1));stateix=np.where(native & (ll!='US'))[0];national=np.where(ll=='US')[0]
  # Kinsa uses its single native national value; no artificial state replication.
  chosen=stateix if len(stateix) else national
  num=valid[:,:,chosen].sum(axis=2);nloc=len(chosen);z=cats(num,nloc)
  future=obs.to_numpy()[None,:]>(issued-pd.Timedelta(days=4)).to_numpy()[:,None];z[future]=0
  usvalid=valid[:,:,national].sum(axis=2) if len(national) else np.zeros(num.shape);us=cats(usvalid,len(national) or 1);us[future]=0
  final=cats(np.isfinite(f[:,chosen]).sum(axis=1),nloc)[None,:]
  units='national series' if not len(stateix) else f'{nloc} native states/DC'
  fig,axes=plt.subplots(2,2,figsize=(15,10),gridspec_kw=dict(height_ratios=[20,1],width_ratios=[5,2]),sharex=True)
  for ax,data,title in [(axes[0,0],z,f'Archived by deadline — {units}'),(axes[0,1],us,'Native US'),(axes[1,0],final,'Frozen final panel'),(axes[1,1],cats(np.isfinite(f[:,national]).sum(axis=1),len(national) or 1)[None,:],'Frozen US final')]:
   ax.imshow(data,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest',origin='lower');ax.set_title(title,fontsize=10)
   ax.set_xticks(np.arange(0,len(obs),5),obs[::5].strftime('%b %d'),rotation=45,ha='right')
   if data.shape[0]>1:ax.set_yticks(np.arange(0,len(issued),4),issued[::4].strftime('%Y-%m-%d'),fontsize=8)
   else:ax.set_yticks([])
  axes[0,0].set_ylabel('Submission round (later dates upward)');axes[1,0].set_xlabel('Observation week ending Saturday');axes[1,1].set_xlabel('Observation week ending Saturday')
  fig.suptitle(n+'\n2025–26 observation weeks, followed through '+issued[-1].strftime('%Y-%m-%d'),fontsize=14)
  fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,labels)],loc='lower center',ncol=3,fontsize=9)
  fig.subplots_adjust(bottom=.15,top=.90,hspace=.15,wspace=.15);fig.savefig(out/(n+'.png'),dpi=140);plt.close(fig)
  missing=[]
  for j in range(len(wi)):
   line=[]
   for k in range(len(ti)):
    states=ll[chosen][~valid[j,k,chosen]].tolist() if not future[j,k] else []
    line.append(states)
    if not future[j,k]:records.append(dict(series=n,issuance=issued[j].date(),actual_cutoff=str(cutoffs[j]),observation_week=obs[k].date(),native_locations=nloc,available=int(num[j,k]),missing_locations=';'.join(states),us_available=bool(usvalid[j,k])))
   missing.append(line)
  payload.append(dict(name=n,native=ll[chosen].tolist(),z=z.tolist(),us=us.tolist(),missing=missing,final=final[0].tolist()))
  series.append(n);print(n,flush=True)
pd.DataFrame(records).to_csv(out/'timeline-cells.csv.gz',index=False)
data=dict(observations=obs.strftime('%Y-%m-%d').tolist(),issuances=issued.strftime('%Y-%m-%d').tolist(),cutoffs=[str(c) for c in cutoffs],colors=colors,labels=labels,series=payload)
(out/'timeline.json').write_text(json.dumps(data))
html='''<!doctype html><meta charset="utf-8"><title>Availability timelines</title><style>body{font:16px system-ui;margin:24px;max-width:1300px}canvas{width:100%;max-width:1200px}#tip{padding:15px;background:#eef2f5;min-height:65px}select{font:inherit;padding:8px}span{display:inline-block;padding:8px;margin:3px}</style><h1>What could we see at each deadline?</h1><p>Choose a series. Read one row to see the observation weeks available at that submission. Later submissions are higher. Hover a square for exact missing states. These are archived reports, not assumed availability or current dashboard history.</p><select id="sel"></select><p id="legend"></p><canvas id="plot" width="1200" height="950"></canvas><div id="tip">Hover a square.</div><p>Counts use native states/DC; Kinsa is national only. Holiday extensions retain their nominal Wednesday row. White means a future observation week. Conflicting finite claims values count as reported without selecting a number.</p><script>const D=DATA;const sel=document.querySelector('#sel'),canvas=document.querySelector('#plot'),ctx=canvas.getContext('2d'),tip=document.querySelector('#tip');D.series.forEach((s,i)=>sel.add(new Option(s.name,i)));document.querySelector('#legend').innerHTML=D.labels.map((l,i)=>'<span style="background:'+D.colors[i]+';color:'+(i===1||i===5?'white':'black')+'">'+l+'</span>').join('');const left=130,top=60,w=1000,h=780,cw=w/D.observations.length,ch=h/D.issuances.length;function draw(){const s=D.series[+sel.value];ctx.clearRect(0,0,1200,950);ctx.font='18px system-ui';ctx.fillText(s.name+' — '+s.native.length+' native locations',left,30);for(let j=0;j<D.issuances.length;j++){const y=top+h-(j+1)*ch;for(let k=0;k<D.observations.length;k++){ctx.fillStyle=D.colors[s.z[j][k]];ctx.fillRect(left+k*cw,y,cw,ch)}if(j%4===0){ctx.fillStyle='black';ctx.font='12px system-ui';ctx.fillText(D.issuances[j],5,y+ch)}}ctx.fillStyle='black';for(let k=0;k<D.observations.length;k+=4){ctx.save();ctx.translate(left+k*cw,top+h+16);ctx.rotate(Math.PI/4);ctx.fillText(D.observations[k],0,0);ctx.restore()}ctx.fillText('Observation week ending Saturday →',left,935)}sel.onchange=draw;canvas.onmousemove=e=>{const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*1200/r.width,y=(e.clientY-r.top)*950/r.height,k=Math.floor((x-left)/cw),j=D.issuances.length-1-Math.floor((y-top)/ch);if(k<0||k>=D.observations.length||j<0||j>=D.issuances.length)return;const s=D.series[+sel.value];tip.textContent='Submission '+D.issuances[j]+'; actual cutoff '+D.cutoffs[j]+'; observation '+D.observations[k]+'. '+D.labels[s.z[j][k]]+'. Missing: '+(s.z[j][k]===0?'not yet an observation':s.missing[j][k].join(', ')||'none')+'. Native US: '+D.labels[s.us[j][k]]};draw();</script>'''
(out/'coverage-explorer.html').write_text(html.replace('DATA',json.dumps(data)))
text='''# Availability over the season: read the staircase

**[Open the interactive heatmap explorer](timeline/coverage-explorer.html)** — hover a square to see the submission deadline, observation week, and exact missing states. Every plot is also shown below and can be downloaded.

**Horizontal axis:** the week the data describe (Saturday ending). **Vertical axis:** the date we were making the forecast; later submissions are higher. Follow one observation week upward to see when it first appeared and whether coverage changed. Follow one row across to see all history available at that deadline. White cells are future observation weeks. Green means complete native geographic coverage; light green means 1–2 missing states/DC, yellow 3–10, orange more than 10, and red no archived values.

The observation season runs August 2, 2025–August 1, 2026; submission rounds continue through September 16, 2026, the frozen panel's last round. Rows retain their Wednesday identifiers, with actual holiday cutoffs used for eligibility. These matrices include **every season observation week by every subsequent deadline**, not just 12-week contexts.

**A current dashboard can show historical values that were not present in our archived release at that time.** These plots establish what our archive contains; a red cell does not prove the upstream dashboard was blank. Conversely, absence from an archive may be a collection gap. No such gap is filled by assumption here. The small final-data strip shows the frozen training panel's retrospective coverage, not a freshly checked dashboard. Claims final-strip gaps may be caused by the identified conflict rule.

Each covariate uses its native geographic support, excluding structurally unsupported states. National Kinsa is shown once, not broadcast to states. US is separate in each static figure. Claims count a finite report even when same-date values conflict; no numerical conflict resolution is invented. ILI/labs/FluSurv are weekly, claims are daily trailing-seven-day percentages sampled on Saturday, Kinsa is a complete weekly mean of daily data, and wastewater is a weekly index from irregular samples.

The six target plots cover admissions and ED for flu (FluSight), COVID and RSV. They use the existing deadline reconstruction combining the three Hubs' historical files and Delphi releases, with the latest eligible statement winning. **They are target-specific availability plots, not independent audits of each Hub's file alone.** Targets use the frozen reconstructed values; no finalized substitutes are added.

[All matrix cells and missing locations](timeline/timeline-cells.csv.gz) · [Counts by lag](full-tables.md) · [Conflict audit](conflict-audit.md).

'''
for n in series:text+=f'## {n}\n\n[![{n} availability timeline](timeline/{n}.png)](timeline/{n}.png)\n\n'
(p/'coverage-timeline.md').write_text(text)
assert len(series)==20
