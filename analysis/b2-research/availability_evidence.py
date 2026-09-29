"""Separate archived value, unknown timing, explicit null and never-observed cells."""
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
p=Path('docs/results/b2-direct-research-v1/availability');out=p/'timeline';a=load('data/processed/panel-b2-deadline.npz');dates=pd.to_datetime(a['dates']);iss=pd.to_datetime(a['issuance_dates']);ti=np.where((dates>='2025-08-02')&(dates<='2026-08-01'))[0];wi=np.where(iss>='2025-08-06')[0];obs=dates[ti];issued=iss[wi];cut=pd.to_datetime(a['forecast_cutoff_utc'][wi],utc=True);locations=a['locations'].astype(str)
labels=['Future observation','Reported value','Timing unknown / archive gap','Explicitly missing, later observed','Explicitly missing, no later value observed','No value anywhere in inspected history','Mixed geographic evidence'];colors=['#ffffff','#238b45','#c3c8cf','#4d9bd5','#cf5060','#eee1cf','#f0c85a'];cm=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,7.5),7)
items=[];totals=[]
for key,nameskey in [('targets','target_names'),('covariates','covariate_names'),('covariates_national','covariate_national_names')]:
 for c,n in enumerate(a[nameskey].astype(str)):
  ll=locations if key!='covariates_national' else np.array(['US']);v=a['asof_'+key][np.ix_(wi,ti)][...,c];known=a['known_report_'+key][np.ix_(wi,ti)][...,c];f=a[key][ti,...,c]
  allv=a['asof_'+key][:,ti][...,c]
  if v.ndim==2:v=v[...,None];known=known[...,None];f=f[:,None];allv=allv[...,None]
  valid=np.isfinite(v);ever=np.isfinite(allv).any(axis=0)|np.isfinite(f)
  if n in ex.DELPHI_COVARIATES:
   g=pd.read_csv(out/(n+'-raw-statements.csv.gz'));r=pd.to_datetime(g.release,utc=True);mid=r.eq(r.dt.normalize());g['release']=r+pd.to_timedelta(mid.astype(int),unit='D')-pd.to_timedelta(mid.astype(int),unit='ns');ix=pd.MultiIndex.from_product([obs.strftime('%Y-%m-%d'),ll]);ever=g.groupby(['day','location'])['count'].max().reindex(ix).fillna(0).gt(0).to_numpy().reshape(len(ti),len(ll))|np.isfinite(f)
   valid[:]=False;known[:]=False
   for j,d in enumerate(cut):
    z=g[g.release<=d].sort_values('release').drop_duplicates(['day','location'],keep='last').set_index(['day','location'])['count'].reindex(ix)
    known[j]=z.notna().to_numpy().reshape(len(ti),len(ll));valid[j]=z.fillna(0).gt(0).to_numpy().reshape(len(ti),len(ll))
  # Finite evidence after the current deadline is observational evidence, not proof of first publication.
  later=np.zeros_like(valid)
  if n in ex.DELPHI_COVARIATES:
   for j,d in enumerate(cut):
    z=g[(g.release>d)&g['count'].gt(0)].groupby(['day','location'])['count'].max().reindex(ix)
    later[j]=z.notna().to_numpy().reshape(len(ti),len(ll))
  else:
   for j,w in enumerate(wi):later[j]=np.isfinite(allv[w+1:]).any(axis=0)
  status=np.full(valid.shape,2,dtype=np.uint8)
  status[~valid & ~ever[None,:,:]]=5
  status[known & ~valid]=4
  status[known & ~valid & later]=3
  status[valid]=1
  future=obs.to_numpy()[None,:]>(issued-pd.Timedelta(days=4)).to_numpy()[:,None];status[np.broadcast_to(future[:,:,None],status.shape)]=0
  native=ever.any(axis=0);chosen=np.where(native & (ll!='US'))[0]
  if not len(chosen):chosen=np.where(ll=='US')[0]
  q=status[:,:,chosen];agg=np.where((q==q[:,:,:1]).all(axis=2),q[:,:,0],6)
  crosses=np.isin(q,[4,5]).any(axis=2)
  fig,ax=plt.subplots(figsize=(12,10));ax.imshow(agg,origin='lower',aspect='auto',cmap=cm,norm=norm,interpolation='nearest');yy,xx=np.where(crosses);ax.scatter(xx,yy,marker='x',s=8,c='#252525',linewidths=.5)
  ax.set_xticks(np.arange(0,len(obs),5),obs[::5].strftime('%b %d'),rotation=45,ha='right');ax.set_yticks(np.arange(0,len(issued),4),issued[::4].strftime('%Y-%m-%d'));ax.set_xlabel('Observation week');ax.set_ylabel('Submission round (later upward)');ax.set_title(n+' — availability evidence\nMixed = locations have different statuses; × = at least one crossed status')
  fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,labels)],loc='lower center',ncol=2,fontsize=9);fig.subplots_adjust(bottom=.22);fig.savefig(out/(n+'-evidence.png'),dpi=140);plt.close(fig)
  items.append(dict(name=n,locations=ll.tolist(),native=native.tolist(),status=status.tolist(),aggregate=agg.tolist(),cross=crosses.tolist()))
  for k,label in enumerate(labels[:-1]):totals.append(dict(series=n,status=label,cells=int((status==k).sum())))
data=dict(observations=obs.strftime('%Y-%m-%d').tolist(),issuances=issued.strftime('%Y-%m-%d').tolist(),cutoffs=[str(d) for d in cut],labels=labels,colors=colors,series=items)
(out/'evidence.json').write_text(json.dumps(data));pd.DataFrame(totals).to_csv(out/'evidence-counts.csv',index=False)
html='''<!doctype html><meta charset="utf-8"><title>Availability evidence</title><style>body{font:16px system-ui;margin:24px;max-width:1300px}select{font:inherit;padding:8px}canvas{width:100%}#tip{background:#eef2f5;padding:16px;min-height:90px}span{display:inline-block;padding:7px;margin:3px}</style><h1>Was it late, unknown, or explicitly missing?</h1><p>Gray means timing is unknown: infer next-season timing from documented periods, not from this gap. Blue requires an explicit missing report followed by a later finite report. × means explicit missing with no later value observed, or no value anywhere in the inspected history; it does not prove permanent absence.</p><section style="background:#eef2f5;padding:16px;margin:16px 0"><h2 style="margin-top:0">How to read the axes</h2><p><strong>X-axis:</strong> the Saturday week the observation describes. <strong>Y-axis:</strong> the nominal Wednesday forecast round; later rounds are higher.</p><p><strong>T−0 is the diagonal boundary:</strong> the Saturday immediately before that Wednesday, four days earlier. On the same row, one cell left is <strong>T−1</strong> (the previous Saturday), and two cells left is T−2. Cells to the right are future observation weeks and are white.</p><p>For <strong>Wednesday January 14, 2026</strong>, T−0 is <strong>Saturday January 10</strong>, and T−1 is <strong>January 3</strong>. Follow a column upward to see when that same observation became available.</p><p>Holiday extensions change the actual cutoff, but keep the same Saturday and nominal Wednesday row. Hover shows the actual cutoff. After the final observation week shown, T−0 lies beyond the right edge.</p></section><select id="series"></select> <select id="location"></select><p id="legend"></p><canvas width="1200" height="950"></canvas><div id="tip">Select a state and hover a cell for its evidence. Aggregate view can contain different statuses.</div><p>States outside native coverage are excluded from aggregate view but selectable individually. Missing reports may reflect suppression or withdrawal. Archives cannot establish that unknown values were genuinely unpublished. No lag is inferred in this evidence view.</p><script>const D=DATA,s=document.querySelector('#series'),l=document.querySelector('#location'),c=document.querySelector('canvas'),ctx=c.getContext('2d'),tip=document.querySelector('#tip');D.series.forEach((v,i)=>s.add(new Option(v.name,i)));document.querySelector('#legend').innerHTML=D.labels.map((v,i)=>'<span style="background:'+D.colors[i]+'">'+([4,5].includes(i)?'× ':'')+v+'</span>').join('');const X=135,Y=55,W=1010,H=780,cw=W/D.observations.length,ch=H/D.issuances.length;function draw(){const a=D.series[+s.value],li=+l.value;ctx.clearRect(0,0,1200,950);ctx.font='17px system-ui';ctx.fillStyle='black';ctx.fillText(a.name+' — '+l.options[l.selectedIndex].text,X,25);for(let j=0;j<D.issuances.length;j++){let y=Y+H-(j+1)*ch;for(let k=0;k<D.observations.length;k++){let z=li<0?a.aggregate[j][k]:a.status[j][k][li];ctx.fillStyle=D.colors[z];ctx.fillRect(X+k*cw,y,cw,ch);if(li<0?a.cross[j][k]:[4,5].includes(z)){ctx.strokeStyle='#222';ctx.beginPath();ctx.moveTo(X+k*cw+4,y+3);ctx.lineTo(X+(k+1)*cw-4,y+ch-3);ctx.moveTo(X+(k+1)*cw-4,y+3);ctx.lineTo(X+k*cw+4,y+ch-3);ctx.stroke()}}if(j%4===0){ctx.fillStyle='black';ctx.font='12px system-ui';ctx.fillText(D.issuances[j],5,y+ch)}}for(let k=0;k<D.observations.length;k+=4){ctx.save();ctx.fillStyle='black';ctx.translate(X+k*cw,Y+H+15);ctx.rotate(Math.PI/4);ctx.fillText(D.observations[k],0,0);ctx.restore()}}function setseries(){l.innerHTML='';l.add(new Option('Native states/DC aggregate',-1));D.series[+s.value].locations.forEach((v,i)=>l.add(new Option(v+(D.series[+s.value].native[i]?'':' (no observed native support)'),i)));draw()}s.onchange=setseries;l.onchange=draw;c.onmousemove=e=>{let r=c.getBoundingClientRect(),k=Math.floor(((e.clientX-r.left)*1200/r.width-X)/cw),j=D.issuances.length-1-Math.floor(((e.clientY-r.top)*950/r.height-Y)/ch);if(k<0||j<0||k>=D.observations.length||j>=D.issuances.length)return;let a=D.series[+s.value],li=+l.value,parts=[];if(li>=0)parts=[a.locations[li]+': '+D.labels[a.status[j][k][li]]];else D.labels.slice(0,6).forEach((label,z)=>{let loc=a.locations.filter((v,i)=>a.native[i]&&(v!=='US'||a.locations.length===1)&&a.status[j][k][i]===z);if(loc.length)parts.push(label+' ('+loc.length+'): '+loc.join(', '))});tip.textContent='Observation '+D.observations[k]+'; submission '+D.issuances[j]+'; cutoff '+D.cutoffs[j]+'. '+parts.join(' | ')};setseries();</script>'''
(out/'evidence.html').write_text(html.replace('DATA',json.dumps(data)))
text='''# Availability evidence: unknown timing, later arrival, and missing values

**[Open the evidence explorer](timeline/evidence.html)**. Select a covariate/target and then a state. The aggregate view can mix statuses across states; hover lists the states in each category.

| Mark | What the archive establishes | Use for next season |
| --- | --- | --- |
| Green | A finite value was reported by this deadline | Observed availability |
| Gray | No eligible report; a value exists elsewhere in the inspected history or final panel | Timing unknown. Infer timing from documented periods, not this gap |
| Blue | An explicit missing statement at this deadline, followed by a later finite report | Documented missing-then-present transition; not necessarily first-ever publication |
| Red × | Explicit missing statement, no later finite report observed in our archive | Missing/withdrawn at that deadline; permanent absence is not proven |
| Beige × | No finite value anywhere in the inspected archive or final panel for that location/week | Never observed in this dataset; may be unsupported or an unresolved archive gap |
| Yellow | Different locations have different evidence | Hover or select a state |
| White | Observation week is still in the future | Not an availability failure |

A value appearing in a later archive **does not by itself prove late publication**. It could be an old value first collected later. We therefore require a preceding explicit missing statement for blue; otherwise timing stays gray. An explicit null can also mean suppression or withdrawal and is not evidence of “never ever.” Crosses are separated by color for that reason. Native unsupported states can be selected individually; aggregate views exclude them.

The classification uses known-report masks in the frozen deadline panel, plus raw claims/ILI/lab/FluSurv statement histories before conflict rejection. Finite conflicting claims count as reported. For those raw sources, later reports are checked through the pinned snapshot; for other sources, through the panel's last available issuance. Final-only values do not prove when a later report occurred. No speculative release delays are filled into this view.

For next-season assumptions, use stable observed release periods to estimate delay, and label any extension over gray periods as inferred. This page does not convert those assumptions into historical observations. The prior coverage heatmaps still answer how many values the archive supplies; this evidence view answers what we can conclude about missing cells.

[Coverage heatmaps](timeline.md) · [Evidence data](timeline/evidence.json) · [Evidence counts](timeline/evidence-counts.csv).

'''
for item in items:text+=f'## {item["name"]}\n\n![Evidence for {item["name"]}](timeline/{item["name"]}-evidence.png)\n\n'
(p/'evidence.md').write_text(text)
print('Wrote 20 evidence matrices, static plots, and interactive state selector.')

# Keep the original user-facing heatmap routes on the evidence classification.
coverage = out/'coverage-explorer.html'
if not coverage.exists() and (out/'explorer.html').exists():
 coverage.write_text((out/'explorer.html').read_text())
(out/'explorer.html').write_text((out/'evidence.html').read_text())
(p/'timeline.md').write_text(text.replace('# Availability evidence: unknown timing, later arrival, and missing values', '# Availability over the season: timing and missing-value evidence').replace('(timeline/evidence.html)', '(timeline/explorer.html)').replace('[Coverage heatmaps](timeline.md)', '[Coverage-count explorer](timeline/coverage-explorer.html)'))
