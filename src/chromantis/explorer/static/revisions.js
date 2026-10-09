/* Percentages use raw units; undefined and unavailable are never treated as zero. */
function revisionPercent(report, reference, denominator = 'report') {
  if (!Number.isFinite(report)) return {status:'missing report'};
  if (!Number.isFinite(reference)) return {status:'missing reference'};
  const divisor = denominator === 'report' ? report : reference;
  if (divisor === 0) return report === reference ? {status:'finite',value:0} : {status:'zero denominator'};
  return {status:'finite',value:100 * (reference-report)/divisor};
}
// Independent rounded bounds keep zero visible without forcing symmetry.
function niceAxis(values) {
  const finite = values.filter(Number.isFinite);
  let low = Math.min(0, ...finite), high = Math.max(0, ...finite);
  if (low === high) { low = -1; high = 1; }
  const padding = (high-low)*0.04;
  if (low < 0) low -= padding;
  if (high > 0) high += padding;
  const raw = (high-low)/6, power = 10**Math.floor(Math.log10(raw));
  const step = [1,2,2.5,5,10].find(n=>n*power>=raw)*power;
  const first = Math.floor(low/step), last = Math.ceil(high/step);
  return {low:first*step, high:last*step, ticks:Array.from({length:last-first+1},(_,i)=>(first+i)*step)};
}
if (typeof module !== 'undefined') module.exports = {revisionPercent, niceAxis};
if (typeof document !== 'undefined') (async () => {
  const $ = id => document.getElementById(id), cache = new Map();
  const query = new URLSearchParams(location.search);
  if (query.get('view') === 'revisions') $('split-section').hidden = true;
  if (query.get('view') === 'splits') $('revision-section').hidden = true;
  const fetchJSON = async path => {const r=await fetch(path); if(!r.ok) throw Error(`${path}: ${r.status}`); return r.json();};
  const decode = text => {const bytes=Uint8Array.from(atob(text),c=>c.charCodeAt(0)), view=new DataView(bytes.buffer); return Float32Array.from({length:bytes.length/4},(_,i)=>view.getFloat32(i*4,true));};
  const m = await fetchJSON('manifest.json'); let data, revisionHits=[], splitHits=[], generation=0;
  if (!m.cv.length) $('split-section').hidden = true;  // no finalization calendars exported (route deleted 2026-10-08)
  const option = (select,value,label=value) => select.add(new Option(label,value));
  m.series.forEach(s=>option($('signal'),s.name));
  [...new Set(m.cv.map(d=>d.weeks))].sort((a,b)=>a-b).forEach(n=>option($('weeks'),n)); $('weeks').value=String(Math.max(...m.cv.map(d=>d.weeks)));
  $('metadata').textContent=`Reference snapshot ${m.truth_day} • ${m.issuances.length} Wednesdays • ${m.experiment}. Frozen reference values are treated as final for this comparison.`;
  function ages(){const old=$('age').value||'0'; $('age').replaceChildren(); for(let a=0;a<+$('weeks').value;a++) option($('age'),a,`T−${(data?.lag||0)+a} (boundary + ${a})`); if(+$('weeks').value>+old) $('age').value=old;}
  function range(all=false){$('start').value=m.issuances[all?0:Math.max(0,m.issuances.length-52)]; $('end').value=m.issuances.at(-1);}
  function canvas(id,height){const el=$(id), width=Math.max(650,el.parentElement.clientWidth); el.width=width; el.height=height; el.style.width=width+'px'; el.style.height=height+'px'; const c=el.getContext('2d');c.fillStyle='#fff';c.fillRect(0,0,width,height);c.font='11px system-ui';return {c,width};}
  const format = v => Number.isFinite(v)?v.toLocaleString(undefined,{maximumSignificantDigits:7}):'unavailable';
  function cell(w,k){const t=m.origins[w]-k, l=$('location').selectedIndex, n=data.locations.length;const report=data.reports[(w*12+k)*n+l], reference=t>=0?data.truth[t*n+l]:NaN;return {w,k,t,report,reference,...revisionPercent(report,reference,$('denominator').value)};}
  function visible(){return m.issuances.map((d,i)=>i).filter(i=>m.issuances[i]>=$('start').value&&m.issuances[i]<=$('end').value).reverse();}
  function rounds(){const old=$('round').value; $('round').replaceChildren();option($('round'),'all','All selected Wednesdays');visible().forEach(w=>option($('round'),w,m.issuances[w]));if([...$('round').options].some(o=>o.value===old))$('round').value=old;}
  function selected(){return $('round').value==='all'?visible():[+$('round').value];}
  function drawRevisions(){
    if($('revision-section').hidden)return;
    const rows=selected(), window=+$('window').value, left=76, right=88, top=45, bottom=495;
    const {c,width}=canvas('revisions',570);revisionHits=[];
    if(!rows.length){c.fillStyle='#222';c.fillText('No Wednesdays in this range.',left,top);return;}
    const day=d=>Date.parse(d+'T00:00:00Z'), week=7*86400000;
    const low=Math.max(0,Math.min(...rows.map(w=>m.origins[w]))-window+1), high=Math.max(...rows.map(w=>m.origins[w]));
    const xmin=day(m.dates[low]), xmax=day(m.issuances[Math.max(...rows)]), x=d=>left+(day(d)-xmin)/(xmax-xmin||week)*(width-left-right);
    const points=rows.map(w=>Array.from({length:window},(_,k)=>cell(w,window-1-k)).filter(v=>v.t>=0));
    const finite=points.flat().filter(v=>v.status==='finite');
    const axis=niceAxis(finite.map(v=>v.value)), y=v=>bottom-(v-axis.low)/(axis.high-axis.low)*(bottom-top);
    const n=data.locations.length, li=$('location').selectedIndex, reference=[];
    for(let t=low;t<=high;t++)reference.push({t,value:data.truth[t*n+li]});
    const referenceAxis=niceAxis(reference.map(v=>v.value)), yr=v=>bottom-(v-referenceAxis.low)/(referenceAxis.high-referenceAxis.low)*(bottom-top);
    c.fillStyle='#374858';c.fillText('Revision (%)',left,20);c.textAlign='right';c.fillText('Reference (raw units)',width-8,20);c.textAlign='left';
    for(const tick of axis.ticks){const yy=y(tick);c.strokeStyle='#e4e9ee';c.beginPath();c.moveTo(left,yy);c.lineTo(width-right,yy);c.stroke();c.fillStyle='#374858';c.textAlign='right';c.fillText(format(tick)+'%',left-8,yy+4);}
    c.textAlign='left';
    for(const tick of referenceAxis.ticks){const yy=yr(tick);c.strokeStyle='#8293a0';c.beginPath();c.moveTo(width-right,yy);c.lineTo(width-right+4,yy);c.stroke();c.fillStyle='#374858';c.fillText(format(tick),width-right+8,yy+4);}
    c.strokeStyle='#8293a0';c.beginPath();c.moveTo(left,y(0));c.lineTo(width-right,y(0));c.stroke();
    for(let t=low;t<=high;t+=Math.max(1,Math.ceil((high-low+1)/6))){c.save();c.translate(x(m.dates[t]),bottom+20);c.rotate(-.35);c.fillStyle='#374858';c.fillText(m.dates[t],0,0);c.restore();}
    c.fillStyle='#374858';c.fillText('Observation date (Saturday); dotted tail ends on forecast Wednesday',left,557);
    c.strokeStyle='#222';c.lineWidth=1.8;c.setLineDash([7,5]);c.beginPath();let connected=false;
    reference.forEach(v=>{if(!Number.isFinite(v.value)){connected=false;return;}if(connected)c.lineTo(x(m.dates[v.t]),yr(v.value));else c.moveTo(x(m.dates[v.t]),yr(v.value));connected=true;});c.stroke();c.setLineDash([]);
    points.forEach((series,i)=>{const w=rows[i], hue=(w*137.508)%360, color=`hsl(${hue} 62% 40%)`;c.strokeStyle=color;c.fillStyle=color;c.lineWidth=rows.length===1?2.5:1.25;c.globalAlpha=rows.length===1?1:.68;c.beginPath();let joined=false;
      series.forEach(v=>{if(v.status!=='finite'){joined=false;return;}const xx=x(m.dates[v.t]), yy=y(v.value);if(joined)c.lineTo(xx,yy);else c.moveTo(xx,yy);joined=true;});c.stroke();
      series.filter(v=>v.status==='finite').forEach(v=>{const xx=x(m.dates[v.t]),yy=y(v.value);c.beginPath();c.arc(xx,yy,rows.length===1?3:1.8,0,Math.PI*2);c.fill();revisionHits.push({...v,x:xx-5,y:yy-5,width:10,height:10});});
      const newest=series.find(v=>v.k===0&&v.status==='finite');if(newest){const xx=x(m.issuances[w]), yy=y(newest.value);c.setLineDash([2,4]);c.beginPath();c.moveTo(x(m.dates[newest.t]),yy);c.lineTo(xx,yy);c.stroke();c.setLineDash([]);c.beginPath();c.moveTo(xx,yy-4);c.lineTo(xx,yy+4);c.stroke();}
      c.globalAlpha=1;
    });
    const all=points.flat();$('formula').textContent=`100 × (reference − Wednesday report) / ${$('denominator').value==='report'?'Wednesday report':'reference'}. Both zero → 0%; other zero denominators → undefined. Axes auto-scale without clipping; revision gridlines follow rounded percentage ticks. Zero is included, not centered.`;
    $('revision-summary').textContent=`${rows.length} forecast rounds · ${window} trailing weeks each · ${finite.length} comparable points (${finite.filter(v=>v.value===0).length} unchanged) · ${all.filter(v=>v.status!=='finite').length} missing/undefined points. Choose one forecast round to isolate its line. Reference snapshot: ${m.truth_day}.`;
  }
  function drawSplits(){if($('split-section').hidden)return;const folds=m.cv.filter(d=>d.protocol===$('protocol').value&&d.weeks===+$('weeks').value), age=+$('age').value, li=$('location').selectedIndex, n=data.locations.length, left=116, top=35, rh=155;
    const {c,width}=canvas('splits',top+folds.length*rh+45), cw=(width-left-15)/m.dates.length;splitHits=[];
    const values=m.dates.map((_,t)=>data.truth[t*n+li]), finite=values.filter(Number.isFinite), min=finite.length?Math.min(0,...finite):0,max=finite.length?Math.max(...finite):1, span=max-min||1;
    folds.forEach((f,r)=>{const y=top+r*rh, layout=data.cv[f.id], roles=new Uint8Array(m.dates.length);for(const[a,b,role]of layout.runs[age][li]) roles.fill(role,a,b);
      c.fillStyle='#374858';c.fillText(f.fold,4,y+15);c.fillText('cutoff',4,y+34);c.fillText(f.fit_cutoff,4,y+49);c.fillText(format(max),4,y+70);
      roles.forEach((role,t)=>{const x=left+t*cw;c.fillStyle=['#eee','#cfe0f3','#f3b6b6','#f6c77b'][role];c.fillRect(x,y,cw+.2,rh-28);splitHits.push({x,y,width:cw,height:rh-28,t,role,fold:f,layout});});
      c.strokeStyle='#202a33';c.lineWidth=1;c.beginPath();let connected=false;values.forEach((v,t)=>{if(!Number.isFinite(v)){connected=false;return;}const x=left+(t+.5)*cw, yy=y+rh-33-(v-min)/span*(rh-40);if(connected)c.lineTo(x,yy);else c.moveTo(x,yy);connected=true;});c.stroke();
      if(layout.status!=='complete'){c.fillStyle='#854321';c.fillText(`No fit: ${layout.reason}`,left+8,y+18);}
      c.fillStyle='#374858';c.fillText(`${roles.filter(x=>x===1).length} training · ${roles.filter(x=>x===3).length} validation weeks · ${roles.filter(x=>x===2).length} held-out weeks`,left,y+rh-10);
    });
    for(let t=0;t<m.dates.length;t+=Math.ceil(m.dates.length/7)){c.fillStyle='#374858';c.fillText(m.dates[t],left+t*cw,top+folds.length*rh+20);}
    $('split-summary').textContent=`Actual ${$('protocol').value} masks, R=${$('weeks').value}, T−${data.lag+age}. Orange marks missing-cell validation inside training; reported correction strength uses up to 52 training weeks. Held-out evaluation stays separate. Four-week maturity/gap; held-out observation dates are purged from training across all reconstructed ages. Colors describe calibration labels at this age; causal triangle updates also use available report pairs during evaluation.`;
  }
  function redraw(){if(data){drawRevisions();drawSplits();}}
  async function selectSignal(){const token=++generation, name=$('signal').value, old=$('location').value||'US';if(!cache.has(name)){const d=await fetchJSON(m.series.find(s=>s.name===name).file);d.truth=decode(d.truth);d.reports=decode(d.reports);cache.set(name,d);}if(token!==generation)return;data=cache.get(name);$('location').replaceChildren();data.locations.forEach(l=>option($('location'),l));$('location').value=data.locations.includes(old)?old:data.locations[0];ages();redraw();}
  function hover(id,hits,describe){for(const event of ['mousemove','click']) $(id).addEventListener(event,e=>{const b=$(id).getBoundingClientRect(), x=(e.clientX-b.left)*$(id).width/b.width,y=(e.clientY-b.top)*$(id).height/b.height;const v=hits().find(v=>x>=v.x&&x<v.x+v.width&&y>=v.y&&y<v.y+v.height);if(v)describe(v);});}
  hover('revisions',()=>revisionHits,v=>{$('revision-detail').textContent=`${data.name} · ${$('location').value}\nWednesday ${m.issuances[v.w]} · observation ${m.dates[v.t]} (T−${v.k})\nReported: ${format(v.report)} → reference: ${format(v.reference)} (${m.truth_day})\n${v.status==='finite'?`${v.value>0?'+':''}${format(v.value)}%`:v.status}`;});
  hover('splits',()=>splitHits,v=>{const d=new Date(m.dates[v.t]+'T00:00:00Z');d.setUTCDate(d.getUTCDate()+4+7*(data.lag+(+$('age').value)));$('split-detail').textContent=`${data.name} · ${$('location').value} · ${v.fold.fold}\nObservation ${m.dates[v.t]} → Wednesday ${d.toISOString().slice(0,10)}\n${['Not used','Training statistics','Held-out evaluation','Missing-cell validation'][v.role]} · reference value: ${format(data.truth[v.t*data.locations.length+$('location').selectedIndex])}${v.layout.status!=='complete'?'\n'+v.layout.reason:''}`;});
  $('signal').onchange=()=>selectSignal().catch(e=>$('error').textContent=e.message);$('location').onchange=redraw;
  for(const id of ['window','round','denominator','protocol','age'])$(id).onchange=redraw;
  for(const id of ['start','end'])$(id).onchange=()=>{rounds();redraw();};
  $('weeks').onchange=()=>{ages();redraw();};$('recent').onclick=()=>{range();rounds();redraw();};$('all').onclick=()=>{range(true);rounds();redraw();};
  $('download').onclick=()=>{const lines=[['signal','location','Wednesday','observation','T_offset','report','reference','percent','status','denominator']];for(const w of selected())for(let k=0;k<+$('window').value;k++){const v=cell(w,k);if(v.t>=0)lines.push([data.name,$('location').value,m.issuances[w],m.dates[v.t],k,Number.isFinite(v.report)?v.report:'',Number.isFinite(v.reference)?v.reference:'',v.value??'',v.status,$('denominator').value]);}const url=URL.createObjectURL(new Blob([lines.map(row=>row.join(',')).join('\n')],{type:'text/csv'})), a=document.createElement('a');a.href=url;a.download=`revisions-${data.name}-${$('location').value}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  window.addEventListener('resize',redraw);range();rounds();await selectSignal();
})().catch(e=>{document.getElementById('error').textContent=e.message;});
