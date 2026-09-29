"""Saved-forecast fans for the three leaders and the overall leader's matched controls.

Seed 42 is fixed before inspecting curves. No refitting, resampling or calibration.
All fan cells use the same frozen Hub support; full-season truth supplies context.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from tapestry.dataset.build import load
from tapestry.dataset.cv import SEASONS, season
from tapestry.evaluation.hubs import CHANNEL, export
from tapestry.evaluation.totals import frozen_cases
from tapestry.data.geography import STATE_FIPS

ROOT = Path('data/experiments/b2-direct-research-v1')
OUT = Path('docs/results/b2-direct-research-v1/fans')
RANK = ROOT / 'ranking-ac92a3c7dbcc'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(RANK / 'configuration_ranking.csv').sort_values('combined_mean')
    assert len(ranked) == 156 and ranked.seeds.eq(3).all()
    design = pd.DataFrame(json.loads((ROOT/'research-design.json').read_text())['rows']).rename(columns={'scenario':'config_id'})
    ranked = ranked.merge(design, on='config_id', validate='one_to_one')
    manifest = json.loads((RANK/'manifest.json').read_text())
    settings = json.loads((ROOT/'experiment.json').read_text())
    panel = load(settings['dataset'])
    frames = {}
    chosen = ranked.head(3).config_id.tolist()
    leader = ranked.iloc[0]
    controls = []
    for source, spatial in [('none', leader.spatial), (leader.sources, 'none')]:
        rows = ranked[(ranked.backbone==leader.backbone)&(ranked.sources==source)&(ranked.spatial==spatial)&(ranked.location_embedding==0)]
        assert len(rows)==1
        controls.append(rows.iloc[0].config_id)
    selected = list(dict.fromkeys(chosen+controls))
    for config in selected:
        matches = [r for r in manifest['runs'] if r['config_id']==config and r['seed']==42]
        assert len(matches)==1
        frames[config]=export(matches[0]['path'])
    ensemble={}
    for case in frozen_cases(settings['frozen']):
        table=pd.read_parquet(Path(settings['frozen'])/case['directory']/'quantiles.parquet')
        ensemble[(case['season'],case['target'])]=table[table.model==case['ensemble']].copy()
    titles={}
    for _,r in ranked[ranked.config_id.isin(selected)].iterrows():
        backbone={'target-multiscale':'Target multiscale','target-mlp':'Target MLP','pathogen-mlp':'Pathogen MLP'}[r.backbone]
        exchange={'pooled':'pooled geography','gated_pool':'gated geography','national_broadcast':'national broadcast','none':'independent locations'}.get(r.spatial,r.spatial)
        titles[r.config_id]=f'{backbone} · {r.sources} covariates\n{exchange} · overall WIS {r.combined_mean:.3f}'
    dates=pd.to_datetime(panel['dates'].astype(str)); labels=np.array([season(d) for d in dates.strftime('%Y-%m-%d')])
    fips={v:k for k,v in STATE_FIPS.items()}|{'US':'US'}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    selections={'best':chosen,'controls':[controls[0],controls[1],chosen[0]]}
    for selection, configs in selections.items():
        for season_label in ['2025-2026'] + (['all'] if selection=='best' else []):
            holds=list(SEASONS) if season_label=='all' else [season_label]
            mask=np.isin(labels,holds)
            ref=[]
            for held in holds:
                ref += sorted(frames[chosen[0]][(held,'wk inc flu hosp')].reference_date.unique())[::4]
            for location in ('US','NC'):
                li=panel['locations'].tolist().index(location)
                for kind in ('hosp','ed'):
                    targets=[t for t in CHANNEL if ('prop ed' in t)==(kind=='ed')]
                    fig,axes=plt.subplots(3,4,figsize=(19,9),sharex=True,sharey='row',squeeze=False)
                    for col,config in enumerate([None,*configs]):
                        for row,target in enumerate(targets):
                            ax=axes[row,col];factor=100 if kind=='ed' else 1
                            ax.plot(dates[mask],panel['targets'][mask,li,CHANNEL[target]]*factor,color='#171717',lw=1.5,zorder=4)
                            for held in holds:
                                key=(held,target)
                                if key not in ensemble:continue
                                support=ensemble[key]
                                support=support[support.location.eq(fips[location])&support.reference_date.isin(ref)]
                                if config is None:part=support
                                else:
                                    part=frames[config][key]
                                    part=part[part.location.eq(fips[location])].merge(support[['reference_date','target_end_date','horizon']].drop_duplicates(),on=['reference_date','target_end_date','horizon'],validate='one_to_one')
                                for _,fan in part.groupby('reference_date'):
                                    fan=fan.sort_values('horizon');x=pd.to_datetime(fan.target_end_date)
                                    color='#686868' if config is None else '#176b98'
                                    ax.fill_between(x,fan['q0.05']*factor,fan['q0.95']*factor,color=color,alpha=.14,lw=0)
                                    ax.fill_between(x,fan['q0.25']*factor,fan['q0.75']*factor,color=color,alpha=.30,lw=0)
                                    ax.plot(x,fan['q0.5']*factor,color=color,lw=1.1)
                            ax.set_ylim(bottom=0)
                            ax.grid(axis='y',alpha=.14)
                            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3 if season_label=='all' else 2))
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
                            ax.set_xlim(dates[mask].min(),dates[mask].max())
                            if col==0:ax.set_ylabel(target.split()[2].upper()+'\n'+('ED visits (%)' if kind=='ed' else 'Weekly admissions'))
                            if row==0:ax.set_title('Hub ensemble' if config is None else titles[config],fontsize=10)
                    handles=[plt.Line2D([],[],color='#171717',lw=1.5),plt.Line2D([],[],color='#176b98',lw=1.1),plt.Rectangle((0,0),1,1,color='#176b98',alpha=.30),plt.Rectangle((0,0),1,1,color='#176b98',alpha=.14)]
                    fig.legend(handles,['Finalized truth','Forecast median','50% interval','90% interval'],loc='lower center',ncol=4,frameon=False)
                    fig.suptitle(f'{location} · {"Admissions" if kind=="hosp" else "ED visits"} · {season_label if season_label!="all" else "three held-out seasons"}\n'+('Top three overall configurations' if selection=='best' else 'Leader and matched source / geographic controls')+'; fixed seed 42; every fourth reference week on frozen Hub support',fontsize=13)
                    fig.tight_layout(rect=(0,.045,1,.925))
                    path=OUT/f'{selection}-{season_label}-{location}-{kind}.png'
                    fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig);print(path,flush=True)
    (OUT/'selection.json').write_text(json.dumps(dict(ranking=str(RANK),selection='Top three configurations by complete three-seed overall mean; matched no-covariate and no-exchange controls for leader',seed=42,evaluation_members=settings['eval_members'],intervals=[50,90],reference_spacing_weeks=4,score_support='Identical frozen target/location/reference/horizon cells for every plotted model and ensemble',selections=selections,titles=titles),indent=2)+'\n')

if __name__=='__main__':main()
