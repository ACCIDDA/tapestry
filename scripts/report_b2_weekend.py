"""Rank completed weekend fits and draw explicit season-specific comparisons."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.experiment.planner import rank
from tapestry.evaluation.totals import season_composites
from tapestry.model.scenario import Scenario

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('-e','--experiment',default='b2-weekend-20261002')
a=p.parse_args()
root=Path('data/experiments')/a.experiment
out=Path('docs/experiments')/a.experiment
out.mkdir(parents=True,exist_ok=True)
ranking=rank(root,allow_incomplete=True,make_plots=False)
design=pd.read_csv(root/'design.csv').rename(columns={'scenario':'config_id'})
scores=pd.read_csv(ranking/'season_scores.csv')
composite=season_composites(scores).merge(design,on='config_id',validate='many_to_one')
composite['training_seasons']=composite.season.map({'2024-2025':'2022-23, 2023-24, 2025-26','2025-2026':'2022-23, 2023-24, 2024-25'})
composite['error_source_season']=composite.season.map({'2024-2025':'2025-26','2025-2026':'2024-25'})
composite['evaluation_inputs']=composite.family.map({'direct':'Wednesday reports on B2 schedule; final proxies for archive gaps','joint':'Wednesday reports on B2 schedule; final proxies for archive gaps','two_stage':'Training-season regression applied to Wednesday reports on B2 schedule; final proxies for archive gaps'})
composite['prediction_labels']='Finalized t+1 to t+4; joint models additionally learn finalized t-3 to t0'
composite['model']=composite.config_id.map(lambda c: (lambda s:f'{s.fit_partition} {s.encoder}; {s.spatial} spatial; covariates {s.covariate_set or "none"}; {s.covariate_encoder} encoding')(Scenario.from_string(c)))
base=composite[composite.treatment.eq('Unchanged finalized training inputs')][['architecture','seed','season','geography','combined']].rename(columns={'combined':'finalized_training_raw_evaluation'})
paired=composite.merge(base,on=['architecture','seed','season','geography'],how='left',validate='many_to_one')
paired['change_percent']=100*(paired.combined/paired.finalized_training_raw_evaluation-1)
paired.to_csv(out/'paired-season-seed-scores.csv',index=False)
scores.merge(design,on='config_id',validate='many_to_one').to_csv(out/'target-season-scores.csv',index=False)
summary=paired.groupby(['architecture','source_b2_label','model','family','treatment','season','training_seasons','error_source_season','evaluation_inputs','prediction_labels','geography'],dropna=False).agg(
    seeds=('seed','nunique'),mean_wis_ratio=('combined','mean'),sd_wis_ratio=('combined','std'),mean_paired_change_percent=('change_percent','mean')).reset_index()
summary.to_csv(out/'summary.csv',index=False)
for season,train,donor in [('2024-2025','2022–23, 2023–24, 2025–26','2025–26'),('2025-2026','2022–23, 2023–24, 2024–25','2024–25')]:
    q=summary.query('season == @season and geography == "all" and seeds == 3').sort_values('mean_wis_ratio').head(20)
    if q.empty: continue
    labels=[f'A{r.architecture}: {r.treatment}\n{r.model}' for r in q.itertuples()]
    fig,ax=plt.subplots(figsize=(17,12))
    y=list(range(len(q)))
    ax.errorbar(q.mean_wis_ratio,y,xerr=q.sd_wis_ratio,fmt='o',capsize=3,color='#287e86')
    ax.set_yticks(y,labels,fontsize=7);ax.invert_yaxis();ax.axvline(1,color='gray',ls='--')
    ax.set_xlabel('WIS / matched Hub WIS; lower is better. Bars show SD across three training seeds.')
    ax.set_title(f'Evaluate {season}; train on {train}; reporting errors from {donor}\nFinalized prediction labels; Wednesday values with B2 availability and explicit final proxies\nSeparate nowcaster models correct evaluation reports; direct and joint models use raw reports',fontsize=10)
    fig.tight_layout();fig.savefig(out/f'leading-models-{season}.png',dpi=160);plt.close(fig)
    q=summary.query('season == @season and geography == "all" and seeds == 3')
    grid=q.pivot(index='treatment',columns='architecture',values='mean_paired_change_percent')
    if grid.notna().any().any():
        fig,ax=plt.subplots(figsize=(18,13))
        im=ax.imshow(grid.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-20,vmax=20)
        ax.set_yticks(range(len(grid)),grid.index,fontsize=8)
        ax.set_xticks(range(len(grid.columns)),[f'A{x}' for x in grid.columns],fontsize=7,rotation=90)
        ax.set_title(f'Evaluate {season}; train on {train}; errors from {donor}\nThree-seed paired WIS change versus the same architecture trained on finalized inputs and evaluated on raw reports\nAll learn finalized labels; B2 availability and final proxies. Two-stage treatments also correct evaluation inputs.',fontsize=10)
        fig.colorbar(im,ax=ax,label='WIS change (%); negative is better; colors capped at ±20%')
        fig.tight_layout();fig.savefig(out/f'paired-treatments-{season}.png',dpi=160);plt.close(fig)
print(summary.sort_values('mean_wis_ratio').head(12).to_string(index=False))
print(out)
