"""Completed B2 study: tables, compact scientific figures and narrative report."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.evaluation.totals import run_scores

OUT=Path('docs/results/b2-direct-research-v1')
RANK=Path('data/experiments/b2-direct-research-v1/ranking-ac92a3c7dbcc')
BACKBONES=['pathogen-mlp','target-mlp','target-multiscale']
BN={'pathogen-mlp':'Pathogen MLP','target-mlp':'Target MLP','target-multiscale':'Target multiscale'}
SN={'none':'None','all':'All seven','kinsa':'Kinsa','ilinet':'ILI','inpatient':'Inpatient claims','outpatient':'Outpatient claims','ww_wval_like':'Wastewater','clinical_lab':'Clinical labs','flusurv':'FluSurv'}
SP={'none':'Independent','pooled':'Pooled','gated_pool':'Gated pool','attention':'Attention','national_broadcast':'National broadcast','pathogen_spatial':'Pathogen tokens','target_spatial':'Target tokens','joint_location_target':'Joint location/target tokens'}


def table(frame):
    clean=frame.fillna('—').astype(str)
    return '| '+' | '.join(clean.columns)+' |\n| '+' | '.join(['---']*len(clean.columns))+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in clean.values)


def main():
    design=pd.DataFrame(json.loads(Path('analysis/b2-research/design.json').read_text())['rows']).rename(columns={'scenario':'config_id'})
    rank=pd.read_csv(RANK/'configuration_ranking.csv').merge(design,on='config_id').sort_values('combined_mean')
    runs=pd.read_csv(RANK/'run_scores.csv');seasons=pd.read_csv(RANK/'season_composite_scores.csv');cells=pd.read_csv(RANK/'season_scores.csv')
    matched=pd.read_csv(OUT/'matched_summary.csv');paired=pd.read_csv(OUT/'paired_seed_scores.csv')
    summary=matched.query('geography=="all" and season=="all" and target=="combined"')
    assert len(rank)==156 and rank.seeds.eq(3).all()
    assert len(runs.query('geography=="all"'))==468 and summary.paired_seeds.eq(3).all()
    rank.to_csv(OUT/'leaderboard.csv',index=False)
    for name in ['run_scores.csv','season_scores.csv','season_composite_scores.csv','manifest.json']:
        shutil.copy2(RANK/name,OUT/('ranking-manifest.json' if name=='manifest.json' else name))
    top=rank.iloc[0];leader=top.config_id
    source=summary.query('kind=="source-addition" and spatial=="none" and location_embedding==0 and not sources.str.startswith("all-minus")',engine='python')
    conditional=summary.query('kind=="source-conditional-addition"')
    spatial=summary.query('kind=="spatial" and location_embedding==0')
    spatial_table=spatial.groupby('spatial').agg(comparisons=('percent_mean','size'),improved=('percent_mean',lambda x:int((x<0).sum())),all_seeds=('seeds_improved',lambda x:int((x==3).sum())),median_percent=('percent_mean','median')).reset_index()
    spatial_table.to_csv(OUT/'spatial-summary.csv',index=False)
    source.to_csv(OUT/'singleton-summary.csv',index=False);conditional.to_csv(OUT/'conditional-summary.csv',index=False)
    coverage=[]
    for row in rank.head(10).itertuples():
        selected=cells[cells.config_id==row.config_id]
        for who in ['model','ensemble']:
            q=run_scores(selected,value=who+'_coverage_95')
            for geo,p in q.groupby('geography'):
                coverage.append(dict(config_id=row.config_id,rank=row.rank,who=who,geography=geo,coverage95_mean=p.combined.mean()))
    pd.DataFrame(coverage).to_csv(OUT/'coverage-summary.csv',index=False)
    target=cells[(cells.config_id==leader)&(cells.geography=='all')&(cells.season=='2025-2026')].groupby('target').agg(wis=('wis_ratio','mean'),coverage95=('model_coverage_95','mean'),ensemble95=('ensemble_coverage_95','mean')).reset_index()
    target.to_csv(OUT/'leader-targets-2025-2026.csv',index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True,layout='constrained')
    for i,row in enumerate(rank.head(12).itertuples()):
        for ax,data in [(axes[0],runs.query('geography=="all"')),(axes[1],seasons.query('geography=="all" and season=="2025-2026"'))]:
            vals=data[data.config_id==row.config_id].combined
            ax.scatter(vals,np.repeat(i,3),s=20,color='#176b98',alpha=.55)
            ax.scatter([vals.mean()],[i],s=65,color='#132b43',marker='D')
    for ax,title in zip(axes,['Overall ranking','2025–26, same configurations']):
        ax.axvline(1,color='#a64738',ls='--',lw=1);ax.set_title(title);ax.set_xlabel('WIS ratio to Hub ensemble (lower is better)');ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(range(12),[f'#{i+1} {BN[r.backbone]}\n{SN[r.sources]} · {SP[r.spatial]}'+(' · ID8' if r.location_embedding else '') for i,r in enumerate(rank.head(12).itertuples())]);axes[0].invert_yaxis()
    fig.suptitle('Three seeds per configuration · dots = seeds, diamonds = means\nTop 12 by overall score; selection uses these same historical seasons')
    fig.savefig(OUT/'leaders.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained',sharey=True)
    sources=['inpatient','outpatient','ww_wval_like','kinsa','ilinet','clinical_lab','flusurv','all']
    for ax,data,key,title in [(axes[0],source,'sources','Add one source to no covariates'),(axes[1],conditional,'label','Add a source back to all-minus-one')]:
        mat=np.full((8,3),np.nan);wins=np.zeros((8,3))
        for i,name in enumerate(sources):
            for j,b in enumerate(BACKBONES):
                q=data[(data[key]==name)&(data.backbone==b)]
                if len(q):mat[i,j]=q.iloc[0].percent_mean;wins[i,j]=q.iloc[0].seeds_improved
        im=ax.imshow(mat,cmap='RdBu_r',vmin=-10,vmax=10,aspect='auto')
        for (i,j),v in np.ndenumerate(mat):
            ax.text(j,i,'—' if np.isnan(v) else f'{v:+.1f}%'+(' *' if wins[i,j]==3 else ''),ha='center',va='center',color='white' if abs(v)>7 else 'black')
        ax.set_xticks(range(3),['Pathogen\nMLP','Target\nMLP','Target\nmultiscale']);ax.set_yticks(range(8),[SN[s] for s in sources]);ax.set_title(title)
    fig.colorbar(im,ax=axes,label='Mean seed-paired WIS change (%) · negative improves',shrink=.7)
    fig.suptitle('Covariate effects without geographic exchange\n* improves in all three seeds; these are optimization replicates, not confidence tests')
    fig.savefig(OUT/'source-effects.png',dpi=170,bbox_inches='tight');plt.close(fig)
    methods=['pooled','national_broadcast','gated_pool','attention','pathogen_spatial','target_spatial','joint_location_target']
    columns=[(b,src) for b in BACKBONES for src in ['none','kinsa','ilinet','all']]
    mat=np.zeros((7,12));wins=np.zeros((7,12))
    for i,method in enumerate(methods):
        for j,(b,src) in enumerate(columns):
            q=spatial[(spatial.backbone==b)&(spatial.sources==src)&(spatial.spatial==method)].iloc[0]
            mat[i,j]=q.percent_mean;wins[i,j]=q.seeds_improved
    fig,ax=plt.subplots(figsize=(15,6),layout='constrained');im=ax.imshow(mat,cmap='RdBu_r',vmin=-15,vmax=15,aspect='auto')
    for (i,j),v in np.ndenumerate(mat):ax.text(j,i,f'{v:+.1f}'+('*' if wins[i,j]==3 else ''),ha='center',va='center',fontsize=8,color='white' if abs(v)>10 else 'black')
    ax.set_yticks(range(7),[SP[x] for x in methods]);ax.set_xticks(range(12),[BN[b]+'\n'+SN[src] for b,src in columns],rotation=35,ha='right')
    for x in [3.5,7.5]:ax.axvline(x,color='white',lw=3)
    fig.colorbar(im,ax=ax,label='Mean paired WIS change (%)',shrink=.8);ax.set_title('Information sharing versus matched independent locations\nNegative improves; * improves in all three seeds; colors clipped at ±15%, labels retain actual values')
    fig.savefig(OUT/'spatial-effects.png',dpi=170,bbox_inches='tight');plt.close(fig)
    rows=[]
    for r in rank.head(10).itertuples():rows.append({'Rank':int(r.rank),'Backbone':BN[r.backbone],'Covariates':SN[r.sources],'Exchange':SP[r.spatial]+(' + ID8' if r.location_embedding else ''),'WIS ± seed SD':f'{r.combined_mean:.3f} ± {r.combined_sd:.3f}','States/DC':f'{r.states_dc_combined_mean:.3f}','US':f'{r.US_combined_mean:.3f}'})
    leaderboard=table(pd.DataFrame(rows))
    spatial_display=spatial_table.set_index('spatial').loc[methods].reset_index()
    spatial_display=pd.DataFrame({'Method':spatial_display.spatial.map(SP),'Improved mean / 12':spatial_display.improved.astype(str)+'/12','Improved all 3 seeds / 12':spatial_display.all_seeds.astype(str)+'/12','Median paired change':spatial_display.median_percent.map(lambda x:f'{x:+.1f}%')})
    comparisons=matched[(matched.config_id==leader)&(matched.geography=='all')&(matched.target=='combined')&(matched.kind=='source-addition')]
    period=table(pd.DataFrame([{'Season':r.season,'No covariates, pooled':f'{r.baseline_mean:.3f}','All covariates, pooled':f'{r.candidate_mean:.3f}','Paired change':f'{r.percent_mean:+.1f}%','Seeds improved':f'{int(r.seeds_improved)}/3'} for r in comparisons.itertuples()]))
    target_display=table(pd.DataFrame([{'2025–26 outcome':r.target.removeprefix('wk inc '),'WIS ratio':f'{r.wis:.3f}','Model 95% coverage':f'{100*r.coverage95:.1f}%','Hub 95% coverage':f'{100*r.ensemble95:.1f}%'} for r in target.itertuples()]))
    singleton=source.pivot(index='sources',columns='backbone',values='percent_mean').reindex(index=sources,columns=BACKBONES)
    singleton=singleton.map(lambda x:f'{x:+.1f}%').rename(index=SN,columns=BN).reset_index().rename(columns={'sources':'Source added'})
    text=f'''# Covariates help with simple geographic pooling; complex target attention does not

**Complete: 468/468 runs, 156 configurations, three seeds and 1,404 held-out-season folds.** Every forecast evaluation used **256 draws**. This is the completed `b2-direct-research-v1` study; no new models were fitted for this report.

The best overall configuration is **six separately fitted target models, a multiscale temporal encoder, all seven covariate groups, and simple geographic pooling**. Its WIS ratio is **1.059 ± 0.049** across seeds, where **1 is the Hub ensemble and lower is better**. It improves on its matching pooled no-covariate control by **9.0% in the mean seed-paired comparison**, with all three seeds improving. **None of the 156 configuration means beats the Hub ensemble overall or on the 2025–26 combined score.**

Three findings should guide the next decision:

- **Keep simple messages in the leading candidate set.** Pooling, national broadcasts and gated pooling usually improve their matched independent-location controls. Same-target token attention improves none of the 12 backbone/source means; joint location/target tokens improve only two.
- **Covariate value depends on the model and the other sources.** Neither Kinsa-only nor ILI-only is a reliable general replacement for the full bundle. Some specific matched comparisons favor them, and source removal shows complementarity. The experiment does not isolate each source inside the winning pooled model.
- **The overall gain is concentrated in 2024–25, and intervals remain too narrow.** For the leader, adding covariates changes 2025–26 from 1.064 to 1.061, improving only one seed. Weighted nominal 95% coverage is **79.3%**, compared with **90.7%** for the Hub ensemble.

The forecasting inputs use the explicitly authorized **2025–26 reporting-regime hypothesis**: available historical vintages first, otherwise lag-gated finalized substitutes when an archive is missing. This matters especially for Kinsa, wastewater and FluSurv. These are exploratory development results under that assumption, not a fully historical real-time backtest or a prospective confirmation.

## Leading models

{leaderboard}

ID8 denotes an eight-dimensional learned location embedding. Scores average target-weighted season scores, with 80% states/DC and 20% native US weight. The displayed SD describes the three fitting seeds, not uncertainty across future seasons. Means of paired percentage changes below are **not** ratios of the displayed means. All configurations have all three seeds; no partial run enters this report.

![Leading models and their 2025–26 scores](leaders.png)

The leader's individual overall seed scores are **1.032, 1.115 and 1.028** (42, 43, 44). The first and third ranked models differ by only 0.021 in their means. Selection among 156 configurations on reused historical seasons makes small rank differences weak evidence of general superiority. [All configurations and their exact scenario strings](leaderboard.csv).

## What covariates help?

### One source at a time, holding geography independent

The table shows mean seed-paired percentage WIS changes relative to the same backbone with no external covariates. Negative improves.

{table(singleton)}

No singleton improves all three independent-location backbones. Adding the complete bundle improves the target multiscale mean in all three seeds, but has little average benefit for the target MLP and slightly worsens the pathogen MLP. **Source relationships and the way information is shared matter more than declaring one universally useful predictor.**

![Source additions and conditional additions](source-effects.png)

### Removing a source from the full bundle

These contrasts also use **independent locations**. Adding a removed source back into the otherwise full bundle gives:

- Inpatient claims: **−5.9%** for target MLP and **−5.4%** for target multiscale, both improving all three seeds.
- Outpatient claims: **−4.8%** for pathogen MLP and **−3.6%** for target MLP, both improving all seeds.
- Wastewater: **−5.0%** for pathogen MLP and **−2.1%** for target multiscale, both improving all seeds; the target MLP mean instead worsens.
- FluSurv: **−7.0%** for target MLP in all seeds, but no consistent benefit in the other backbones.

These results suggest complementarity, not a source ranking independent of architecture. Kinsa's conditional mean is mildly favorable in all three backbones but improves only **2/3, 2/3 and 1/3** seeds, respectively. Its contribution is therefore uncertain. Source effects include the change in model input features/parameters; there is no parameter-count-matched noise-source control.

### Kinsa-only and ILI-only with geographic exchange

The clearest favorable Kinsa-only contrast is **target MLP with national broadcasting**: adding Kinsa improves **2.4%**, in all three seeds, to WIS **1.105**. Kinsa is already broadcast from its national series to every location before the model; this result is not obtained by restricting Kinsa to the US output.

For **target MLP with pooled geography**, adding ILI alone improves **2.3%**, in all seeds, to WIS **1.093**. In this same architecture, all covariates score **1.126** and do not improve the pooled no-covariate mean. Thus “more sources” is not automatically better. Both promising single-source results remain above the Hub ensemble overall and are selected comparisons within a broad screen.

[Every singleton comparison](singleton-summary.csv) · [Conditional source comparisons](conditional-summary.csv) · [All paired seed effects](paired_seed_scores.csv).

## Passing information across locations and targets

Each method is compared against independent locations with the same backbone, source bundle and zero location-ID embedding. There are **12 comparisons per method** (three backbones × none/Kinsa/ILI/all covariates).

{table(spatial_display)}

![Matched information-sharing effects](spatial-effects.png)

Simple pooling shares the observed-state mean and the native US context; national broadcasting shares the US context; gated pooling learns how strongly each recipient uses pooled and national messages. **Gated pooling is the most consistent by the count of improved means (11/12); plain pooling produces the best overall configuration.** Full location attention is mixed, especially in the multiscale models.

The failure of scoped target-token methods is specific to these implementations and settings. Every model already sees all six local target histories, and the scoped tokens also carry selected covariates. Separate target fits do not imply absence of cross-target inputs. The study does **not** show that all cross-pathogen information is useless, nor does it compare adjacent-state graphs, geographic distance or a jointly optimized multi-model ensemble.

Adding an eight-dimensional location-ID embedding **worsens every independent-location mean (12/12)**. With location attention it improves 6/12 means, including a large recovery for multiscale/no-covariates, but is not a dependable default. Existing population and native-US features are retained in both embedding conditions. [Detailed spatial counts](spatial-summary.csv) · [Embedding comparisons](embedding.png).

## Where the leader improves—and where it does not

This is the clean covariate contrast for the best configuration: **target multiscale with pooled geography, all covariates versus none**.

{period}

The overall advantage is mainly the **20.2% paired reduction in 2024–25**. There is no consistent recent-season gain: the 2025–26 difference is only **−0.4%**, with **one of three seeds improving**. The best 2025–26-only configuration has a mean of **1.008**, but ranks 146th overall; this season-specific selection is not a replacement recommendation.

Adding pooling to the all-covariate multiscale model is a separate contrast: **1.125 → 1.059**, mean paired improvement **5.7%**, two seeds improving. It improves 2025–26 by 2.8% in the paired mean, also two seeds. Do not add these percentages as independent causal contributions.

### Outcomes and interval coverage

{target_display}

The leader beats the ensemble for **2025–26 flu admissions, flu ED and RSV ED**, but loses for COVID admissions/ED and RSV admissions. The combined 2025–26 score remains 1.061. Across the complete weighted task set, its 95% coverage is **79.3%** (states/DC **77.0%**, US **88.5%**); corresponding ensemble coverage is **90.7%**. The WIS decomposition shows smaller dispersion but larger penalties for both underprediction and overprediction than the ensemble. This supports investigating uncertainty calibration, not treating the best point on this ranking as ready for deployment. No calibration was fitted here.

## Fan plots: best three models

**Columns:** Hub ensemble; rank 1 target multiscale/all/pooled; rank 2 target MLP/all/national broadcast; rank 3 target multiscale/all/gated pool. Fans use **seed 42 fixed in advance**, not the best seed, and the saved 256-draw evaluation. Black is finalized truth; colored lines are medians; dark/light bands are 50%/90% intervals. ED is displayed as percent of visits. Forecasts are shown every fourth reference week and only on the same frozen Hub target/location/reference/horizon support. These plots illustrate individual fits; the ranking averages three seeds.

### United States, 2025–26

[![US admissions fans](fans/best-2025-2026-US-hosp.png)](fans/best-2025-2026-US-hosp.png)

[![US ED fans](fans/best-2025-2026-US-ed.png)](fans/best-2025-2026-US-ed.png)

### North Carolina, 2025–26

North Carolina is the same preselected example state used in preceding reports; it was not chosen from this ranking or for visually favorable performance.

[![NC admissions fans](fans/best-2025-2026-NC-hosp.png)](fans/best-2025-2026-NC-hosp.png)

[![NC ED fans](fans/best-2025-2026-NC-ed.png)](fans/best-2025-2026-NC-ed.png)

### Matched controls and all seasons

The control figures put **pooled/no covariates**, **all covariates/no exchange**, and **all covariates/pooled** beside the ensemble. They hold the target-multiscale backbone and seed fixed.

[![US admissions, matched controls](fans/controls-2025-2026-US-hosp.png)](fans/controls-2025-2026-US-hosp.png)

[US ED controls](fans/controls-2025-2026-US-ed.png) · [NC admissions controls](fans/controls-2025-2026-NC-hosp.png) · [NC ED controls](fans/controls-2025-2026-NC-ed.png).

Full three-season fans for the top three: [US admissions](fans/best-all-US-hosp.png), [US ED](fans/best-all-US-ed.png), [NC admissions](fans/best-all-NC-hosp.png), [NC ED](fans/best-all-NC-ed.png). [Exact fan selections and conventions](fans/selection.json).

## What this experiment supports next

Keep **target multiscale + all sources + pooling** as the overall development leader, alongside **target MLP + all sources + national broadcast** and the simpler **target MLP + ILI + pooling**. Avoid defaulting to target-token attention or location embeddings without a specific matched benefit. Source-removal experiments inside the winning pooled model would be needed to identify its essential predictors; the current independent-location ablations cannot answer that question for it.

The immediate unresolved issue is robustness on the most recent season and interval calibration. A prospective season or untouched rolling holdout is needed before calling the recipe a demonstrated winner. The earlier B0 value near 0.889 used a different input/data protocol; this experiment is not a matched reproduction, and its difference cannot be assigned to covariates or one training choice. No additional training or calibration was launched for this analysis.

## Availability assumptions and interpretation

**Subsequent raw-data audit:** same-report-date conflicts in claims were converted to missing by the shared extractor, affecting both deadline and final arrays. This understates reported claims availability and limits claims attribution and revision estimates. The [corrected availability audit](availability/conflict-audit.md) separates finite source reports from retained pipeline inputs. Saved runs have not been changed; the performance effect of resolving these conflicts remains unmeasured.


[Detailed 2025–26 covariate availability by observation lag and revisions to final](availability/index.md) counts actual archived reports separately from assumed availability.

Training uses complete finalized target and covariate histories after holding out the evaluation/validation weeks. Normalization uses fitting data only. Artificial missingness affects target histories in 50% of training examples; covariates receive no extra artificial dropout. All three backbones use cap 300/patience 30, including families whose historical best used cap 100. Finality flags, B0 normalization and fixed calendar, current validation draws and smaller loss scale remain as planned.

Forecasting uses actual deadline vintages when available, then finalized historical proxies only where there was no known report/null and the assumed release preceded the holiday-adjusted cutoff. Known withdrawals, structurally missing native coverage and audited missing Missouri ED inputs are not filled. Release-delay assumptions are **targets 4 days; Kinsa 1; inpatient claims 3; outpatient flu/COVID 0/1; ILI/labs/FluSurv 6; wastewater flu/COVID 13 and RSV 9**. The 2025–26-derived median schedule is explicitly backcast onto earlier seasons; it is an availability assumption informed by that season, not an independently estimated historical schedule.

Among available input cells in 12-week contexts, **Kinsa, wastewater and FluSurv are 100% finalized proxies in 2023–24 and 2024–25**. In 2025–26, proxy shares remain **67.6% for Kinsa, 53–55% for wastewater and 29.1% for FluSurv**. These unweighted input-context counts include all modeled origins and repeat observations across contexts; they are not percentages of scored forecast cells. Finalized substitutes can contain later revisions; wastewater's final index can also use retrospectively updated baselines. Consequently, covariate benefit is conditional on these assumptions.

[Availability fractions by source/season](availability-summary.csv) · [Release lags and exact proxy counts](../b2-research/data/operational-source-lags.csv) · [Full data policy and checksums](../b2-research/data/operational-policy.json) · [Design and launch commands](../../design/b2-direct-research.md).

## Evidence and reproduction

The complete ranking is **`ranking-ac92a3c7dbcc`**; the earlier partial `ranking-e54000acc7af` is excluded. All 468 seed runs and all 267 planned matched contrasts are complete. Three seeds represent fitting variability, not three independent epidemic replications. No confidence claim or multiple-comparison correction is attached to the exploratory means.

[Ranking manifest](ranking-manifest.json) · [All run scores](run_scores.csv) · [Season composites](season_composite_scores.csv) · [Target-season scores](season_scores.csv) · [Coverage aggregates](coverage-summary.csv) · [All matched results](research.md) · [Research provenance](research-provenance.json).

```bash
# From /proj/jlessler/projects/tapestry-all/tapestry on Longleaf.
.venv/bin/python -m tapestry.experiment.planner status -e b2-direct-research-v1
.venv/bin/python -m tapestry.experiment.planner rank -e b2-direct-research-v1 --no-plots
.venv/bin/python analysis/b2-research/report.py --experiment b2-direct-research-v1 --ranking data/experiments/b2-direct-research-v1/ranking-ac92a3c7dbcc
.venv/bin/python analysis/b2-research/fans.py
.venv/bin/python analysis/b2-research/write_report.py
```
'''
    (OUT/'index.md').write_text(text)
    (OUT/'analysis-summary.json').write_text(json.dumps(dict(runs=468,configurations=156,folds=1404,eval_members=256,ranking=str(RANK),best_config=leader,best_wis=float(top.combined_mean),best_seed_sd=float(top.combined_sd),means_below_hub=int((rank.combined_mean<1).sum()),planned_comparisons=len(summary),fan_seed=42,report='index.md'),indent=2)+'\n')
    print(OUT/'index.md')

if __name__=='__main__':main()
