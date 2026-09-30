"""Insert the completed analysis into the standard report's preserved write-up."""
from pathlib import Path
import pandas as pd
from tapestry.evaluation.plots import WRITEUP_START, WRITEUP_END
out=Path('analysis/b-2-t0-report');page=Path('docs/results/b-2-t0/index.md')
r=pd.read_csv(out/'named_ranking.csv');e=pd.read_csv(out/'matched_effects.csv');ci=pd.read_csv(out/'b0_seed_comparison.csv').set_index('config_id')
p=Path((out/'ranking-path.txt').read_text().strip());s=pd.read_csv(p/'season_composite_scores.csv');s=s[s.geography=='all'];sm=s.groupby(['config_id','season']).mean(numeric_only=True)
labels={v.config_id:v.label for v in r.itertuples()}
names={'C1':'Pathogen MLP; distance sharing; no covariates','C2':'Target MLP; all covariates except outpatient; no spatial sharing','C3':'Target MLP; neighbor sharing; Kinsa'}
lines=['### Main findings', '',
'All **927 runs are complete: 309 configurations × 3 seeds (42–44)**. The final ranking is `ranking-edfe99ea0c26`. **281/309 (90.9%)** configurations beat the Hub ensemble on the equally weighted two-season mean: **300/309 (97.1%)** in 2024–25 and **181/309 (58.6%)** in 2025–26. All counts use configuration means across three seeds.', '',
'**The sweep finds several competitive recipes, but no convincing improvement over B0.** C1 has the lowest observed mean; C2 has the smallest seed spread among the top three and the best 2025–26 score of those three. The first two are only 0.45% and 0.05% below the reproduced B0 two-season score. These differences are much smaller than seed uncertainty.', '',
'| Model | 2024–25 | 2025–26 | Combined mean ± seed SD |', '|---|---:|---:|---:|']
for v in r.head(3).itertuples():
 a=sm.loc[v.config_id];lines.append(f'| {v.label}: {names[v.label]} | {a.loc["2024-2025","combined"]:.4f} | {a.loc["2025-2026","combined"]:.4f} | {v.combined_mean:.4f} ± {v.combined_sd:.4f} |')
lines+=['| B0 reproduced reference | 0.7485 | 0.9003 | 0.8244 ± 0.0254 |','',
'All three selected current models use 20% training masking. Fans above use **seed 42**, with predictive 50%/90% intervals; they do not pool seeds and do not show confidence intervals for performance. Heatmaps average all three seeds. Fans include the Hub ensemble, US and North Carolina, admissions and ED separately. Grey heatmap cells have no frozen evaluation support.', '',
'### Why the leaders differ', '',
'| 2025–26 target | C1 | C2 | C3 |','|---|---:|---:|---:|']
for target in [c for c in s.columns if c.startswith('wk inc')]:
 vals=[sm.loc[(v.config_id,'2025-2026'),target] for v in r.head(3).itertuples()]
 lines.append('| '+target.removeprefix('wk inc ')+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
lines+=['',
'C1 is particularly strong on RSV admissions and ED, but COVID ED is worse than the ensemble (1.261). C2 improves all six targets against the ensemble in 2025–26, including COVID ED (0.952), while sacrificing some of C1’s RSV advantage. C3’s strength is 2024–25; its 2025–26 COVID ED remains weak (1.200). There is no uniform winner across targets or seasons.', '',
'### What helps or hurts: matched comparisons', '',
'Each comparison below changes only the named factor and holds **every other Scenario field fixed**. A context is one otherwise identical configuration. Deltas are changes in the combined relative-WIS score, not percentages; negative is better. We average the matched deltas within each seed, then form an exploratory 95% Student-t interval across the three seed averages (2 degrees of freedom). The context counts show consistency, not independent sample sizes.', '',
'| Change from reference | Matched contexts | Mean Δ WIS ratio | 95% seed-only interval | Contexts improved |','|---|---:|---:|---:|---:|']
select=[('backbone','target-multiscale'),('backbone','target-mlp'),('covariate_encoder','raw'),('spatial','attention'),('spatial','national_broadcast'),('spatial','gated_pool'),('spatial','pooled'),('spatial','distance'),('sources','outpatient'),('sources','all'),('sources','kinsa'),('sources','ilinet'),('mask_rate','0.2'),('signal_features','multiscale'),('coordinates','True')]
for factor,value in select:
 z=e[(e.factor==factor)&(e.value.astype(str)==value)].iloc[0]
 lines.append(f'| {factor}: {z.reference} → {value} | {z.contexts} | {z.mean_delta:+.4f} | [{z.seed_ci_low:+.4f}, {z.seed_ci_high:+.4f}] | {z.improved_contexts}/{z.contexts} |')
lines+=['',
'**Most consistent evidence within this sweep:** the multiscale CNN underperforms pathogen MLP in 90/103 matched contexts; raw covariate encoding underperforms summary encoding in 25/27; national broadcast underperforms no spatial sharing in 58/63; attention in 11/12; gated pooling in 11/12. Their seed-only intervals remain on the harmful side. This supports avoiding those choices as defaults in this protocol, but it is exploratory evidence rather than a multiplicity-adjusted claim about future seasons.', '',
'**Inputs are selective, not “more is better.”** Adding all covariates worsens 26/30 matched contexts (mean +0.0499), although its seed-only interval includes zero. Kinsa helps 21/30 and ILINet 18/30, with intervals also crossing zero: promising but context-dependent. Outpatient alone harms all three backbone comparisons (mean +0.0930). Removing outpatient from the full bundle improves all three backbones: −0.0288 for pathogen MLP, −0.0980 for target MLP, and −0.0048 for the multiscale CNN. That is a result for this source representation and protocol, not proof outpatient information is intrinsically unhelpful.', '',
'**Weak or mixed evidence:** target-wise versus pathogen-wise MLP fitting, coordinates, distance/neighbor sharing, smoothed/shared encoders, and engineered signal features have intervals spanning zero. C1’s distance sharing does not establish a general distance-sharing benefit: its average effect is approximately zero across matched contexts. All three winners use 20% masking, yet increasing masking from zero to 20% worsens 40/54 matched contexts on average (+0.0154; interval includes zero). Winner features should not be interpreted as individually beneficial.', '',
'![Matched source and backbone heatmap](source-backbone-heatmap.png)', '',
'The supplementary heatmap holds spatial sharing at none, masking at 0.2, coordinates off, signal features at none and covariate encoder at summary. It compares only combinations actually run.', '',
'### Worst configurations', '',
'| Label | Backbone | Covariates | Spatial sharing | Covariate encoder | Masking | Combined mean ± SD |','|---|---|---|---|---|---:|---:|']
for v in r.tail(5).iloc[::-1].itertuples():
 lines.append(f'| {v.label} | {v.backbone} | {v.sources} | {v.spatial} | {v.covariate_encoder} | {v.mask_rate:.0%} | {v.combined_mean:.4f} ± {v.combined_sd:.4f} |')
lines+=['',
'The three worst combine pathogen MLP, all covariates, raw covariate encoding and national broadcast. All three masking choices perform poorly, so this is not explained by a single masking setting. Matched comparisons independently implicate both raw encoding and national broadcast; their interaction is not separately estimated here.', '',
'### Confidence and interpretation', '',
'| Comparison with B0 | Mean difference | 95% seed-only interval | Seeds better than B0 |','|---|---:|---:|---:|']
for v in r.head(3).itertuples():
 z=ci.loc[v.config_id];lines.append(f'| {v.label} − B0 | {z.mean_delta:+.4f} | [{z.lo:+.4f}, {z.hi:+.4f}] | {int(z.better_seeds)}/3 |')
lines+=['',
'**Confidence in beating B0 is low.** Every interval above crosses zero widely. C1 beats B0 in two of three seed-ID comparisons; C2 in only one. A small rank difference is insufficient to declare either superior. C2’s smaller SD describes fitting stability on these tasks, not a guarantee of better generalization.', '',
'**Confidence in the broad harmful patterns is stronger, but conditional.** The matched-direction counts and seed intervals support the CNN/raw-encoder/broad-pooling penalties within the tested settings. Only three seeds are available; Student-t intervals assume independent approximately normal seed differences and cannot verify that assumption. We do not treat correlated configurations or locations as independent replicates. No multiple-comparison adjustment was applied.', '',
'**This is season-held-out CV, not an untouched final test.** The held-out season is excluded from training and early stopping, but the 2024–25 fold can train on 2025–26. All 309 models were selected and compared using these same CV outcomes. These intervals omit model-selection optimism, temporal sampling uncertainty and performance on genuinely new seasons. The safest conclusion is a competitive cluster of MLP recipes, not a newly proven champion.', '',
'### Scope, assumptions and reproducibility', '',
'- The request is interpreted as reporting the completed sweep; no models were retrained. The standard report job was **3061288**.',
'- Current and B0 target histories are finalized. Current training adds 2022–23 and changes other training settings; B0 used 2,048 evaluation samples versus 256 here. The B0 reference is a recipe comparison, not an isolated architecture ablation.',
'- The metric is the location-relative WIS ratio on frozen identical tasks: US 20%, states/DC 80%; admissions weight 1, ED 0.5; equal season weights. 2024–25 has flu/COVID admissions only; 2025–26 has all six targets. The older three-season B0 average is not compared directly with this two-season average.',
'- Complete B0.1 sweep comparison: 114/172 (66.3%) beat the ensemble on the same two-season mean, versus 281/309 (90.9%) here. Different configuration sets and training panels prevent attributing this difference to one change.',
'- Commands and analysis sources: `analysis/b-2-t0-report/commands.sh`, `build_report.py`, `analyze.py`, and `write_analysis.py`. Standard manager: `.venv/bin/python -m tapestry.experiment.planner status -e b-2-t0`; ranking: `.venv/bin/python -m tapestry.experiment.planner rank -e b-2-t0`. The report wrapper retains top-three fan/heatmap selection; default `rank` selects only the best model.',
'- [Matched effects](matched_effects.csv), [B0 seed comparisons](b0_seed_comparison.csv), and [named final ranking](named_ranking.csv) accompany this report. No visual inspection of generated plots was performed, following project instructions.', '']
coverage_path = out/'coverage_summary.csv'
if coverage_path.exists():
    coverage = pd.read_csv(coverage_path)
    lines += ['### Predictive interval coverage', '',
              'Coverage uses the same location, target and season weights as WIS, averaged over three seeds on the two shared evaluation seasons. These are empirical predictive coverages, not confidence levels for score differences.', '',
              '| Model | 50% interval | 80% interval | 90% interval | 95% interval |',
              '|---|---:|---:|---:|---:|']
    for _, row in coverage.iterrows():
        lines.append('| '+row['model']+' | '+' | '.join(f"{100*row[str(c)]:.1f}%" for c in (50,80,90,95))+' |')
    lines += ['', 'All model families shown under-cover. C1 improves modestly over B0, but its nominal 90% interval covers only 80.8% overall. The ensemble also under-covers at 90% and 95%, but is closer to nominal than these models. C1 COVID ED in 2025–26 is particularly weak: 54.3% coverage for the 90% interval; C2 improves it to 66.9%. WIS superiority does not imply calibrated intervals. Undercoverage may reflect narrow intervals, forecast bias or both.', '',
              'Protocol distinction: the current sweep is B-2 T-0, not the earlier availability-restricted B1. B0 and B-2 both supply finalized target histories through the latest context week. Audited B1 variants withheld inputs missing from the Wednesday archive and some used recent vintages/nowcasting, making the usable context much poorer. B-2 adds the older 2022–23 training season, optional covariates and artificial training masks; outer evaluation has no artificial masks. This adds historical volume, not a more recent season. B-2 scores two seasons rather than the original three; excluding 2023–24 substantially changes the headline comparison.', '']
text=page.read_text();before,rest=text.split(WRITEUP_START,1);_,after=rest.split(WRITEUP_END,1)
page.write_text(before+WRITEUP_START+'\n\n'+'\n'.join(lines)+'\n'+WRITEUP_END+after)
for name in ['source-backbone-heatmap.png','matched_effects.csv','b0_seed_comparison.csv','named_ranking.csv']:
 (page.parent/name).write_bytes((out/name).read_bytes())
print(page)
