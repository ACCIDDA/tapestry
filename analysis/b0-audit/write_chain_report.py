"""Write the single final narrative from completed consecutive comparisons."""
from pathlib import Path
import json,re
import pandas as pd
root=Path('docs/results/b1-to-b0-chain')
assert json.loads((root/'completion.json').read_text())['endpoint_verified']
f=pd.read_csv(root/'chain_table.csv',dtype={'stage':str}).set_index('stage')
assert f.loc['07','mean']==f.loc['09','mean']
check=pd.read_csv(root/'representation_comparison.csv');assert len(check)==9 and check.identical_forecasts.all() and check.max_parameter_difference.eq(0).all()
labels=[('00','Starting point: B1-derived direct forecaster'),('01','Restore B0 input normalization'),('02','Remove artificial missing inputs during training'),('03','Remove the “this value is final” indicator'),('03b','Restore complete inputs **at forecast time only**'),('04','Restore complete histories **during training**'),('05','Restore B0 validation weeks'),('06','Restore B0 validation simulations'),('07','Restore B0 training-error weight → **B0**')]
lines=['| Consecutive change | Overall error | Change from preceding row | 2025–26 error |','| --- | ---: | ---: | ---: |']
previous=None
for stage,label in labels:
 r=f.loc[stage];rounded=round(r['mean'],5)
 delta='—' if previous is None else f'{rounded-previous:+.5f}'
 lines.append(f'| {label} | {rounded:.5f} | {delta} | {r.season2025_mean:.5f} |')
 previous=rounded
f.loc[[stage for stage,_ in labels]].to_csv(root/'report_table.csv')
text='\n'.join(lines)+'''

**Lower is better; 1 is the Hub ensemble benchmark.** Relative weighted interval score (WIS) measures forecast error, including uncertainty intervals. Numbers average three fits with different random seeds. Each row changes the preceding row; negative differences are improvements.

**1. Starting point: B1-derived direct forecaster.** Three networks—flu, COVID and RSV—learn from two seasons to predict admissions and emergency-department (ED) visit proportions for the next four weeks using 12 weeks of histories; the remaining season evaluates them. This starting recipe restricts inputs to reports present in the checked archives by each deadline, adds artificial missingness during training, includes a “this value is final” indicator, and omits B0’s fitted input normalization. All subsequent rows keep the original dataset, network recipe, evaluation cases and forecast simulation draws fixed, with no extra predictors or separate recent-history estimation stage.

**2. Restore B0 input normalization.** Admissions inputs are divided by each location’s historical high level, while transformed ED inputs are centered around their historical average and divided by their variability, using training data only. This puts different histories on comparable numerical scales without changing forecast units or scoring weights. It improves 2025–26 error from **1.35764 to 1.27626**, especially COVID ED, but worsens the overall mean, so its benefit depends on the season.

**3. Remove artificial missing inputs during training.** Previously, half the training examples received an additional pattern of hidden observations, teaching the model to forecast from incomplete histories; this step stops that deliberate hiding while retaining genuine archive gaps. Error worsens in all three repeated fits, both overall and in 2025–26. That supports keeping this training exercise under the input restrictions tested here.

**4. Remove the “this value is final” indicator.** This removes the extra input telling the model which supplied observations are finalized; the ordinary indication of whether an observation is present remains. Because every supplied value in these experiments is finalized, the extra indicator adds no new information, but removing it changes the input representation and therefore the fitted model. Overall error worsens **1.19484 → 1.27085**, with inconsistent effects across repeated fits, while the 2025–26 mean improves slightly.

**5. Restore complete inputs at forecast time only.** Keep the fitted models unchanged, but give them every observation present in the finalized history when making forecasts, including observations absent from the deadline archives. Overall error improves **1.27085 → 1.07026**, driven by the older seasons; 2025–26 instead worsens slightly, **1.32718 → 1.33836**. This is consistent with the corrected 2025–26 scored dates already having every latest admissions input and nearly every latest ED input.

**6. Restore complete histories during training.** Now also train with complete finalized histories, retaining the complete forecast inputs from the preceding row. This improves 2025–26 **1.33836 → 0.90652**, across all six outcomes and all three repeated fits. The likely benefit is much richer training information and better-informed normalization: the older archive supplies no latest-week ED inputs in 2023–24 and only about 13% in 2024–25, despite observations existing in finalized history.

**7. Restore B0 validation weeks.** Validation temporarily hides historical weeks during an initial fit to choose training duration, after which the final fit uses all training weeks. This step changes the hidden 2023–24 dates: the winter block moves from **December 23, 30 and January 6**, around the flu/COVID peak, to **January 20, 27 and February 3**, during the later decline, keeping peak observations visible to the initial fit. Overall error improves **0.93339 → 0.91476** in all three repeated fits, although the 2025–26 aggregate improvement comes from only one fit, so later validation is not established as universally better.

**8. Restore B0 validation simulations.** The model produces simulated possible outcomes to estimate its validation error and choose training duration; this step replaces the newer simulation draws with B0’s draws. Both versions use a fixed set across candidate training durations, and the forecast simulations used for final evaluation remain unchanged. The effect is negligible: overall error changes **0.91476 → 0.91540**, while 2025–26 remains **0.88568**.

**9. Restore B0 training-error weight → B0.** This approximately triples each pathogen model’s combined training error, preserving the relative importance of admissions, ED, seasons and locations; it changes neither input normalization nor final scoring weights. The optimizer limits unusually large parameter updates, and numerical behavior can also make rescaling the error change the fitted model. This reaches B0’s **0.88889** overall error, but worsens 2025–26 **0.88568 → 0.90025** in every repeated fit, so recovering B0 does not mean every B0 choice helps the latest season.

![Consecutive comparison](../b1-to-b0-chain/chain.png)

The complete validation-calendar change is below; dates in the other two seasons are unchanged.

| Block | Newer calendar: weeks ending | B0 calendar: weeks ending | Epidemic timing |
| --- | --- | --- | --- |
| Autumn 2023 | Sep 2, 9, 16 | Sep 30; Oct 7, 14 | Before the main winter peaks |
| Winter 2023–24 | Dec 23, 30; Jan 6 | Jan 20, 27; Feb 3 | Newer block straddles flu/COVID peaks; B0 uses the later decline |
| Spring 2024 | Apr 13, 20, 27 | May 11, 18, 25 | Both after the winter peaks |

National flu admissions peaked **December 30**, COVID admissions **January 6**; flu/COVID ED peaked December 30 and RSV ED November 25. B0’s calendar selects longer training for 2025–26: average passes through the data rise **53 → 68** for flu, **54 → 131** for COVID and **37 → 104** for RSV. Keeping peak observations visible is a plausible explanation, but changing validation weeks also changes the data used to fit normalization, so these mechanisms were not isolated separately.

![Validation dates relative to epidemic peaks](../b1-to-b0-chain/validation-timing.png)

**Conclusion:** a season resembling 2025–26 supports training on complete histories rather than recreating old archive gaps. It does **not** justify copying every B0 choice: masking helps, normalization helps the recent season, and B0’s larger error weight hurts that season. The best recent-season row is **0.88568**, before the weight change. These are sequential effects; combining their best elements remains untested.

<details>
<summary><strong>Assumptions, corrected holiday deadlines, and every missing 2025–26 week/location</strong></summary>

'''
inv=(root/'availability_inventory.md').read_text().split('\n',2)[2]
inv=re.sub(r'\]\(([^):]+\.(?:csv|json))\)',r'](../b1-to-b0-chain/\1)',inv)
text+='**Scope:** all visible values are finalized; preliminary-value revisions are untested. Only 2025–26 availability has corrected deadlines; earlier seasons retain the saved Wednesday archive. These retrospective season comparisons do not establish future superiority. Scoring gives states/DC 80%, native US 20%, admissions twice ED, and seasons equal weight. Available benchmark outcomes are flu admissions in 2023–24, flu/COVID admissions in 2024–25, and all six in 2025–26.\n\n'+inv+'\n</details>\n\n[Table data](../b1-to-b0-chain/report_table.csv) · [Individual fits](../b1-to-b0-chain/run_scores.csv) · [Outcome-specific effects](../b1-to-b0-chain/target_effects.csv) · [Validation dates and peaks](../b1-to-b0-chain/validation_week_peak_positions.csv).\n'
Path('docs/results/b0-reproduction/index.md').write_text(text)
print('Main narrative words:',len(text.split('<details>')[0].split()))
