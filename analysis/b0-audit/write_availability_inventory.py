"""Readable exhaustive missing-week inventory; rows remain available as CSV."""
from pathlib import Path
import pandas as pd,json,hashlib
import numpy as np
from tapestry.dataset.build import load
out=Path('docs/results/b1-to-b0-chain')
f=pd.read_csv(out/'deadline_latest_cells.csv');missing=f.loc[~f.present].copy();scored=missing.loc[missing.scored_origin].copy();scored.to_csv(out/'missing_scored_latest_2025-2026.csv',index=False)
short={'wk inc flu hosp':'Flu admissions','wk inc covid hosp':'COVID admissions','wk inc rsv hosp':'RSV admissions','wk inc flu prop ed visits':'Flu ED','wk inc covid prop ed visits':'COVID ED','wk inc rsv prop ed visits':'RSV ED'}
lines=['# Missing latest-week inputs in 2025–26','', 'These are observation weeks (Saturday week ends), not submission dates. The latest expected input is the Saturday seven days before the forecast reference date, even when a holiday moves submission later. Each row was checked at the appropriate Hub deadline, including Christmas and New Year extensions. Presence means a report exists in the checked Hub files, raw NSSP files (CSV and Parquet), or dated Delphi archives. Absence here does not establish absence from every possible upstream or private source.','', 'For scored forecasts, all latest admissions inputs are present. The ED cases below are all in Missouri. Different target counts reflect different scoring calendars, not different statewide ED reporting coverage.','', '| Scored outcome | Location | Missing observation weeks |','| --- | --- | --- |']
for target,g in scored.groupby('target'):
 assert set(g.location)=={'MO'}
 lines.append(f"| {short[target]} | Missouri | {', '.join(g.observation_week)} |")
lines+=['','The complete season covers 53 observation weeks, August 2, 2025 through August 1, 2026. The following also includes dates outside the scored forecasts. Those calendar checks use the usual weekly cutoff and do not imply that a Hub submission was required. All 52 means the 50 states, District of Columbia and the native United States series.','', '| Outcomes | Missing locations | Missing observation weeks |','| --- | --- | --- |']
groups={}
for target,g in missing.groupby('target'):
 by_locations={}
 for week,gg in g.groupby('observation_week'):
  key=tuple(sorted(gg.location));by_locations.setdefault(key,[]).append(week)
 for locations,weeks in by_locations.items():groups.setdefault((locations,tuple(weeks)),[]).append(short[target])
for (locations,weeks),targets in groups.items():
 label='All 52 modeled locations' if len(locations)==52 else ', '.join(sorted(f.loc[f.location.isin(locations),'location_name'].unique()))
 lines.append(f"| {', '.join(targets)} | {label} | {', '.join(weeks)} |")
lines+=['','All modeled locations: '+', '.join(sorted(f.location_name.unique()))+'.','', 'For the forecast reference date December 27, 2025, the latest expected observation week was December 20; COVID/RSV submissions closed December 29 and FluSight December 30. For reference date January 3, 2026, the latest expected observation week was December 27; deadlines were January 4 and January 5 respectively. All cutoffs are 23:00 Eastern. At each earlier common deadline, all three latest ED series were present at every modeled location except Missouri. The extra FluSight day changes no input-presence mask.','', '[Every missing latest-input case, with location and deadline](missing_latest_2025-2026.csv) · [Every missing cell in the 12-week input histories](missing_context_2025-2026.csv) · [All audited latest-input cases, including those present](deadline_latest_cells.csv) · [Git source versions](deadline_sources.csv).']
(out/'availability_inventory.md').write_text('\n'.join(lines)+'\n')
p=load('data/audits/b0/original-panel-deadline2025.npz');q=load('data/audits/b0/original-panel-commondeadline2025.npz');assert np.array_equal(np.isfinite(p['asof_targets']),np.isfinite(q['asof_targets']));np.testing.assert_equal(p['targets'],q['targets'])
(out/'common_cutoff_equivalence.json').write_text(json.dumps(dict(availability_masks_identical=True,finalized_values_identical=True,pinned_dataset_sha256=hashlib.sha256(Path('data/audits/b0/original-panel-deadline2025.npz').read_bytes()).hexdigest(),numerical_input_policy='finalized_available: uses only as-of presence then finalized values',earliest_deadline_satisfied_for_all_input_channels=True),indent=2)+'\n')
print('Missing latest:',len(missing),'scored:',len(scored),'inventory groups:',len(groups))
