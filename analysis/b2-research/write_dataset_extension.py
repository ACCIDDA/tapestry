from pathlib import Path
import json,pandas as pd
p=Path('docs/results/dataset-2022-extension');r=json.loads((p/'verification.json').read_text());x=pd.read_csv(p/'coverage.csv');changes=pd.read_csv(p/'finalized-nhsn-changes.csv')
def table(frame):return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])
s=f'''# Dataset extended with the 2022–23 season

The working base dataset, `data/processed/panel.npz`, now covers **{r['start']} through {r['end']}** ({r['weeks']} weekly observations; {r['issuances']} nominal Wednesday rounds). The first 12 weeks precede the first Saturday of 2022–23 (August 6, 2022). They are stored as context; the existing cross-validation policy excludes seasons outside the declared training seasons from fitted histories.

**2022–23 is added to training**, while the evaluation seasons remain 2023–24, 2024–25 and 2025–26. Missing outcomes are masked by the existing loss policy. A fourth scored fold and its Hub benchmark were not created. No models were trained or rescored by this dataset update.

## Target and covariate coverage in 2022–23

Counts are retrospective finite values in the rebuilt dataset. “Weeks with values” means at least one native location is present, not complete state coverage. Missing covariates and outcomes remain NaN. National Kinsa is stored once.

{table(x[['series','weeks_with_values','season_weeks','finite_cells','total_cells','first_week','last_week']].fillna('—'))}

## Finalized NHSN policy

Finite nonnegative CDC finalized admission counts take precedence for retrospective target values; native `USA` maps to `US`. Where the finalized snapshot has no finite value, the previous archive-resolved value remains. The override is applied to final targets only, never historical as-of arrays. This preserves the distinction between complete retrospective training information and evidence of historical availability.

Changes to NHSN values within the previously existing calendar are recorded below; these are consequences of explicit finalized-source precedence, not of adding the season alone.

{table(changes)}

All overlapping covariates, ED final values and historical as-of arrays were verified unchanged. Future observation weeks remain empty in every historical as-of array. Training-role checks confirm the added season is available for fitting and each evaluation season remains excluded from its own fit. The source snapshot IDs and policy are recorded in dataset metadata; truth resolution remains pinned to September 22, 2026 to avoid mixing this extension with a source refresh.

The seven covariate groups retain the existing extraction rules. In particular claims conflict rejection has **not** been resolved by this extension, so finalized covariate histories can still have those known gaps. No assumed availability is inserted into the base archive panel. The separate `panel-b2-deadline.npz` and `panel-b2-operational.npz` used by the completed experiment are unchanged and must be rebuilt against this expanded base before a new operational experiment uses 2022–23. Saved experiment pins and reported scores are not modified.

SHA256: `{r['sha256']}`.

[Coverage CSV](coverage.csv) · [Verification record](verification.json) · [Finalized NHSN changes](finalized-nhsn-changes.csv).

Rebuild the base panel with the same truth cutoff:

```bash
.venv/bin/python -m tapestry.dataset.build build --start 2022-05-14 --truth-day 2026-09-22 --workers 3 --output data/processed/panel.npz
```
'''
(p/'index.md').write_text(s)
