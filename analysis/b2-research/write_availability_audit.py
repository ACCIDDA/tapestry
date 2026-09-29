from pathlib import Path
import pandas as pd
p=Path('docs/results/b2-direct-research-v1/availability')
s=pd.read_csv(p/'summary.csv');a=pd.read_csv(p/'conflict-audit.csv')
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
rows=[]
for name,g in a.groupby('covariate',sort=False):
 r=g[g.lag.eq(1)].iloc[0];assert r.pipeline_available+r.conflicting==r.raw_finite
 rows.append([name,int(r.opportunities),f'{int(r.pipeline_available)} ({100*r.pipeline_available/r.opportunities:.1f}%)',f'{int(r.raw_finite)} ({100*r.raw_finite/r.opportunities:.1f}%)',int(r.conflicting)])
source=[]
for name,g in s.groupby('covariate',sort=False):
 raw=a[a.covariate.eq(name)].set_index('lag');g=g.set_index('lag');n=int(g.loc[0,'opportunities'])
 freq='Daily → Saturday trailing-7-day value' if name.startswith(('inpatient','outpatient')) else 'Daily → complete 7-day mean' if name=='kinsa_ili' else 'Weekly index; irregular raw samples' if name.startswith('nwss') else 'Weekly'
 source.append([name,freq,n]+[f'{int(raw.loc[k,"raw_finite"] if len(raw) else g.loc[k,"available"])} ({100*(raw.loc[k,"raw_finite"] if len(raw) else g.loc[k,"available"])/n:.1f}%)' for k in range(4)])
(p/'conflict-audit.md').write_text(f'''# Why the previous availability table was low

**A processing rule was counting conflicting reports as unavailable.** The table was numerically faithful to the model panel, but it understated source availability for claims. We should have labeled it “values retained by the pipeline,” not simply “reported values available.”

## Corrected source-availability interpretation

At each deadline, find the latest archived statement date for that observation week and location. Count it as reported if **at least one finite native value exists on that date**, including dates with multiple conflicting finite values. This does not choose one conflicting number to feed a model. Explicit all-null latest reports stay unavailable. For the seven Delphi covariates, this audit rereads raw archives before the conflict rule; wastewater/Kinsa retain their previously reconstructed archive counts.

Counts are location-weeks, not finite revision pairs. Each row's denominator is **53 deadlines × fixed native locations**. For example 52 locations × 53 = 2,756; Kinsa is 53 national weeks. Latest means the preceding Saturday, not the newest arbitrary daily observation. Cutoffs retain holiday extensions. Archive gaps remain in the denominator.

{table(['Covariate','Frequency / model input','Location-weeks','Latest','One week earlier','Two earlier','Three earlier'],source)}

## How much the conflict rule removed

The following is **one week earlier**. “Finite raw report” is not a claim that the exact latest value can be uniquely recovered from the date-only archive.

{table(['Covariate','Location-weeks','Retained in pipeline','Finite raw report','Discarded conflicts'],rows)}

The affected extraction function is `tapestry.dataset.extract._without_conflicts`: different values sharing (source tier, report time, observation day, location) become NaN. It runs before both historical resolution and final-panel resolution. In the audited deadline cells, the counted conflicts contain only finite values (no mixed finite/null statements). Thus this is value ambiguity, not evidence that no observation was published.

Concrete raw example: **North Carolina outpatient flu**, observation day **2026-01-10**, report date **2026-01-20**, has values **0.832603 and 0.834298**. Both are available before the January 21 deadline, but our pipeline discards the cell. The raw file preserves only a date for these reports; file ordering has not been verified as revision ordering. Choosing the last CSV row would therefore introduce an unsupported assumption.

## ILI's remaining low percentage is visible in the archive

For lag 1, the 53 weekly rounds break down as follows: **9 rounds × 52 locations; 6 × 0; 9 × 30; 1 × 31; 28 × 44 = 2,001**, or 72.6% of 2,756. The six zero rounds are October 8–November 12, 2025. The 28 rounds from January 28 onward have 44/52, or 84.6%, native locations.

In a direct raw-file check for the observation week January 24, 2026, AK, CT, HI, NY, OK, OR, SD and UT lack eligible statewide values in the panel. The raw state ILI archive contains no AK observation for that week. A fresh read-only Delphi archive request for report times after January 29 likewise returned no AK rows. This supports an archive/source-coverage gap, not an arithmetic denominator error. It does not establish whether another upstream CDC resource could supply those values. No attribution to a shutdown or genuine nonreporting is asserted from these checks alone.

Kinsa and wastewater additionally lack early-season vintage archives, and FluSurv has a substantial archival gap. The season-wide table is not a pure reporting-delay estimate. It also includes off-season dates and fixed native geographies rather than conditioning on only dates/locations that happened to report.

## Consequences for revision estimates and experiments

The earlier revision table uses the **unambiguous subset retained by the old pipeline**. Because the same conflict rule also removes final values, claims revision sample sizes are much smaller than raw availability. Those conditional numbers remain reproducible but do not measure revision magnitude over all published values. Recovering a defensible within-date ordering, or an explicitly defined aggregation with a sensitivity analysis, is necessary before replacing them with all-report revision estimates.

This audit **does not change the frozen training panels, resolve ambiguous values arbitrarily, or rerun models**. The old B2 scores are still scores of the saved fits; claims-source attribution and the comparison to B0 now have an additional identified data-processing limitation. Its performance effect has not been measured.

[All 12-lag conflict counts](conflict-audit.csv) · [Original panel availability and conditional revisions](index.md) · [Weekly original counts](by-week.csv). Per-source compressed statement tables alongside this report contain deadline/final values, raw min/max/count, and the conflict flags.

Reproduce with:

```bash
.venv/bin/python analysis/b2-research/covariate_availability.py
.venv/bin/python analysis/b2-research/audit_covariate_conflicts.py
.venv/bin/python analysis/b2-research/write_availability_audit.py
```
''')
print(table(['Covariate','N','Pipeline lag1','Raw lag1','Conflicts'],rows))
index=p/'index.md';text=index.read_text();start='<!-- raw-audit:start -->';end='<!-- raw-audit:end -->'
if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
intro=f'''{start}
## Full tables: all 14 covariates and 12 lags

[Availability, unique missing states and distinct missing weeks](full-tables.md) · [All-lag revisions](full-revisions.md) · [Per-state counts and exact missing dates](missing-by-state-lag.csv).

Every covariate link in the full table opens its state-by-lag missing-week counts. DC and national gaps are separate; unsupported geographies are listed explicitly.

## Correction: reported values versus retained values

**The low claims availability was partly a processing problem.** Multiple different finite values on the same report date were being converted to missing by the pipeline. The corrected source-availability counts below include those published values; they do not pick an arbitrary value for modeling. [Detailed diagnosis and evidence](conflict-audit.md).

“Location-weeks checked” = **native locations × 53 deadlines**: 52 × 53 = 2,756 for a source covering all states, DC and US; 1 × 53 = 53 for national Kinsa. It is not a count of revision pairs. The denominator also includes archive gaps and off-season weeks.

{table(['Covariate','Frequency / model input','Location-weeks checked','Latest','One week earlier','Two earlier','Three earlier'],source)}

Claims are daily but the model selects Saturday's trailing-seven-day percentage. ILI/labs/FluSurv are weekly. Kinsa daily values form a complete weekly mean. Wastewater indices are weekly aggregates of irregular measurements. “Latest” refers to that weekly input, not the newest arbitrary daily report.

**ILI's 72.6% is a different issue:** six zero-report weeks and reduced state coverage are present in the raw archive. Kinsa/wastewater also lack early-season vintage coverage. These season-wide percentages therefore do not measure reporting delay alone. The old revision figures below remain conditional on the unambiguous retained subset and need a conflict-resolution policy before they can describe all claims reports.
{end}
'''
pos=text.index('## What the counts mean');text=text[:pos]+intro+'\n'+text[pos:];index.write_text(text)
