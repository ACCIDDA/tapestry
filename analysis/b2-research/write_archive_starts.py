from pathlib import Path
import pandas as pd
p=Path('docs/data/availability');x=pd.read_csv(p/'raw-archive-starts.csv');ww=pd.read_csv(p/'raw-nwss-starts.csv')
rows=[]
for r in x.itertuples():
 if r.covariate.startswith('nwss'):continue
 rows.append([r.covariate,r.earliest_observation,r.earliest_finite_report])
for r in ww.itertuples():rows.append([r.source.replace('signal=','NWSS raw '),r.earliest_observation,r.earliest_finite_report])
table='| Covariate | Earliest observation date | Earliest archived report date |\n| --- | --- | --- |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows)
s='''# Where the raw covariate histories begin

These dates scan all downloaded native geographies for each modeled signal, with **no state selection, Saturday-only selection, season/deadline cutoff, fill-method filter, age-group filter, or conflicting-value rejection**. Earliest finite values are reported below; the CSV also records the first raw report including null rows. Acquisition scope still limits what exists in our downloaded archives.

**Observation date** is the day/week the measurement describes. **Report date** is the earliest release timestamp carried by our archive for a finite value. Neither is our download date. An archive beginning later than its observations contains retrospective history; its first report date is not proof the source was unavailable before that date.

'''+table+'''

Kinsa here is the raw daily PopHIVE/Kinsa series, before requiring complete seven-day weeks. NWSS rows are raw sewershed concentrations, before deriving weekly state indices or imposing site-count/history requirements. The modeled wastewater indices therefore have different earliest observation weeks; see the raw-date CSV for their separate entries.

A long archive history does not imply continuous coverage. In particular FluSurv has a major gap from November 2020 to November 2025, then to February 2026. First-date summaries cannot establish availability in each intervening season. ILINet's earliest archive entry can also be national rather than state-level.

[All signal dates and raw row counts](raw-archive-starts.csv) · [Raw NWSS dates](raw-nwss-starts.csv) · [Availability evidence](evidence.md).
'''
main=Path('docs/data/index.md')
body=s.split('\n',1)[1].strip().replace('(raw-archive-starts.csv)', '(availability/raw-archive-starts.csv)').replace('(raw-nwss-starts.csv)', '(availability/raw-nwss-starts.csv)').replace('(evidence.md)', '(availability/evidence.md)')
text=main.read_text();start='<!-- source-history:start -->';end='<!-- source-history:end -->'
a=text.index(start);b=text.index(end)+len(end)
main.write_text(text[:a]+start+'\n## First observation and first archived report dates\n\n'+body+'\n'+end+text[b:]);print(table)
