"""Freeze pre-modern ILI reports for transfer without evaluation-season labels."""
from pathlib import Path
import pandas as pd
import numpy as np
from tapestry.experiment.provenance import sha256, save
sources=sorted(Path('data/raw/delphi_fluview_ilinet').rglob('signal=ili/geo_type=state/archive.csv.gz'))
source=sources[-1]
df=pd.read_csv(source)
df=df[(df.report_time<'2022-08-01') & (df.reference_time<'2022-08-01')]
df=df.sort_values('report_time').drop_duplicates(['reference_time','geo_value'],keep='last')
a=df.pivot(index='reference_time',columns='geo_value',values='value')
a.index=pd.to_datetime(a.index)
a=a.reindex(pd.date_range(a.index.min(),a.index.max(),freq='W-SAT'))
# Percent -> proportion. Preserve missing locations/weeks.
path=Path('data/processed/historical_ili.npz');path.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(path,values=a.to_numpy(dtype=np.float32)/100,dates=a.index.strftime('%Y-%m-%d').to_numpy(dtype='U10'),locations=a.columns.to_numpy(dtype='U8'))
save(path.with_suffix('.json'),dict(source=str(source),source_sha256=sha256(source),report_cutoff='2022-08-01 exclusive',reference_start=str(a.index.min().date()),reference_end=str(a.index.max().date()),observed_cells=int(a.notna().sum().sum()),locations=len(a.columns),unit='proportion',output_sha256=sha256(path)))
print(path.with_suffix('.json').read_text())
