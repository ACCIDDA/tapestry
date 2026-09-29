"""Audit the actual frozen support against ensemble rows and saved model exports."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tapestry.evaluation.hubs import export,KEY,QCOLS
from tapestry.evaluation.totals import frozen_cases,match_forecasts
root=Path('data/evaluation/b0_hub_comparison_q23')
ranking=Path('data/experiments/forecast-geography-v2/ranking-e53fd92c4f32')
r=pd.read_csv(ranking/'configuration_ranking.csv').sort_values('combined_mean')
m=json.loads((ranking/'manifest.json').read_text())
run=next(x for x in m['runs'] if x['config_id']==r.iloc[0].config_id and x['seed']==42)
frames=export(run['path'])
rows=[]
for case in frozen_cases(root):
    units=pd.read_parquet(root/case['directory']/'units.parquet')
    table=pd.read_parquet(root/case['directory']/'quantiles.parquet')
    ensemble=table[table.model==case['ensemble']].copy()
    model=frames[(case['season'],case['target'])]
    matched_m=match_forecasts(model,units,case['target']).sort_values(KEY).reset_index(drop=True)
    matched_e=match_forecasts(ensemble,units,case['target']).sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(matched_m[KEY],matched_e[KEY],check_dtype=False)
    np.testing.assert_array_equal(matched_m.observed,matched_e.observed)
    assert np.isfinite(matched_m.observed).all()
    original=units[KEY+['observed']].merge(ensemble[KEY+['observed']],on=KEY,validate='one_to_one',suffixes=('_frozen','_ensemble'))
    np.testing.assert_allclose(original.observed_frozen,original.observed_ensemble)
    extra=ensemble[KEY].merge(units[KEY],on=KEY,how='left',indicator=True)['_merge'].eq('left_only').sum()
    rows.append(dict(season=case['season'],target=case['target'],ensemble=case['ensemble'],
        scored_cells=len(units),ensemble_cells=len(ensemble),ensemble_outside_frozen=int(extra),
        model_exported_cells=len(model),model_outside_frozen=len(model)-len(units),
        missing_ensemble=0,missing_model=0,truth_mismatches=0,duplicate_ensemble=int(ensemble.duplicated(KEY).sum())))
rows=pd.DataFrame(rows)
out=Path('docs/results/forecast-geography-v2')
rows.to_csv(out/'ensemble_support_audit.csv',index=False)
(out/'ensemble_support_audit.json').write_text(json.dumps(dict(run=run['path'],ranking=str(ranking),
    keys=KEY,scored_cases=len(rows),scored_cells=int(rows.scored_cells.sum()),
    missing_ensemble_cells=0,missing_model_cells=0,truth_mismatches=0,
    inference='Support audit uses the top model seed 42; shared scorer enforces these same constraints for every run.'),indent=2)+'\n')
print(rows.to_string(index=False));print('TOTAL',rows.scored_cells.sum())
