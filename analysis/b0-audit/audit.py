"""Reproducible saved-forecast audit; plan/run/status/rank never train or edit original runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tapestry.evaluation.hubs import export, KEY, QCOLS
from tapestry.evaluation.quantiles import LEVELS
from tapestry.evaluation.totals import (frozen_cases, match_forecasts, case_cells, cells_totals, rank)

ROOT = Path('data/audits/b0')
OUT = Path('docs/results/b0-audit')
FROZEN = Path('data/evaluation/b0_hub_comparison_q23')

def plan():
    runs = [dict(config_id=r['label'], seed=r['seed'], path=r['extracted'], original=r['path'])
            for r in json.loads((ROOT/'selected.json').read_text())]
    experiment = Path('data/experiments/forecast-geography-v2')
    ranking = pd.read_csv(next(experiment.glob('ranking-*/configuration_ranking.csv')))
    for label, scenario in [('latest-best',ranking.iloc[0].config_id),
                            ('latest-control','ed_transform=logit,epochs=300,patience=30,fit_partition=pathogen,supplied_final=1,mask_rate=0.5,input_mode=finalized_available')]:
        from tapestry.model.scenario import Scenario
        folder = experiment/Scenario.from_string(scenario).run_id
        for seed in (42,43,44):
            src = sorted((folder/f's{seed}').glob('attempt-*'))[-1]
            runs.append(dict(config_id=label,seed=seed,path=str(src),original=str(src)))
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'plan.json').write_text(json.dumps(runs,indent=2)+'\n')
    print(json.dumps(runs,indent=2))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    diagnostics=[]; truths=[]; hashes=[]
    cases=[]
    for case in frozen_cases(FROZEN):
        units=pd.read_parquet(FROZEN/case['directory']/'units.parquet')
        q=pd.read_parquet(FROZEN/case['directory']/'quantiles.parquet')
        ens=match_forecasts(q[q.model==case['ensemble']],units,case['target'])
        cases.append((case,units,ens))
        for filename in ('units.parquet','quantiles.parquet'):
            f=FROZEN/case['directory']/filename
            hashes.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    for item in json.loads((ROOT/'plan.json').read_text()):
        label,seed=item['config_id'],item['seed']; src=Path(item['path'])
        frames=export(src); parts=[]
        for f in src.glob('eval_*/forecasts.npz'):
            hashes.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        for case,units,ens in cases:
            pred=frames[(case['season'],case['target'])]
            model=match_forecasts(pred,units,case['target'])
            cells=case_cells(model,ens,case); parts.append(cells)
            # Independent pinball implementation, separate from interval-score code.
            m=model.sort_values(KEY); error=m.observed.to_numpy()[:,None]-m[QCOLS].to_numpy()
            independent=2*np.maximum(error*LEVELS,error*(LEVELS-1)).mean(axis=1)
            err=float(np.max(np.abs(independent-cells.model_wis.to_numpy())))
            assert np.allclose(independent,cells.model_wis,rtol=1e-12,atol=1e-10)
            matched=units.merge(pred,on=KEY,validate='one_to_one')
            diff=matched.model_original_truth-matched.observed
            truths.append(dict(config_id=label,seed=seed,target=case['target'],season=case['season'],
                               n=len(matched),original_mask_missing=int((~matched.model_original_mask).sum()),
                               truth_mae=float(diff.abs().mean()),truth_max=float(diff.abs().max())))
            diagnostics.append(dict(config_id=label,seed=seed,target=case['target'],season=case['season'],n=len(cells),pinball_max_error=err))
        cells=pd.concat(parts,ignore_index=True)
        dest=ROOT/'rescored'/label/f's{seed}';dest.mkdir(parents=True,exist_ok=True)
        totals=cells_totals(cells);totals.to_csv(dest/'totals.csv',index=False)
        old=src/('historical_totals.csv' if label.startswith('b0-') else 'totals.csv')
        prior=pd.read_csv(old); keys=['target','season','location','horizon']
        joined=totals.merge(prior,on=keys,suffixes=('_new','_old'),validate='one_to_one')
        assert len(joined)==len(totals)==len(prior)
        for who in ('model','ensemble'):
            delta=(joined[f'{who}_wis_new']-joined[f'{who}_wis_old']).abs().max()
            assert np.allclose(joined[f'{who}_wis_new'], joined[f'{who}_wis_old'], rtol=1e-12, atol=1e-10)
            print(label,seed,who,'max total difference',delta,flush=True)
        cells.to_parquet(dest/'cells.parquet',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'arithmetic_checks.csv',index=False)
    pd.DataFrame(truths).to_csv(OUT/'truth_comparison.csv',index=False)
    (OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')

def ranking():
    runs=[dict(config_id=r['config_id'],seed=r['seed'],path=ROOT/'rescored'/r['config_id']/f"s{r['seed']}")
          for r in json.loads((ROOT/'plan.json').read_text())]
    print(rank(runs,OUT)[['config_id','combined_mean','combined_sd','states_dc_combined_mean','US_combined_mean']].to_string(index=False))

def status():
    for r in json.loads((ROOT/'plan.json').read_text()):
        f=ROOT/'rescored'/r['config_id']/f"s{r['seed']}"/'totals.csv'
        print('complete' if f.exists() else 'pending',r['config_id'],r['seed'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plan','run','status','rank'])
    {'plan':plan,'run':run,'status':status,'rank':ranking}[p.parse_args().command]()
