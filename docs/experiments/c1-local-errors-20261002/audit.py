"""Check the actual nowcast residual banks and draw historical training examples."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tapestry.dataset import cv
from tapestry.dataset.build import load
from tapestry.dataset.reporting_error import ReportingErrors
from tapestry.experiment.augmentation import donor_corrections
from tapestry.experiment.replay import cached_nowcasts
from tapestry.model.scenario import Scenario

root=Path('data/experiments/c1-local-errors-20261002')
out=Path(__file__).parent
settings=json.loads((root/'experiment.json').read_text())
panel=load(settings['dataset'])
scenarios=[Scenario.from_string(s) for s in pd.read_csv(root/'jobs.csv').scenario]
# Runtime provenance verification of the unchanged nowcaster cache, before GPU use.
cached_nowcasts(panel,root,settings['dataset_sha256'])
records=[]
fig,axes=plt.subplots(2,2,figsize=(14,8))
for held_out in scenarios[0].scored_seasons:
    for inner in (True,False):
        roles=cv.week_roles(panel['dates'],scenarios[0],held_out)
        keep=(roles=='fit') if inner else np.isin(roles,['fit','validation'])
        corrections=donor_corrections(panel,scenarios[0],held_out,keep,root,settings['dataset_sha256'])
        fold=cv.fold(panel,scenarios[0],held_out,inner=inner)
        for scenario in scenarios:
            bank=ReportingErrors(panel,scenario,held_out,keep,corrections)
            assert bank.error_dates.isdisjoint(set(panel['dates'][~keep]))
            augmented=bank.batch(fold.train,np.random.default_rng(42))
            count=changed=0
            for original,new in zip(fold.train,augmented):
                for name in ('available','target_available','target_values','Y'):
                    np.testing.assert_array_equal(original[name],new[name])
                assert np.isfinite(new['values']).all()
                assert (new['values']>=0).all() and (new['values'][:,3:]<=1).all()
                count+=original['available'].sum()
                changed+=((original['values']!=new['values'])&original['available']).sum()
            records.append(dict(evaluation_season=held_out,stage='inner' if inner else 'full',
                method=scenario.reporting_method,strength=scenario.reporting_strength,
                donor_season=bank.donor_season,donor_windows=len(bank.donors),
                residual_evidence_fraction=float((bank.support&bank.visible).mean()),
                changed_available_fraction=float(changed/count),all_checks_passed=True))
            if held_out=='2025-2026' and not inner:
                examples=[(i,e) for i,e in enumerate(fold.train) if cv.season(e['context_dates'][-1])=='2023-2024']
                us=fold.train[0]['locations'].index('US')
                index,e=max(examples,key=lambda pair:pair[1]['values'][-1,0,us])
                for ax,(signal,location) in zip(axes.flat,[(0,'US'),(0,'NC'),(1,'US'),(1,'NC')]):
                    li=e['locations'].index(location)
                    if scenario==scenarios[0]:
                        ax.plot(e['context_dates'],e['values'][:,signal,li],color='black',lw=2,label='Finalized history')
                    label=scenario.reporting_method+(' (half strength)' if scenario.reporting_strength==.5 else '')
                    ax.plot(e['context_dates'],augmented[index]['values'][:,signal,li],label=label)
                    ax.set_title(f'{location}: '+('flu' if signal==0 else 'COVID')+' admissions')
                    ax.tick_params(axis='x',rotation=45)
                    ax.set_ylabel('Weekly admissions')
            print(records[-1],flush=True)
pd.DataFrame(records).to_csv(out/'input-audit.csv',index=False)
axes[0,0].legend(fontsize=8)
fig.suptitle('C1 training-input examples: one seeded error draw on 2023–24 finalized history\nErrors from 2024–25; C1 trains on 2022–25, predicts final outcomes, evaluates actual nowcast inputs in 2025–26')
fig.tight_layout();fig.savefig(out/'training-input-examples.png',dpi=170);plt.close(fig)
