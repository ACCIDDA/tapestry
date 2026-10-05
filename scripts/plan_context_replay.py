"""Plan scheduled-availability C1 replay with matched target-history corrections."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from tapestry.model.scenario import Scenario
from tapestry.experiment.planner import seed_state
from tapestry.experiment.provenance import sha256, save

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('-e','--experiment',default='context-replay-v1-20261001')
p.add_argument('--nowcaster',default='conditional_chain')
p.add_argument('--growth',type=float,default=.2)
p.add_argument('--penalty',type=float,default=10.)
p.add_argument('--reference',default='context-nowcast-v2-20261001')
p.add_argument('--growth-weight',type=float,default=1.)
p.add_argument('--residual-halflife',type=float,default=26.)
p.add_argument('--features',default='basic')
p.add_argument('--gate',default='none')
p.add_argument('--strength',type=float,default=1.)
p.add_argument('--candidate',help='Run ID within --reference; use its exact nowcaster settings')
p.add_argument('--uncertainty',type=float,nargs='*',default=[],help='Additional causal joint-history uncertainty strengths')
p.add_argument('--seasonal-uncertainty',type=float,nargs='*',default=[],help='Additional uncertainty around the seasonal center')
p.add_argument('--additional-strengths',type=float,nargs='*',default=[],help='Other completed correction strengths within the same reference experiment')
p.add_argument('--eval-members',type=int,default=256)
a=p.parse_args()
reference=Path('data/experiments')/a.reference
reference_jobs=pd.read_csv(reference/'jobs.csv')
if a.candidate:
    matches=[Scenario.from_string(s) for s in reference_jobs.scenario if Scenario.from_string(s).run_id==a.candidate]
    if len(matches)!=1:raise ValueError('Candidate run ID not unique in reference')
    selected=matches[0]
    for argument,field in [('nowcaster','model'),('growth','growth'),('penalty','penalty'),
        ('growth_weight','growth_weight'),('residual_halflife','residual_halflife'),('features','features'),('gate','gate'),('strength','strength')]:
        setattr(a,argument,getattr(selected,'finalization_'+field))
source=Path('data/experiments/b-2-t0')
settings=json.loads((source/'experiment.json').read_text())
base=Scenario.from_string(pd.read_csv('docs/experiments/b-2-t0/named_ranking.csv').iloc[0].config_id)
folder=Path('data/experiments')/a.experiment
if (folder/'jobs.csv').exists():
    raise SystemExit('Already planned; use status/run to resume without re-pinning')
checkpoints={}
for seed in (42,43,44):
    attempt,_,complete=seed_state(source,base.scenario_string,seed)
    if not complete:raise ValueError(f'Missing C1 seed {seed}')
    for fold in base.scored_seasons:
        path=attempt/f'eval_{fold}'/'model.pt'
        if json.loads((path.parent/'manifest.json').read_text())['dataset_sha256']!=settings['dataset_sha256']:
            raise ValueError('C1 dataset mismatch')
        checkpoints[f'{base.run_id}/s{seed}/{fold}']=dict(path=str(path),sha256=sha256(path))
arms=[('finalized','selected',.2),('vintage','selected',.2),('nowcast','selected',.2),('nowcast',a.nowcaster,a.growth)]
scenarios=[replace(base,replay_from=str(source),replay_inputs=arm,replay_flags='available',
    replay_schedule=True,replay_nowcaster=model,replay_growth=growth,
    replay_penalty=a.penalty if model=='context_residual' else 10.,
    replay_growth_weight=a.growth_weight if model=='context_residual' else 1.,
    replay_residual_halflife=a.residual_halflife if model=='context_residual' else 26.,
    replay_features=a.features if model=='context_residual' else 'basic',
    replay_gate=a.gate if model=='context_residual' else 'none',
    replay_strength=a.strength if model=='context_residual' else 1.).scenario_string for arm,model,growth in arms]
point=Scenario.from_string(scenarios[-1])
scenarios.extend(replace(point,replay_uncertainty=strength).scenario_string for strength in a.uncertainty)
seasonal=Scenario.from_string(scenarios[2])
scenarios.extend(replace(seasonal,replay_uncertainty=strength).scenario_string for strength in a.seasonal_uncertainty)
for strength in a.additional_strengths:
    scenarios.extend(replace(point,replay_strength=strength,replay_uncertainty=u).scenario_string
                     for u in [0.,*a.uncertainty])
reference=Path('data/experiments')/a.reference
reference_jobs=pd.read_csv(reference/'jobs.csv')
matches=[s for s in reference_jobs.scenario if Scenario.from_string(s).finalization_model==a.nowcaster
    and Scenario.from_string(s).finalization_features==a.features
    and Scenario.from_string(s).finalization_gate==a.gate
    and Scenario.from_string(s).finalization_strength==a.strength
    and Scenario.from_string(s).finalization_growth_weight==a.growth_weight
    and Scenario.from_string(s).finalization_residual_halflife==a.residual_halflife
    and (Scenario.from_string(s).finalization_penalty==a.penalty if a.nowcaster=='context_residual' else Scenario.from_string(s).finalization_growth==a.growth)]
if len(matches)!=1:raise ValueError('Need exactly one matching completed nowcaster')
records=[dict(root=str(reference),scenario=matches[0])]
for strength in a.additional_strengths:
    s=replace(Scenario.from_string(matches[0]),finalization_strength=strength).scenario_string
    if s not in set(reference_jobs.scenario):raise ValueError(f'No reference for strength {strength}')
    records.append(dict(root=str(reference),scenario=s))
for record in records:
    if not seed_state(reference,record['scenario'],42)[2]:
        raise ValueError('Nowcaster reference is not complete')
subprocess.run([sys.executable,'-m','tapestry.experiment.planner','plan','-e',a.experiment,
    '-s',*scenarios,'--seeds','42','43','44','--device','cuda','--eval-members',str(a.eval_members),
    '--dataset',settings['dataset'],'--frozen',settings['frozen']],check=True)
save(folder/'replay-source.json',dict(source=str(source),dataset_sha256=settings['dataset_sha256'],
    checkpoints=checkpoints,labels={base.run_id:'C1'},
    nowcaster_reference=records[0],nowcaster_references=records))
print(f'Pinned six C1 CV checkpoints; planned {len(scenarios)*3} runs / {len(scenarios)*6} fold evaluations.')
