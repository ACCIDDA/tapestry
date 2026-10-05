"""Plan 48 B2 architectures x 30 revision treatments x three seeds."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from tapestry.model.scenario import Scenario
from tapestry.experiment.provenance import save, sha256


def architectures():
    ranking = pd.read_csv('docs/experiments/b-2-t0/named_ranking.csv')
    candidates = []
    seen = set()
    for row in ranking.itertuples():
        s = replace(Scenario.from_string(row.config_id), supplied_final=False, mask_rate=0.,
                    reporting_missingness=False, reporting_augmentation='vintage')
        if s.scenario_string not in seen:
            candidates.append((s, row.label)); seen.add(s.scenario_string)
    # Keep the 16 strongest distinct architectures, then cover the original
    # architecture/source choices by greedy categorical distance, rank as tie break.
    chosen, rest = candidates[:16], candidates[16:]
    axes = ['encoder','spatial','fit_partition','covariate_set','covariate_encoder',
            'signal_features','coordinates','lookback','width','decoder']
    while len(chosen) < 48:
        def distance(item):
            return min(sum(getattr(item[0],k) != getattr(other[0],k) for k in axes) for other in chosen)
        best = max(range(len(rest)), key=lambda i:(distance(rest[i]),-i))
        chosen.append(rest.pop(best))
    return chosen


def treatments():
    result=[]
    def add(label,family='direct',**kw):
        result.append((label,dict(weekend_family=family,**kw)))
    add('Unchanged finalized training inputs', reporting_probability=0.)
    for strength in (.25,.5,1.):
        add(f'Local seasonal log errors strength {strength}', reporting_strength=strength)
    for strength in (.5,1.):
        add(f'Half clean episodes; local log errors strength {strength}', reporting_strength=strength,reporting_probability=.5)
    for method in ('calendar_log','synchronous_log'):
        for strength in (.5,1.):
            add(f'{method} strength {strength}',reporting_method=method,reporting_strength=strength)
    add('Local phase matching without calendar',reporting_method='phase_log')
    add('Local log errors random strength zero to one',reporting_random_strength=True)
    for features,penalty,strength in [('phase',1.,1.),('phase',10.,1.),('phase',100.,1.),
        ('phase',10.,.25),('phase',10.,.5),('age',10.,1.),('phase_local',1.,1.),
        ('phase_local',10.,1.),('phase_local',100.,1.)]:
        add(f'Nowcaster {features}; ridge {penalty}; correction {strength}',family='two_stage',
            correction_features=features,correction_penalty=penalty,correction_strength=strength,nowcast_weeks=4)
    add('Nowcaster phase; synchronous errors; half correction',family='two_stage',
        correction_strength=.5,reporting_method='synchronous_log',nowcast_weeks=4)
    for name,kw in [('local full',{}),('local half',dict(reporting_strength=.5)),
        ('synchronous full',dict(reporting_method='synchronous_log')),
        ('local random',dict(reporting_random_strength=True))]:
        for weight in (.25,1.):
            add(f'Joint recent reconstruction and forecast; {name}; reconstruction weight {weight}',
                family='joint',nowcast_weeks=4,joint_weight=weight,**kw)
    assert len(result)==30
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-e','--experiment',default='b2-weekend-20261002')
    p.add_argument('--describe',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Three model families, one architecture, 2 epochs, three seeds')
    a=p.parse_args()
    configs=[]; rows=[]
    bases=architectures()
    arms=treatments()
    if a.smoke:
        bases=bases[:1]; arms=[arms[3],arms[13],arms[24]]
    for i,(base,label) in enumerate(bases):
        for j,(treatment,changes) in enumerate(arms):
            s=replace(base,**changes)
            if a.smoke:
                s=replace(s,epochs=2,patience=0,members=8,validation_members=16)
            configs.append(s.scenario_string)
            rows.append(dict(run_id=s.run_id,architecture=i+1,source_b2_label=label,
                family=s.weekend_family,treatment=treatment,scenario=s.scenario_string))
    assert len(configs)==len(set(configs))
    print(f'{len(configs)} configurations; {len(configs)*3} seeds; {len(configs)*6} scored folds')
    if a.describe:
        print(pd.DataFrame(rows).groupby('family').size().to_string()); return
    folder=Path('data/experiments')/a.experiment
    if (folder/'jobs.csv').exists(): raise SystemExit('Already planned; use status/run to resume')
    settings=json.loads(Path('data/experiments/b-2-t0/experiment.json').read_text())
    if sha256(settings['dataset']) != settings['dataset_sha256']: raise ValueError('B2 dataset changed')
    subprocess.run([sys.executable,'-m','tapestry.experiment.planner','plan','-e',a.experiment,
        '-s',*configs,'--seeds','42','43','44','--device','cuda','--eval-members','256',
        '--dataset',settings['dataset'],'--frozen',settings['frozen']],check=True)
    pd.DataFrame(rows).to_csv(folder/'design.csv',index=False)
    save(folder/'protocol.json',dict(source_b2_configurations=309,configurations=len(configs),
        seeds=[42,43,44],scored_folds=len(configs)*6,architecture_selection='Top 16 unique after removing masking/finality; 32 greedy diverse original B2 architectures',
        availability='B2 source schedule, no random input masking, no transported missingness; absent archived evaluation cells use frozen-final proxies',
        folds={'2024-2025':dict(training=['2022-2023','2023-2024','2025-2026'],errors='2025-2026',retrospective=True),
               '2025-2026':dict(training=['2022-2023','2023-2024','2024-2025'],errors='2024-2025',retrospective=False)},
        labels='Finalized values; forecast t+1 to t+4; joint models also reconstruct t-3 to t0',
        dataset_sha256=settings['dataset_sha256']))

if __name__=='__main__': main()
