"""Plan inference/scoring jobs for every stage under a common evaluation seed rule."""
from pathlib import Path
from dataclasses import replace
import json,shutil,subprocess,sys,shlex
from tapestry.experiment.planner import read_jobs
from tapestry.model.scenario import Scenario

stages=sys.argv[1:] or ['00','01','02','03','03b','04','05','06','07','09']
for stage in stages:assert not (Path('data/experiments')/f'b1-b0-score-{stage}').exists(),'Existing plans: inspect/resume, never re-plan'
for stage in stages:
 source={'03b':'b1-b0-chain-03','04':'b0-current-training-normalized','05':'b0-fitting-split','06':'b0-fitting-split-draws'}.get(stage,f'b1-b0-chain-{stage}')
 job=next(j for j in read_jobs(Path('data/experiments')/source) if stage!='04' or 'input_mode=finalized_available' not in j['scenario'])
 scenario=Scenario.from_string(job['scenario'])
 if stage=='03b':scenario=replace(scenario,input_mode='finalized')
 dataset='data/audits/b0/original-panel-deadline2025.npz' if stage in ['00','01','02','03','03b'] else 'data/audits/b0/original-panel-unified.npz'
 name=f'b1-b0-score-{stage}';root=Path('data/experiments')/name
 command=[sys.executable,'-m','tapestry.experiment.planner','plan','-e',name,'-s',scenario.scenario_string,'--seeds','42','43','44','--device','cuda','--dataset',dataset,'--eval-members','2048']
 print(shlex.join(command),flush=True);subprocess.run(command,check=True)
 reference=Path('data/experiments/b0-current-training-normalized/code');shutil.rmtree(root/'code');shutil.copytree(reference,root/'code')
 p=root/'code/src/tapestry/experiment/planner.py';s=p.read_text();anchor="    if args.command == 'fit':\n";assert s.count(anchor)==1
 s=s.replace(anchor,anchor+'        from tapestry.chain_reforecast import fit as reforecast\n        reforecast(args)\n        return\n');p.write_text(s)
 shutil.copy2('analysis/b0-audit/chain_reforecast.py',root/'code/src/tapestry/chain_reforecast.py')
 (root/'reforecast-protocol.json').write_text(json.dumps(dict(stage=stage,source_experiment=source,source_scenario=job['scenario'],evaluation_seed_offset=1000,weight_updates=False,change='Shared B0 evaluation seed; stage03b alone restores forecast inputs while retaining stage03 weights',plan_command=command),indent=2)+'\n')
