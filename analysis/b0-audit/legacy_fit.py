"""Run untouched historical B0 training behind the current experiment manager.

Only layout is adapted: top-level eval directories link to legacy outputs. The
current frozen scorer evaluates the original quantiles without modification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def fit(args):
    experiment = Path(__file__).resolve().parents[3]
    # This module is copied to EXPERIMENT/code/src/tapestry/legacy_reproduction.py.
    config_file = experiment/'legacy-configs.json'
    configs = json.loads(config_file.read_text())
    from tapestry.model.scenario import Scenario
    scenario = Scenario.from_string(args.scenario)
    config = configs[scenario.fit_partition]
    output = Path(args.output)
    old_output = output/'legacy'
    flags = []
    for name in ('epochs','patience','lookback','width','batch_size','members','lr','count_transform',
                 'ed_transform','loss_weights','encoder','spatial','heads','decoder','noise','us_error',
                 'head_sharing','location_embedding','fit_partition','validation_members','weight_decay','latent'):
        flags += ['--'+name.replace('_','-'),str(config[name])]
    for name in ('geography','dynamics'):
        if config[name]:flags += ['--'+name]
    flags += ['--annual-calendar' if config['annual_calendar'] else '--no-annual-calendar']
    env=dict(os.environ, PYTHONPATH=str(experiment/'legacy-code/src'))
    command=[sys.executable,'-m','tapestry.models.season_cv','--dataset',args.dataset,
             '--population-file','data/audits/b0/populations-from-checkpoint.csv','--eval-members',str(args.eval_members),
             '--device',args.device,'--seed',str(args.seed),'--output',str(old_output),*flags]
    subprocess.run(command,env=env,check=True)
    legacy=json.loads((old_output/'manifest.json').read_text())
    for held in ('2023-2024','2024-2025','2025-2026'):
        (output/f'eval_{held}').symlink_to(Path('legacy')/f'eval_{held}',target_is_directory=True)
    (output/'manifest.json').write_text(json.dumps(dict(scenario=args.scenario,seed=args.seed,
        folds=['2023-2024','2024-2025','2025-2026'], protocol='untouched_B0_legacy_training',
        legacy_manifest='legacy/manifest.json',dataset_sha256=legacy['dataset_sha256'],
        eval_members=args.eval_members,command=command),indent=2)+'\n')
    from tapestry.evaluation.totals import score_run
    score_run(output,args.frozen)
