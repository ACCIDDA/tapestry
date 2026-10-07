"""Compare evaluation sample counts using the same trained B2 C1 checkpoints."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shlex
import subprocess
import sys

from tapestry.experiment.planner import check_pinned_inputs, seed_state
from tapestry.experiment.provenance import save, sha256
from tapestry.model.scenario import Scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='b2-eval-samples-20261005')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--eval-members', type=int, nargs='+', default=[256, 2048])
    args = parser.parse_args()
    source = Path('data/experiments/b-2-t0')
    settings = json.loads((source / 'experiment.json').read_text())
    check_pinned_inputs(settings)
    base = Scenario.from_string(
        'ed_transform=logit,spatial=distance,epochs=300,patience=30,'
        'fit_partition=pathogen,supplied_final=1,mask_rate=0.2,'
        'covariate_encoder=summary,input_mode=scheduled_final,'
        'input_normalization=b0,evaluation_seasons=recent_two')
    replay = replace(base, replay_from=str(source), replay_inputs='finalized')
    checkpoints = {}
    for seed in args.seeds:
        attempt, _, complete = seed_state(source, base.scenario_string, seed)
        if not complete:
            raise ValueError(f'Missing completed B2 C1 checkpoint for seed {seed}')
        for season in base.scored_seasons:
            path = attempt / f'eval_{season}' / 'model.pt'
            metadata = json.loads((path.parent / 'manifest.json').read_text())
            if metadata['dataset_sha256'] != settings['dataset_sha256']:
                raise ValueError('Source checkpoint used a different dataset')
            checkpoints[f'{base.run_id}/s{seed}/{season}'] = dict(
                path=str(path), sha256=sha256(path))
    names = [f'{args.prefix}-n{members}' for members in args.eval_members]
    for name in names:
        if (Path('data/experiments') / name).exists():
            raise ValueError(f'{name} already exists; use manager status/run to resume')
    for name, members in zip(names, args.eval_members):
        command = [sys.executable, '-m', 'tapestry.experiment.planner', 'plan',
                   '-e', name, '-s', replay.scenario_string,
                   '--seeds', *map(str, args.seeds), '--device', 'cuda',
                   '--eval-members', str(members), '--dataset', settings['dataset'],
                   '--frozen', settings['frozen']]
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        save(Path('data/experiments') / name / 'replay-source.json', dict(
            source=str(source), dataset_sha256=settings['dataset_sha256'],
            checkpoints=checkpoints, labels={base.run_id: 'B2 C1'},
            purpose='Evaluation sample count only; fixed trained weights and original inputs',
            evaluation_rng_seeds=[seed + 1000 for seed in args.seeds]))


if __name__ == '__main__':
    main()
