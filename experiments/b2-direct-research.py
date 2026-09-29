"""Matched covariate attribution and information-sharing study; plan via the manager."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from tapestry.model.scenario import Scenario

NAME = 'b2-direct-research-v1'
DATASET = 'data/processed/panel-b2-operational.npz'
SOURCES = ('inpatient', 'outpatient', 'ww_wval_like', 'kinsa', 'ilinet', 'clinical_lab', 'flusurv')
ALL = '+'.join(SOURCES)
BACKBONES = [('pathogen-mlp', 'mlp', 'pathogen'), ('target-mlp', 'mlp', 'target'),
             ('target-multiscale', 'multiscale_conv', 'target')]
SPATIAL = ('none', 'pooled', 'attention', 'national_broadcast', 'gated_pool',
           'pathogen_spatial', 'target_spatial', 'joint_location_target')


def design():
    base = Scenario(input_mode='vintaged', asof_weeks=12, training_inputs='finalized',
                    input_normalization='b0', validation_calendar='b0', ed_transform='logit',
                    supplied_final=True, mask_rate=.5, epochs=300, patience=30,
                    members=128, validation_members=256, covariate_encoder='summary')
    rows = {}
    def add(scenario, family, backbone, sources):
        key = scenario.scenario_string
        row = rows.setdefault(key, dict(scenario=key, run_id=scenario.run_id, backbone=backbone,
                                       sources=sources, spatial=scenario.spatial,
                                       location_embedding=scenario.location_embedding, families=[]))
        row['families'].append(family)
    sources = [('', 'none')] + [(s, s) for s in SOURCES] + [(ALL, 'all')]
    sources += [('+'.join(s for s in SOURCES if s != omitted), 'all-minus-' + omitted) for omitted in SOURCES]
    for name, encoder, partition in BACKBONES:
        backbone = replace(base, encoder=encoder, fit_partition=partition)
        for groups, label in sources:
            add(replace(backbone, covariate_set=groups), 'source-attribution', name, label)
        for groups, label in [('', 'none'), ('kinsa', 'kinsa'), ('ilinet', 'ilinet'), (ALL, 'all')]:
            for spatial in SPATIAL:
                add(replace(backbone, covariate_set=groups, spatial=spatial), 'information-sharing', name, label)
            for spatial in ('none', 'attention'):
                add(replace(backbone, covariate_set=groups, spatial=spatial, location_embedding=8),
                    'location-embedding', name, label)
    assert len(rows) == 156
    return list(rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--experiment', default=NAME)
    parser.add_argument('--dataset', default=DATASET)
    args = parser.parse_args()
    rows = design()
    seeds = [42, 43, 44]
    if args.smoke:
        seeds = [42]
        args.experiment = 'b2-direct-research-smoke-v1'
        selected = [('pathogen-mlp', 'none'), ('target-mlp', 'national_broadcast'),
                    ('target-multiscale', 'joint_location_target'), ('pathogen-mlp', 'target_spatial')]
        rows = [dict(r) for name, spatial in selected for r in rows
                if r['backbone'] == name and r['spatial'] == spatial and r['sources'] == 'all'
                and r['location_embedding'] == 0]
        for row in rows:
            s = replace(Scenario.from_string(row['scenario']), epochs=2, patience=1)
            row.update(scenario=s.scenario_string, run_id=s.run_id)
    total = len(rows) * len(seeds)
    out = Path('analysis/b2-research/smoke-design.json' if args.smoke else 'analysis/b2-research/design.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(experiment=args.experiment, configurations=len(rows), seeds=seeds,
                                  runs=total, folds=total * 3, eval_members=256, rows=rows), indent=2) + '\n')
    command = [sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', args.experiment,
               '-s', *[r['scenario'] for r in rows], '--seeds', *map(str, seeds), '--device', 'cuda',
               '--eval-members', '256', '--dataset', args.dataset]
    if args.plan:
        folder = Path('data/experiments') / args.experiment
        if folder.exists():
            raise RuntimeError('Experiment already exists; use status to resume, never re-plan an existing study')
        subprocess.run(command, check=True)
        (folder / 'research-design.json').write_text(out.read_text())
    print(json.dumps(dict(configurations=len(rows), runs=total, folds=total * 3, eval_members=256,
                          design=str(out), planned=args.plan)))

if __name__ == '__main__':
    main()
