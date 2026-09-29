"""B-2: scheduled final inputs, two held-out seasons, three seeds per configuration."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tapestry.model.scenario import Scenario

SOURCES = ('inpatient', 'outpatient', 'ww_wval_like', 'kinsa', 'ilinet', 'clinical_lab', 'flusurv')
ALL = '+'.join(SOURCES)
BACKBONES = [('pathogen-mlp', 'mlp', 'pathogen'), ('target-mlp', 'mlp', 'target'),
             ('target-multiscale', 'multiscale_conv', 'target')]


def design():
    base = Scenario(input_mode='scheduled_final', evaluation_seasons='recent_two',
                    input_normalization='b0', ed_transform='logit', supplied_final=True,
                    mask_rate=.2, epochs=300, patience=30, members=128,
                    validation_members=256, covariate_encoder='summary')
    rows = {}
    def add(s, family, backbone, sources):
        row = rows.setdefault(s.scenario_string, dict(scenario=s.scenario_string, run_id=s.run_id,
                              backbone=backbone, sources=sources, spatial=s.spatial,
                              covariate_encoder=s.covariate_encoder, mask_rate=s.mask_rate,
                              coordinates=s.coordinates, signal_features=s.signal_features, families=[]))
        if family not in row['families']:
            row['families'].append(family)
    sources = [('', 'none')] + [(s, s) for s in SOURCES] + [(ALL, 'all')]
    sources += [('+'.join(s for s in SOURCES if s != omitted), 'all-minus-' + omitted) for omitted in SOURCES]
    for name, encoder, partition in BACKBONES:
        backbone = replace(base, encoder=encoder, fit_partition=partition)
        for groups, label in sources:
            add(replace(backbone, covariate_set=groups), 'source-attribution', name, label)
        for groups, label in [('', 'none'), ('kinsa', 'kinsa'), ('ilinet', 'ilinet'), (ALL, 'all')]:
            for spatial in ('none', 'pooled', 'attention', 'national_broadcast', 'gated_pool',
                            'neighbors', 'distance', 'gravity'):
                add(replace(backbone, covariate_set=groups, spatial=spatial), 'information-sharing', name, label)
            for spatial in ('none', 'distance'):
                add(replace(backbone, covariate_set=groups, spatial=spatial, coordinates=True), 'coordinates', name, label)
        for cov_encoder in ('raw', 'smooth', 'summary', 'shared'):
            for spatial in ('none', 'neighbors', 'national_broadcast'):
                for mask in (0., .1, .2):
                    add(replace(backbone, covariate_set=ALL, covariate_encoder=cov_encoder,
                                spatial=spatial, mask_rate=mask), 'encoding-and-masking', name, 'all')
        for features in ('multiscale', 'smooth_multiscale'):
            for spatial in ('none', 'neighbors', 'national_broadcast'):
                for mask in (0., .1, .2):
                    add(replace(backbone, covariate_set=ALL, signal_features=features, spatial=spatial,
                                mask_rate=mask), 'signal-transforms', name, 'all')
    return list(rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--experiment', default='b-2-t0')
    parser.add_argument('--dataset', default='data/processed/panel.npz')
    args = parser.parse_args()
    rows = design()
    if args.smoke:
        args.experiment += '-smoke'
        rows = [dict(r) for r in rows if r['sources'] == 'all' and r['backbone'] == 'pathogen-mlp'
                and r['spatial'] in ('none', 'neighbors', 'distance', 'gravity', 'national_broadcast')
                and (r['signal_features'] == 'none' or r['spatial'] == 'neighbors')
                and r['mask_rate'] == .2 and r['covariate_encoder'] == 'summary' and not r['coordinates']]
        for row in rows:
            s = replace(Scenario.from_string(row['scenario']), epochs=2, patience=1, members=8, validation_members=8)
            row.update(scenario=s.scenario_string, run_id=s.run_id)
    seeds = [42, 43, 44]
    specification = dict(experiment=args.experiment, target_availability='T-0', configurations=len(rows), seeds=seeds, runs=len(rows)*3,
                  folds=len(rows)*6, eval_members=256, rows=rows)
    out = Path('analysis') / args.experiment / 'design.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specification, indent=2)+'\n')
    if args.plan:
        folder = Path('data/experiments') / args.experiment
        if folder.exists():
            raise RuntimeError('Experiment exists; use manager status to resume')
        subprocess.run([sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', args.experiment,
                        '-s', *[r['scenario'] for r in rows], '--seeds', *map(str, seeds), '--device', 'cuda',
                        '--eval-members', '256', '--dataset', args.dataset], check=True)
        (folder/'research-design.json').write_text(out.read_text())
    print(json.dumps({k:v for k,v in specification.items() if k != 'rows'}))

if __name__ == '__main__':
    main()
