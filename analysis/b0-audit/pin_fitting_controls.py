"""Pin two B0 fitting interventions onto the completed normalized control's code.

Only experiment snapshots are edited. The supervised-loss multiplier is unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


ARMS = {'split': (True, False), 'draws': (False, True), 'split-draws': (True, True)}
REFERENCE = Path('data/experiments/b0-current-training-normalized/code')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def pin(arm):
    root = Path('data/experiments') / f'b0-fitting-{arm}'
    assert (root / 'experiment.json').exists(), 'Plan through the manager first'
    assert not list(root.glob('*/s*/attempt-*')), 'Never alter an existing attempt snapshot'
    assert not (root / 'fitting-intervention.json').exists(), 'Already pinned; do not re-plan'
    split, draws = ARMS[arm]
    destination = root / 'code'
    shutil.rmtree(destination)
    shutil.copytree(REFERENCE, destination)
    relative_cv = Path('src/tapestry/dataset/cv.py')
    relative_fit = Path('src/tapestry/experiment/training.py')
    if split:
        p = destination / relative_cv
        source = replace_once(p.read_text(),
            "                start = season_start(int(label[:4]))\n"
            "                position = np.array([(date.fromisoformat(dates[i][:10]) - start).days // 7 for i in weeks])\n",
            "                # Experimental B0 split: count from first observed week of each season.\n"
            "                position = np.arange(len(weeks))\n")
        p.write_text(source)
    if draws:
        p = destination / relative_fit
        source = p.read_text()
        source = replace_once(source,
            '    optimizer = torch.optim.Adam(model.parameters(), lr=scenario.lr, weight_decay=scenario.weight_decay)\n',
            '''    # B0's fixed validation latent draws, generated on CPU separately from training.
    # This audit is restricted to the saved global-noise/no-covariate recipe.
    validation_z = []
    if selecting:
        assert scenario.noise == 'global' and scenario.us_error == 'none'
        assert not scenario.covariate_set and 'history_samples' not in validation[0]
        generator = torch.Generator().manual_seed(seed + 2000)
        for ids in torch.arange(len(validation)).split(scenario.batch_size):
            validation_z.append(torch.randn(scenario.validation_members, len(ids),
                                            model.config['latent'], generator=generator).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=scenario.lr, weight_decay=scenario.weight_decay)
''')
        source = replace_once(source,
            '''            with torch.no_grad(), torch.random.fork_rng(devices=[torch.device(device)] if str(device).startswith("cuda") else []):
                torch.manual_seed(seed + 900000)
                for ids in torch.arange(len(validation), device=device).split(scenario.batch_size):
                    samples = training_samples(model, validation, ids, vvalues[ids], vavailable[ids],
                                               vknown_final[ids], vcal[ids], None if vcov is None else vcov[ids],
                                               scenario.validation_members, scenario.input_mode == 'vintaged')
''',
            '''            with torch.no_grad():
                for ids, z in zip(torch.arange(len(validation), device=device).split(scenario.batch_size), validation_z):
                    samples = model(values=vvalues[ids], available=vavailable[ids],
                                    known_final=vknown_final[ids], calendar=vcal[ids], z=z,
                                    locations=list(validation[0]['locations']),
                                    vintaged=scenario.input_mode == 'vintaged')
''')
        p.write_text(source)
    changed = []
    for original in sorted((REFERENCE / 'src').rglob('*.py')):
        relative = original.relative_to(REFERENCE)
        if digest(original) != digest(destination / relative):
            changed.append(str(relative))
    expected = ([str(relative_cv)] if split else []) + ([str(relative_fit)] if draws else [])
    assert sorted(changed) == sorted(expected)
    assert 'options.update(input_scales(' in (destination / relative_fit).read_text()
    record = dict(arm=arm,reference=str(REFERENCE),b0_validation_weeks=split,
                  b0_validation_draws=draws,loss_multiplier='unchanged current-code weights',
                  episode_filter='No-op on audited finalized episodes; no filtering patch',
                  changed_files=changed,
                  reference_hashes={str(p.relative_to(REFERENCE)):digest(p)
                                    for p in sorted((REFERENCE / 'src').rglob('*.py'))},
                  snapshot_hashes={str(p.relative_to(destination)):digest(p)
                                   for p in sorted((destination / 'src').rglob('*.py'))})
    (root / 'fitting-intervention.json').write_text(json.dumps(record, indent=2) + '\n')
    print(root / 'fitting-intervention.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=ARMS)
    pin(parser.parse_args().arm)
