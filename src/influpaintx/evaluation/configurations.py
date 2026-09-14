"""Stable, extensible identifiers for a formulation and its seeded realization."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def identify(run, family='B0'):
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    config = dict(manifest['config'])
    seed = config.pop('seed')
    for key in ('output', 'device', 'dataset', 'population_file'):
        config.pop(key, None)
    # Unknown future configuration fields participate automatically in identity.
    identity = dict(schema=1, family=family, config=config,
                    dataset_sha256=manifest['dataset_sha256'],
                    code_sha256={Path(k).name: v for k, v in manifest['code_sha256'].items()},
                    populations=manifest.get('folds', [{}])[0].get('experiment', {}).get('populations'))
    full_hash = digest(identity)
    config_id = f'{family}-{full_hash[:12]}'
    return dict(config_id=config_id, model_id=f'{config_id}-s{seed}', seed=seed,
                identity_sha256=full_hash, identity=identity, run=str(run.resolve()), label=run.name)
