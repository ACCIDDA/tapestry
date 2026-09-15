"""Identifiers: the configuration is its scenario string; a run adds `:s<seed>`.

Code, data, and git versions are provenance, not identity; the manager warns
when compared runs come from different commits.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from tapestry.models.scenarios import TrainingScenario


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def identify(run, relative_to=None):
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    scenario = TrainingScenario.from_config(manifest['config'])
    seed = manifest['config']['seed']
    config_id = scenario.scenario_string
    return dict(config_id=config_id, model_id=f'{config_id}:s{seed}', seed=seed,
                run=os.path.relpath(run.resolve(), Path(relative_to).resolve()) if relative_to else str(run),
                label=manifest.get('scenario_name', config_id), scenario=asdict(scenario),
                provenance=dict(dataset_sha256=manifest['dataset_sha256'], git=manifest.get('git'),
                                eval_members=manifest['config'].get('eval_members'),
                                code_sha256={Path(k).name: v for k, v in manifest['code_sha256'].items()}))
