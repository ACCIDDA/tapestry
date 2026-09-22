"""Identifiers: the configuration is its scenario string; a run adds `:s<seed>`.

Code, data, and git versions are provenance, not identity.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from tapestry.model.scenario import Scenario


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def identify(run, relative_to=None):
    """Identify a saved run from its manifest."""
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    scenario = Scenario.from_string(manifest['scenario'])
    seed = manifest['seed']
    config_id = scenario.scenario_string or 'default'
    return dict(config_id=config_id, model_id=f'{config_id}:s{seed}', seed=seed,
                run=os.path.relpath(run.resolve(), Path(relative_to).resolve()) if relative_to else str(run),
                label=manifest.get('run_id', config_id), scenario=asdict(scenario),
                provenance=dict(git=manifest.get('git'), eval_members=manifest.get('eval_members')))
