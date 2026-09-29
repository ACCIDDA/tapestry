"""Scientific preflight: validation dates, no-label episodes and fixed loss weights."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from tapestry.dataset import cv
from tapestry.dataset.build import load
from tapestry.experiment.planner import read_jobs, pinned_inputs
from tapestry.model.objective import LOSS_WEIGHTS, loss_cell_weights
from tapestry.model.scenario import Scenario

ROOT = Path('docs/results/b0-fitting-controls')
ROOT.mkdir(parents=True, exist_ok=True)
BASE = Path('data/experiments/b0-current-training-normalized')
scenario = Scenario(ed_transform='logit', epochs=300, patience=30, fit_partition='pathogen')
panel = load('data/audits/b0/original-panel-unified.npz')
settings = json.loads((BASE / 'experiment.json').read_text())
rows = []
dates = np.asarray(panel['dates']).astype(str)
labels = np.array([cv.season(d) for d in dates])
spec = importlib.util.spec_from_file_location('tapestry.dataset.audit_cv',
    'data/experiments/b0-fitting-split/code/src/tapestry/dataset/cv.py')
legacy_cv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy_cv
spec.loader.exec_module(legacy_cv)

# Read the actual B0 constants and validation-draw function, without importing its old CLI.
legacy_root = Path('data/experiments/b0-exact-reproduction/legacy-code/src/tapestry/models')
if not legacy_root.exists():
    legacy_root = Path('data/audits/b0/old-src/tapestry/models')
legacy_source = (legacy_root / 'season_cv.py').read_text()
constants = {}
for node in ast.parse(legacy_source).body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.startswith('VALIDATION_'):
                constants[target.id] = ast.literal_eval(node.value)
            elif isinstance(target, ast.Tuple) and all(isinstance(n, ast.Name) and n.id.startswith('VALIDATION_') for n in target.elts):
                constants.update(zip([n.id for n in target.elts], ast.literal_eval(node.value)))
assert (scenario.validation_spacing, scenario.validation_offset, scenario.validation_weeks) == (
    constants['VALIDATION_SPACING'], constants['VALIDATION_OFFSET'], constants['VALIDATION_WEEKS'])

for policy, module in [('current', cv), ('b0', legacy_cv)]:
    for held in cv.SEASONS:
        inner = module.fold(panel, scenario, held, inner=True)
        full = module.fold(panel, scenario, held)
        if policy == 'b0':
            hidden = []
            for season in cv.SEASONS:
                if season == held:
                    continue
                weeks = np.flatnonzero(labels == season)
                position = np.arange(len(weeks)) % constants['VALIDATION_SPACING']
                use = (position >= constants['VALIDATION_OFFSET']) & (position < constants['VALIDATION_OFFSET'] + constants['VALIDATION_WEEKS'])
                hidden.extend(dates[weeks[use]])
            assert inner.info['validation_weeks'] == sorted(hidden)
        hidden = set(inner.info['validation_weeks'])
        for e in inner.train:
            assert all(cv.season(d) != held for i, d in enumerate(e['context_dates']) if e['available'][i].any())
            assert all(d not in hidden for i, d in enumerate(e['context_dates']) if e['available'][i].any())
            assert all(cv.season(d) != held and d not in hidden
                       for i, d in enumerate(e['target_dates']) if e['target_available'][i].any())
        for phase, episodes in [('selection', inner.train), ('validation', inner.validation), ('refit', full.train)]:
            mask = np.stack([e['target_available'] for e in episodes])
            weights = loss_cell_weights(episodes, LOSS_WEIGHTS[scenario.loss_weights])
            for pathogen, channels in [('flu', [0, 3]), ('covid', [1, 4]), ('rsv', [2, 5])]:
                removed = int((~mask[:, :, channels].any(axis=(1, 2, 3))).sum())
                mass = float(weights[:, :, channels].sum())
                rows.append(dict(policy=policy, held_out=held, phase=phase, pathogen=pathogen,
                                 episodes=len(episodes), empty_episodes=removed, loss_mass=mass))
                assert removed == 0, 'Filtering is not a no-op: add the episode-filter factor before launching'
                assert np.isclose(mass, 1 / 3, atol=1e-6), mass

run_source = (legacy_root / 'run.py').read_text()
node = next(n for n in ast.parse(run_source).body if isinstance(n, ast.FunctionDef) and n.name == 'validation_draws')
namespace = dict(torch=torch, VALIDATION_MEMBERS=256)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<historical validation_draws>', 'exec'), namespace)
for component in range(3):
    seed = 42 + 10000 * component
    model = SimpleNamespace(config=dict(latent=16, noise='global', us_error='none'))
    args = SimpleNamespace(seed=seed, validation_members=256, batch_size=8, device='cpu')
    episodes = inner.validation
    historical = namespace['validation_draws'](model, episodes, args)
    generator = torch.Generator().manual_seed(seed + 2000)
    for ids, old_z, _, _ in historical:
        z = torch.randn(256, len(ids), 16, generator=generator)
        assert torch.equal(z, old_z)

for arm in ('split', 'draws', 'split-draws'):
    root = Path('data/experiments') / f'b0-fitting-{arm}'
    planned = json.loads((root / 'experiment.json').read_text())
    for key in ('dataset', 'dataset_sha256', 'frozen', 'frozen_manifest_sha256', 'population_sha256', 'eval_members', 'device'):
        assert planned[key] == settings[key], (arm, key)
    assert all(planned[k] == v for k, v in pinned_inputs(planned).items())
    jobs = read_jobs(root)
    assert len(jobs) == 1 and jobs[0]['seeds'] == [42, 43, 44]
    assert jobs[0]['scenario'] == scenario.scenario_string
    relative = 'src/tapestry/model/objective.py'
    assert (root / 'code' / relative).read_bytes() == (BASE / 'code' / relative).read_bytes()

pd.DataFrame(rows).to_csv(ROOT / 'preflight_episode_weights.csv', index=False)
(ROOT / 'preflight.json').write_text(json.dumps(dict(
    baseline=str(BASE), filtering_removes_zero_episodes=True, loss_mass_per_pathogen=1/3,
    b0_validation_dates_match=True, held_out_and_validation_weeks_excluded=True,
    b0_validation_draws_match=True, input_and_recipe_hashes_match=True,
    factors=['validation calendar', 'validation random draws'],
    multiplier_intervention=False, new_runs=9), indent=2) + '\n')
print(pd.DataFrame(rows).groupby(['policy', 'phase']).agg(
    cases=('empty_episodes', 'size'), removed=('empty_episodes', 'sum'),
    min_loss_mass=('loss_mass', 'min'), max_loss_mass=('loss_mass', 'max')).to_string())
print('Preflight passed: no episode-filter intervention is necessary on these data.')
