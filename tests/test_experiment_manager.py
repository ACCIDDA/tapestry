import json
from pathlib import Path
import subprocess

import pytest

pytest.importorskip('torch')
from tapestry.models import manager
from tapestry.models.scenarios import ESSENTIAL, TrainingScenario, get_scenarios, get_training_scenario


def artifacts(output):
    output.mkdir(parents=True)
    (output / 'scores.csv').write_text('test\n1\n')
    (output / 'manifest.json').write_text(json.dumps({'folds': [dict(eval_season=s) for s in manager.SEASONS]}))
    for season in manager.SEASONS:
        folder = output / f'eval_{season}'
        folder.mkdir()
        for name in ('model.pt', 'forecasts.npz', 'training.json', 'scores.csv'):
            (folder / name).write_text('test')


def test_scenario_roundtrip_and_comparison_controls():
    grid = get_scenarios('grid')
    essential = get_scenarios()
    assert len(grid) == 289 and len(set(grid.values())) == 289
    assert len(essential) == 14 and set(essential.values()) <= set(grid.values())
    for scenario in grid.values():
        assert TrainingScenario.from_string(scenario.scenario_string) == scenario
        assert len((scenario.scenario_string + '::s42').encode()) < 255
    for name in ('state_us', 'residual2', 'latent32', 'balanced', 'flu_only', 'conv_h12', 'conv_h26'):
        candidate, control, _ = ESSENTIAL[name]
        baseline = essential[control]
        assert sum(getattr(candidate, field) != getattr(baseline, field)
                   for field in candidate.__dataclass_fields__) == 1
    with pytest.raises(ValueError):
        get_training_scenario('typo')
    with pytest.raises(ValueError):
        TrainingScenario.from_string(essential['anchor'].scenario_string.replace('geography=1', 'geography=2'))


def test_run_resume_and_protocol_collision(tmp_path, monkeypatch):
    definition = dict(dataset='input.npz', population_file='population.csv', runtime={})
    scenarios = {'anchor': get_training_scenario('anchor')}
    runs, selected = manager.prepare(tmp_path, definition, scenarios, [42, 43])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        artifacts(Path(command[command.index('--output') + 1]))

    monkeypatch.setattr(manager.subprocess, 'run', run)
    manager.execute(tmp_path, definition, runs, selected)
    assert len(calls) == 2
    manager.execute(tmp_path, definition, runs, selected)
    assert len(calls) == 2  # no refit on resume
    record = runs[selected[0]]
    assert json.loads((Path(record['output']) / 'manifest.json').read_text())['scenario_string'] == record['scenario_string']
    with pytest.raises(ValueError, match='changed'):
        manager.prepare(tmp_path, dict(definition, dataset='different.npz'), scenarios, [42])
    (Path(record['output']) / 'eval_2023-2024/model.pt').unlink()
    with pytest.raises(ValueError, match='artifacts'):
        manager.execute(tmp_path, definition, runs, selected)


def test_failure_retry_preserves_partial_attempt(tmp_path, monkeypatch):
    definition = dict(dataset='input.npz', population_file='population.csv', runtime={})
    runs, selected = manager.prepare(tmp_path, definition, {'baseline': get_training_scenario('baseline')}, [42])

    def fail(command, **kwargs):
        out = Path(command[command.index('--output') + 1])
        out.mkdir()
        (out / 'partial').write_text('preserve')
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(manager.subprocess, 'run', fail)
    assert manager.execute(tmp_path, definition, runs, selected, keep_going=True) == selected
    record = runs[selected[0]]
    previous = Path(record['output'])
    assert record['status'] == 'failed'

    def succeed(command, **kwargs):
        artifacts(Path(command[command.index('--output') + 1]))

    monkeypatch.setattr(manager.subprocess, 'run', succeed)
    manager.execute(tmp_path, definition, runs, selected)
    assert record['status'] == 'complete'
    assert len(record['attempts']) == 2
    assert (previous / 'partial').read_text() == 'preserve'
    assert Path(record['output']) != previous


def test_path_validation_and_lock(tmp_path):
    for name in ('../escape', '.', 'a/b', ''):
        with pytest.raises(ValueError):
            manager.experiment_folder(tmp_path, name)
    with manager.locked(tmp_path):
        with pytest.raises(RuntimeError, match='Another manager'):
            with manager.locked(tmp_path):
                pass


def test_protocol_hashes_inputs_and_compare_rejects_incomplete(tmp_path):
    from argparse import Namespace
    dataset, population, frozen = tmp_path / 'data.npz', tmp_path / 'population.csv', tmp_path / 'frozen'
    dataset.write_bytes(b'input')
    population.write_text('test')
    frozen.mkdir()
    (frozen / 'manifest.json').write_text('{}')
    args = Namespace(dataset=dataset, population_file=population, frozen=frozen, epochs=1,
                     width=8, batch_size=8, members=2, eval_members=4, lr=.001, device='cpu')
    definition = manager.protocol(args)
    manager.prepare(tmp_path, definition, {'baseline': get_training_scenario('baseline')}, [42])
    with pytest.raises(ValueError, match='incomplete'):
        manager.compare(tmp_path)
    dataset.write_bytes(b'changed')
    assert manager.protocol(args) != definition
    (frozen / 'manifest.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='scoring inputs changed'):
        manager.compare(tmp_path)


def test_compare_dispatches_completed_runs_to_epibench(tmp_path, monkeypatch):
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    definition = dict(frozen=str(frozen), frozen_files_sha256={}, code_sha256={})
    runs, selected = manager.prepare(tmp_path, definition, {'baseline': get_training_scenario('baseline')}, [42])
    out = tmp_path / 'output'
    artifacts(out)
    runs[selected[0]].update(status='complete', output=str(out))
    manager.save(tmp_path / 'runs.json', runs)
    calls = []
    monkeypatch.setattr(manager.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    destination = manager.compare(tmp_path)
    assert calls[0][2] == 'tapestry.evaluation.sweep'
    assert str(out) in calls[0] and str(destination) in calls[0]
    assert json.loads((tmp_path / 'comparison.json').read_text())['status'] == 'complete'


def test_real_three_fold_run_with_all_new_switches_and_resume(tmp_path):
    from dataclasses import replace
    from datetime import date, timedelta
    import numpy as np
    import torch
    from tapestry.model_data import FinalizedDataset
    from tapestry.models import B0

    days = tuple((date(2023, 9, 2) + timedelta(weeks=i)).isoformat() for i in range(157))
    panel = np.ones((len(days), 6, 2, 2), dtype=np.float32)
    panel[:, :, 0] *= 2 + np.arange(len(days), dtype=np.float32)[:, None, None] % 12
    panel[:, 3:, 0] *= .001
    dataset = tmp_path / 'data.npz'
    FinalizedDataset(panel, days, ('AL', 'US'), {}).save(dataset)
    population = tmp_path / 'population.csv'
    population.write_text('location,population\nAL,5000000\nUS,330000000\n')
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    (frozen / 'manifest.json').write_text('{}')
    scenario = replace(TrainingScenario(), lookback=26, encoder='conv', heads='state_us',
                       decoder='residual2', latent=32, loss_weights='balanced_admissions')
    arguments = ['-e', 'smoke', '--root', str(tmp_path), '--scenario', scenario.scenario_string,
                 '--dataset', str(dataset), '--population-file', str(population), '--frozen', str(frozen),
                 '--epochs', '1', '--width', '8', '--members', '2', '--eval-members', '4', '--seeds', '42']
    manager.main(['plan', *arguments])
    manager.main(['run', *arguments])
    records = json.loads((tmp_path / 'smoke/runs.json').read_text())
    record, = records.values()
    assert record['status'] == 'complete'
    output = Path(record['output'])
    loss_scales = []
    for season in manager.SEASONS:
        checkpoint = torch.load(output / f'eval_{season}/model.pt', weights_only=True)
        model = B0(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        assert model.config['encoder'] == 'conv' and model.config['heads'] == 'state_us'
        assert model.config['decoder'] == 'residual2' and model.config['latent'] == 32
        assert np.isfinite(checkpoint['metadata']['history']).all()
        with np.load(output / f'eval_{season}/forecasts.npz') as forecasts:
            assert np.isfinite(forecasts['quantiles']).all()
            assert forecasts['quantiles'].shape[0] == 5
        loss_scales.append(checkpoint['state_dict']['scale'])
    before = (output / 'scores.csv').stat().st_mtime_ns
    manager.main(['run', *arguments])
    assert (output / 'scores.csv').stat().st_mtime_ns == before
