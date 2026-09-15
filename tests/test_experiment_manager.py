"""Scenario strings, the sweep grid, and the per-task experiment layout with resume."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')

from tapestry.models import manager
from tapestry.models.scenarios import ANCHOR, BASELINE, CROSS_ANCHORS, GRID, TrainingScenario, get_scenarios, ofat


def test_short_strings_round_trip_and_reject_typos():
    scenarios = {*get_scenarios('essential').values(), *get_scenarios('grid').values()}
    assert len({scenario.scenario_string for scenario in scenarios}) == len(scenarios)
    assert all(TrainingScenario.from_string(s.scenario_string) == s for s in scenarios)
    text = ANCHOR.scenario_string
    assert text == ('b0:h12:tr_4rt:ed_lin:geo1:dyn1:lw_first:enc_mlp:sp_none:hd_sh:dec_leg:nz_glob:'
                    'z16:w64:ep50:pat0:bs8:m8:lr0.001')
    for bad in (text.replace('tr_4rt', 'tr_fourth'), text.replace(':h12', ':h012'), text.replace(':z16', ''),
                text.replace('b0:', 'b1:'), text.replace('w64:ep50', 'ep50:w64'), text + ':x1',
                text.replace('geo1', 'geo2'), text.replace('lr0.001', 'lr1e-3'), text.replace('pat0', 'pat'),
                text.replace('nz_glob', 'nz_global'), text.replace('ep50:pat0', 'ep50:pat50')):
        with pytest.raises(ValueError):
            TrainingScenario.from_string(bad)


def test_grid_crosses_every_factor_once():
    grid = get_scenarios('grid')
    assert len(grid) == 4097
    assert grid['baseline'] == replace(BASELINE, loss_weights='objective')
    sweep = [scenario for name, scenario in grid.items() if name != 'baseline']
    assert len(set(sweep)) == 4096
    assert {s.loss_weights for s in grid.values()} == {'objective'}
    assert all(s.geography and s.width == 64 and s.lr == .001 for s in sweep)
    for field, values in GRID.items():
        observed = Counter((s.epochs, s.patience) if field == 'stopping' else getattr(s, field) for s in sweep)
        assert observed == {value: 4096 // len(values) for value in values}


def test_crosses_cover_each_axis_without_factorial_combinations():
    scenarios = get_scenarios('crosses')
    candidates = set(scenarios.values())
    assert len(candidates) == len(scenarios) < 60
    assert set(CROSS_ANCHORS.values()) <= candidates
    for anchor in CROSS_ANCHORS.values():
        for field, values in {**GRID, 'count_transform': ('raw', *GRID['count_transform']),
                              'ed_transform': ('linear', *GRID['ed_transform']),
                              'geography': (False, True)}.items():
            for value in values:
                options = dict(zip(('epochs', 'patience'), value)) if field == 'stopping' else {field: value}
                assert replace(anchor, **options) in candidates
    for candidate in candidates:
        assert candidate.loss_weights == 'objective'
        assert any(len({('stopping' if key in ('epochs', 'patience') else key)
                        for key, value in asdict(candidate).items() if value != getattr(anchor, key)}) <= 1
                   for anchor in CROSS_ANCHORS.values())


def test_flags_parse_back_to_the_same_scenario():
    from tapestry.models.season_cv import build_parser
    extra = replace(ANCHOR, width=32, epochs=300, patience=20, batch_size=4, members=4, lr=3e-4,
                    spatial='attention', noise='local', count_transform='log1p', ed_transform='logit')
    for scenario in (BASELINE, *get_scenarios('essential').values(), extra):
        assert TrainingScenario.from_config(vars(build_parser().parse_args(scenario.flags()))) == scenario


def test_ofat_names_single_changes():
    assert ofat(ANCHOR, width=(32, 64, 128), heads=('state_us',)) == {
        'width_32': replace(ANCHOR, width=32), 'width_128': replace(ANCHOR, width=128),
        'heads_su': replace(ANCHOR, heads='state_us')}


def test_array_commands_chunk_indices_below_the_slurm_cap():
    assert manager.ranges([5, 0, 1, 2, 7, 8]) == '0-2,5,7-8'
    assert manager.array_commands([0, 1, 2, 999, 1000, 1001, 2500], 'sweep') == [
        'sbatch --array=0-2,999 --export=ALL,OFFSET=0 scripts/b0_sweep.sbatch sweep',
        'sbatch --array=0-1 --export=ALL,OFFSET=1000 scripts/b0_sweep.sbatch sweep',
        'sbatch --array=500 --export=ALL,OFFSET=2000 scripts/b0_sweep.sbatch sweep']


def write_cv(output, seed, scenario=ANCHOR):
    for season in manager.SEASONS:
        for name in ('model.pt', 'forecasts.npz', 'training.json', 'scores.csv'):
            path = output / f'eval_{season}' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('x')
    (output / 'scores.csv').write_text('x')
    (output / 'manifest.json').write_text(json.dumps(dict(
        config=dict(asdict(scenario), seed=seed), dataset_sha256='d', code_sha256={},
        folds=[dict(eval_season=season) for season in manager.SEASONS])))


def write_frozen(folder, quantiles):
    folder.mkdir(parents=True)
    (folder / 'manifest.json').write_text(json.dumps(dict(quantiles=quantiles, cases=[])))


def test_plan_run_resume_and_status(tmp_path, monkeypatch):
    from tapestry.evaluation.hubs import QCOLS
    monkeypatch.chdir(tmp_path)
    calls, fail = [], {43}

    def fake_run(command, log):
        output = Path(command[command.index('--output' if '--output' in command else '--run') + 1])
        if 'tapestry.evaluation.totals' in command:
            (output / 'totals.csv').write_text('x')
            return
        seed = int(command[command.index('--seed') + 1])
        calls.append(seed)
        if seed in fail:
            raise RuntimeError('fold failed')
        write_cv(output, seed)

    monkeypatch.setattr(manager, 'execute', fake_run)
    monkeypatch.setattr(manager, 'git_state', lambda: dict(git_commit='abc', git_dirty=False))
    write_frozen(Path('frozen'), QCOLS)
    write_frozen(Path('five'), QCOLS[::5])
    folder = Path('data/experiments/demo')
    manager.main(['plan', '-e', 'demo', '--scenario', 'anchor', 'baseline', '--seeds', '42', '43', '--frozen', 'five'])
    with pytest.raises(ValueError, match='rebuild frozen support'):
        manager.main(['run', '-e', 'demo', '--task', '0'])
    assert calls == []
    manager.main(['plan', '-e', 'demo', '--scenario', 'anchor', 'baseline', '--seeds', '42', '43', '--frozen', 'frozen'])
    with pytest.raises(SystemExit):
        manager.main(['run', '-e', 'demo', '--task', '0', '--keep-going'])
    rows = {(row['name'], row['seed']): row for row in manager.collect(folder)}
    assert {key: row['status'] for key, row in rows.items()} == {
        ('anchor', 42): 'complete', ('anchor', 43): 'failed', ('baseline', 42): 'planned', ('baseline', 43): 'planned'}
    assert rows['anchor', 42]['attempt'] == f'{ANCHOR.scenario_string}/s42/attempt-001'
    assert rows['anchor', 42]['git_commit'] == 'abc'
    assert (folder / 'runs.csv').is_file()
    manifest = json.loads((folder / rows['anchor', 42]['attempt'] / 'cv/manifest.json').read_text())
    assert manifest['scenario_name'] == 'anchor'
    record = json.loads((folder / rows['anchor', 42]['attempt'] / 'run.json').read_text())
    assert record['scoring_command'][-2:] == ['--frozen', 'frozen']

    # A run without totals is not complete and is retried.
    (folder / rows['anchor', 42]['attempt'] / 'cv/totals.csv').unlink()
    assert not manager.seed_state(folder, ANCHOR.scenario_string, 42)[2]

    # Extending keeps task numbers; rerunning reuses complete seeds and retries in a new attempt.
    manager.main(['plan', '-e', 'demo', '--scenario', 'state_us', 'anchor', '--seeds', '44', '--frozen', 'frozen'])
    assert [(job['task'], job['name'], job['seeds']) for job in manager.read_jobs(folder)] == [
        (0, 'anchor', [42, 43, 44]), (1, 'baseline', [42, 43]), (2, 'state_us', [44])]
    calls.clear()
    fail.clear()
    manager.main(['run', '-e', 'demo', '--task', '0'])
    assert calls == [42, 43, 44]
    rows = {(row['name'], row['seed']): row for row in manager.collect(folder)}
    assert rows['anchor', 43]['status'] == 'complete'
    assert rows['anchor', 43]['attempt'].endswith('s43/attempt-002')
    assert manager.pending_tasks(rows.values()) == [1, 2]
