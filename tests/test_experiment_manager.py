"""Scenario strings and the per-task experiment layout with resume."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')

from tapestry.models import manager
from tapestry.models.scenarios import ANCHOR, BASELINE, TrainingScenario, get_scenarios, ofat


def test_short_strings_round_trip_and_reject_typos():
    scenarios = {*get_scenarios('essential').values(), *get_scenarios('grid').values()}
    assert len({scenario.scenario_string for scenario in scenarios}) == len(scenarios)
    assert all(TrainingScenario.from_string(s.scenario_string) == s for s in scenarios)
    text = ANCHOR.scenario_string
    assert text == 'b0:h12:tr_4rt:geo1:dyn1:lw_first:enc_mlp:hd_sh:dec_leg:z16:w64:ep50:bs8:m8:lr0.001'
    for bad in (text.replace('tr_4rt', 'tr_fourth'), text.replace(':h12', ':h012'), text.replace(':z16', ''),
                text.replace('b0:', 'b1:'), text.replace('w64:ep50', 'ep50:w64'), text + ':x1',
                text.replace('geo1', 'geo2'), text.replace('lr0.001', 'lr1e-3')):
        with pytest.raises(ValueError):
            TrainingScenario.from_string(bad)


def test_flags_parse_back_to_the_same_scenario():
    from tapestry.models.season_cv import build_parser
    extra = replace(ANCHOR, width=32, epochs=5, batch_size=4, members=4, lr=3e-4)
    for scenario in (BASELINE, *get_scenarios('essential').values(), extra):
        assert TrainingScenario.from_config(vars(build_parser().parse_args(scenario.flags()))) == scenario


def test_ofat_names_single_changes():
    assert ofat(ANCHOR, width=(32, 64, 128), heads=('state_us',)) == {
        'width_32': replace(ANCHOR, width=32), 'width_128': replace(ANCHOR, width=128),
        'heads_su': replace(ANCHOR, heads='state_us')}


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


def test_plan_run_resume_and_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls, fail = [], {43}

    def fake_cv(command, log):
        seed = int(command[command.index('--seed') + 1])
        calls.append(seed)
        if seed in fail:
            raise RuntimeError('fold failed')
        write_cv(Path(command[command.index('--output') + 1]), seed)

    monkeypatch.setattr(manager, 'execute', fake_cv)
    monkeypatch.setattr(manager, 'git_state', lambda: dict(git_commit='abc', git_dirty=False))
    folder = Path('data/experiments/demo')
    manager.main(['plan', '-e', 'demo', '--scenario', 'anchor', 'baseline', '--seeds', '42', '43'])
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

    # Extending keeps task numbers; rerunning reuses complete seeds and retries in a new attempt.
    manager.main(['plan', '-e', 'demo', '--scenario', 'state_us', 'anchor', '--seeds', '44'])
    assert [(job['task'], job['name'], job['seeds']) for job in manager.read_jobs(folder)] == [
        (0, 'anchor', [42, 43, 44]), (1, 'baseline', [42, 43]), (2, 'state_us', [44])]
    calls.clear()
    fail.clear()
    manager.main(['run', '-e', 'demo', '--task', '0'])
    assert calls == [43, 44]
    rows = {(row['name'], row['seed']): row for row in manager.collect(folder)}
    assert rows['anchor', 43]['status'] == 'complete'
    assert rows['anchor', 43]['attempt'].endswith('s43/attempt-002')
    assert manager.pending_tasks(rows.values()) == [1, 2]
