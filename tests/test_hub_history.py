"""Scientific invariants for Git-derived release dates and whole-file vintages."""
import os
import subprocess

from tapestry.data.catalog import get_spec
from tapestry.data.repository import RawDataRepository
from tapestry.data.selection import SelectedData
from tapestry.data.sources.hub_history import write_history, HISTORY_FILE
from tapestry.data.sources.hubverse import HubMirror
from tapestry.explorer.index import ExplorerIndex
import numpy as np
import pandas as pd

from tapestry.dataset.extract import Revisions, resolve, revisions


def test_git_committer_cutoff_revisions_and_whole_snapshot_deletion(tmp_path):
    repo = tmp_path / 'git'
    repo.mkdir()
    def git(*args, stamp=None):
        env = dict(os.environ, GIT_AUTHOR_DATE='2023-01-01T12:00:00Z',
                   GIT_COMMITTER_DATE=stamp or '2023-11-21T12:00:00Z')
        return subprocess.check_output(['git', '-C', str(repo), *args], env=env, stderr=subprocess.DEVNULL).decode().strip()
    git('init', '-b', 'main')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Test')
    target = repo / 'target-data' / 'truth-Incident Hospitalizations.csv'
    target.parent.mkdir()
    target.write_text('date,location,value\n2023-11-04,37,10\n2023-11-11,37,20\n')
    git('add', '.'); git('commit', '-m', 'First publication')
    # A new commit without a target change still has the previous target state.
    git('commit', '--allow-empty', '-m', 'No target changes', stamp='2023-11-22T12:00:00Z')
    # The obsolete alias remains on disk; deleting its successor must not revive it.
    target = repo / 'target-data' / 'target-hospital-admissions.csv'
    target.write_text('target_end_date,location,value\n2023-11-11,37,30\n')
    git('add', '.'); git('commit', '-m', 'Revise and retract', stamp='2023-11-23T12:00:00Z')
    git('rm', str(target)); git('commit', '-m', 'Remove target', stamp='2023-11-24T12:00:00Z')
    spec = get_spec('hub_flusight_current')
    raw = RawDataRepository(tmp_path / 'data')
    raw.initialize({spec.key: spec})
    mirror = HubMirror(repo / '.git')
    tip = mirror.resolve('main')
    with raw.begin_snapshot(spec) as snapshot:
        history = write_history(mirror, spec, tip, snapshot)
        snapshot.commit(selector={'commit': tip}, source_state={'commit': tip, 'commit_time': mirror.commit_time(tip)})
    assert [r['rows'] for r in history['releases']] == [2, 1, 0]
    records = list(SelectedData(raw.root).iter_records(available_by='2023-11-22'))
    assert len(records) == 2
    assert all(r.available_at.startswith('2023-11-21') for r in records)
    assert list(SelectedData(raw.root).iter_records(available_by='2023-11-20')) == []
    archive = revisions('nhsn_flu_admissions', raw.root)
    dates = ('2023-11-04', '2023-11-11')
    at = lambda archive, day: resolve(archive, day, dates, ('NC',))[:, 0]
    np.testing.assert_array_equal(at(archive, '2023-11-22'), [10, 20])
    np.testing.assert_array_equal(at(archive, '2023-11-23'), [np.nan, 30])
    assert np.isnan(at(archive, '2023-11-24')).all()
    # Native as_of snapshots remain authoritative where they establish coverage;
    # Git must not resurrect native omissions or nulls, nor override native values.
    native = pd.DataFrame(dict(tier=['hub'], release=np.array(['2023-11-22'], 'datetime64[ns]'), day=['2023-11-04'],
                               location=['NC'], value=[11.]))
    rows = pd.concat([archive.rows, native], ignore_index=True).sort_values(['day', 'location', 'release'])
    np.testing.assert_array_equal(at(Revisions(rows, archive.git_releases), '2023-11-23'), [11, np.nan])
    index = ExplorerIndex(raw.root)
    index.build(progress=lambda _: None)
    with index.connect() as db:
        sid = db.execute('SELECT id FROM series WHERE source_path=?', (HISTORY_FILE,)).fetchone()[0]
    def points(day):
        return index.data('NC', [sid], as_of=day)['series'][0]['points']
    assert points('2023-11-20') == []
    assert [p[1] for p in points('2023-11-22')] == [10, 20]
    assert [p[1] for p in points('2023-11-23')] == [30]
    assert points('2023-11-24') == []
