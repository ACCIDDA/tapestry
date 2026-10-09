"""Scientific equivalence of compact historical availability and final values."""
from datetime import date, timedelta
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from chromantis.explorer.export import compact_history
from chromantis.explorer.index import ExplorerIndex


def test_compaction_preserves_asof_values_retractions_and_final(tmp_path):
    connection = sqlite3.connect(':memory:')
    connection.executescript('''
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE series (id INTEGER, dataset_key TEXT, source_path TEXT,
                             full_snapshots INTEGER, snapshot_id TEXT);
        CREATE TABLE releases (dataset_key TEXT, source_path TEXT, vintage TEXT);
        INSERT INTO series VALUES (1, 'claims', 'claims.csv', 0, 'source');
        INSERT INTO series VALUES (2, 'hub', 'snapshots.csv', 1, 'source');
    ''')
    rows = []
    def add(sid, event, release, value, samples=1):
        rows.append(dict(series_id=sid, state='NC', event_date=event,
                         release_time=release, value=value, samples=samples,
                         source_snapshot='source'))
    # Duplicate contributions must be averaged, not arbitrarily selected.
    add(1, '2026-06-01', '2026-09-15T12:00:00', 3.)
    add(1, '2026-06-01', '2026-09-16T12:00:00', 10.)
    add(1, '2026-06-01', '2026-09-16T12:00:00', 20.)
    add(1, '2026-06-01', '2026-09-18T12:00:00', 30.)
    add(1, '2026-06-01', '2026-09-19T12:00:00', None, 0)
    add(1, '2026-06-01', '2026-09-20T12:00:00', 40.)
    add(1, '2026-06-01', '2026-09-21T12:00:00', 50.123456789)
    # An unchanged early release must carry forward, without copying every week.
    for release in ('2026-09-15', '2026-09-16', '2026-09-18', '2026-09-19', '2026-09-21'):
        add(1, '2026-06-02', release, 7.)
    # A whole-snapshot omission removes the old observation, including when
    # intermediate snapshots themselves are dropped.
    for release, events in [('2026-09-15', ['2026-06-01']),
                            ('2026-09-16', ['2026-06-02']),
                            ('2026-09-17', ['2026-06-01']),
                            ('2026-09-19', ['2026-06-02']),
                            ('2026-09-20', ['2026-06-03'])]:
        connection.execute('INSERT INTO releases VALUES (?, ?, ?)', ('hub', 'snapshots.csv', release))
        for event in events:
            add(2, event, release, 2.)
    schema = pa.schema([('series_id', pa.int64()), ('state', pa.string()),
                        ('event_date', pa.string()), ('release_time', pa.string()),
                        ('value', pa.float64()), ('samples', pa.int64()),
                        ('source_snapshot', pa.string())], metadata={b'build_id': b'check'})
    path = tmp_path / 'revisions.parquet'
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    compact_history(connection, path, progress=lambda _: None)
    compact = pq.read_table(path).to_pylist()
    assert len(compact) < len(rows)
    assert pq.read_schema(path).metadata == schema.metadata

    def resolve(records, sid, cutoff):
        eligible = [r for r in records if r['series_id'] == sid and
                    (cutoff is None or r['release_time'] <= cutoff + 'T23:59:59.999999')]
        if sid == 2:
            vintages = [r[0] for r in connection.execute('SELECT vintage FROM releases')
                        if cutoff is None or r[0] <= cutoff + 'T23:59:59.999999']
            latest = max(vintages, default=None)
            eligible = [r for r in eligible if r['release_time'] == latest]
        grouped = {}
        for r in eligible:
            key = (r['event_date'], r['release_time'])
            total, n = grouped.get(key, (0., 0))
            grouped[key] = (total + (r['value'] or 0.), n + r['samples'])
        by_event = {}
        for (event, release), (total, n) in sorted(grouped.items()):
            by_event[event] = total / n if n else None
        return {event: value for event, value in by_event.items() if value is not None}

    days = [date(2026, 9, 12) + timedelta(days=i) for i in range(22)]
    for sid in (1, 2):
        for cutoff in [d.isoformat() for d in days if d.weekday() in (2, 5)] + [None]:
            assert resolve(compact, sid, cutoff) == resolve(rows, sid, cutoff)
    assert resolve(compact, 1, '2026-09-19') == {'2026-06-02': 7.}
    assert resolve(compact, 1, None)['2026-06-01'] == 50.123456789
    assert ExplorerIndex._cutoff('2026-09-18')[0] == '2026-09-16'
    assert ExplorerIndex._cutoff('2026-09-20')[0] == '2026-09-19'
    assert ExplorerIndex._cutoff('latest') == (None, None)
