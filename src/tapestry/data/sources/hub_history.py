"""Complete Git vintages for canonical Hub CSV targets without row release dates.

Paths are explicit: forecasts, oracle outputs, dated backup copies, and derived
rates are not independent truth products. Earlier aliases follow preferred names.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
from datetime import datetime, timezone
from ..geography import STATE_FIPS

HISTORY_FILE = 'git-history.ndjson.gz'
HISTORY_INDEX = 'git-history.json'
HISTORY_TARGETS = {
    'hub_flusight_current': {
        'wk inc flu hosp': ('target-data/target-hospital-admissions.csv',
                           'target-data/truth-Incident Hospitalizations.csv'),
        'wk inc flu prop ed visits': ('target-data/target-ed-visits-prop.csv',
                                     'target-data/target-ed-visits.csv'),
    },
    'hub_covid_current': {'wk inc covid hosp': ('target-data/covid-hospital-admissions.csv',)},
    # This repository's canonical target history has native as_of timestamps.
    'hub_rsv_current': {},
    'hub_flusight_legacy': {'Incident Hospitalizations': ('data-truth/truth-Incident Hospitalizations.csv',)},
    'hub_covid_legacy': {f'{kind} {measure}': (f'data-truth/truth-{kind} {measure}.csv',)
                         for kind in ('Incident', 'Cumulative')
                         for measure in ('Cases', 'Deaths', 'Hospitalizations')},
}


def write_history(mirror, spec, tip, snapshot):
    """Export full file states on first-parent changes through a pinned commit.

Commit time is a proxy for repository publication, not provider release time.
Same-time commits resolve to the last first-parent state. Backdated children
cannot become available before their parents' target-file states.
"""
    targets = HISTORY_TARGETS.get(spec.key, {})
    paths = tuple(path for alternatives in targets.values() for path in alternatives)
    commits = mirror.run('log', '--first-parent', '--reverse', '--format=%H %cI', tip, '--', *paths).splitlines() if paths else []
    plans = {}
    previous_time = ''
    for line in commits:
        commit, stamp = line.split(' ', 1)
        release = datetime.fromisoformat(stamp).astimezone(timezone.utc).isoformat()
        release = max(release, previous_time)
        plans[release] = commit
        previous_time = release
    value_column = 'observation' if spec.key.endswith('_current') else 'value'
    releases, cached, total = [], {}, 0
    best_rank = {target: len(alternatives) - 1 for target, alternatives in targets.items()}
    output = snapshot.path(HISTORY_FILE, media_type='application/gzip')
    with gzip.open(output, 'wt', encoding='utf-8') as stream:
        previous_state = None
        for release, commit in plans.items():
            entries = mirror._tree_entries(commit, paths)
            blobs = {path: sha for _, kind, sha, path in entries if kind == 'blob'}
            chosen = {}
            for target, alternatives in targets.items():
                # Once a successor appears, an old alias cannot resurrect it
                # after removal (both filenames sometimes coexist in the repo).
                chosen[target] = next((path for path in alternatives[:best_rank[target] + 1] if path in blobs), None)
                if chosen[target]:
                    best_rank[target] = alternatives.index(chosen[target])
            state = tuple((target, path, blobs[path]) for target, path in chosen.items() if path)
            if state == previous_state:
                continue
            previous_state = state
            count, files = 0, []
            for target, path, sha in state:
                if cached.get(path, (None,))[0] != sha:
                    content = mirror.read_file(commit, path)
                    if content.startswith(b'version https://git-lfs.github.com/spec/v1'):
                        raise ValueError(f'Git LFS payload missing for {spec.key}/{path} at {commit}')
                    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
                    columns = set(reader.fieldnames or ())
                    if columns & {'as_of', 'report_time', 'release_date'}:
                        raise ValueError(f'{path} has native release dates; configure it as native vintages, not Git fallback')
                    if not {'location', 'value'} <= columns or not columns.intersection({'date', 'target_end_date'}):
                        raise ValueError(f'Unsupported historical target schema in {path}: {sorted(columns)}')
                    rows = []
                    for row in reader:
                        # Canonical scope is native states/DC/US, never county aggregation.
                        loc = row['location']
                        if loc != 'US' and loc not in STATE_FIPS:
                            continue
                        day = row.get('date') or row['target_end_date']
                        if row.get('date') and row.get('target_end_date') and row['date'] != row['target_end_date']:
                            raise ValueError(f'Conflicting event dates in {path} at {commit}')
                        rows.append(dict(row, date=day, target=target, **{value_column: row['value']}))
                    cached[path] = sha, rows
                rows = cached[path][1]
                files.append(dict(path=path, blob=sha, target=target, rows=len(rows)))
                for row in rows:
                    record = dict(row, _git_release=release, _git_commit=commit, _git_path=path)
                    stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                    count += 1
            total += count
            # Empty releases are meaningful: they remove a whole former snapshot.
            releases.append(dict(release_time=release, commit=commit, files=files, rows=count))
    index = dict(version=1, tip=tip, repository=spec.source_url, rows=total, releases=releases,
                 availability='First-parent committer timestamp in UTC; conservative nondecreasing target-state times. Proxy for repository publication, not provider release.',
                 selection='Complete native state/DC/US target-file snapshots; preferred filenames replace aliases. Native as_of files remain separate and authoritative where covered.')
    snapshot.write_json(HISTORY_INDEX, index)
    print(f'{spec.key}: {len(releases)} Git target releases, {total:,} rows', flush=True)
    return index


def backfill_history(repository, spec):
    """Add history to a new immutable acquisition, preserving the saved Git tip."""
    from .hubverse import HubMirror
    import shutil
    previous = repository.latest(spec.key)
    source = repository.snapshot_path(previous)
    tip = previous.source_state['commit']
    mirror = HubMirror(repository.mirrors_dir / f'{spec.key}.git')
    with repository.begin_snapshot(spec) as snapshot:
        for file in previous.files:
            if file.path not in {HISTORY_FILE, HISTORY_INDEX}:
                shutil.copyfile(source / file.path, snapshot.path(file.path, rows=file.rows, media_type=file.media_type))
        index = write_history(mirror, spec, tip, snapshot)
        return snapshot.commit(selector=dict(previous.selector, git_history=True),
            source_state=dict(previous.source_state, git_history_releases=len(index['releases'])))
