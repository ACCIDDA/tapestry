"""B1: pinned Wednesday snapshots and reference finals, materialized once as arrays."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

from tapestry.data.geography import STATE_NAMES, observation_geography
from tapestry.data.selection import SelectedData, describe, source_signal, _timestamp
from .finalized import CHANNELS, saturday, season
from tapestry.models.provenance import CALENDAR_START, SEASONS

SOURCES = ('hub_flusight_current', 'hub_covid_current', 'hub_rsv_current', 'delphi_nhsn', 'delphi_nssp')
ORIGINS = ('totalconfflunewadm', 'totalconfc19newadm', 'totalconfrsvnewadm',
           'percent_visits_influenza', 'percent_visits_covid', 'percent_visits_rsv')
SIGNALS = ('confirmed_admissions_flu_ew', 'confirmed_admissions_covid_ew', 'confirmed_admissions_rsv_ew',
           'pct_ed_visits_influenza', 'pct_ed_visits_covid', 'pct_ed_visits_rsv')
# These are requested week STARTS converted explicitly, not source date heuristics.
START = '2023-08-05'
ARCHIVE_START = (date.fromisoformat(START) + timedelta(days=4)).isoformat()
DEFAULT_DATASET = 'data/processed/build_b1_wednesday_calendar.npz'
RSV_ED_START = '2023-11-11'
OFFSETS = (-1, 0, 1, 2, 3, 4)  # relative to preceding Saturday
REASONS = ('available', 'no_archive_coverage', 'hub_missing_or_retracted', 'source_null_or_invalid', 'outside_support')


def release_time(value):
    """Naive timestamps are UTC; compare complete Wednesday calendar days in UTC."""
    return _timestamp(str(value)).isoformat()


def cutoff_time(day):
    return release_time(str(day) + 'T23:59:59.999999+00:00')


def output_dates(issuance, lookback=12):
    day = date.fromisoformat(issuance)
    if day.weekday() != 2:
        raise ValueError('B1 issuance must be Wednesday')
    if lookback < 2:
        raise ValueError('B1 context needs at least two weeks')
    end = day - timedelta(days=4)
    return (tuple((end - timedelta(weeks=i)).isoformat() for i in reversed(range(lookback))),
            tuple((end + timedelta(weeks=i)).isoformat() for i in OFFSETS))


class VintageArchive:
    """Resolve full Hub releases and per-observation Delphi revisions separately.

    Coverage is conservative: once a channel/location appears, its weekly period
    from the earliest event onward is covered, even if later snapshots omit rows.
    Interior holes and trailing/retracted observations never trigger fallback.
    Only releases eligible at the cutoff establish that coverage.
    """
    def __init__(self):
        self.hub = {}
        self.delphi = {}
        self.provenance = [dict(source='none', release=None, snapshot_id=None)]
        self._provenance_ids = {}
        self.manifests = []
        self.audit = []

    def provenance_id(self, **record):
        record['fallback_reason'] = ('absent_hub_historical_coverage'
                                     if record['source'].startswith('delphi_') else None)
        key = json.dumps(record, sort_keys=True)
        if key not in self._provenance_ids:
            self._provenance_ids[key] = len(self.provenance)
            self.provenance.append(record)
        return self._provenance_ids[key]

    def add(self, source, release, day, channel, location, value, snapshot_id='', source_path=''):
        release = release_time(release)
        saturday(day)  # Both weekly archives use week ends; never shift Sundays.
        pid = self.provenance_id(source=source, release=release, snapshot_id=snapshot_id, source_path=source_path)
        cell = (day, channel, location)
        if source.startswith('hub_'):
            rows = self.hub.setdefault(source, {}).setdefault(release, {})
            key = cell
        else:
            rows = self.delphi.setdefault(cell, {})
            key = release
        # Quarantine same-release conflicts, including null versus a value.
        if key in rows and rows[key][0] != value:
            value = None
        rows[key] = (value, pid)

    def resolve(self, cutoff):
        cutoff = cutoff_time(cutoff)
        hub, coverage, latest = {}, {}, {}
        for source, releases in self.hub.items():
            eligible = sorted(r for r in releases if r <= cutoff)
            if not eligible:
                continue
            selected = eligible[-1]
            latest[source] = self.provenance_id(source=source, release=selected,
                snapshot_id=self.provenance[next(iter(releases[selected].values()))[1]]['snapshot_id'],
                source_path='full snapshot')
            for release in eligible:
                for day, c, loc in releases[release]:
                    key = (c, loc)
                    coverage[key] = min(day, coverage.get(key, day))
            hub.update(releases[selected])
        delphi = {}
        for cell, revisions in self.delphi.items():
            eligible = [r for r in revisions if r <= cutoff]
            if eligible:
                delphi[cell] = revisions[max(eligible)]
        return hub, coverage, latest, delphi

    def panel(self, dates, locations, state):
        hub, coverage, latest, delphi = state
        values = np.zeros((len(dates), 6, len(locations)), np.float32)
        available = np.zeros_like(values, dtype=bool)
        provenance = np.zeros_like(values, dtype=np.int32)
        reasons = np.full_like(values, 1, dtype=np.uint8)
        for t, day in enumerate(dates):
            for c in range(6):
                for l, loc in enumerate(locations):
                    if day < START or (c == 5 and day < RSV_ED_START):
                        reasons[t, c, l] = 4
                        continue
                    cell = day, c, loc
                    covered = (c, loc) in coverage and day >= coverage[c, loc]
                    if covered:
                        source = SOURCES[c % 3]
                        value, pid = hub.get(cell, (None, latest.get(source, 0)))
                        reason = 2 if value is None else 0
                    else:
                        value, pid = delphi.get(cell, (None, 0))
                        reason = (3 if pid else 1) if value is None else 0
                    provenance[t, c, l] = pid
                    reasons[t, c, l] = reason
                    if value is not None:
                        values[t, c, l], available[t, c, l] = value, True
        return values, available, provenance, reasons


def read_archive(data_root):
    archive = VintageArchive()
    selected = SelectedData(data_root)
    seen_manifests = set()
    for key in SOURCES:
        for table in selected.selected_tables(dataset_key=key):
            signal = source_signal(table.source_path)
            if key.startswith('delphi_') and signal and signal not in SIGNALS:
                continue
            manifest = Path(data_root) / 'raw' / key / 'snapshots' / table.snapshot_id / 'manifest.json'
            if str(manifest) not in seen_manifests:
                archive.manifests.append(dict(source=key, snapshot_id=table.snapshot_id,
                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
                seen_manifests.add(str(manifest))
            identities = {}
            for row in table.iter_rows():
                if key.startswith('delphi_'):
                    row_signal = source_signal(table.source_path, row)
                    if row_signal not in SIGNALS:
                        continue
                    # The acquisition requests native (unfilled) observations only.
                    if str(row.get('fill_method', 'none')).lower() not in ('source', 'none', '', 'null'):
                        continue
                    c = SIGNALS.index(row_signal)
                else:
                    target = str(row.get('target', row.get('target_variable', '')))
                    if target not in identities:
                        origin = describe(key, 'observation', table.source_path, row)['origin_column']
                        identities[target] = ORIGINS.index(origin) if origin in ORIGINS else None
                    c = identities[target]
                    if c is None:
                        continue
                loc = observation_geography(row, table.geographic_resolutions)
                if loc not in STATE_NAMES and loc != 'US':
                    continue
                release = row.get(table.vintage_column)
                if not release:
                    continue  # retrieval time cannot establish historical availability
                day = saturday(row.get(table.event_date_column) or row.get('target_end_date') or row.get('date')).isoformat()
                if day < START:
                    continue
                raw = row.get('observation' if key.startswith('hub_') else 'value')
                try:
                    value = float(raw)
                    upper = (1 if key.startswith('hub_') else 100) if c >= 3 else float('inf')
                    if not np.isfinite(value) or not 0 <= value <= upper:
                        value = None
                except (TypeError, ValueError):
                    value = None
                if c >= 3 and key.startswith('delphi_') and value is not None:
                    value /= 100
                archive.add(key, release, day, c, loc, value, table.snapshot_id, table.source_path)
    archive.audit = selected.audit
    return archive


@dataclass
class WednesdayDataset:
    arrays: dict
    metadata: dict

    @property
    def locations(self):
        return tuple(self.arrays['locations'].tolist())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            np.savez_compressed(stream, **self.arrays, metadata=json.dumps(self.metadata))
        path.with_suffix('.json').write_text(json.dumps(self.metadata, indent=2) + '\n')

    @classmethod
    def load(cls, path, *, archive=False):
        with np.load(path, allow_pickle=False) as data:
            result = cls({k: data[k] for k in data.files if k != 'metadata'}, json.loads(str(data['metadata'])))
        return result if archive else result.model_view()

    @property
    def calendar_weeks(self):
        candidates = self.metadata.get('calendar_weeks', sorted(set(self.arrays['target_dates'].flat)))
        start = self.metadata.get('calendar_start', CALENDAR_START)
        return tuple(str(d) for d in candidates if str(d) >= start and season(date.fromisoformat(str(d))) in SEASONS)

    def model_view(self):
        """One calendar filter for folds, diagnostics and prediction; archive stays intact."""
        if self.metadata.get('view') == 'model_calendar':
            return self
        a, weeks = self.arrays, self.calendar_weeks
        context = np.isin(a['context_dates'], weeks)
        targets = np.isin(a['target_dates'], weeks)
        rows = context.any(1) & targets.any(1)
        arrays = {key: value[rows].copy() if key != 'locations' else value.copy() for key, value in a.items()}
        context, targets = context[rows], targets[rows]
        reasons = list(self.metadata.get('reasons', REASONS))
        if 'outside_model_calendar' not in reasons:
            reasons.append('outside_model_calendar')
        reason = reasons.index('outside_model_calendar')
        for key in ('X_values', 'X_available', 'X_provenance'):
            arrays[key][~context] = 0
        arrays['X_reason'][~context] = reason
        for prefix, permitted in (('Y_recent', targets[:, :2]), ('Y_future', targets[:, 2:])):
            arrays[prefix][~permitted] = 0
            arrays[prefix + '_valid'][~permitted] = False
        arrays['Y_provenance'][~targets] = 0
        arrays['Y_reason'][~targets] = reason
        metadata = dict(self.metadata, view='model_calendar', calendar_weeks=list(weeks), reasons=reasons,
                        archive_issuances=len(a['issuance_dates']), model_issuances=int(rows.sum()))
        return WednesdayDataset(arrays, metadata)

    def episodes(self, *, start=None, end=None, target_start=None, target_end=None, supervised=True,
                 min_availability=0.):
        """`min_availability` drops episodes whose mean X_available falls below it. The
        archive begins before several channels exist, so early Wednesdays carry as little
        as one of six channels; keeping them trains the model on near-empty inputs."""
        a = self.model_view().arrays
        for i, issuance in enumerate(a['issuance_dates']):
            if (start and issuance < start) or (end and issuance > end):
                continue
            if min_availability and a['X_available'][i].mean() < min_availability:
                continue
            y = np.concatenate((a['Y_recent'][i], a['Y_future'][i]))
            valid = np.concatenate((a['Y_recent_valid'][i], a['Y_future_valid'][i])).copy()
            for h, day in enumerate(a['target_dates'][i]):
                if (target_start and day < target_start) or (target_end and day > target_end):
                    valid[h] = False
            if not a['X_available'][i].any() or (supervised and not valid.any()):
                continue
            yield dict(X=np.stack((a['X_values'][i], a['X_available'][i]), axis=2),
                       Y=np.stack((np.where(valid, y, 0), valid), axis=2),
                       issuance_date=str(issuance), context_dates=tuple(a['context_dates'][i]),
                       target_dates=tuple(a['target_dates'][i]), locations=self.locations,
                       season=season(date.fromisoformat(a['context_dates'][i, -1])), index=i)


def build_wednesday(data_root='data', *, start=ARCHIVE_START, end, truth_cutoff, lookback=12,
                    locations=None, archive=None, calendar_start=None, calendar_dataset=None):
    """Explicit issuance range and pinned truth cutoff; allow incomplete leading context."""
    archive = read_archive(data_root) if archive is None else archive
    if calendar_start is not None and calendar_dataset is not None:
        raise ValueError('Choose --calendar-start or --calendar-dataset, not both')
    calendar = dict(calendar_start=calendar_start or CALENDAR_START)
    saturday(calendar['calendar_start'])
    if calendar_dataset is not None:
        from .finalized import FinalizedDataset
        reference = FinalizedDataset.load(calendar_dataset)
        calendar = dict(calendar_start=reference.dates[0], calendar_weeks=list(reference.dates),
                        calendar_source=dict(path=str(calendar_dataset),
                            sha256=hashlib.sha256(Path(calendar_dataset).read_bytes()).hexdigest()))
    locations = tuple(locations or (*sorted(STATE_NAMES), 'US'))
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    output_dates(start, lookback)
    output_dates(end, lookback)
    if first > last or date.fromisoformat(truth_cutoff) < last:
        raise ValueError('Require start <= end <= truth cutoff')
    truth_state = archive.resolve(truth_cutoff)
    rows = []
    for offset in range(0, (last - first).days + 1, 7):
        issuance = (first + timedelta(days=offset)).isoformat()
        context, targets = output_dates(issuance, lookback)
        x, a, xp, xr = archive.panel(context, locations, archive.resolve(issuance))
        y, valid, yp, yr = archive.panel(targets, locations, truth_state)
        rows.append(dict(issuance_dates=issuance, context_dates=context, target_dates=targets,
            X_values=x, X_available=a, X_provenance=xp, X_reason=xr,
            Y_recent=y[:2], Y_recent_valid=valid[:2], Y_future=y[2:], Y_future_valid=valid[2:],
            Y_provenance=yp, Y_reason=yr))
    arrays = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    arrays['locations'] = np.asarray(locations)
    has_inputs = arrays['X_available'].any(axis=(1, 2, 3))
    has_labels = (arrays['Y_recent_valid'].any(axis=(1, 2, 3))
                  | arrays['Y_future_valid'].any(axis=(1, 2, 3)))
    if not (has_inputs & has_labels).any():
        raise ValueError(
            f'No usable B1 episodes for {start} through {end} '
            f'(truth cutoff {truth_cutoff}): {int(has_inputs.sum())}/{len(rows)} '
            f'episodes have Wednesday inputs and {int(has_labels.sum())}/{len(rows)} '
            f'have reference labels, with no usable overlap. '
            f'Check data root {str(data_root)!r}, source snapshots, dates and locations. '
            f'Acquire the required Hub/Delphi archives or copy an existing precomputed '
            f'B1 dataset. No output files were written.')
    metadata = dict(version=2, model='B1', **calendar, history_mode='strict_wednesday',
        channels=CHANNELS, units=['admissions'] * 3 + ['proportion'] * 3,
        lookback=lookback, truth_cutoff=truth_cutoff, start=start, end=end,
        offsets_from_preceding_saturday=OFFSETS, hub_offsets=[-2, -1, 0, 1, 2, 3],
        cutoff_policy='End of Wednesday UTC; naive timestamps treated as UTC. No intraday deadline claim.',
        source_date_policy='Native Hub event dates and Delphi weekly reference_time are Saturday week ends; reject other weekdays.',
        support_start=START, rsv_ed_start=RSV_ED_START,
        truth_policy='Reference finals: latest eligible Hub full snapshot; Delphi latest revision only outside Hub coverage.',
        coverage_policy='Per channel/location earliest event in any eligible Hub release onward; holes and retractions remain missing.',
        fallback_policy='A Delphi provenance entry means absent Hub historical coverage; explicit Hub missing cells never fall back.',
        reasons=REASONS, provenance=archive.provenance, source_manifests=archive.manifests, selection_audit=archive.audit)
    dataset = WednesdayDataset(arrays, metadata)
    if not any(dataset.episodes()):
        raise ValueError('No usable B1 episodes inside the model calendar; check calendar and archive dates. No output files were written.')
    return dataset


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--start', default=ARCHIVE_START, help='First archived Wednesday; distinct from the model calendar')
    parser.add_argument('--end', required=True, help='Last Wednesday issuance')
    parser.add_argument('--truth-cutoff', required=True)
    parser.add_argument('--lookback', type=int, default=12)
    parser.add_argument('--calendar-start', help='First modelled Saturday; defaults to the shared B0/B1 start')
    parser.add_argument('--calendar-dataset', help='Use the exact calendar of this B0 NPZ, recording its hash')
    parser.add_argument('--output', default=DEFAULT_DATASET)
    args = parser.parse_args(argv)
    try:
        dataset = build_wednesday(args.data_root, start=args.start, end=args.end,
            truth_cutoff=args.truth_cutoff, lookback=args.lookback,
            calendar_start=args.calendar_start, calendar_dataset=args.calendar_dataset)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    dataset.save(args.output)
    view = dataset.model_view()
    print(json.dumps(dict(output=args.output, archive_shape=list(dataset.arrays['X_values'].shape),
        model_shape=list(view.arrays['X_values'].shape), calendar_start=view.calendar_weeks[0],
        usable_episodes=sum(1 for _ in dataset.episodes()),
        observed_by_channel=view.arrays['X_available'].sum((0, 1, 3)).tolist())))


if __name__ == '__main__':
    main()
