"""Count B1 episode/input losses with explicit, aligned finalized comparators.

Run from the repo root after building the Wednesday dataset. Pass the actual B0
NPZ for a B0 comparison; its hash is recorded separately from B1's truth policy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tapestry.model_data.finalized import CHANNELS, FinalizedDataset
from tapestry.model_data.wednesday import DEFAULT_DATASET, WednesdayDataset
from tapestry.models.provenance import CALENDAR_START


def count(value):
    return int(np.count_nonzero(value))


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def audit(ds, train_end, b0=None, b0_start=None):
    ds = ds.model_view()
    b0_start = b0_start or (b0.dates[0] if b0 is not None else ds.metadata.get('calendar_start', CALENDAR_START))
    a = ds.arrays
    available = a['X_available']
    labels = np.concatenate((a['Y_recent_valid'], a['Y_future_valid']), axis=1)
    # The same pinned reference panel is repeated in overlapping output windows.
    # Reconstruct its mask by exact date, checking repeated entries agree.
    by_date = {}
    for days, masks in zip(a['target_dates'], labels):
        for day, mask in zip(days, masks):
            if day in by_date and not np.array_equal(by_date[day], mask):
                raise ValueError(f'Inconsistent pinned reference support: {day}')
            by_date[day] = mask
    missing = set(a['context_dates'].flat) - by_date.keys()
    if any(day in ds.calendar_weeks for day in missing):
        raise ValueError('Reference labels do not span the requested context; rebuild a wider issuance range')
    empty = np.zeros_like(available[0, 0])
    finalized = np.array([[by_date.get(day, empty) for day in days] for days in a['context_dates']])
    has_input = available.any((1, 2, 3))
    has_final_input = finalized.any((1, 2, 3))
    sections = {}
    for name, rows, end in (
        ('full_requested_period', np.ones(len(available), bool), None),
        ('short_run_fitting_partition', a['issuance_dates'] <= train_end, train_end),
    ):
        permitted = labels.copy()
        if end:
            permitted &= (a['target_dates'] <= end)[:, :, None, None]
        has_label = permitted.any((1, 2, 3))
        baseline = rows & has_final_input & has_label
        actual = rows & has_input & has_label
        lost = baseline & ~actual
        total_input = count(finalized[rows])
        missing_input = count(finalized[rows] & ~available[rows])
        per_channel = []
        for c, channel in enumerate(CHANNELS):
            total = count(finalized[rows, :, c])
            hidden = count(finalized[rows, :, c] & ~available[rows, :, c])
            per_channel.append(dict(channel=channel, finalized_input_cells=total,
                unavailable_wednesday_cells=hidden, percent_unavailable=percent(hidden, total),
                wednesday_only_cells=count(available[rows, :, c] & ~finalized[rows, :, c]),
                eligible_output_cells=count(permitted[baseline, :, c]),
                output_cells_lost_with_episodes=count(permitted[lost, :, c]),
                retained_episodes_without_any_focal_history=count(
                    actual & permitted[:, :, c].any((1, 2)) & ~available[:, :, c].any((1, 2)))))
        sections[name] = dict(candidate_issuances=count(rows), label_end=end,
            finalized_eligible_episodes=count(baseline), wednesday_eligible_episodes=count(actual),
            lost_episodes=count(lost), lost_episode_percent=percent(count(lost), count(baseline)),
            lost_issuance_dates=a['issuance_dates'][lost].tolist(),
            finalized_input_cells=total_input, unavailable_wednesday_cells=missing_input,
            input_cell_percent_unavailable=percent(missing_input, total_input),
            eligible_output_cells=count(permitted[baseline]),
            output_cells_lost_with_episodes=count(permitted[lost]),
            target_location_examples=count(permitted[baseline].any(1)),
            target_location_examples_lost=count(permitted[lost].any(1)),
            by_channel=per_channel)
    result = dict(truth_cutoff=ds.metadata['truth_cutoff'],
        calendar_weeks=list(ds.calendar_weeks), calendar_source=ds.metadata.get('calendar_source'),
        archive_issuances=ds.metadata.get('archive_issuances'), model_issuances=len(available),
        issuance_range=[str(a['issuance_dates'][0]), str(a['issuance_dates'][-1])],
        lookback=int(available.shape[1]), locations=len(ds.locations),
        definitions=dict(episode='One issuance containing all six targets and 52 locations.',
            input_cell='One issuance/context-week/channel/location; repeated weeks count in every window.',
            output_cell='One issuance/output-week/channel/location with a valid label; not an independent epidemiological replicate.',
            target_location_example='One issuance/channel/location with at least one valid output week.',
            finalized_comparator='Same B1 pinned labels/source policy/support, used counterfactually as context. Isolates vintage availability.',
            masking='Natural availability only, before artificial training dropout.',
            empty_history='Missing focal histories remain trainable through other channels; even all-missing local histories are retained when the overall episode has inputs.'),
        sections=sections, source_manifests=ds.metadata['source_manifests'])
    # This counterfactual is used only to measure support, never saved as model inputs.
    from tapestry.models.b1_seasons import fold
    from tapestry.models.provenance import SEASONS
    counterfactual = WednesdayDataset(dict(a, X_available=finalized), ds.metadata)
    folds = {}
    for held in SEASONS:
        real = fold(ds, held)
        final = fold(counterfactual, held)
        parts = {}
        for name, actual, reference in zip(('fitting', 'validation', 'evaluation'), real[:3], final[:3]):
            actual_dates = {e['issuance_date'] for e in actual}
            lost = [e['issuance_date'] for e in reference if e['issuance_date'] not in actual_dates]
            parts[name] = dict(finalized_eligible_episodes=len(reference), wednesday_eligible_episodes=len(actual),
                               lost_episodes=len(lost), lost_issuance_dates=lost)
        folds[held] = parts
    result['season_cv'] = folds
    from tapestry.models.b1_report import history_support
    result['history_support'] = history_support(ds)
    if b0 is not None:
        if b0.locations != ds.locations:
            raise ValueError('B0 and B1 location orders differ')
        b0_x, b0_y = [], []
        for days in a['context_dates']:
            e = b0.query(str(days[-1]), lookback=available.shape[1])
            # Recreate B0's original dataset-start padding from a wider saved panel.
            e['X'][np.array(e['context_dates']) < b0_start] = 0
            e['X'][~np.isin(e['context_dates'], ds.calendar_weeks)] = 0
            e['Y'][~np.isin(e['target_dates'], ds.calendar_weeks)] = 0
            b0_x.append(e['X'][:, :, 1].astype(bool))
            b0_y.append(e['Y'][:, :, 1].astype(bool))
        b0_x, b0_y = np.array(b0_x), np.array(b0_y)
        comparisons = {}
        for name, end in (('full_period', None), ('short_run_fitting_partition', train_end)):
            rows = a['context_dates'][:, -1] >= b0_start
            b0_valid, b1_valid = b0_y.copy(), a['Y_future_valid'].copy()
            if end:
                rows &= a['issuance_dates'] <= end
                target_ok = (a['target_dates'][:, 2:] <= end)[:, :, None, None]
                b0_valid &= target_ok
                b1_valid &= target_ok
            before = rows & b0_x.any((1, 2, 3)) & b0_valid.any((1, 2, 3))
            after = rows & has_input & b1_valid.any((1, 2, 3))
            comparisons[name] = dict(b0_episodes=count(before), b1_episodes=count(after),
                lost_episodes=count(before & ~after), gained_episodes=count(after & ~before),
                lost_percent=percent(count(before & ~after), count(before)),
                lost_issuance_dates=a['issuance_dates'][before & ~after].tolist())
        result['b0_comparison'] = dict(start=b0_start,
            interpretation='Supplied B0 NPZ, same four future horizons, model calendar and issuance dates. Source/support differences remain; this is not an isolated effect of vintage availability.',
            snapshots=[dict(dataset=p['dataset'], snapshot_id=p['snapshot_id']) for p in b0.metadata['provenance']],
            sections=comparisons)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--train-end', default='2025-05-28')
    parser.add_argument('--b0-dataset')
    parser.add_argument('--output', default='data/processed/b1-audit/training-support.json')
    args = parser.parse_args()
    ds = WednesdayDataset.load(args.dataset)
    b0 = FinalizedDataset.load(args.b0_dataset) if args.b0_dataset else None
    result = audit(ds, args.train_end, b0)
    result['dataset_sha256'] = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    if args.b0_dataset:
        result['b0_dataset_sha256'] = hashlib.sha256(Path(args.b0_dataset).read_bytes()).hexdigest()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {k: v for k, v in part.items() if k != 'by_channel'}
                      for name, part in result['sections'].items()}, indent=2))


if __name__ == '__main__':
    main()
