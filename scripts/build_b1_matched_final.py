"""Matched-final control: the B1 dataset with recent vintages replaced by their finals.

The ladder measures what B1's modelling additions buy. A separate question is what
the Wednesday *inputs* cost, and B0's historical scores cannot answer it: B0 has
different training labels and different usable support, so any difference mixes the
input change with those.

This control changes exactly one thing. It keeps B1's episodes, labels, availability
mask, folds and scoring support bit-for-bit, and replaces only the *values* of recent
cells that carry a historical Wednesday version with the pinned reference final for
the same cell. Nothing is added or removed, so a paired run against `B1-fromB0-refit`
isolates the vintage effect.

Why this needs no archive rebuild: `context_dates[-2:]` equals `target_dates[:2]` for
every episode, so `Y_recent` already holds the pinned final for precisely the cells
that carry vintages. The substitution is therefore exact and verifiable in-place,
rather than a second pass over the source archive that could drift.

Cells whose final is unavailable (`Y_recent_valid` false) keep their vintage value and
are reported, never silently invented.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tapestry.model_data.wednesday import WednesdayDataset


def matched_final_arrays(arrays):
    """Replace recent Wednesday-version values with their pinned finals.

    Returns (new arrays, audit). `X_available` and `X_final` are deliberately left
    untouched: flipping availability would change episode support and stop the
    control being paired.
    """
    out = {key: value.copy() for key, value in arrays.items()}
    recent = np.s_[:, -2:]
    available, final = arrays['X_available'][recent], arrays['X_final'][recent]
    truth, truth_valid = arrays['Y_recent'], arrays['Y_recent_valid']

    vintage = available & ~final
    replaceable = vintage & truth_valid
    unresolved = vintage & ~truth_valid

    values = out['X_values'][recent]
    before = values[replaceable].copy()
    values[replaceable] = truth[replaceable]
    out['X_values'][:, -2:] = values

    # The substituted cells are now reference finals; say so, so a model that reads
    # the finality flag sees a consistent world rather than a final labelled vintage.
    flags = out['X_final'][recent]
    flags[replaceable] = True
    out['X_final'][:, -2:] = flags

    delta = truth[replaceable] - before
    audit = dict(
        recent_cells=int(available.size),
        vintage_cells=int(vintage.sum()),
        replaced_cells=int(replaceable.sum()),
        unresolved_vintage_cells=int(unresolved.sum()),
        unchanged_by_replacement=int((delta == 0).sum()),
        under_reported_cells=int((delta > 0).sum()),
        over_reported_cells=int((delta < 0).sum()),
        mean_absolute_revision=float(np.abs(delta).mean()) if delta.size else 0.,
    )
    return out, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='data/processed/build_b1_wednesday_calendar.npz')
    parser.add_argument('--output', default='data/processed/build_b1_matched_final.npz')
    args = parser.parse_args(argv)

    source = Path(args.source)
    dataset = WednesdayDataset.load(source, archive=True)
    arrays, audit = matched_final_arrays(dataset.arrays)

    # Everything except recent input values must be untouched; assert rather than trust.
    for key, value in dataset.arrays.items():
        if key == 'X_values':
            assert np.array_equal(value[:, :-2], arrays[key][:, :-2]), 'older context changed'
        elif key == 'X_final':
            assert np.array_equal(value[:, :-2], arrays[key][:, :-2]), 'older finality changed'
        else:
            assert np.array_equal(value, arrays[key]), f'{key} changed; the control must stay paired'

    metadata = dict(dataset.metadata)
    metadata.update(
        control='matched_final_recent_inputs',
        control_source=dict(path=str(source),
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        control_policy=('Recent Wednesday versions replaced by the pinned reference final for the '
                        'same cell; availability, episodes, labels and support unchanged, so runs '
                        'pair one-to-one with the Wednesday dataset. Vintage cells with no '
                        'available final keep their version and are counted in control_audit.'),
        control_audit=audit)
    WednesdayDataset(arrays, metadata).save(args.output)

    digest = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    print(json.dumps(dict(output=args.output, sha256=digest, **audit), indent=2))


if __name__ == '__main__':
    main()
