"""Export the revision explorer: every signal's reported values by Wednesday against its reference values.

Run: python -m chromantis.explorer.revisions [--dataset data/processed/panel.npz] [--output docs/data/revisions]
This reads the pinned panel only; it neither fits nor scores models.

History: until 2026-10-08 this also exported the training/held-out calendars of the
signal-finalization experiments (`reporting-triangle-*`); that training route was
deleted, so the explorer's split view is hidden when no calendars are exported.
"""
import argparse
import base64
import json
from pathlib import Path
import shutil

import numpy as np

from chromantis.dataset.build import load, context_end, LAG_ONE_COVARIATES, PANEL_DATASET
from chromantis.experiment.provenance import sha256


def packed(values):
    return base64.b64encode(np.asarray(values, dtype='<f4').tobytes()).decode('ascii')


def signals(panel):
    """(name, truth [date, location], as-of [issuance, date, location], locations); national covariates once, as US."""
    for name_key, value_key, asof_key in (
        ('target_names', 'targets', 'asof_targets'),
        ('covariate_names', 'covariates', 'asof_covariates'),
        ('covariate_national_names', 'covariates_national', 'asof_covariates_national'),
    ):
        for k, name in enumerate(panel[name_key].astype(str)):
            national = value_key == 'covariates_national'
            truth, asof = panel[value_key][..., k], panel[asof_key][..., k]
            yield name, (truth[:, None] if national else truth), (asof[..., None] if national else asof), \
                np.array(['US']) if national else panel['locations'].astype(str)


def build(dataset, output):
    panel = load(dataset)
    metadata = json.loads(str(panel['metadata']))
    dates, issues = panel['dates'].astype(str), panel['issuance_dates'].astype(str)
    ends = np.array([context_end(d) for d in issues])
    origin = np.searchsorted(dates, ends)
    origin[ends < dates[0]] = -1
    week_index = origin[:, None] - np.arange(12)[None, :]
    output.mkdir(parents=True, exist_ok=True)
    static = Path(__file__).parent / 'static'
    shutil.copy2(static / 'revisions.html', output / 'index.html')
    shutil.copy2(static / 'revisions.js', output / 'revisions.js')
    shutil.copy2(static / 'revision-comparison.html', output / 'comparison.html')
    catalog = []
    for name, truth, asof, locations in signals(panel):
        values = asof[np.arange(len(issues))[:, None], week_index.clip(0)]
        values = np.where((week_index >= 0)[..., None], values, np.nan)
        payload = dict(name=name, locations=locations.tolist(), truth=packed(truth), reports=packed(values),
                       lag=int(name in LAG_ONE_COVARIATES), cv={})
        (output / f'{name}.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
        catalog.append(dict(name=name, file=f'{name}.json', locations=locations.tolist()))
    manifest = dict(dates=dates.tolist(), issuances=issues.tolist(), origins=origin.tolist(),
                    truth_day=metadata['truth_day'], dataset_sha256=sha256(dataset),
                    experiment=Path(dataset).name, series=catalog, cv=[],
                    cutoff_policy='Wednesday panel snapshot; no additional publication-time reconstruction',
                    encoding='base64 little-endian float32; reports[issuance, T-offset 0..11, location]; truth[date, location]')
    (output / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
    print(json.dumps(dict(output=str(output), signals=len(catalog), Wednesdays=len(issues), reference_day=metadata['truth_day'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=PANEL_DATASET)
    parser.add_argument('--output', type=Path, default=Path('docs/data/revisions'))
    args = parser.parse_args()
    build(args.dataset, args.output)


if __name__ == '__main__':
    main()
