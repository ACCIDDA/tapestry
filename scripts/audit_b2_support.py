"""Count B2 fold availability before component-specific target filtering."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.model_data.b2 import B2Dataset
from tapestry.models.b1_seasons import SEASONS, fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/processed/build_b2.npz'))
    parser.add_argument('--output', type=Path, default=Path('docs/results/B2-screen'))
    args = parser.parse_args()
    ds = B2Dataset.load(args.dataset)
    rows = []
    for mode in ('finalized', 'wednesday'):
        for held in SEASONS:
            fitting, _, refit, evaluation, _ = fold(ds, held, input_mode=mode, allow_empty_context=True)
            masks = [np.stack([e['C'][:, :, 1].astype(bool) for e in episodes])
                     for episodes in (refit, evaluation)]
            train, evaluate = [mask.any(axis=(0, 1)) for mask in masks]
            for group, indices in (('claims', slice(0, 4)), ('wastewater', slice(4, 10)),
                                   ('kinsa', slice(10, 11))):
                rows.append(dict(input_mode=mode, held_out=held, group=group,
                                 fitting_origins=len(fitting), refit_origins=len(refit),
                                 evaluation_origins=len(evaluation),
                                 refit_pairs=int(train[indices].sum()),
                                 evaluation_pairs=int(evaluate[indices].sum()),
                                 shared_pairs=int((train[indices] & evaluate[indices]).sum())))
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / 'support_by_fold.csv', index=False)
    (args.output / 'support_manifest.json').write_text(json.dumps(dict(
        dataset=str(args.dataset), dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        definition='A pair is a covariate channel/location with any available cell over origins and history weeks. '
                   'Counts precede component-specific supervised-target filtering and are upper bounds for a component. '
                   'Shared pairs require any refit support and any evaluation support, not simultaneous observation dates.',
        covariates=ds.metadata['covariate_names']), indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
