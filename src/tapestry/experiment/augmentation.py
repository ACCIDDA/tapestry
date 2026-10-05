"""Fold-purged reporting augmentation and real-vintage evaluation for B2 retraining."""
from pathlib import Path
import fcntl
import json

import numpy as np

from tapestry.dataset import cv
from tapestry.dataset.build import covariate_names_for
from tapestry.dataset.reporting_error import ReportingErrors
from .provenance import sha256
from .replay import cached_nowcasts, replay_episode, selected_nowcasts


def donor_corrections(panel, scenario, held_out, keep, folder, dataset_hash):
    """Cache causal residual predictions after excluding outer/inner reference dates."""
    stage = 'inner' if scenario.patience and np.array_equal(keep, cv.week_roles(panel['dates'], scenario, held_out) == 'fit') else 'full'
    path = folder / f'augmentation-nowcasts-{held_out}-{stage}.npz'
    stamp = dict(dataset_sha256=dataset_hash, keep=np.flatnonzero(keep).tolist(),
        model_sha256=sha256(Path(__file__).parents[1] / 'model' / 'finalization.py'),
        replay_sha256=sha256(Path(__file__).with_name('replay.py')))
    with path.with_suffix('.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if path.exists():
            with np.load(path) as saved:
                if json.loads(str(saved['metadata'])) != stamp:
                    raise ValueError('Augmentation correction cache changed; create a new experiment')
                return saved['predictions']
        permitted = panel['dates'][keep].astype(str)
        donor_season = max(cv.season(d) for d in permitted)
        wanted = [str(np.datetime64(d) + np.timedelta64(4, 'D')) for d in permitted if cv.season(d) == donor_season]
        predictions = selected_nowcasts(cv.masked(panel, keep), wanted)
        np.savez_compressed(path, predictions=predictions, metadata=json.dumps(stamp))
        return predictions


def prepare(panel, scenario, held_out, full, inner, seed, folder, dataset_hash):
    roles = cv.week_roles(panel['dates'], scenario, held_out)
    keep = np.isin(roles, ['fit', 'validation'])
    def bank(allowed):
        corrections = donor_corrections(panel, scenario, held_out, allowed, folder, dataset_hash) if scenario.reporting_augmentation == 'nowcast' else None
        return ReportingErrors(panel, scenario, held_out, allowed, corrections)
    outer = bank(keep)
    selection = bank(roles == 'fit') if inner else None
    if inner:
        # One fixed random validation intervention: identical at every epoch and arm.
        inner.validation = selection.batch(inner.validation, np.random.default_rng(seed + 800000))
    corrections = cached_nowcasts(panel, folder, dataset_hash) if scenario.reporting_augmentation == 'nowcast' else None
    full.score = [replay_episode(e, panel, covariate_names_for(scenario.covariate_set),
                                scenario.reporting_augmentation, corrections, 'off') for e in full.score]
    metadata = dict(full=outer.metadata, selection=selection.metadata if selection else None,
                    validation='fixed donor draw from inner-fitting-only library',
                    scoring='real held-out Wednesday vintages; final labels and frozen support unchanged')
    return outer, selection, metadata
