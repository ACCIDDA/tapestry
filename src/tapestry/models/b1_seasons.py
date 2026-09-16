"""B0's leave-one-season-out protocol, applied to B1's Wednesday episodes.

B0 (`season_cv`) works on a finalized `[week, channel, value/mask, location]` panel:
a fold holds out one season, and early stopping hides three consecutive target weeks
out of every sixteen inside each training season, zeroing them in the fit's inputs,
labels and scales. B1's unit is a Wednesday issuance whose 12-week context is already
materialized, so the same rules are applied per episode:

- Fold membership and hidden weeks are decided by **target date**, using B0's
  `season()` and its `VALIDATION_WEEKS/SPACING/OFFSET` pattern, so the two models
  hold out the same calendar weeks.
- A held-out or hidden week is removed from labels **and** zeroed inside every
  fitting episode's context (`X_available -> False`), mirroring B0 zeroing its
  panel. Episodes are retained rather than dropped; B1 already models missing inputs.
- Validation episodes score only hidden weeks, as in B0.

Assumption: B1 conditions on real Wednesday vintages, so zeroing a context week
removes information that was genuinely available at issuance. This is the
protocol's cost, accepted to match B0's leakage rule exactly.
"""
from datetime import date

import numpy as np

from tapestry.model_data.finalized import season
from .season_cv import SEASONS, VALIDATION_WEEKS, VALIDATION_SPACING, VALIDATION_OFFSET


def hidden_weeks(ds, held_out):
    """B0's 3-in-16 hidden target weeks inside each training season of a fold.

    Uses the dataset's own ordered target weeks so the pattern indexes weeks, not
    issuances, exactly as B0 indexes its panel rows.
    """
    weeks = sorted({str(day) for row in ds.arrays['target_dates'] for day in row})
    labels = [season(date.fromisoformat(day)) for day in weeks]
    hidden = set()
    for label in SEASONS:
        if label == held_out:
            continue
        index = [i for i, value in enumerate(labels) if value == label]
        position = np.arange(len(index)) % VALIDATION_SPACING
        chosen = (position >= VALIDATION_OFFSET) & (position < VALIDATION_OFFSET + VALIDATION_WEEKS)
        hidden.update(weeks[i] for i, take in zip(index, chosen) if take)
    return hidden


def _mask_context(episode, blocked):
    """Zero blocked context weeks, as B0 zeroes hidden/held-out weeks in its panel."""
    x = episode['X'].copy()
    for j, day in enumerate(episode['context_dates']):
        if str(day) in blocked:
            x[j] = 0
    return x


def _keep_labels(episode, x, allowed):
    """Retain only labels whose target week is in `allowed`; drop unsupervised episodes."""
    y = episode['Y'].copy()
    for h, day in enumerate(episode['target_dates']):
        if str(day) not in allowed:
            y[h] = 0
    if not y[:, :, 1].any() or not x[:, :, 1].any():
        return None
    return {**episode, 'X': x, 'Y': y}


def fold(ds, held_out):
    """Return (fitting, validation, evaluation) episodes for one held-out season.

    Fitting and validation come from the two training seasons with held-out and
    hidden weeks zeroed everywhere. Evaluation episodes keep their real Wednesday
    context (nothing is hidden at prediction time) and score only held-out weeks.
    """
    if held_out not in SEASONS:
        raise ValueError(f'Unknown season {held_out}; expected one of {SEASONS}')
    weeks = sorted({str(day) for row in ds.arrays['target_dates'] for day in row})
    label = {day: season(date.fromisoformat(day)) for day in weeks}
    hidden = hidden_weeks(ds, held_out)
    training_weeks = {day for day in weeks if label[day] in SEASONS and label[day] != held_out}
    held_weeks = {day for day in weeks if label[day] == held_out}
    # A fold's fit never sees the held-out season or its own validation weeks.
    blocked = held_weeks | hidden
    inner = training_weeks - hidden

    fitting, validation, evaluation = [], [], []
    for episode in ds.episodes(supervised=False):
        masked = _mask_context(episode, blocked)
        kept = _keep_labels(episode, masked, inner)
        if kept is not None:
            fitting.append(kept)
        kept = _keep_labels(episode, masked, hidden)
        if kept is not None:
            validation.append(kept)
        # Evaluation keeps the unmodified Wednesday information state.
        kept = _keep_labels(episode, episode['X'], held_weeks)
        if kept is not None:
            evaluation.append(kept)
    if not fitting or not validation or not evaluation:
        raise ValueError(f'No fitting, validation or evaluation episodes for held-out {held_out}')
    info = dict(held_out=held_out, hidden_target_weeks=sorted(hidden),
                training_weeks=len(training_weeks), inner_weeks=len(inner),
                held_out_weeks=len(held_weeks), fitting_episodes=len(fitting),
                validation_episodes=len(validation), evaluation_episodes=len(evaluation))
    return fitting, validation, evaluation, info
