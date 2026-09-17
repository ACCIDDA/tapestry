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

All input fields, including known-final flags, are zeroed for blocked weeks.
Hidden reference finals therefore cannot bypass the nowcast during validation.
"""
from datetime import date

import numpy as np

from tapestry.model_data.finalized import season
from .season_cv import SEASONS, VALIDATION_WEEKS, VALIDATION_SPACING, VALIDATION_OFFSET


# B0's panel begins at the first available September 2023 week and its season 1 is
# therefore 48 weeks, not a full 52. B1's archive reaches back to 2023-08-05. That
# is a data-availability difference, not a season definition, so the shared fold
# calendar comes from the dataset's pinned B0 weeks (or the shared default) and
# B1 drops the four earlier weeks. Otherwise
# B1's 2023-2024 fold would both train on more data and, because the 3-in-16
# pattern counts weeks from the start of a season, hide a different set of weeks
# (measured: an offset of exactly four weeks in two of the three folds).


def season_weeks(ds):
    """The fold calendar: target weeks inside the three modelled seasons, in order."""
    return list(ds.calendar_weeks)


def hidden_weeks(ds, held_out):
    """B0's 3-in-16 hidden target weeks inside each training season of a fold.

    Uses the ordered season calendar so the pattern indexes weeks, not issuances,
    exactly as B0 indexes its panel rows.
    """
    weeks = season_weeks(ds)
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
    hidden weeks zeroed everywhere. Evaluation keeps the dataset's retrospective
    context (including supplied finals) and scores only held-out weeks.
    """
    if held_out not in SEASONS:
        raise ValueError(f'Unknown season {held_out}; expected one of {SEASONS}')
    ds = ds.model_view()
    all_weeks = sorted({str(day) for row in ds.arrays['target_dates'] for day in row})
    weeks = season_weeks(ds)
    label = {day: season(date.fromisoformat(day)) for day in weeks}
    hidden = hidden_weeks(ds, held_out)
    training_weeks = {day for day in weeks if label[day] != held_out}
    held_weeks = {day for day in weeks if label[day] == held_out}
    # A fold's fit never sees the held-out season or its own validation weeks.
    # Weeks outside the three modelled seasons are blocked too: B0's panel does
    # not contain them, so leaving them visible would give B1 extra history.
    outside = set(all_weeks) - set(weeks)
    blocked = held_weeks | hidden | outside
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
        # Evaluation keeps the retrospective conditioning state: nothing is hidden
        # at prediction time except weeks B0's panel does not have either.
        kept = _keep_labels(episode, _mask_context(episode, outside), held_weeks)
        if kept is not None:
            evaluation.append(kept)
    if not fitting or not validation or not evaluation:
        raise ValueError(f'No fitting, validation or evaluation episodes for held-out {held_out}')
    info = dict(held_out=held_out, hidden_target_weeks=sorted(hidden),
                training_weeks=len(training_weeks), inner_weeks=len(inner),
                held_out_weeks=len(held_weeks), fitting_episodes=len(fitting),
                validation_episodes=len(validation), evaluation_episodes=len(evaluation),
                # Auditable against B0's fold: the two must hold out the same calendar.
                season_weeks=len(weeks), weeks_outside_seasons=sorted(outside))
    return fitting, validation, evaluation, info
