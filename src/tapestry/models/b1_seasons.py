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


def _origin_in(episode, allowed):
    """B0 selects origins by calendar membership of the context-ending Saturday.

    B1's episode is a Wednesday issuance, so the fold membership of an origin is
    decided by `context_dates[-1]`, never by the issuance date's own season.
    """
    return str(episode['context_dates'][-1]) in allowed


def _keep_labels(episode, x, allowed, direct=True):
    """Retain only labels whose target week is in `allowed`; drop unsupervised episodes.

    `direct` restricts the "is anything still supervised?" test to the four forecast
    slots, because the direct task never scores the two recent ones. Without it, an
    origin whose forecasts all fall outside the partition survives on recent labels
    alone, which B0's four-horizon panel cannot do.
    """
    y = episode['Y'].copy()
    for h, day in enumerate(episode['target_dates']):
        if str(day) not in allowed:
            y[h] = 0
    supervised = y[2:] if direct else y
    if not supervised[:, :, 1].any() or not x[:, :, 1].any():
        return None
    return {**episode, 'X': x, 'Y': y}


def fold(ds, held_out, min_availability=0., direct=True):
    """Return (fitting, validation, refit, evaluation) episodes for one held-out season.

    Four explicit partitions, each reconstructed from the original episodes rather
    than from an already-masked intermediate, mirroring B0's `season_cv`:

    - *fitting* (inner): context and labels restricted to training weeks minus the
      hidden early-stopping weeks. Chooses the epoch count.
    - *validation*: context is the full training season **including earlier hidden
      weeks**, as in B0's `validation_split`, but only hidden weeks carry labels.
      Origins are training origins whose *supervised* targets include a hidden week.
      `direct` selects those slots: B1 episodes carry two recent (nowcast) targets
      before the four forecast horizons, and the direct task supervises only the
      latter, so a hidden week reachable solely in a recent slot is not a B0
      validation origin. The two-stage task supervises both.
    - *refit*: context and labels over all training weeks, hidden ones restored.
      This is what the finally reported model is fitted on, for the epoch count
      selected on *validation*.
    - *evaluation*: the dataset's retrospective context, scoring held-out weeks only.

    `min_availability` drops sparse episodes from the three fitting partitions only.
    Evaluation is never filtered: the scored tasks must stay identical to B0's,
    or the ensemble-relative comparison stops being like-for-like.
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
    inner = training_weeks - hidden
    # Per-partition context blocks. Inner hides the validation weeks; validation and
    # refit may condition on them; all three hide the held-out season and any week
    # outside B0's panel.
    inner_blocked = held_weeks | hidden | outside
    training_blocked = held_weeks | outside

    fitting, validation, refit, evaluation = [], [], [], []
    for episode in ds.episodes(supervised=False):
        inner_context = _mask_context(episode, inner_blocked)
        training_context = _mask_context(episode, training_blocked)
        sparse = min_availability and inner_context[:, :, 1].mean() < min_availability
        if _origin_in(episode, inner) and not sparse:
            kept = _keep_labels(episode, inner_context, inner, direct)
            if kept is not None:
                fitting.append(kept)
        # B0's validation origins are the training origins that can reach a hidden
        # week, and they condition on the whole training season.
        if _origin_in(episode, training_weeks) and not sparse:
            supervised = [str(d) for d in episode['target_dates'][2:]] if direct \
                else [str(d) for d in episode['target_dates']]
            if hidden.intersection(supervised):
                kept = _keep_labels(episode, training_context, hidden, direct)
                if kept is not None:
                    validation.append(kept)
            kept = _keep_labels(episode, training_context, training_weeks, direct)
            if kept is not None:
                refit.append(kept)
        # Evaluation keeps the retrospective conditioning state: nothing is hidden
        # at prediction time except weeks B0's panel does not have either.
        if _origin_in(episode, held_weeks):
            kept = _keep_labels(episode, _mask_context(episode, outside), held_weeks, direct)
            if kept is not None:
                evaluation.append(kept)
    if not fitting or not validation or not refit or not evaluation:
        raise ValueError(f'No fitting, validation, refit or evaluation episodes for held-out {held_out}')
    info = dict(held_out=held_out, hidden_target_weeks=sorted(hidden),
                training_weeks=len(training_weeks), inner_weeks=len(inner),
                held_out_weeks=len(held_weeks), fitting_episodes=len(fitting),
                validation_episodes=len(validation), refit_episodes=len(refit),
                evaluation_episodes=len(evaluation),
                fitting_origins=sorted(str(e['context_dates'][-1]) for e in fitting),
                validation_origins=sorted(str(e['context_dates'][-1]) for e in validation),
                refit_origins=sorted(str(e['context_dates'][-1]) for e in refit),
                evaluation_origins=sorted(str(e['context_dates'][-1]) for e in evaluation),
                # Auditable against B0's fold: the two must hold out the same calendar.
                season_weeks=len(weeks), weeks_outside_seasons=sorted(outside))
    return fitting, validation, refit, evaluation, info
