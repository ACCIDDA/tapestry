"""Leave-one-season-out cross-validation by masking the panel (replaces `dataset/splits.py`).

Policy (the pre-refactor `models/season_cv.py` one, restored 2026-09-22 after the
refactor's episode-level split let training episodes carry held-out-season labels
and kept validation weeks in inputs and labels; see
docs/design/restructure-2026-unified.md §3 and its decision log):

- A fold copies the panel and makes every week outside the two training seasons
  unavailable -- in targets, covariates and the as-of overlay (overlay cells by their
  reference week). Training episodes are cut from that masked panel, with origins
  in training weeks only, so held-out weeks are absent from inputs, labels, loss
  scales and covariate standardization.
- Inner early-stopping fit (`inner=True`): additionally hide 3 consecutive weeks of
  every 16, starting at week 4 of each training season (weeks 4-6, 20-22, 36-38).
  Validation episodes are cut from the training panel (hidden weeks visible), with
  origins in the four training weeks before each hidden week, and only hidden
  weeks are scored as labels.
- The refit after epoch selection uses the fold's training panel without the
  validation mask (`inner=False`).
- Score episodes come from the unmasked panel with origins in the held-out season;
  earlier weeks are allowed as context, only labels inside the held-out season count.
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .build import covariate_names_for, overlay_dates
from .episodes import episodes, restrict_labels

SEASONS = ('2023-2024', '2024-2025', '2025-2026')
# Early stopping hides weeks 4-6, 20-22 and 36-38 of each training season.
VALIDATION_WEEKS, VALIDATION_SPACING, VALIDATION_OFFSET = 3, 16, 4
HORIZON_WEEKS = 4


def season(day):
    """CDC epiweek 31-30, as used throughout the project (not challenge dates)."""
    def boundary(year):
        jan4 = date(year, 1, 4)
        return jan4 - timedelta(days=(jan4.weekday() + 1) % 7) + timedelta(weeks=30)
    day = day if isinstance(day, date) else date.fromisoformat(str(day)[:10])
    year = day.year if day >= boundary(day.year) else day.year - 1
    return f'{year}-{year + 1}'


def masked(panel, keep):
    """Copy of `panel` in which every calendar week not in `keep` (bool [T]) is unavailable.

    Overlay cells are masked by their reference week; reference weeks outside the
    calendar are already unavailable.
    """
    dates = [str(d) for d in panel['dates']]
    kept = {d for d, k in zip(dates, keep) if k}
    out = dict(panel)
    out['targets'] = np.where(keep[:, None, None], panel['targets'], np.nan).astype(np.float32)
    out['covariates'] = np.where(keep[:, None, None], panel['covariates'], np.nan).astype(np.float32)
    out['covariate_mask'] = panel['covariate_mask'] & keep[:, None, None]
    out['covariates_national'] = np.where(keep[:, None], panel['covariates_national'], np.nan).astype(np.float32)
    for name in ('asof_targets', 'asof_covariates', 'asof_covariates_national'):
        depth = panel[name].shape[1]
        visible = np.array([[d in kept for d in overlay_dates(str(i), depth)] for i in panel['issuance_dates']])
        shape = visible.shape + (1,) * (panel[name].ndim - 2)
        out[name] = np.where(visible.reshape(shape), panel[name], np.nan).astype(np.float32)
    return out


def validation_weeks(dates, held_out):
    """[T] bool: the hidden early-stopping weeks of each training season."""
    labels = np.array([season(d) for d in dates])
    hidden = np.zeros(len(dates), dtype=bool)
    for label in SEASONS:
        if label != held_out:
            weeks = np.flatnonzero(labels == label)
            position = np.arange(len(weeks)) % VALIDATION_SPACING
            hidden[weeks[(position >= VALIDATION_OFFSET) & (position < VALIDATION_OFFSET + VALIDATION_WEEKS)]] = True
    return hidden


@dataclass
class Fold:
    train: list
    validation: list | None
    score: list | None
    info: dict


def fold(panel, scenario, held_out, inner=False):
    """Episodes for one leave-one-season-out fold; `inner=True` gives the early-stopping split."""
    if held_out not in SEASONS:
        raise ValueError(f'Unknown season {held_out}; expected one of {SEASONS}')
    dates = np.array([str(d) for d in panel['dates']])
    labels = np.array([season(d) for d in dates])
    training = np.isin(labels, [s for s in SEASONS if s != held_out])
    hidden = validation_weeks(dates, held_out) if inner else np.zeros(len(dates), dtype=bool)
    keep = training & ~hidden
    cut = lambda p: episodes(p, scenario.lookback, scenario.input_mode, covariate_names_for(scenario.covariate_set))
    kept = set(dates[keep])
    train = [e for e in cut(masked(panel, keep)) if e['context_dates'][-1] in kept]
    validation = score = None
    info = dict(held_out=held_out, inner=inner, training_weeks=int(training.sum()), fit_weeks=int(keep.sum()))
    if inner:
        origins = np.zeros(len(dates), dtype=bool)
        for i in np.flatnonzero(hidden):
            origins[max(i - HORIZON_WEEKS, 0):i] = True
        origins = set(dates[origins & training])
        hidden_dates = set(dates[hidden])
        validation = [e for e in cut(masked(panel, training)) if e['context_dates'][-1] in origins]
        validation = [e for e in (restrict_labels(e, hidden_dates) for e in validation) if e is not None]
        info.update(validation_weeks=sorted(hidden_dates), validation_episodes=len(validation))
        if not validation:
            raise ValueError(f'No validation episodes for held-out {held_out}')
    else:
        season_dates = set(dates[labels == held_out])
        score = [e for e in cut(panel) if e['context_dates'][-1] in season_dates]
        score = [e for e in (restrict_labels(e, season_dates) for e in score) if e is not None]
        info.update(score_episodes=len(score), first_score_origin=score[0]['context_dates'][-1] if score else None)
        if not score:
            raise ValueError(f'No score episodes for held-out {held_out}')
    info['train_episodes'] = len(train)
    if not train:
        raise ValueError(f'No training episodes for held-out {held_out}')
    return Fold(train, validation, score, info)
