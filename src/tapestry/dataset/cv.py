"""Leave-one-season-out cross-validation by masking the panel (replaces `dataset/splits.py`).

Policy (the pre-refactor `models/season_cv.py` one, restored 2026-09-22 after the
refactor's episode-level split let training episodes carry held-out-season labels
and kept validation weeks in inputs and labels; see
docs/design/restructure-2026-unified.md §3 and its decision log):

- A fold copies the panel and makes every week outside the training seasons (including the added 2022–23 season)
  unavailable -- in targets, covariates and the as-of arrays (by reference week). Training episodes are cut from that masked panel, with origins
  in training weeks only, so held-out weeks are absent from inputs, labels, loss
  scales and covariate standardization.
- Inner early-stopping fit (`inner=True`, scenarios with `patience > 0`):
  additionally hide `validation_weeks` consecutive weeks of every
  `validation_spacing`, starting at week `validation_offset` of each training
  season (Scenario fields since 2026-09-22; defaults 3/16/4 = weeks 4-6, 20-22,
  36-38). Week positions count from the season's first epiweek (CDC week 31,
  `season_start`), not from the first calendar week, so every season hides the same
  weeks; 2023-24 starts before the calendar (2023-09-02 is its 5th week), and its
  positions before the calendar start simply do not exist (fixed 2026-09-22).
  Validation episodes are cut from the training panel (hidden weeks
  visible), with origins selected by the task horizons: four weeks before each hidden
  week for forecasting, or at/after it for recent-week nowcasting. Only hidden
  weeks are scored as labels.
- The refit after epoch selection uses the fold's training panel without the
  validation mask (`inner=False`).
- Score episodes come from the unmasked panel with origins in the held-out season;
  earlier weeks are allowed as context, only labels inside the held-out season count.
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .build import covariate_names_for
from .episodes import episodes, restrict_labels

SEASONS = ('2023-2024', '2024-2025', '2025-2026')
TRAINING_SEASONS = ('2022-2023', *SEASONS)


def season_start(year):
    """The Sunday starting CDC epiweek 31 of `year`: first day of season `year`-`year + 1`."""
    jan4 = date(year, 1, 4)
    return jan4 - timedelta(days=(jan4.weekday() + 1) % 7) + timedelta(weeks=30)


def season(day):
    """CDC epiweek 31-30, as used throughout the project (not challenge dates)."""
    day = day if isinstance(day, date) else date.fromisoformat(str(day)[:10])
    year = day.year if day >= season_start(day.year) else day.year - 1
    return f'{year}-{year + 1}'


def masked(panel, keep):
    """Copy of `panel` in which every calendar week not in `keep` (bool [T]) is unavailable.

    As-of arrays are masked by reference week (their calendar-week axis 1).
    """
    out = dict(panel)
    for name in ('targets', 'covariates', 'covariates_national'):
        out[name] = np.where(keep.reshape(-1, *(1,) * (panel[name].ndim - 1)), panel[name], np.nan).astype(np.float32)
    for name in ('asof_targets', 'asof_covariates', 'asof_covariates_national'):
        shape = (1, -1, *(1,) * (panel[name].ndim - 2))
        out[name] = np.where(keep.reshape(shape), panel[name], np.nan).astype(np.float32)
    return out


def week_roles(dates, scenario, held_out):
    """[T] role of each calendar week in one fold, exactly as `fold` uses them.

    'score': the held-out season; 'unused': weeks in neither training nor evaluation seasons; 'validation': the
    early-stopping weeks hidden from the inner fit (only when `scenario.patience > 0`;
    the refit trains on them); 'fit': the remaining training-season weeks. Validation
    positions count weeks from the season's first epiweek (`season_start`).
    """
    if scenario.task == 'pipeline':
        scenario = scenario.stage('forecast')
    dates = np.asarray(dates).astype(str)
    labels = np.array([season(d) for d in dates])
    roles = np.where(labels == held_out, 'score', np.where(np.isin(labels, TRAINING_SEASONS), 'fit', 'unused')).astype('U10')
    if scenario.patience:
        for label in TRAINING_SEASONS:
            if label != held_out:
                weeks = np.flatnonzero(labels == label)
                start = season_start(int(label[:4]))
                position = np.array([(date.fromisoformat(dates[i][:10]) - start).days // 7 for i in weeks])
                if scenario.validation_calendar == 'b0':
                    position = np.arange(len(weeks))
                position %= scenario.validation_spacing
                hidden = (position >= scenario.validation_offset) & \
                    (position < scenario.validation_offset + scenario.validation_weeks)
                roles[weeks[hidden]] = 'validation'
    return roles


@dataclass
class Fold:
    train: list
    validation: list | None
    score: list | None
    info: dict


def fold(panel, scenario, held_out, inner=False):
    """Episodes for one leave-one-season-out fold; `inner=True` gives the early-stopping split."""
    if scenario.task == 'pipeline':
        scenario = scenario.episode_scenario()
    if held_out not in SEASONS:
        raise ValueError(f'Unknown season {held_out}; expected one of {SEASONS}')
    if inner and not scenario.patience:
        raise ValueError('The inner early-stopping fold needs patience > 0')
    dates = np.array([str(d) for d in panel['dates']])
    roles = week_roles(dates, scenario, held_out)
    training = np.isin(roles, ['fit', 'validation'])
    hidden = (roles == 'validation') if inner else np.zeros(len(dates), dtype=bool)
    keep = training & ~hidden
    cut = lambda p, mode=scenario.input_mode: episodes(p, scenario.lookback, mode, covariate_names_for(scenario.covariate_set),
                             scenario.lookback if scenario.task != 'forecast' else scenario.asof_weeks,
                             horizons=scenario.horizons)
    kept = set(dates[keep])
    train_mode = 'finalized' if scenario.training_inputs == 'finalized' else scenario.input_mode
    train = [e for e in cut(masked(panel, keep), train_mode) if e['context_dates'][-1] in kept]
    validation = score = None
    info = dict(held_out=held_out, inner=inner, training_weeks=int(training.sum()), fit_weeks=int(keep.sum()))
    if inner:
        origins = np.zeros(len(dates), dtype=bool)
        for i in np.flatnonzero(hidden):
            for horizon in scenario.horizons:
                origin = i - horizon
                if 0 <= origin < len(dates):
                    origins[origin] = True
        origins = set(dates[origins & training])
        hidden_dates = set(dates[hidden])
        validation = [e for e in cut(masked(panel, training)) if e['context_dates'][-1] in origins]
        validation = [e for e in (restrict_labels(e, hidden_dates) for e in validation) if e is not None]
        info.update(validation_weeks=sorted(hidden_dates), validation_episodes=len(validation))
        if not validation:
            raise ValueError(f'No validation episodes for held-out {held_out}')
    else:
        season_dates = set(dates[roles == 'score'])
        score = [e for e in cut(panel) if e['context_dates'][-1] in season_dates]
        score = [e for e in (restrict_labels(e, season_dates) for e in score) if e is not None]
        info.update(score_episodes=len(score), first_score_origin=score[0]['context_dates'][-1] if score else None)
        if not score:
            raise ValueError(f'No score episodes for held-out {held_out}')
    info['train_episodes'] = len(train)
    if not train:
        raise ValueError(f'No training episodes for held-out {held_out}')
    return Fold(train, validation, score, info)
