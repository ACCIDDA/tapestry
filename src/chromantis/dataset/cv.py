"""Cross-validation by masking the panel, for the problem's fold kind (`problem.Problem`).

Fold kinds (2026-10-09; before, only leave-one-season-out existed):
- `leave_one_season_out`: folds are CDC seasons (epiweek 31-30); a fold trains on the other
  declared training seasons.
- `leave_one_period_out`: folds are declared date periods (e.g. outbreaks); a fold trains on
  the other declared training periods. Weeks outside every period are unused.
- `rolling_origin`: folds are declared evaluation windows; a fold trains on every week
  before its window, so nothing after the forecast window's start is ever used.
In every kind, "season" below means the fold's period.

- A fold copies the panel and makes every week outside the training seasons (including 2022–23)
  unavailable -- in targets, covariates and the as-of arrays (by reference week). Training episodes are cut from that masked panel, with origins
  in training weeks only, so held-out weeks are absent from inputs, labels, loss
  scales and covariate standardization.
- Inner early-stopping fit (`inner=True`, scenarios with `patience > 0`):
  additionally hide `validation_weeks` consecutive weeks of every
  `validation_spacing`, starting at week `validation_offset` of each training
  season (defaults 3/16/4 = weeks 4-6, 20-22,
  36-38). Week positions count from the season's first epiweek (CDC week 31,
  `season_start`), not from the first calendar week, so every season hides the same
  weeks; positions before the panel calendar start simply do not exist.
  Validation episodes are cut from the training panel (hidden weeks
  visible), with origins selected by the scenario horizons (four weeks before each
  hidden week; `joint` also at/after it for its reconstruction weeks). Only hidden
  weeks are scored as labels.
- The refit after epoch selection uses the fold's training panel without the
  validation mask (`inner=False`).
- Score episodes come from the unmasked panel with origins in the held-out season;
  earlier weeks are allowed as context, only labels inside the held-out season count.
  Forecasts are scored on Wednesday reports (`input_mode='reported'`,
  2026-10-05 standard evaluation), whatever the training inputs, with the inputs
  visible at the forecast Hub's own deadline (`score_episodes(..., hub)`), or on
  `synthetic` histories that `fit.py` vintages artificially (`evaluation_inputs=prescribed`),
  or, for problems without the reporting-error stage, on finalized histories on the
  source schedule (`evaluation_inputs=finalized`).
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .build import for_hub
from .episodes import episodes, restrict_labels

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
    asof = [name for name in panel if name.startswith(('asof_', 'hub_')) and 'asof_' in name]
    for name in asof:  # the main as-of arrays and each Hub's own-deadline rows (dataset.build.for_hub)
        shape = (1, -1, *(1,) * (panel[name].ndim - 2))
        out[name] = np.where(keep.reshape(shape), panel[name], np.nan).astype(np.float32)
    return out


def training_seasons(problem, scenario, held_out):
    """Folds fitted in one fold: every other season, or (`training_window=recent2`) the two before it.

    `rolling_origin` has no training folds: it reports the weeks before the window instead."""
    window = getattr(scenario, 'training_window', 'all')
    if problem.fold_kind == 'rolling_origin':
        return (f'before {problem.periods[held_out][0]}',)
    if window == 'recent2':
        i = problem.training_folds.index(held_out)
        return problem.training_folds[max(0, i - 2):i]
    permitted = tuple(s for s in problem.training_folds if s != held_out)
    return permitted[-2:] if window == 'last2' else permitted


def week_roles(dates, problem, scenario, held_out):
    """[T] role of each calendar week in one fold, exactly as `fold` uses them.

    'score': the held-out season; 'unused': weeks in neither training nor evaluation seasons; 'validation': the
    early-stopping weeks hidden from the inner fit (only when `scenario.patience > 0`;
    the refit trains on them); 'fit': the remaining training-season weeks. Validation
    positions count weeks from the season's first epiweek (`season_start`).
    """
    dates = np.asarray(dates).astype(str)
    labels = problem.fold_labels(dates)
    training = problem.training_weeks(dates, held_out, getattr(scenario, 'training_window', 'all'))
    roles = np.where(labels == held_out, 'score', np.where(training, 'fit', 'unused')).astype('U10')
    if problem.fold_kind == 'rolling_origin':
        # One training block: every week before the window; validation positions count from its first week.
        labels = np.where(training, '', labels)
        blocks = {'': date.fromisoformat(str(dates[training][0])[:10]) - timedelta(days=6)} if training.any() else {}
    else:
        blocks = {label: problem.fold_start(label) for label in training_seasons(problem, scenario, held_out)}
    if scenario.patience:
        for label, start in blocks.items():
            if label != held_out:
                weeks = np.flatnonzero(training & (labels == label))
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


def score_inputs(problem):
    """Standard evaluation inputs: Wednesday reports, histories to be vintaged artificially, or
    finalized histories on the source schedule (problems without the reporting-error stage)."""
    return {'prescribed': 'synthetic', 'reported': 'reported', 'finalized': 'scheduled_final'}[problem.evaluation_inputs]


def score_episodes(panel, problem, scenario, held_out, hub=None):
    """Held-out-season score episodes on inputs visible at `hub`'s deadlines."""
    mode = score_inputs(problem)
    dates = np.array([str(d) for d in panel['dates']])
    season_dates = set(dates[week_roles(dates, problem, scenario, held_out) == 'score'])
    lookback = max(12,scenario.lookback) if problem.evaluation_inputs == 'prescribed' else scenario.lookback
    hub = hub or next((value for value in problem.target_hubs if value), 'flusight')
    score = episodes(for_hub(panel, hub) if problem.reporting_errors else panel, problem, problem.input_names(scenario.input_set), lookback, mode,
                     problem.covariate_names(scenario.covariate_set),
                     horizons=problem.model_horizons(scenario))
    score = [e for e in score if e['context_dates'][-1] in season_dates]
    score = [e for e in (restrict_labels(e, season_dates) for e in score) if e is not None]
    if not score:
        raise ValueError(f'No score episodes for held-out {held_out}')
    return score


def fold(panel, problem, scenario, held_out, inner=False, inputs=None):
    """Episodes for one leave-one-season-out fold; `inner=True` gives the early-stopping split.

    Training and validation episodes use `inputs` (default `scenario.input_mode`); `score`
    holds the FluSight-deadline score episodes; other Hubs come from `score_episodes`."""
    if held_out not in problem.folds:
        raise ValueError(f'Unknown fold {held_out}; expected one of {problem.folds}')
    if inner and not scenario.patience:
        raise ValueError('The inner early-stopping fold needs patience > 0')
    dates = np.array([str(d) for d in panel['dates']])
    roles = week_roles(dates, problem, scenario, held_out)
    training = np.isin(roles, ['fit', 'validation'])
    hidden = (roles == 'validation') if inner else np.zeros(len(dates), dtype=bool)
    keep = training & ~hidden
    cut = lambda p: episodes(p, problem, problem.input_names(scenario.input_set), scenario.lookback,
                             inputs or scenario.input_mode, problem.covariate_names(scenario.covariate_set),
                             horizons=problem.model_horizons(scenario))
    kept = set(dates[keep])
    train = [e for e in cut(masked(panel, keep)) if e['context_dates'][-1] in kept]
    validation = score = None
    info = dict(held_out=held_out, training_seasons=list(training_seasons(problem, scenario, held_out)), inner=inner, training_weeks=int(training.sum()), fit_weeks=int(keep.sum()))
    if inner:
        origins = np.zeros(len(dates), dtype=bool)
        for i in np.flatnonzero(hidden):
            for horizon in problem.model_horizons(scenario):
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
        score = score_episodes(panel, problem, scenario, held_out)
        info.update(score_episodes=len(score), score_inputs=score_inputs(problem),
                    first_score_origin=score[0]['context_dates'][-1])
    info['train_episodes'] = len(train)
    if not train:
        raise ValueError(f'No training episodes for held-out {held_out}')
    return Fold(train, validation, score, info)
