"""Leave-one-season-out cross-validation by masking the panel.

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
  visible), with origins selected by the task horizons: four weeks before each hidden
  week for forecasting, or at/after it for recent-week nowcasting. Only hidden
  weeks are scored as labels.
- The refit after epoch selection uses the fold's training panel without the
  validation mask (`inner=False`).
- Score episodes come from the unmasked panel with origins in the held-out season;
  earlier weeks are allowed as context, only labels inside the held-out season count.
  Forecasts are always scored on Wednesday reports (`input_mode='reported'`,
  2026-10-05 standard evaluation), whatever the training inputs, with the inputs
  visible at the forecast Hub's own deadline (`score_episodes(..., hub)`).
  Nowcast and pipeline tasks keep their own dated inputs.
"""
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .build import covariate_names_for, for_hub, HUBS
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
    asof = [name for name in panel if name.startswith(('asof_', 'hub_')) and 'asof_' in name]
    for name in asof:  # the main as-of arrays and each Hub's own-deadline rows (dataset.build.for_hub)
        shape = (1, -1, *(1,) * (panel[name].ndim - 2))
        out[name] = np.where(keep.reshape(shape), panel[name], np.nan).astype(np.float32)
    return out


def training_seasons(scenario, held_out):
    """Seasons fitted in one fold: every other season, or (`training_window=recent2`) the two before it."""
    if getattr(scenario, 'training_window', 'all') == 'recent2':
        i = TRAINING_SEASONS.index(held_out)
        return TRAINING_SEASONS[max(0, i - 2):i]
    return tuple(s for s in TRAINING_SEASONS if s != held_out)


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
    seasons = training_seasons(scenario, held_out)
    roles = np.where(labels == held_out, 'score', np.where(np.isin(labels, seasons), 'fit', 'unused')).astype('U10')
    if scenario.patience:
        for label in seasons:
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


def score_inputs(scenario):
    """Standard evaluation inputs: Wednesday reports for forecasts; nowcasts keep their own."""
    return 'reported' if scenario.task == 'forecast' else scenario.input_mode


def score_episodes(panel, scenario, held_out, hub=HUBS[0], legacy_inputs=False):
    """Held-out-season score episodes on inputs visible at `hub`'s deadlines.

    `legacy_inputs=True` cuts them with the scenario's own `input_mode` instead; only
    fixed-checkpoint replay (`experiment.replay`) uses it to reproduce old B2 inputs."""
    mode = scenario.input_mode if legacy_inputs else score_inputs(scenario)
    if scenario.task == 'pipeline':
        scenario = scenario.episode_scenario()
        mode = scenario.input_mode
    dates = np.array([str(d) for d in panel['dates']])
    season_dates = set(dates[week_roles(dates, scenario, held_out) == 'score'])
    source = for_hub(panel, hub) if mode == 'reported' else panel
    score = episodes(source, scenario.lookback, mode, covariate_names_for(scenario.covariate_set),
                     scenario.lookback if scenario.task != 'forecast' else scenario.asof_weeks,
                     horizons=scenario.horizons)
    score = [e for e in score if e['context_dates'][-1] in season_dates]
    score = [e for e in (restrict_labels(e, season_dates) for e in score) if e is not None]
    if not score:
        raise ValueError(f'No score episodes for held-out {held_out}')
    if mode == 'reported' and scenario.input_mode == 'vintaged':
        # The known-final flag keeps the meaning it had in training (2026-10-05): vintaged
        # training marks its newest `asof_weeks` weeks non-final and older weeks final.
        # Other training modes marked every available cell final, as reported episodes do.
        recent = min(scenario.asof_weeks, scenario.lookback)
        for e in score:
            if recent:
                e['known_final'][-recent:] = False
    return score


def fold(panel, scenario, held_out, inner=False, legacy_inputs=False):
    """Episodes for one leave-one-season-out fold; `inner=True` gives the early-stopping split.

    `score` holds the FluSight-deadline score episodes; other Hubs come from `score_episodes`."""
    original = scenario
    if scenario.task == 'pipeline':
        scenario = scenario.episode_scenario()
    if held_out not in scenario.scored_seasons:
        raise ValueError(f'Unknown season {held_out}; expected one of {SEASONS}')
    if inner and not scenario.patience:
        raise ValueError('The inner early-stopping fold needs patience > 0')
    dates = np.array([str(d) for d in panel['dates']])
    if scenario.evaluation_seasons == 'recent_two':
        missing = set(TRAINING_SEASONS) - {season(d) for d in dates}
        if missing:
            raise ValueError(f'Two-season protocol requires the expanded panel; absent seasons: {sorted(missing)}')
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
    info = dict(held_out=held_out, training_seasons=list(training_seasons(scenario, held_out)), inner=inner, training_weeks=int(training.sum()), fit_weeks=int(keep.sum()))
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
        score = score_episodes(panel, original, held_out, HUBS[0], legacy_inputs)
        info.update(score_episodes=len(score), score_inputs='scenario' if legacy_inputs else score_inputs(original),
                    first_score_origin=score[0]['context_dates'][-1])
    info['train_episodes'] = len(train)
    if not train:
        raise ValueError(f'No training episodes for held-out {held_out}')
    return Fold(train, validation, score, info)
