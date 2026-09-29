"""Dated, named native-unit histories connecting independently fitted models."""
from dataclasses import dataclass
from datetime import date, timedelta

import torch
from torch import nn

from tapestry.dataset.build import CHANNELS
from tapestry.dataset.episodes import calendar as calendar_features


def _dates(context_dates, issuances, episodes, weeks):
    days = tuple(tuple(str(d) for d in row) for row in context_dates)
    issued = tuple(str(d) for d in issuances)
    if len(days) != episodes or len(issued) != episodes or any(len(row) != weeks for row in days):
        raise ValueError('Dates must match episode/history axes; one issuance per episode is required')
    for row, issuance in zip(days, issued):
        parsed = [date.fromisoformat(d) for d in row]
        cutoff = date.fromisoformat(issuance)
        if cutoff.weekday() != 2 or parsed[-1] != cutoff - timedelta(days=4):
            raise ValueError('Context must end on the Saturday preceding its Wednesday issuance')
        if any(b - a != timedelta(weeks=1) for a, b in zip(parsed, parsed[1:])):
            raise ValueError('Context dates must be consecutive weeks in ascending order')
    return days, issued


@dataclass
class CovariateHistory:
    """Named as-of inputs [episode, week, covariate, value/available, location]."""
    values: torch.Tensor
    names: tuple
    locations: tuple
    context_dates: tuple
    issuances: tuple

    def __post_init__(self):
        self.names, self.locations = tuple(self.names), tuple(self.locations)
        if self.values.ndim != 5 or self.values.shape[2:] != (len(self.names), 2, len(self.locations)):
            raise ValueError('Covariate shape must match named covariate/location axes')
        if len(set(self.names)) != len(self.names) or len(set(self.locations)) != len(self.locations):
            raise ValueError('Covariate and location names must be unique')
        self.context_dates, self.issuances = _dates(self.context_dates, self.issuances, *self.values.shape[:2])


@dataclass
class HistorySamples:
    """Joint draws [member, episode, week, target, location]; masks omit member axis."""
    values: torch.Tensor
    available: torch.Tensor
    known_final: torch.Tensor
    estimated: torch.Tensor
    locations: tuple
    context_dates: tuple
    issuances: tuple
    targets: tuple = CHANNELS
    provenance: dict | None = None

    def __post_init__(self):
        self.locations, self.targets = tuple(self.locations), tuple(self.targets)
        if self.values.ndim != 5 or self.values.shape[0] < 1 or self.values.shape[3:] != (len(CHANNELS), len(self.locations)):
            raise ValueError('Histories must have member/episode/week/target/location axes')
        if self.targets != CHANNELS or len(set(self.locations)) != len(self.locations):
            raise ValueError('Use panel target order and unique location IDs')
        for mask in (self.available, self.known_final, self.estimated):
            if mask.shape != self.values.shape[1:] or mask.dtype != torch.bool:
                raise ValueError('History masks must be boolean and exactly match episode/week/target/location axes')
        if (self.known_final & (self.estimated | ~self.available)).any() or (self.estimated & ~self.available).any():
            raise ValueError('Final and estimated cells must be available; estimated cells cannot be known-final')
        self.context_dates, self.issuances = _dates(self.context_dates, self.issuances, *self.values.shape[1:3])


def _calendar(context_dates, reference, supplied=None):
    expected = torch.as_tensor(calendar_features([row[-1] for row in context_dates]), device=reference.device)
    if supplied is not None and (supplied.shape != expected.shape or not torch.allclose(supplied, expected, atol=1e-6, rtol=0)):
        raise ValueError('Calendar features do not match the history dates')
    return expected


def _covariates(covariates, model, locations, context_dates, issuances):
    names = model.config['covariate_names']
    if not names:
        return None
    if not isinstance(covariates, CovariateHistory):
        raise ValueError('Provide named, dated CovariateHistory for this model')
    weeks = len(context_dates[0])
    if (covariates.locations != tuple(locations) or covariates.issuances != tuple(issuances)
            or tuple(row[-weeks:] for row in covariates.context_dates) != tuple(context_dates)):
        raise ValueError('Covariate locations, dates or issuance do not match target history')
    if not set(names) <= set(covariates.names):
        raise ValueError(f'Missing model covariates: {sorted(set(names) - set(covariates.names))}')
    return covariates.values[:, -weeks:, [covariates.names.index(name) for name in names]]


def reconstruct(nowcaster, *, values, available, locations, context_dates, issuances, members=8,
                known_final=None, covariates=None, calendar=None, provenance=None):
    """Reconstruct recent weeks using the nowcaster's lookback; preserve longer history."""
    horizons = nowcaster.config['horizons']
    lookback = nowcaster.config['lookback']
    if horizons != list(range(1 - len(horizons), 1)) or not 1 <= len(horizons) <= lookback:
        raise ValueError('Nowcaster must predict consecutive completed weeks ending at horizon 0')
    if values.ndim != 4 or values.shape[1] < lookback:
        raise ValueError('Input history is shorter than the nowcaster lookback')
    if list(locations) != list(nowcaster.config['location_ids']):
        raise ValueError('Location order differs from nowcaster')
    final = torch.zeros_like(available) if known_final is None else known_final.clone()
    source = HistorySamples(values[None], available, final, torch.zeros_like(available),
                            locations, context_dates, issuances)
    dates = tuple(row[-lookback:] for row in source.context_dates)
    draws = nowcaster(values=values[:, -lookback:], available=available[:, -lookback:],
                      known_final=final[:, -lookback:], calendar=_calendar(dates, values, calendar),
                      locations=list(locations), members=members,
                      covariates=_covariates(covariates, nowcaster, locations, dates, source.issuances), vintaged=True)
    histories = values.unsqueeze(0).expand(members, *values.shape).clone()
    histories[:, :, -len(horizons):] = draws
    visible = available.clone()
    visible[:, -len(horizons):] = True
    final[:, -len(horizons):] = False
    estimated = torch.zeros_like(available)
    estimated[:, -len(horizons):] = True
    return HistorySamples(histories, visible, final, estimated, locations, source.context_dates,
                          source.issuances, provenance=provenance)


def forecast_histories(forecaster, histories, calendar=None, covariates=None):
    """One conditional future per intact history draw, using the forecaster's lookback."""
    lookback = forecaster.config['lookback']
    if histories.values.shape[2] < lookback:
        raise ValueError('History is shorter than forecaster lookback')
    if list(histories.locations) != list(forecaster.config['location_ids']):
        raise ValueError('History location order differs from forecaster')
    if any(h <= 0 for h in forecaster.config['horizons']):
        raise ValueError('Forecaster must predict future weeks')
    dates = tuple(row[-lookback:] for row in histories.context_dates)
    cal = _calendar(dates, histories.values, calendar)
    cov = _covariates(covariates, forecaster, histories.locations, dates, histories.issuances)
    values = histories.values[:, :, -lookback:]
    m, n, p, c, l = values.shape
    repeat = lambda x: x.unsqueeze(0).expand(m, *x.shape).reshape(m * n, *x.shape[1:])
    output = forecaster(values=values.reshape(m * n, p, c, l),
                        available=repeat(histories.available[:, -lookback:]),
                        known_final=repeat(histories.known_final[:, -lookback:]),
                        estimated=repeat(histories.estimated[:, -lookback:]), calendar=repeat(cal),
                        locations=list(histories.locations), members=1, vintaged=True,
                        covariates=None if cov is None else repeat(cov))
    return output[0].reshape(m, n, *output.shape[2:])


class NowcastForecast(nn.Module):
    """Independent stages; input history spans the longer of their two lookbacks."""
    def __init__(self, nowcaster, forecaster):
        super().__init__()
        if nowcaster.config['location_ids'] != forecaster.config['location_ids']:
            raise ValueError('Both stages must use the same location order')
        self.nowcaster, self.forecaster = nowcaster, forecaster
        names = list(dict.fromkeys(nowcaster.config['covariate_names'] + forecaster.config['covariate_names']))
        self.config = dict(forecaster.config, covariate_names=names,
                           lookback=max(nowcaster.config['lookback'], forecaster.config['lookback']))

    def forward(self, *, values, available, locations, context_dates, issuances, members=8,
                known_final=None, covariates=None, calendar=None, vintaged=True):
        if not vintaged:
            raise ValueError('Pipeline expects as-of reports')
        histories = reconstruct(self.nowcaster, values=values, available=available,
                                calendar=calendar, locations=locations, members=members,
                                context_dates=context_dates, issuances=issuances,
                                known_final=known_final, covariates=covariates)
        return forecast_histories(self.forecaster, histories, calendar, covariates)
