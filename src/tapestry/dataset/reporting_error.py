"""Training-only, local seasonal bootstrap of reporting-error windows.

Errors retain their age, signal and location alignment. Final epidemic trajectories
and labels are never resampled. See the dated experiment protocol for assumptions.
"""
import warnings

import numpy as np

from . import cv
from .build import covariate_names_for
from .episodes import episodes


def peak_scales(panel, keep):
    """Season/location/signal peaks from permitted fitting dates only."""
    labels = np.array([cv.season(d) for d in panel['dates']])
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for label in sorted(set(labels[keep])):
            use = keep & (labels == label)
            target = np.nanmax(panel['targets'][use], axis=0).T
            state = np.nanmax(panel['covariates'][use], axis=0).T
            national = np.nanmax(panel['covariates_national'][use], axis=0)
            result[label] = (np.nan_to_num(target), np.nan_to_num(state), np.nan_to_num(national))
    return result


def phase(episode, peak):
    """Per-signal, per-location recent log level and two-week log change."""
    values, valid = episode['values'], episode['available']
    scale = np.maximum(peak, 1e-8)
    def mean(start, end):
        ok = valid[start:end]
        return np.divide(np.where(ok, values[start:end], 0).sum(0), ok.sum(0),
                         out=np.full(values.shape[1:], np.nan, np.float32), where=ok.sum(0) > 0) / scale
    recent, prior = mean(-2, None), mean(-4, -2)
    recent, prior = np.log(recent + .05), np.log(prior + .05)
    return np.concatenate((recent / np.log(2), (recent - prior) / .5), axis=0)


def season_position(day):
    """Fraction of the actual 52/53-week CDC season, wrapping at its boundary."""
    year = int(cv.season(day)[:4])
    start, end = cv.season_start(year), cv.season_start(year + 1)
    return (np.datetime64(day) - np.datetime64(start)).astype(int) / (end - start).days


def transport(values, error, floor):
    """Invert the signed log ratio; zero residuals preserve inputs exactly."""
    changed = np.maximum(0, (values + floor) * np.exp(error) - floor)
    return np.where(error == 0, values, changed)


class ReportingErrors:
    """A fold-specific empirical distribution, sampled when a minibatch is queried."""
    def __init__(self, panel, scenario, held_out, keep, corrections=None):
        from tapestry.experiment.replay import replay_episode
        self.names = list(covariate_names_for(scenario.covariate_set))
        self.probability = scenario.reporting_probability
        self.random_strength = scenario.reporting_random_strength
        self.recent = scenario.reporting_recent
        self.mode = scenario.reporting_augmentation
        self.method = scenario.reporting_method
        self.strength = scenario.reporting_strength
        self.transport_missingness = scenario.reporting_missingness
        self.peaks = peak_scales(panel, keep)
        self.donor_season = max(self.peaks)
        # Donor seasons (2026-10-06): the latest permitted season only, or every
        # permitted season with recency weights 1, 1/2, 1/4, ... (latest first).
        ordered = sorted(self.peaks, reverse=True)
        pooled = getattr(scenario, 'vintage_seasons', 'latest') == 'all'
        self.season_weight = {label: .5 ** i for i, label in enumerate(ordered)} if pooled else {self.donor_season: 1.}
        self.state_names = list(panel['covariate_names'])
        self.national_names = list(panel['covariate_national_names'])
        self.unit_floor = np.array([1. if str(n).startswith('nhsn_') else .0001
                                   for n in panel['target_names']], np.float32)[:, None]
        # Mask before extracting errors, matching features, or nowcast residuals.
        masked = cv.masked(panel, keep)
        allowed = set(panel['dates'][keep].astype(str))
        final = episodes(masked, scenario.lookback, 'scheduled_final', self.names)
        self.donors, features, errors, supports, visible, cov_errors, cov_supports, cov_visible = [], [], [], [], [], [], [], []
        donor_weights = []
        self.error_dates = set()
        # Signals with any archived report per season (2023-24 has no COVID/RSV archive).
        # Only reports issued during that season count (later issuances also revise its weeks).
        labels = np.array([cv.season(d) for d in panel['dates'].astype(str)])
        issued = np.array([cv.season(d) for d in panel['issuance_dates'].astype(str)])
        archived = {label: np.isfinite(masked['asof_targets'][issued == label][:, labels == label]).any(axis=(0, 1, 2))
                    for label in self.season_weight}
        first_date = str(panel['dates'][0])
        for e in final:
            day = e['context_dates'][-1]
            origin_season = cv.season(day)
            if day not in allowed or origin_season not in self.season_weight or e['context_dates'][0] < first_date:
                continue
            raw = replay_episode(e, masked, self.names, 'vintage')
            # Exclude pre-archive/onboarding windows, not ordinary latest-week delays.
            # Seasons archiving all six signals keep the original rule (every signal reported in
            # the last four weeks); partially archived seasons (2023-24) need flu admissions only.
            recent = raw['available'][-4:].any(axis=(0, 2))
            if not (recent.all() if archived[origin_season].all() else recent[0]):
                continue
            observed = replay_episode(e, masked, self.names, 'nowcast', corrections) if self.mode == 'nowcast' else raw
            # Error reference weeks must also belong to the donor season; an
            # origin just after its boundary must not import an older season.
            donor_dates = np.array([cv.season(d) == origin_season for d in e['context_dates']])
            supported = e['available'] & donor_dates[:, None, None]
            floor = self.target_floor(e)
            ok = supported & observed['available']
            error = ((observed['values'] - e['values']) / np.maximum(e['values'], floor)
                     if self.method == 'local_additive' else
                     np.log((observed['values'] + floor) / (e['values'] + floor)))
            errors.append(np.where(ok, error, 0))
            supports.append(supported)
            visible.append(observed['available'])
            features.append(phase(e, self.peaks[origin_season][0]))
            self.donors.append(day)
            donor_weights.append(self.season_weight[origin_season])
            self.error_dates.update(d for d in e['context_dates'] if d in allowed and cv.season(d) == origin_season)
            if self.names:
                f, x = e['covariates'], observed['covariates']
                support = f[..., 1, :].astype(bool) & donor_dates[:, None, None]
                floor = np.maximum(.05 * self.cov_peak(self.donor_season)[None], 1e-8)
                ok = support & x[..., 1, :].astype(bool)
                error = ((x[..., 0, :] - f[..., 0, :]) / np.maximum(f[..., 0, :], floor)
                         if self.method == 'local_additive' else
                         np.log((x[..., 0, :] + floor) / (f[..., 0, :] + floor)))
                cov_errors.append(np.where(ok, error, 0))
                cov_supports.append(support)
                cov_visible.append(x[..., 1, :].astype(bool))
        if len(self.donors) < 8:
            raise ValueError(f'Only {len(self.donors)} reporting windows in {self.donor_season}')
        self.features = np.stack(features)
        self.donor_log_weight = np.log(np.array(donor_weights))
        self.positions = np.array([season_position(d) for d in self.donors])
        self.errors, self.support, self.visible = map(np.stack, (errors, supports, visible))
        if self.names:
            self.cov_errors, self.cov_support, self.cov_visible = map(np.stack, (cov_errors, cov_supports, cov_visible))
        self.metadata = dict(mode=self.mode, transport_missingness=self.transport_missingness, donor_season=self.donor_season, donor_origins=self.donors,
            permitted_error_dates=sorted(self.error_dates), windows=len(self.donors),
            held_out=held_out, target_evidence_fraction=float(self.support.mean()),
            donor_seasons={k: v for k, v in self.season_weight.items()},
            donor_windows_by_season={k: sum(cv.season(d) == k for d in self.donors) for k in self.season_weight},
            version='local_seasonal_log_v2',
            method=self.method, strength=self.strength, probability=self.probability,
            random_strength=self.random_strength, recent_weeks=self.recent,
            matching='per location: eight nearest local six-signal log-level/log-change and circular calendar windows; exp(-distance/2) bootstrap',
            bandwidths='log level log(2); two-week log change 0.5; calendar eight weeks',
            dependence=('one donor date shared across all locations and signals' if self.method == 'synchronous_log' else
                        'one joint age/signal window per location; cross-location donor synchrony not retained'),
            error_scale=('(nowcast_or_report-final)/max(final,floor)' if self.method == 'local_additive' else
                         'log((nowcast_or_report + floor)/(final + floor))'),
            floor='max(5% reference-season location/signal peak, one count or 0.0001 proportion)',
            unsupported_cells='unchanged; never infer missingness from absent final truth',
            labels='unchanged', finality_channel=False)

    def target_floor(self, episode):
        # Context may straddle seasons: use the reference week's season, not
        # the origin's season, for both extracting and applying each residual.
        fallback = self.peaks[cv.season(episode['context_dates'][-1])][0]
        return np.stack([np.maximum(.05 * self.peaks.get(cv.season(d), (fallback,))[0],
                                    self.unit_floor) for d in episode['context_dates']])

    def cov_peak(self, label):
        _, state, national = self.peaks[label]
        return np.stack([state[self.state_names.index(n)] if n in self.state_names else
                         np.full(state.shape[-1], national[self.national_names.index(n)]) for n in self.names])

    def draw(self, episode, rng):
        label = cv.season(episode['context_dates'][-1])
        peak = self.peaks[label][0]
        feature = phase(episode, peak)
        difference = (self.features - feature[None]) ** 2
        valid = np.isfinite(difference)
        distance = np.divide(np.where(valid, difference, 0).sum(1), valid.sum(1),
                             out=np.full((len(self.donors), feature.shape[-1]), np.inf), where=valid.sum(1) > 0)
        delta = np.abs(self.positions - season_position(episode['context_dates'][-1]))
        calendar = (np.minimum(delta, 1 - delta) * 52 / 8) ** 2
        # If local phase is absent, use calendar only. No other location donates.
        distance = np.where(np.isfinite(distance), distance, 0) + calendar[:, None]
        if self.method == 'phase_log':
            distance -= calendar[:, None]
        # Older donor seasons are down-weighted inside the exp(-distance/2) kernel.
        distance = distance - 2 * self.donor_log_weight[:, None]
        if self.method == 'synchronous_log':
            distance = np.broadcast_to(distance.mean(axis=1)[:, None], distance.shape)
        if self.method == 'calendar_log':
            distance = np.broadcast_to(calendar[:, None], distance.shape)
        # A same-season origin cannot donate its own revision pattern.
        eligible = np.array([d != episode['context_dates'][-1] for d in self.donors])
        candidates = np.flatnonzero(eligible)
        locations = np.arange(feature.shape[-1])
        nearest = candidates[np.argsort(distance[candidates], axis=0, kind='stable')[:8]]
        weights = np.exp(-.5 * (distance[nearest, locations] - distance[nearest[0], locations]))
        cumulative = np.cumsum(weights / weights.sum(0), axis=0)
        choice = (rng.random(len(locations))[None] > cumulative).sum(0).clip(max=len(nearest)-1)
        donor = nearest[choice, locations]
        if self.method == 'synchronous_log':
            donor[:] = donor[0]
        active = self.probability == 1 or (self.probability > 0 and rng.random() < self.probability)
        multiplier = (rng.uniform(0, 1) if self.random_strength else 1.) if active else 0.
        def errors(array):
            out = local(array).copy() * multiplier
            if self.recent:
                out[:-self.recent] = 0
            return out
        # Advanced indexing produces [location, age, signal].
        local = lambda array: np.moveaxis(array[donor, :, :, locations], 0, -1)
        values = self.apply(episode['values'], errors(self.errors), self.target_floor(episode))
        values[:, 3:] = np.minimum(1, values[:, 3:])
        available = episode['available'].copy()
        if self.transport_missingness:
            available &= ~local(self.support) | local(self.visible)
        values = np.where(available, values, 0).astype(np.float32)
        out = dict(episode, values=values, available=available,
                   known_final=np.zeros_like(available))
        if self.names:
            cov = episode['covariates'].copy()
            floor = np.maximum(.05 * self.cov_peak(label)[None], 1e-8)
            cov_error = errors(self.cov_errors)
            # Shared national covariates must stay identical across locations.
            for k, name in enumerate(self.names):
                if name in self.national_names and name not in self.state_names:
                    us = episode['locations'].index('US')
                    cov_error[:, k] = self.cov_errors[donor[us], :, k, us][:, None] * multiplier
                    if self.recent:
                        cov_error[:-self.recent, k] = 0
            values = self.apply(cov[..., 0, :], cov_error, floor)
            # These selected source units, including WVAL-like indices, are nonnegative.
            values = np.maximum(0, values)
            available = cov[..., 1, :].astype(bool)
            if self.transport_missingness:
                available &= ~local(self.cov_support) | local(self.cov_visible)
            out['covariates'] = np.stack((np.where(available, values, 0), available), axis=-2).astype(np.float32)
        return out

    def apply(self, values, errors, floor):
        errors = errors * self.strength
        if self.method == 'local_additive':
            return np.maximum(0, values + errors * np.maximum(values, floor))
        return transport(values, errors, floor)

    def batch(self, episodes_, rng):
        return [self.draw(e, rng) for e in episodes_]
