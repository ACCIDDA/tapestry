"""Local seasonal bootstrap of reporting-error windows: artificial preliminary histories.

Errors retain their age, signal and location alignment. Final epidemic trajectories
and labels are never resampled. See the dated experiment protocol for assumptions.
`ReportingErrors` serves training (artificial training histories, correction-tree
examples) and, with a prescribed `error_reference`, artificial evaluation histories.
"""
import warnings

import numpy as np

from . import cv
from .build import covariate_names_for
from .episodes import episodes, select_covariates


def archived_report(episode, panel, covariate_names):
    """The episode's context as archived at its Wednesday issuance (unpublished cells unavailable).

    Labels and their masks are unchanged. Moved from the deleted fixed-checkpoint replay
    module (`replay_episode(..., 'vintage')`) on 2026-10-08."""
    dates = panel['dates'].astype(str)
    issue = str(np.datetime64(episode['context_dates'][-1]) + np.timedelta64(4, 'D'))
    issues = panel['issuance_dates'].astype(str)
    wi = np.searchsorted(issues, issue)
    if wi >= len(issues) or issues[wi] != issue:
        raise ValueError(f'No matching vintage for {issue}')
    ti = np.searchsorted(dates, episode['context_dates'])
    if (ti >= len(dates)).any() or not np.array_equal(dates[ti], episode['context_dates']):
        raise ValueError('Report context does not align to panel dates')
    values = np.moveaxis(panel['asof_targets'][wi, ti].copy(), -1, -2)
    result = dict(episode, values=np.nan_to_num(values), available=np.isfinite(values),
                  known_final=np.zeros_like(values, dtype=bool), issuance=issue)
    if covariate_names:
        v, a = select_covariates(panel['asof_covariates'][wi, ti], panel['asof_covariates_national'][wi, ti],
                                 panel['covariate_names'], panel['covariate_national_names'], panel['locations'],
                                 covariate_names)
        result['covariates'] = np.stack((v, a), axis=-2)
    return result


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


def phase(episode, peak, channels=None):
    """Per-signal, per-location recent log level and two-week log change."""
    values, valid = episode['values'], episode['available'].copy()
    if channels is not None:
        valid[:, [c for c in range(6) if c not in channels]] = False
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
    """A fold-specific empirical distribution, sampled when a minibatch is queried.

    `draw` applies the scenario's scope (2026-10-05 B3 pilot, merged here 2026-10-08):
    `actual_share` uses the actual archived report instead of an artificial draw,
    `error_scope=early_actual` uses actual reports in the donor season, and
    `error_signals` limits which signals are perturbed."""
    def __init__(self, panel, scenario, held_out, keep):
        self.scenario = scenario
        self.names = list(covariate_names_for(scenario.covariate_set))
        self.probability = scenario.reporting_probability
        self.random_strength = scenario.reporting_random_strength
        self.recent = scenario.reporting_recent
        self.method = scenario.reporting_method
        self.strength = scenario.reporting_strength
        self.transport_missingness = scenario.reporting_missingness
        self.peaks = peak_scales(panel, keep)
        self.reference = scenario.error_reference
        self.channels = list({'all':range(6),'flu':[0,3],'flu_hosp':[0],'flu_ed':[3],
                              'flu_covid':[0,1,3,4],'flu_rsv':[0,2,3,5]}[scenario.pathogen_inputs])
        self.default_peaks = tuple(np.maximum.reduce([p[i] for p in self.peaks.values()]) for i in range(3))
        self.donor_season = max(self.peaks)
        # Donor seasons (2026-10-06): the latest permitted season only, or every
        # permitted season with recency weights 1, 1/2, 1/4, ... (latest first).
        ordered = sorted(self.peaks, reverse=True)
        pooled = scenario.error_seasons == 'all'
        self.season_weight = {label: .5 ** i for i, label in enumerate(ordered)} if pooled else {self.donor_season: 1.}
        if self.reference:
            # Reporting calibration is prescribed separately from epidemic fitting.
            # No held-out epidemic trajectory is added to the forecaster's train set.
            keep = np.array([cv.season(d) == self.reference for d in panel['dates']])
            self.reference_peaks = peak_scales(panel, keep)
            self.donor_season = self.reference
            self.season_weight = {self.reference: 1.}
        else:
            self.reference_peaks = self.peaks
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
            raw = archived_report(e, masked, self.names)
            # Exclude pre-archive/onboarding windows, not ordinary latest-week delays.
            # Seasons archiving all six signals keep the original rule (every signal reported in
            # the last four weeks); partially archived seasons (2023-24) need flu admissions only.
            recent = raw['available'][-4:].any(axis=(0, 2))
            if not (recent.all() if archived[origin_season].all() else recent[0]):
                continue
            observed = raw
            # Error reference weeks must also belong to the donor season; an
            # origin just after its boundary must not import an older season.
            donor_dates = np.array([cv.season(d) == origin_season for d in e['context_dates']])
            supported = e['available'] & donor_dates[:, None, None]
            floor = self.target_floor(e, reference=True)
            ok = supported & observed['available']
            error = ((observed['values'] - e['values']) / np.maximum(e['values'], floor)
                     if self.method == 'local_additive' else
                     np.log((observed['values'] + floor) / (e['values'] + floor)))
            errors.append(np.where(ok, error, 0))
            supports.append(supported)
            visible.append(observed['available'])
            features.append(phase(e, self.default_peaks[0], self.channels))
            self.donors.append(day)
            donor_weights.append(self.season_weight[origin_season])
            self.error_dates.update(d for d in e['context_dates'] if d in allowed and cv.season(d) == origin_season)
            if self.names:
                f, x = e['covariates'], observed['covariates']
                support = f[..., 1, :].astype(bool) & donor_dates[:, None, None]
                floor = np.maximum(.05 * self.cov_peak(self.donor_season, reference=True)[None], 1e-8)
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
        self.metadata = dict(mode='vintage', transport_missingness=self.transport_missingness, donor_season=self.donor_season, donor_origins=self.donors,
            permitted_error_dates=sorted(self.error_dates), windows=len(self.donors),
            held_out=held_out, target_evidence_fraction=float(self.support.mean()),
            donor_seasons={k: v for k, v in self.season_weight.items()},
            donor_windows_by_season={k: sum(cv.season(d) == k for d in self.donors) for k in self.season_weight},
            version='prescribed_revision_v4' if self.reference else 'local_seasonal_log_v3',
            error_reference=self.reference, matching_channels=self.channels,
            method=self.method, strength=self.strength, probability=self.probability,
            random_strength=self.random_strength, recent_weeks=self.recent,
            matching='eight nearest permitted-signal log-level/log-change windows; exp(-distance/2) bootstrap; optional circular calendar',
            bandwidths='log level log(2); two-week log change 0.5; calendar eight weeks',
            dependence=('one donor date shared across all locations and signals' if self.method in ('synchronous_log','synchronous_phase_log') else
                        'one joint age/signal window per location; cross-location donor synchrony not retained'),
            error_scale=('(nowcast_or_report-final)/max(final,floor)' if self.method == 'local_additive' else
                         'log((nowcast_or_report + floor)/(final + floor))'),
            floor='max(5% reference-season location/signal peak, one count or 0.0001 proportion)',
            unsupported_cells=('nearest genuinely observed age/signal/location residual; national or native-location pool if needed' if self.reference else
                               'unchanged; never infer missingness from absent final truth'),
            donor_exclusion=('exclude the closest seasonal release position in every recipient season; overlapping error windows allowed in the prescribed calibration bank' if self.reference else 'different origin'),
            labels='unchanged', finality_channel=False, signals=scenario.error_signals, scope=scenario.error_scope)
        # Actual archived reports by origin (finalized value where nothing was archived).
        self.reported = {e['context_dates'][-1]: e for e in episodes(
            masked, scenario.lookback, 'reported', self.names, horizons=tuple(range(1 - scenario.lookback, 5)))}

    def target_floor(self, episode, reference=False):
        # Context may straddle seasons: use the reference week's season, not
        # the origin's season, for both extracting and applying each residual.
        peaks = self.reference_peaks if reference else self.peaks
        fallback = self.default_peaks[0]
        return np.stack([np.maximum(.05 * peaks.get(cv.season(d), (fallback,))[0],
                                    self.unit_floor) for d in episode['context_dates']])

    def cov_peak(self, label, reference=False):
        _, state, national = (self.reference_peaks if reference else self.peaks).get(label, self.default_peaks)
        return np.stack([state[self.state_names.index(n)] if n in self.state_names else
                         np.full(state.shape[-1], national[self.national_names.index(n)]) for n in self.names])

    def draw(self, episode, rng):
        day = episode['context_dates'][-1]
        scenario = self.scenario
        actual = scenario.actual_share and day in self.reported and rng.random() < scenario.actual_share
        if actual or (scenario.error_scope == 'early_actual' and cv.season(day) == self.donor_season):
            report = self.reported[day]
            return dict(episode, **{k: report[k] for k in ('values', 'available', 'known_final', 'covariates', 'filled')
                                    if k in report})
        out = self.artificial(episode, rng)
        if scenario.error_signals == 'admissions':
            out['values'][:, 3:] = episode['values'][:, 3:]
            out['available'][:, 3:] = episode['available'][:, 3:]
        if scenario.error_signals in ('admissions', 'targets') and 'covariates' in episode:
            out['covariates'] = episode['covariates']
        return out

    def artificial(self, episode, rng):
        label = cv.season(episode['context_dates'][-1])
        feature = phase(episode, self.default_peaks[0], self.channels)
        difference = (self.features - feature[None]) ** 2
        valid = np.isfinite(difference)
        distance = np.divide(np.where(valid, difference, 0).sum(1), valid.sum(1),
                             out=np.full((len(self.donors), feature.shape[-1]), np.inf), where=valid.sum(1) > 0)
        delta = np.abs(self.positions - season_position(episode['context_dates'][-1]))
        calendar = (np.minimum(delta, 1 - delta) * 52 / 8) ** 2
        # If local phase is absent, use calendar only. No other location donates.
        distance = np.where(np.isfinite(distance), distance, 0) + calendar[:, None]
        if self.method in ('phase_log','synchronous_phase_log'):
            distance -= calendar[:, None]
        # Older donor seasons are down-weighted inside the exp(-distance/2) kernel.
        distance = distance - 2 * self.donor_log_weight[:, None]
        if self.method in ('synchronous_log','synchronous_phase_log'):
            distance = np.broadcast_to(distance.mean(axis=1)[:, None], distance.shape)
        if self.method == 'calendar_log':
            distance = np.broadcast_to(calendar[:, None], distance.shape)
        eligible = np.array([d != episode['context_dates'][-1] for d in self.donors])
        if self.reference:
            # The 2025 reporting distribution is prescribed separately from epidemic
            # fitting. Exclude the same seasonal release position in EVERY season,
            # including 2025, so January never loses its entire winter donor pool.
            # Only errors are transported; donor epidemic values/labels are not.
            # Adjacent releases may contain overlapping reference weeks: this is a
            # conditional reporting-process experiment, not held-out calibration.
            eligible = np.ones(len(self.donors), dtype=bool)
            eligible[np.argmin(np.minimum(delta, 1 - delta))] = False
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            raise ValueError('No reporting donors disjoint from the recipient history and future labels')
        locations = np.arange(feature.shape[-1])
        nearest = candidates[np.argsort(distance[candidates], axis=0, kind='stable')[:8]]
        weights = np.exp(-.5 * (distance[nearest, locations] - distance[nearest[0], locations]))
        cumulative = np.cumsum(weights / weights.sum(0), axis=0)
        choice = (rng.random(len(locations))[None] > cumulative).sum(0).clip(max=len(nearest)-1)
        donor = nearest[choice, locations]
        if self.method in ('synchronous_log','synchronous_phase_log'):
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
        target_errors = errors(self.errors)
        if self.reference:
            # A missing archived donor cell is never silently treated as zero revision.
            observed = self.support & self.visible
            chosen = local(observed)
            us = episode['locations'].index('US')
            for age, channel, location in zip(*np.where(episode['available'] & ~chosen)):
                if channel not in self.channels:
                    continue
                pool = candidates[observed[candidates,age,channel,location]]
                if len(pool):
                    order = pool[np.argsort(distance[pool,location],kind='stable')[:8]]
                    d = order[rng.integers(len(order))]
                    residual = self.errors[d,age,channel,location]
                else:
                    pool = candidates[observed[candidates,age,channel,us]]
                    if len(pool):
                        residual = self.errors[rng.choice(pool),age,channel,us]
                    else:
                        pool = self.errors[candidates,age,channel][observed[candidates,age,channel]]
                        if not len(pool):
                            raise ValueError(f'No genuine revision evidence at age {len(target_errors)-1-age}, channel {channel}')
                        residual = rng.choice(pool)
                target_errors[age,channel,location] = residual * multiplier
            # Donor matching and transport use only the configured pathogen inputs.
            target_errors[:,[c for c in range(6) if c not in self.channels]] = 0
        values = self.apply(episode['values'], target_errors, self.target_floor(episode))
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
