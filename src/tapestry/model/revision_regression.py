"""Shrunk log-revision regression with age, local phase and calendar predictors.

Fit only on synthetic reports of permitted training trajectories. No finalized
level, peak date or future growth is used as a prediction feature.
"""
import numpy as np
from tapestry.dataset.cv import season
from tapestry.dataset.episodes import calendar


class RevisionRegression:
    def __init__(self, weeks=4, penalty=10., strength=1., features='phase'):
        self.weeks, self.penalty, self.strength, self.features = weeks, penalty, strength, features

    def design(self, e, channel):
        value = e['values'][:, channel].astype(float)
        valid = e['available'][:, channel]
        unit = 1. if channel < 3 else .0001
        scale = np.maximum(np.max(np.where(valid, value, 0), axis=0), unit)
        floor = np.maximum(.05 * scale, unit)
        log = np.log((value + floor) / (scale + floor))
        recent = np.where(valid[-4:], log[-4:], 0)
        level = recent.mean(0)
        growth = (recent[-1] + recent[-2] - recent[-3] - recent[-4]) / 2
        acceleration = recent[-1] - 2 * recent[-2] + recent[-3]
        cal = calendar([e['context_dates'][-1]])[0]
        rows = []
        for age in range(self.weeks):
            age_hot = np.broadcast_to(np.eye(self.weeks)[age], (value.shape[1], self.weeks))
            parts = [age_hot]
            if self.features != 'age':
                phase = np.stack((level, growth, acceleration, valid[-4:].mean(0),
                                  np.full(len(level), cal[0]), np.full(len(level), cal[1]),
                                  np.full(len(level), np.exp(-abs(cal[2]) * 26 / 2))), axis=1)
                parts.append((age_hot[:, :, None] * phase[:, None, :]).reshape(len(level), -1))
            if self.features == 'phase_local':
                # Location effects shrink toward the shared age/phase correction.
                parts.append(np.eye(value.shape[1]))
            rows.append(np.concatenate(parts, axis=1))
        return np.stack(rows), floor

    def fit(self, examples):
        self.coefficients = []
        self.training_seasons = sorted({season(truth['context_dates'][-1]) for _, truth in examples})
        for k in range(6):
            xs, ys, weights = [], [], []
            seasons = [season(t['context_dates'][-1]) for _, t in examples]
            for (reported, truth), label in zip(examples, seasons):
                x, floor = self.design(reported, k)
                indices = np.arange(-1, -self.weeks - 1, -1)
                ok = reported['available'][indices, k] & truth['available'][indices, k]
                y = np.log((truth['values'][indices, k] + floor) / (reported['values'][indices, k] + floor))
                xs.append(x[ok]); ys.append(y[ok])
                weights.append(np.full(ok.sum(), 1. / seasons.count(label)))
            x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
            if not len(y):
                self.coefficients.append(np.zeros(x.shape[-1])); continue
            w *= len(w) / w.sum()
            ridge = np.eye(x.shape[1]) * self.penalty
            self.coefficients.append(np.linalg.solve(x.T @ (w[:, None] * x) + ridge, x.T @ (w * y)))
        return self

    def apply(self, e):
        values = e['values'].copy()
        for k in range(6):
            x, floor = self.design(e, k)
            correction = np.clip(x @ self.coefficients[k], -np.log(4), np.log(4)) * self.strength
            indices = np.arange(-1, -self.weeks - 1, -1)
            values[indices, k] = np.maximum(0, (values[indices, k] + floor) * np.exp(correction) - floor)
        values[:, 3:] = np.minimum(1, values[:, 3:])
        return dict(e, values=np.where(e['available'], values, 0).astype(np.float32),
                    known_final=np.zeros_like(e['available']))

    def record(self):
        return dict(weeks=self.weeks, penalty=self.penalty, strength=self.strength, features=self.features,
                    training_seasons=self.training_seasons, coefficients=[x.tolist() for x in self.coefficients])
