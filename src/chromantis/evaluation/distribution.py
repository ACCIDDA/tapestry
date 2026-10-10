"""Marginal coverage, four-week rectangular coverage, and sampled admission totals.

Computed on the first four problem horizons for each trained target."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .quantiles import LEVELS
from .standard import quantile_scores, COVERAGE


def distribution_scores(run):
    from .ranking import views
    from chromantis.model.scenario import Scenario
    from chromantis.problem import Problem
    manifest=json.loads((Path(run)/'manifest.json').read_text())
    scenario = Scenario.from_string(manifest['scenario'])
    problem = Problem.load(manifest['problem'])
    weights=problem.target_weights(scenario.loss_weights)
    rows = []
    for history, view in views(run).items():
        for season in json.loads((view/'manifest.json').read_text())['folds']:
            with np.load(view/f'eval_{season}'/'forecasts.npz') as f:
                q, y, mask = f['quantiles'][:, :, :4], f['truth'][:, :4], f['mask'][:, :4].astype(bool)
                if y.shape[1] < 4:
                    raise ValueError('Distribution diagnostics require at least four future weeks')
                for channel, signal in enumerate(problem.target_signals):
                    if not weights[channel]:
                        continue
                    target=signal.hub_target or signal.name
                    for li, loc in enumerate(f['locations']):
                        valid = mask[:, :, channel, li].all(1)
                        if not valid.any():
                            continue
                        pred = q[:, :, :, channel, li][:, valid]
                        truth = y[valid, :, channel, li]
                        weekly = quantile_scores(pred.reshape(len(LEVELS), -1).T, truth.reshape(-1))
                        metrics = dict(weekly_wis=weekly.wis.mean())
                        for coverage in COVERAGE:
                            k = np.flatnonzero(np.isclose(LEVELS, (1-coverage/100)/2))[0]
                            inside = (truth >= pred[k]) & (truth <= pred[-1-k])
                            metrics[f'weekly_coverage_{coverage}'] = inside.mean()
                            metrics[f'all_four_weeks_coverage_{coverage}'] = inside.all(1).mean()
                        if channel == 0 and 'target_0_sum_quantiles' in f:
                            summed = f['target_0_sum_quantiles'][:, :, li][:, valid].T
                            scored = quantile_scores(summed, truth.sum(1))
                            metrics['four_week_sum_wis'] = scored.wis.mean()
                            for coverage in COVERAGE:
                                metrics[f'four_week_sum_coverage_{coverage}'] = scored[f'covered_{coverage}'].mean()
                        rows.append(dict(history=history, season=season, target=target,
                                         location=str(loc), windows=int(valid.sum()), **metrics))
    return pd.DataFrame(rows)
