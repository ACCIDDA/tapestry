"""Marginal coverage, four-week rectangular coverage, and sampled admission totals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .quantiles import LEVELS
from .standard import quantile_scores, COVERAGE


def distribution_scores(run):
    from .ranking import views
    from chromantis.model.scenario import Scenario
    from chromantis.model.objective import LOSS_WEIGHTS
    scenario = Scenario.from_string(json.loads((Path(run)/'manifest.json').read_text())['scenario'])
    rows = []
    for history, view in views(run).items():
        for season in json.loads((view/'manifest.json').read_text())['folds']:
            with np.load(view/f'eval_{season}'/'forecasts.npz') as f:
                q, y, mask = f['quantiles'], f['truth'], f['mask'].astype(bool)
                if y.shape[1] != 4:
                    raise ValueError('Distribution diagnostics require four future weeks')
                for channel, target in [(0, 'flu_admissions'), (3, 'flu_ed')]:
                    if not LOSS_WEIGHTS[scenario.loss_weights][channel]:continue
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
                        if channel == 0 and 'flu_admission_sum_quantiles' in f:
                            summed = f['flu_admission_sum_quantiles'][:, :, li][:, valid].T
                            scored = quantile_scores(summed, truth.sum(1))
                            metrics['four_week_sum_wis'] = scored.wis.mean()
                            for coverage in COVERAGE:
                                metrics[f'four_week_sum_coverage_{coverage}'] = scored[f'covered_{coverage}'].mean()
                        rows.append(dict(history=history, season=season, target=target,
                                         location=str(loc), windows=int(valid.sum()), **metrics))
    return pd.DataFrame(rows)
