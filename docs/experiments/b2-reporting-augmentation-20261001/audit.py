"""Describe revision distributions and synthetic histories, without fitting or scoring."""
from dataclasses import replace
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chromantis.dataset.build import load
from chromantis.dataset import cv
from chromantis.dataset.reporting_error import ReportingErrors
from chromantis.model.scenario import Scenario

ROOT = Path(__file__).parent
panel = load('data/processed/panel.npz')
scenario = replace(Scenario.from_string(pd.read_csv('docs/experiments/b-2-t0/named_ranking.csv').iloc[0].config_id),
                   supplied_final=False, reporting_augmentation='vintage')
dates = panel['dates'].astype(str)
rows = []
for season in ('2023-2024', '2024-2025', '2025-2026'):
    in_season = np.array([cv.season(d) == season for d in dates])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        peak = np.nanmax(panel['targets'][in_season], axis=0)
    for k, name in enumerate(panel['target_names']):
        for age in (0, 1, 2, 4, 7, 11):
            error, missing, support = [], [], []
            for wi, issue in enumerate(panel['issuance_dates']):
                t = np.searchsorted(dates, str(np.datetime64(issue) - np.timedelta64(4, 'D'))) - age
                if t < 0 or t >= len(dates) or not in_season[t]:
                    continue
                final, raw = panel['targets'][t, :, k], panel['asof_targets'][wi, t, :, k]
                ok = np.isfinite(final)
                missing.extend((~np.isfinite(raw[ok])).tolist())
                paired = ok & np.isfinite(raw)
                scale = np.maximum(np.abs(final), .05 * peak[:, k])
                error.extend(((raw[paired] - final[paired]) / np.maximum(scale[paired], 1e-8)).tolist())
                support.append(int(paired.sum()))
            q = np.quantile(error, [.1, .5, .9]) if error else [np.nan]*3
            rows.append(dict(season=season, signal=name, age=age, cells=len(error),
                missing=np.mean(missing) if missing else np.nan, q10=q[0], median=q[1], q90=q[2]))
frame = pd.DataFrame(rows)
frame.to_csv(ROOT / 'revision-distributions.csv', index=False)
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for ax, name in zip(axes.flat, panel['target_names']):
    for season, color in [('2024-2025', '#d88729'), ('2025-2026', '#247a9c')]:
        d = frame[(frame.signal == name) & (frame.season == season)]
        ax.plot(d.age, d['median'], color=color, label=season)
        ax.fill_between(d.age, d.q10, d.q90, color=color, alpha=.18)
    ax.axhline(0, color='grey', lw=.7)
    ax.set_title(str(name).replace('_', ' '), fontsize=10)
    ax.set_xlabel('Weeks before latest context week')
    ax.set_ylabel('Relative error (peak floor 5%)')
axes[0, 0].legend()
fig.suptitle('Real revision patterns: median and 10–90% range of reported cells')
fig.tight_layout(); fig.savefig(ROOT / 'revision-distributions.png', dpi=160); plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True, sharey=True)
for ax, name in zip(axes.flat, panel['target_names']):
    for season in ('2024-2025', '2025-2026'):
        d = frame[(frame.signal == name) & (frame.season == season)]
        ax.plot(d.age, 100*d.missing, marker='o', label=season)
    ax.set_title(str(name).replace('_', ' '), fontsize=10)
    ax.set_xlabel('Weeks before latest context week'); ax.set_ylabel('Missing reports (%)')
axes[0, 0].legend(); fig.tight_layout(); fig.savefig(ROOT / 'reporting-availability.png', dpi=160); plt.close(fig)
held = '2025-2026'
roles = cv.week_roles(dates, scenario, held)
bank = ReportingErrors(panel, scenario, held, np.isin(roles, ['fit', 'validation']))
fold = cv.fold(panel, scenario, held)
rng = np.random.default_rng(42)
li = list(panel['locations']).index('US')
old = [e for e in fold.train if cv.season(e['context_dates'][-1]) == '2023-2024']
peak = max(old, key=lambda e: e['values'][-1, 0, li])
center = next(i for i,e in enumerate(old) if e is peak)
chosen = [old[max(0,center-5)], peak, old[min(len(old)-1,center+5)]]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, e in zip(axes, chosen):
    draws = [bank.draw(e, rng) for _ in range(80)]
    for draw in draws[:20]:
        values = np.where(draw['available'][:,0,li],draw['values'][:,0,li],np.nan)
        ax.plot(range(-11,1), values, alpha=.15, color='#247a9c')
        assert np.array_equal(draw['target_values'], e['target_values'])
        assert np.array_equal(draw['target_available'], e['target_available'])
    ax.plot(range(-11,1), e['values'][:,0,li], color='black', lw=2, label='Final history')
    ax.set_title('US flu · origin ' + e['context_dates'][-1]); ax.set_xlabel('Context week')
axes[0].set_ylabel('Weekly admissions'); axes[0].legend()
fig.suptitle('Stochastic historical inputs around the epidemic peak; labels unchanged')
fig.tight_layout(); fig.savefig(ROOT/'synthetic-peak-windows.png',dpi=160); plt.close(fig)
print(frame[(frame.age==0)&(frame.season!='2023-2024')].to_string(index=False))
print('Donor windows:', len(bank.donors), bank.donor_season)
