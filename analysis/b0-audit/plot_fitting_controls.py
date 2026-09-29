"""Plots from completed fitting controls; no image inspection."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

out = Path('docs/results/b0-fitting-controls')
labels = {'current': 'Current validation', 'split': 'B0 validation weeks',
          'draws': 'B0 validation randomness', 'split-draws': 'Both B0 changes'}
scores = pd.read_csv(out / 'run_scores.csv').query("geography == 'all'")
seasons = pd.read_csv(out / 'season_composite_scores.csv').query("geography == 'all'")
old = pd.read_csv('docs/results/b0-reproduction/run_scores.csv').query(
    "geography == 'all' and config_id == 'old-training-l40'")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, frame, title in [(axes[0], scores, 'All three seasons'),
                         (axes[1], seasons[seasons.season.eq('2025-2026')], '2025–26 held-out season')]:
    for seed, part in frame.groupby('seed'):
        values = part.set_index('config_id').loc[list(labels), 'combined']
        ax.plot(range(4), values, 'o-', alpha=.65, linewidth=1.2, label=f'Seed {seed}')
    mean = frame.groupby('config_id').combined.mean().loc[list(labels)]
    ax.plot(range(4), mean, 'D--', color='black', label='Mean', linewidth=2)
    for i, value in enumerate(mean):
        ax.annotate(f'{value:.3f}', (i, value), xytext=(0, 10), textcoords='offset points', ha='center')
    ax.axhline(1, color='.6', linestyle=':', label='Ensemble parity')
    ax.set_xticks(range(4), ['Current', 'B0 weeks', 'B0 randomness', 'Both'], rotation=15)
    ax.set_title(title)
    ax.set_ylabel('Relative WIS (lower is better)')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].axhline(old.combined.mean(), color='#777777', linestyle='--', label='Original B0 / L40')
axes[0].legend(fontsize=8, frameon=False, loc='best')
fig.suptitle('Fitting controls · normalized finalized inputs · loss multiplier unchanged')
fig.tight_layout()
fig.savefig(out / 'fitting-controls.png', dpi=180)
fig.savefig(out / 'fitting-controls.svg')
plt.close(fig)

cells = pd.read_csv(out / 'season_scores.csv').query("geography == 'all'")
wide = cells.groupby(['season', 'target', 'config_id']).wis_ratio.mean().unstack().loc[:, list(labels)]
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(wide.to_numpy(), cmap='RdBu_r', vmin=.7, vmax=1.3, aspect='auto')
ax.set_xticks(range(4), ['Current', 'B0 weeks', 'B0 randomness', 'Both'])
ax.set_yticks(range(len(wide)), [f"{s} · {t.replace('wk inc ', '').replace('prop ed visits','ED').replace('hosp','admissions')}" for s, t in wide.index])
for i in range(len(wide)):
    for j in range(4):
        value = wide.iloc[i, j]
        ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='white' if value < .78 or value > 1.22 else 'black')
fig.colorbar(im, ax=ax, label='Relative WIS (color range 0.7–1.3)', shrink=.7)
ax.set_title('Fitting controls · mean of three seeds on every supported target–season')
fig.tight_layout()
fig.savefig(out / 'target-season.png', dpi=180)
plt.close(fig)
print(out / 'fitting-controls.png')
