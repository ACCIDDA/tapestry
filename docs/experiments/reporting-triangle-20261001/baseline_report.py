"""Summarize the stored v11 point-baseline comparison without rerunning fits."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path('data/experiments/reporting-triangle-v11-20261001')
OUTPUT = Path(__file__).parent
scores = pd.read_csv(ROOT / 'finalization-ranking/signal-scores.csv')
reported = scores[scores.kind == 'reported']
# Fold/seed means within signal and age, then equal signal/age weights.
means = reported.groupby(['cv', 'signal', 'age', 'method']).normalized_mae.mean()
summary = means.groupby(['cv', 'method']).mean().unstack()
summary['model_to_baseline'] = summary.prediction / summary.baselinenowcast
summary['reduction_pct'] = 100 * (1 - summary.model_to_baseline)
summary.to_csv(OUTPUT / 'baselinenowcast-summary.csv')
coverage = []
for path in sorted(ROOT.glob('*/s42/attempt-002/eval_*/baselinenowcast_scores.json')):
    counts = json.loads(path.read_text())['status_counts']
    coverage.append(dict(fold=path.parent.name.removeprefix('eval_'), **counts))
pd.DataFrame(coverage).to_csv(OUTPUT / 'baselinenowcast-coverage.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
for ax, cv in zip(axes[0], ['rolling', 'season']):
    by_age = means.loc[cv].groupby(['age', 'method']).mean().unstack()
    for method, label in [('prediction', 'Chromantis'), ('baselinenowcast', 'baselinenowcast point'),
                          ('persistence', 'Unchanged report')]:
        ax.plot(by_age.index, by_age[method], marker='o', markersize=3, label=label)
    ax.set(title=cv.capitalize(), xlabel='Weeks behind signal boundary', ylabel='Normalized MAE')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('Reported cells: equal signal and location weights; lower is better')
fig.tight_layout()
fig.savefig(OUTPUT / 'baselinenowcast-comparison.png', dpi=170)
plt.close(fig)
print(summary.to_string())
