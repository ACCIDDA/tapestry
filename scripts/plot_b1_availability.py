"""Plot archived report availability at each Wednesday cutoff; no model scoring."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

source = Path('data/processed/b1-audit/source-coverage-locations.csv')
out = Path('docs/results/b1-reporting-regimes')
f = pd.read_csv(source, parse_dates=['issuance_date'])
f = f[(f.location != 'US') & f.context_offset.isin([-1, -2])].copy()
assert f.location.nunique() == 51
assert not f.duplicated(['issuance_date', 'target', 'location', 'context_offset']).any()
# Compare the two different observation weeks visible at the SAME Wednesday.
pairs = f.pivot(index=['issuance_date', 'target', 'location'], columns='context_offset', values='version_available')
# The first model-calendar Wednesday lacks the preceding out-of-calendar week.
# Omit incomplete comparison dates rather than treating unaudited slots as absence.
incomplete = pairs[pairs.isna().any(axis=1)].reset_index()[['issuance_date', 'target']].drop_duplicates()
valid = pairs.reset_index().merge(incomplete.assign(incomplete=True), on=['issuance_date', 'target'], how='left')
pairs = valid[valid.incomplete.isna()].drop(columns='incomplete').set_index(['issuance_date', 'target', 'location'])
assert pairs.notna().all().all()
pairs = pairs.rename(columns={-1: 'latest', -2: 'preceding'})
pairs['older_only'] = (pairs.preceding == 1) & (pairs.latest == 0)
weekly = pairs.groupby(['issuance_date', 'target']).agg(
    locations=('latest', 'size'), latest=('latest', 'sum'),
    preceding=('preceding', 'sum'), older_only=('older_only', 'sum')).reset_index()
assert (weekly.locations == 51).all()
for column in ['latest', 'preceding', 'older_only']:
    weekly[column + '_pct'] = 100 * weekly[column] / 51
weekly.to_csv(out / 'wednesday-availability.csv', index=False)

fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True, sharey=True)
blue, orange = '#0072B2', '#D55E00'
for row, (pathogen, name) in enumerate([('flu', 'Influenza'), ('covid', 'COVID-19'), ('rsv', 'RSV')]):
    for col, (prefix, suffix, system) in enumerate([('nhsn', 'admissions', 'NHSN admissions'), ('nssp', 'proportion', 'NSSP ED')]):
        ax = axes[row, col]
        q = weekly[weekly.target == f'{prefix}_{pathogen}_{suffix}'].sort_values('issuance_date')
        x = q.issuance_date
        # Filled band is an exact per-location event, not a difference of percentages.
        ax.fill_between(x, 0, 100, where=q.older_only_pct > 0, step='post', color='#F0E442', alpha=.23, linewidth=0)
        ax.step(x, q.preceding_pct, where='post', color=orange, lw=2.2)
        ax.step(x, q.latest_pct, where='post', color=blue, lw=1.6)
        ax.set_title(f'{name} — {system}', loc='left', fontsize=12)
        ax.set_ylim(-4, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        if col == 0:
            ax.set_ylabel('States/DC with an archived report (%)')
        if row == 2:
            ax.set_xlabel('Wednesday forecast date')
        if row == 0 and col == 1:
            ax.annotate('Older week available;\nlatest week unavailable',
                        xy=(pd.Timestamp('2025-01-15'), 3),
                        xytext=(pd.Timestamp('2024-08-01'), 47), fontsize=10,
                        arrowprops=dict(arrowstyle='->', color='#444444'),
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
fig.suptitle('What was available when we forecast on Wednesday?', fontsize=17, y=.985)
fig.legend(handles=[Line2D([0], [0], color=blue, lw=2, label='Latest Saturday: 4 days earlier'),
                    Line2D([0], [0], color=orange, lw=2, label='Previous Saturday: 11 days earlier'),
                    Patch(facecolor='#F0E442', alpha=.3, label='Some locations have only the older week')],
           loc='upper center', bbox_to_anchor=(.5, .949), ncol=3, frameon=False)
fig.text(.5, .015, 'Hub + Git archives + Delphi, available by that Wednesday. Supplied finals are excluded.\n0% means no eligible archived report, not zero disease or zero revision; archive gaps may differ from publication gaps.',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0, .06, 1, .90))
fig.savefig(out / 'wednesday-availability.png', dpi=170)
plt.close(fig)
(out / 'wednesday-availability-manifest.json').write_text(json.dumps({
    'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'assumptions': [
        'X axis is Wednesday issuance, comparing latest and preceding observation weeks at the same cutoff.',
        'Availability is any eligible archived report in the provider audit (Hub/Git/Delphi), before B1 precedence; no final-label eligibility requirement.',
        'Dates missing either age in the audit are omitted, including the first calendar boundary; they are not counted as unavailable.',
        'Denominator is all 51 states/DC; US excluded. Supplied-final fallbacks do not count as archived reports.',
        'Yellow means at least one same location has an 11-day report but no 4-day report on that Wednesday.',
        'Solid step lines carry each weekly status to the next Wednesday for display; no claim about intervening release days.',
        'Archive availability cannot establish actual publication absence. No visual inspection or model scoring.'
    ]}, indent=2) + '\n')
print('Wrote', out / 'wednesday-availability.png')
