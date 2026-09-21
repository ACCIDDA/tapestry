import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

res = pd.read_csv('indices_all.csv', parse_dates=['week_end'])
res = res[res.week_end >= '2021-07-01']

GEOS = [('US', 'United States'), ('nc', 'North Carolina'),
        ('ca', 'California'), ('ri', 'Rhode Island')]
ROWS = [
    ('wval_like',    'exp((x − p10) / sd)\nmedian over sites',        'log',    (0.25, 120), 1.0),
    ('pct_rank',     'percentile of x in\nsite history',              'linear', (0, 1),      0.5),
    ('robust_z',     '(x − median) / IQR\nmedian over sites',         'linear', (-2.6, 3.2), 0.0),
    ('conc_matched', 'wval_like, restricted to\nthe flowpop panel',   'log',    (0.25, 120), 1.0),
    ('flowpop_wval', 'wval_like on flowpop_lin\nsame panel as above', 'log',    (0.25, 120), 1.0),
    ('wval_popw',    'wval_like scores\npopulation-weighted median',  'log',    (0.25, 120), 1.0),
]
PATH = [('covid', 'SARS-CoV-2', '#B4472A'),
        ('flu',   'Influenza A', '#2F6EA8'),
        ('rsv',   'RSV',         '#4F8A54')]
TICKS = [0.5, 1, 2, 5, 10, 20, 50]

fig, axes = plt.subplots(len(ROWS), len(GEOS), figsize=(21, 15), sharex=True, squeeze=False)
for i, (key, formula, scale, ylim, ref) in enumerate(ROWS):
    for j, (code, title) in enumerate(GEOS):
        ax = axes[i][j]
        for pk, plabel, colour in PATH:
            d = res[(res.key == key) & (res.state == code) &
                    (res.pathogen == pk)].sort_values('week_end')
            if d.empty:
                continue
            y = d['index'].to_numpy(dtype=float).copy()
            y[(d['week_end'].diff().dt.days > 21).to_numpy()] = np.nan
            ax.plot(d['week_end'], y, color=colour, lw=1.1, label=plabel)
        ax.set_yscale(scale)
        ax.set_ylim(*ylim)
        if scale == 'log':
            ax.yaxis.set_major_locator(FixedLocator(TICKS))
            ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in TICKS]))
        ax.axhline(ref, color='0.55', lw=0.8, ls='--', zorder=0)
        ax.grid(True, color='0.92', lw=0.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9)
        if i == 0:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        if j == 0:
            ax.set_ylabel(f"{key}\n{formula}", fontsize=9.5, linespacing=1.5, color='0.15')
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncols=3, loc='upper left',
           bbox_to_anchor=(0.004, 0.955), fontsize=11)
fig.suptitle('Candidate state wastewater activity indices — Delphi NWSS snapshot 2026-09-18\n'
             'rows are definitions, columns are geographies; dashed line is each index’s neutral level',
             x=0.005, ha='left', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.925])
fig.savefig('indices_all.png', dpi=130)
print('wrote indices_all.png')
