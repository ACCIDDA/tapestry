"""Check plots for one FluSight submission CSV (same layout as the InfluPaint plot50-95 PDFs).

One PDF per target, one row per location: left, the whole season with previous seasons
dashed grey (shifted by whole 52-week years onto the current season's dates); right, the last
twelve observed weeks and the forecast. Bands are the submitted 50/80/95% intervals, the line
is the median. Observations are the Hub target data in the local Hub clone (latest values,
not the reports the model saw), so it also checks the file against what the Hub will score.

    .venv/bin/python -m chromantis.production intervals production/output/2026-10-07/Chromantis/2026-10-10-ACCIDDA-Chromantis.csv

Several CSVs for the same reference date are overlaid for comparison (50% and 95% bands and
medians, one color per file, labels from --labels) and written next to the first file as
<first>-vs-<others>-<target>-plot50-95.pdf:

    .venv/bin/python -m chromantis.production intervals A.csv B.csv --labels System2 B7
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

TARGETS = {'wk inc flu hosp': ('target-hospital-admissions.csv', 'hosp', 'Weekly flu admissions'),
           'wk inc flu prop ed visits': ('target-ed-visits-prop.csv', 'ed', 'Flu share of ED visits')}
BANDS = [(0.025, 0.975, .15), (0.1, 0.9, .25), (0.25, 0.75, .4)]
COMPARE_BANDS = [(0.025, 0.975, .12), (0.25, 0.75, .3)]
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # validated categorical slots, fixed order
PAST_SEASONS = 4


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csv', type=Path, nargs='+')
    p.add_argument('--labels', nargs='+', help='One label per CSV (default: file stems)')
    p.add_argument('--hub', type=Path, default=Path('production/hubs/FluSight-forecast-hub'))
    args = p.parse_args(argv)
    subs = [pd.read_csv(f, dtype={'location': str}, parse_dates=['target_end_date', 'reference_date']) for f in args.csv]
    labels = args.labels or [f.stem for f in args.csv]
    if len(labels) != len(subs) or len({s.reference_date.iloc[0] for s in subs}) != 1:
        raise ValueError('Give one label per CSV; all CSVs must share one reference date')
    compare = len(subs) > 1
    sub = subs[0]
    reference = sub.reference_date.iloc[0]
    season_start = pd.Timestamp(reference.year if reference.month >= 8 else reference.year - 1, 8, 1)
    names = pd.read_csv(args.hub / 'auxiliary-data/locations.csv', dtype={'location': str}).set_index('location').location_name
    for target, (file, short, label) in TARGETS.items():
        forecast = sub[sub.target == target]
        if forecast.empty:
            continue
        truth = pd.read_csv(args.hub / 'target-data' / file, dtype={'location': str}, parse_dates=['date'])
        locations = sorted(forecast.location.unique(), key=lambda l: (l != 'US', l))
        fig, axes = plt.subplots(len(locations), 2, figsize=(10, 2.6 * len(locations)), squeeze=False)
        for row, location in zip(axes, locations):
            qs = []
            for other in subs:
                q = other[(other.target == target) & (other.location == location)].pivot(index='target_end_date', columns='output_type_id', values='value')
                q.columns = q.columns.astype(float).round(3)
                qs.append(q)
            observed = truth[truth.location == location].set_index('date').value.sort_index()
            current = observed[observed.index >= season_start]
            for ax, zoom in zip(row, (False, True)):
                past_max = []
                if not zoom:
                    for k in range(1, PAST_SEASONS + 1):
                        shift = pd.Timedelta(weeks=52 * k)
                        past = observed[(observed.index >= season_start - shift) & (observed.index < season_start - shift + pd.Timedelta(weeks=52))]
                        if len(past):
                            past_max.append(past.max())
                            ax.plot(past.index + shift, past.values, color='grey', ls='--', lw=.5,
                                    label='previous seasons' if k == 1 else None)
                for k, q in enumerate(qs):
                    color = COLORS[k] if compare else 'darkblue'
                    for lo, hi, alpha in (COMPARE_BANDS if compare else BANDS):
                        ax.fill_between(q.index, q[lo], q[hi], color=color, alpha=alpha, lw=0,
                                        label=f'{labels[k] + " " if compare else ""}{round((hi - lo) * 100)}% interval')
                    ax.plot(q.index, q[0.5], color=color, marker='.', lw=1, label=f'{labels[k]} median' if compare else 'median')
                shown = current[current.index >= reference - pd.Timedelta(weeks=12)] if zoom else current
                ax.plot(shown.index, shown.values, color='black', marker='.', lw=.5, ms=4, label='Hub target data')
                ax.axvline(reference, color='k', lw=.8, ls='-.')
                start = reference - pd.Timedelta(weeks=12) if zoom else season_start
                end = qs[0].index.max() + pd.Timedelta(days=4) if zoom else season_start + pd.Timedelta(weeks=52)
                ax.set_xlim(start, end)
                visible = [s[(s.index >= start) & (s.index <= end)].max() for s in (shown, *(q[0.975] for q in qs))] + past_max
                ax.set_ylim(0, 1.05 * max(v for v in visible if pd.notna(v)) or 1)
                ax.set_title(f'{names.get(location, location)}' + (' (zoom)' if zoom else ''), fontsize=9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
                ax.tick_params(labelsize=7, axis='both')
                ax.tick_params(axis='x', rotation=45)
                ax.spines[['top', 'right']].set_visible(False)
            row[0].set_ylabel(label, fontsize=8)
        axes[0, 0].legend(fontsize=7, loc='upper left')
        title = ' vs '.join(labels) if compare else args.csv[0].stem
        fig.suptitle(f'{title}: {target}, reference date {reference.date()}', y=1.0)
        fig.tight_layout()
        stem = args.csv[0].stem + ''.join(f'-vs-{l}' for l in labels[1:]) if compare else args.csv[0].stem
        path = args.csv[0].with_name(f'{stem}-{short}-plot50-{"95" if compare else "80-95"}.pdf')
        fig.savefig(path)
        plt.close(fig)
        print(path)


if __name__ == '__main__':
    main()
