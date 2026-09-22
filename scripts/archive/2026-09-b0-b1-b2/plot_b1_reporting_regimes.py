"""Describe B1 vintage revisions by report age and reporting regime; no model scoring."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from tapestry.model_data.wednesday import WednesdayDataset
from tapestry.model_data.finalized import CHANNELS, season
from datetime import date

OUT = Path('docs/results/b1-reporting-regimes')
OUT.mkdir(parents=True, exist_ok=True)
PATH = Path('data/processed/build_b1_wednesday_calendar.npz')
ds = WednesdayDataset.load(PATH)
a = ds.arrays
rows = []
for i, issuance in enumerate(a['issuance_dates']):
    for h in range(2):
        day = str(a['target_dates'][i, h])
        for c, target in enumerate(CHANNELS):
            for l, location in enumerate(ds.locations):
                valid = bool(a['Y_recent_valid'][i, h, c, l])
                available = bool(a['X_available'][i, h-2, c, l])
                supplied = bool(a['X_final'][i, h-2, c, l])
                source = ds.metadata['provenance'][int(a['X_provenance'][i, h-2, c, l])]['source']
                rows.append(dict(date=day, issuance=str(issuance), season=season(date.fromisoformat(day)),
                                 target=target, age=11-7*h, location=location,
                                 geography='US' if location == 'US' else 'states',
                                 valid=valid, reported=valid and available and not supplied,
                                 source=source, report=float(a['X_values'][i, h-2, c, l]),
                                 final=float(a['Y_recent'][i, h, c, l])))
f = pd.DataFrame(rows)
f['date'] = pd.to_datetime(f.date)
f['regime'] = np.select([f.date < '2024-05-01', f.date < '2024-11-01'],
                        ['Previous regime', 'Voluntary period'], default='New mandate')
f['delta'] = f.final - f.report
# Pair ages on exactly the same observation/location and selected archive provider.
r = f[f.reported].copy()
keys = ['date', 'target', 'location', 'source']
common = r[r.age == 4][keys].merge(r[r.age == 11][keys], on=keys, validate='one_to_one')
paired = r.merge(common, on=keys, validate='many_to_one')

def stats(g):
    total = g.final.sum()
    return dict(cells=len(g), weeks=g.date.nunique(), locations=g.location.nunique(),
                final_total=total,
                net_pct=100*g.delta.sum()/total if total > 0 else np.nan,
                absolute_pct=100*g.delta.abs().sum()/total if total > 0 else np.nan)

def summarize(frame, keys):
    return pd.DataFrame([{**dict(zip(keys, k)), **stats(g)} for k, g in frame.groupby(keys)])

weekly_all = summarize(r, ['date', 'target', 'age', 'geography'])
weekly = summarize(paired, ['date', 'target', 'age', 'geography'])
summary = summarize(paired, ['regime', 'season', 'target', 'age', 'geography'])
source = summarize(paired, ['regime', 'season', 'target', 'age', 'geography', 'source'])
coverage = f.groupby(['date', 'season', 'regime', 'target', 'age', 'geography']).agg(
    eligible_finals=('valid', 'sum'), genuine_reports=('reported', 'sum')).reset_index()
paired_counts = paired.groupby(['date', 'target', 'age', 'geography']).size().rename('paired_reports').reset_index()
coverage = coverage.merge(paired_counts, on=['date', 'target', 'age', 'geography'], how='left')
coverage['paired_reports'] = coverage.paired_reports.fillna(0).astype(int)
for name, frame in [('weekly-paired', weekly), ('weekly-all-reports', weekly_all), ('regime-season-summary', summary),
                    ('source-summary', source), ('coverage', coverage)]:
    frame.to_csv(OUT / f'{name}.csv', index=False)

# NHSN left, NSSP right; separate pathogens and ages. No smoothing or gap filling.
colors = {4: '#0072B2', 11: '#D55E00'}
targets = [(next(t for t in CHANNELS if 'nhsn' in t and p in t),
            next(t for t in CHANNELS if 'nssp' in t and p in t))
           for p in ['flu', 'covid', 'rsv']]
calendar = pd.DatetimeIndex(sorted(f.date.unique()))
start, end = calendar.min(), calendar.max()

for metric, ylabel, title, filename in [
    ('net_pct_all', 'Net revision / final total (%)', 'All genuine reports: each age on its available support', 'net-revisions-all-reports.png'),
    ('net_pct', 'Net revision / final total (%)', 'Signed revisions: how much is added to the preliminary report?', 'net-revisions.png'),
    ('absolute_pct', 'Absolute revision / final total (%)', 'Absolute revisions: how much changes, regardless of direction?', 'absolute-revisions.png'),
    ('coverage', 'Locations with genuine report (%)', 'Available preliminary reports in the B1 archive', 'report-coverage.png'),
]:
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    for row, (p, pair) in enumerate(zip(['Influenza', 'COVID-19', 'RSV'], targets)):
        for col, target in enumerate(pair):
            ax = axes[row, col]
            q = coverage if metric == 'coverage' else (weekly_all if metric == 'net_pct_all' else weekly)
            q = q[(q.target == target) & (q.geography == 'states')]
            offscale = None
            for age in [4, 11]:
                v = q[q.age == age].set_index('date').reindex(calendar)
                y = 100*v.genuine_reports/51 if metric == 'coverage' else v['net_pct' if metric == 'net_pct_all' else metric]
                # Display-only omission requested by the user; retain the data in all summaries.
                if target == 'nhsn_covid_admissions' and age == 4 and metric != 'coverage':
                    outlier_date = pd.Timestamp('2024-11-16')
                    offscale = (outlier_date, float(y.loc[outlier_date]))
                    y = y.copy()
                    y.loc[outlier_date] = np.nan
                ax.plot(calendar, y, color=colors[age], lw=1.3, ls='-' if age == 4 else '--')
            if metric == 'coverage':
                v = q[q.age == 4].set_index('date').reindex(calendar)
                ax.plot(calendar, 100*v.eligible_finals/51, color='#666666', lw=1, ls=':')
                ax.set_ylim(-3, 103)
            else:
                ax.axhline(0, color='#777777', lw=.6)
            if offscale is not None:
                outlier_date, value = offscale
                lower, upper = ax.get_ylim()
                ax.set_ylim(lower, upper)
                boundary = lower if value < lower else upper
                ax.annotate(f'Nov 16, 2024: {value:+.1f}% (off scale)',
                            xy=(outlier_date, boundary),
                            xytext=(18, 34 if value < lower else -34),
                            textcoords='offset points', color=colors[4], fontsize=9,
                            va='bottom' if value < lower else 'top',
                            arrowprops=dict(arrowstyle='->', color=colors[4], lw=1.4),
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
            if col == 0:
                ax.axvspan(pd.Timestamp('2024-05-01'), pd.Timestamp('2024-11-01'), color='#E69F00', alpha=.16)
                for d in ['2024-05-01', '2024-11-01']:
                    ax.axvline(pd.Timestamp(d), color='#777777', lw=.9)
            for d in ['2024-08-01', '2025-08-01', '2026-08-01']:
                ax.axvline(pd.Timestamp(d), color='#bbbbbb', lw=.6, ls=':')
            ax.set_title(f'{p} — {"NHSN admissions" if col == 0 else "NSSP ED proportions"}')
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', alpha=.2)
            ax.set_xlim(start, end)
    handles = [Line2D([0], [0], color=colors[4], label='Latest week: 4 days old'),
               Line2D([0], [0], color=colors[11], ls='--', label='Preceding week: 11 days old')]
    if metric == 'coverage':
        handles.append(Line2D([0], [0], color='#666666', ls=':', label='Locations with reference final (4-day rows)'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .935), ncol=3)
    subtitle = ('All 51 states/DC locations; archive availability is not hospital participation.' if metric == 'coverage'
                else ('States/DC; each age has its own support. Gaps mean unavailable reports, not zero revision.'
                      if metric == 'net_pct_all' else
                      'States/DC; ages matched by observation week, location and provider. Supplied finals excluded.'))
    fig.suptitle(title + '\n' + subtitle, fontsize=14, y=.99)
    fig.text(.5, .012, 'Observation week-ending date · NHSN orange: voluntary period, May–October 2024; new mandate from November 1 (user-provided boundaries).', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .89))
    fig.savefig(OUT / filename, dpi=170)
    plt.close(fig)

metadata = dict(dataset=str(PATH), sha256=hashlib.sha256(PATH.read_bytes()).hexdigest(),
                truth_cutoff=ds.metadata['truth_cutoff'], paired_report_cells=len(paired),
                regime_boundaries_source='User instruction, 2026-09-18; not independently verified in this analysis.',
                assumptions=[
                    'Regime assigned by observation week-ending date; boundary weeks can straddle policies.',
                    'NHSN boundaries are not NSSP policy boundaries; NSSP tables use these dates only for comparison.',
                    'Matched plots pair ages on observation/location/provider; all-reports plot uses each age own support. Unavailable dates remain gaps.',
                    'Net = 100 sum(final-report)/sum(final); absolute = 100 sum(abs(final-report))/sum(final).',
                    'States/DC pooled; US separate in CSVs. Admissions use count mass; ED sums equally weighted location proportions, not national ED visits.',
                    'Coverage = locations with genuine report and final / 51; it measures B1 archive availability, not hospital participation.',
                    'Pinned reference finals may revise. Zero finals retained; zero total gives undefined metric.',
                    'No causal test of reporting regime or effect on trained models. Source selection and calendar support may change.',
                    'COVID admissions age-4 outlier on 2024-11-16 omitted from plotted lines and axis scaling, annotated with its value; retained in all CSVs and summaries.',
                    'No visual inspection of generated graphs; no training or model scoring.'
                ])
(OUT / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Canonical targets:', list(CHANNELS))
print(summary[(summary.geography == 'states') & summary.target.str.startswith('nhsn')][
    ['regime', 'season', 'target', 'age', 'cells', 'weeks', 'net_pct', 'absolute_pct']].round(2).to_string(index=False))
print('Wrote', OUT)
