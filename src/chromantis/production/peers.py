"""Plot a submission against same-week Hub forecasts and the preceding season.

One PDF page and PNG per location: admissions/ED rows, full season/zoom columns.
Only saved forecasts are read; no fitting, forecasting, scoring or publication.
Example: python -m chromantis.production peers forecast.csv candidate.csv
"""
import argparse
from datetime import date, timedelta, datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

DEFAULT_LOCATIONS = ['US', 'CA', 'TX', 'NY', 'NC', 'IA']
DEFAULT_PEERS = ['Google_SAI-FluEns', 'OHT_JHU-nbxd', 'CMU-TimeSeries',
                 'UGA_flucast-INFLAenza', 'UMass-flusion', 'NAU-vulPES']
TARGETS = {'wk inc flu hosp': ('target-hospital-admissions.csv', 'Weekly flu admissions', 1),
           'wk inc flu prop ed visits': ('target-ed-visits-prop.csv', 'Flu ED visits (%)', 100)}
KEYS = ['reference_date', 'target', 'horizon', 'target_end_date', 'location']
COLORS = ['#c33f30', '#009999', '#7854a5', '#ed8c20']


def week_one(year):
    jan4 = date(year, 1, 4)
    return jan4 - timedelta(days=(jan4.weekday()+1) % 7)


def week_id(day):
    year = day.year
    if day < week_one(year):
        year -= 1
    elif day >= week_one(year+1):
        year += 1
    return year, (day-week_one(year)).days//7+1


def align_previous(day):
    year, week = week_id(day.date())
    start = week_one(year+1) + timedelta(weeks=week-1)
    return pd.NaT if start >= week_one(year+2) else pd.Timestamp(start+timedelta(days=6))


def read_medians(raw, suffix, reference=None):
    d = pd.read_parquet(BytesIO(raw)) if suffix == '.parquet' else pd.read_csv(BytesIO(raw), dtype={'location': str})
    d = d[(d.output_type == 'quantile') & d.target.isin(TARGETS)].copy()
    d['output_type_id'] = pd.to_numeric(d.output_type_id)
    d['horizon'] = pd.to_numeric(d.horizon)
    d = d[(d.output_type_id == .5) & d.horizon.isin([0, 1, 2, 3])].copy()
    for key in ['reference_date', 'target_end_date']:
        d[key] = pd.to_datetime(d[key])
    if reference is not None and not d.reference_date.eq(reference).all():
        raise ValueError('Comparison forecasts must have the same reference date')
    d['location'] = d.location.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(2)
    d['value'] = pd.to_numeric(d.value)
    if d.duplicated(KEYS).any():
        raise ValueError('Duplicate forecast tasks')
    if not (d.target_end_date == d.reference_date + pd.to_timedelta(d.horizon*7, unit='D')).all():
        raise ValueError('Forecast horizon/date mismatch')
    if not np.isfinite(d.value).all() or (d.value < 0).any() or (d.loc[d.target == 'wk inc flu prop ed visits', 'value'] > 1).any():
        raise ValueError('Invalid forecast values: require nonnegative counts and ED proportions in [0, 1]')
    return d[KEYS+['value']]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', nargs='+', type=Path, help='Primary forecast, then optional local candidates')
    p.add_argument('--labels', nargs='+', help='One plot label per local forecast')
    p.add_argument('--hub', type=Path, default=Path('production/hubs/FluSight-forecast-hub'))
    p.add_argument('--hub-ref', default='HEAD', help='Local Hub commit/ref; no network fetch is performed')
    p.add_argument('--out', type=Path, help='Default: <primary CSV stem>-comparison/')
    p.add_argument('--locations', nargs='+', default=DEFAULT_LOCATIONS, help='Abbreviations/FIPS, or all')
    p.add_argument('--peer-models', nargs='*', default=DEFAULT_PEERS, help='Named Hub models drawn individually')
    p.add_argument('--peer-csv', action='append', type=Path, default=[], help='Additional peer CSV/Parquet, e.g. a downloaded pending submission')
    p.add_argument('--zoom-weeks', type=int, default=8, help='Weeks before reference date shown in zoom')
    args = p.parse_args(argv)
    if args.zoom_weeks < 1 or (args.labels and len(args.labels) != len(args.csv)):
        p.error('Use a positive zoom window and one label per local CSV')
    labels = args.labels or [f.stem[11:] if len(f.stem) > 11 and f.stem[10] == '-' else f.stem for f in args.csv]
    if len(set(labels)) != len(labels):
        p.error('Local forecast labels must be unique; use --labels')
    out = args.out or args.csv[0].with_name(args.csv[0].stem+'-comparison')
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *cmd: subprocess.check_output(['git', '-C', str(args.hub), *cmd])
    commit = git('rev-parse', f'{args.hub_ref}^{{commit}}').decode().strip()
    sources, frames, warnings = [], [], []
    reference = None

    def record(path, raw, role, model):
        sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), role=role, model=model))

    for path, label in zip(args.csv, labels):
        raw = path.read_bytes()
        d = read_medians(raw, path.suffix, reference)
        if d.empty or d.reference_date.nunique() != 1:
            raise ValueError(f'{path}: need one nonempty reference date')
        reference = d.reference_date.iloc[0]
        d['model'], d['role'] = label, 'local'
        frames.append(d)
        record(path.resolve(), raw, 'local', label)
        companion = path.with_suffix('.json')
        if companion.exists():
            meta = json.loads(companion.read_text())
            if meta.get('output_sha256') and meta['output_sha256'] != sources[-1]['sha256']:
                warnings.append(f'{label}: companion JSON hash does not match; model description not reused.')
            else:
                sources[-1]['definition'] = meta.get('definition')
                sources[-1]['operational_panel_sha256'] = meta.get('operational_panel_sha256')
    if reference.dayofweek != 5:
        raise ValueError('FluSight reference date must be Saturday')
    local_models = {path.stem[11:] for path in args.csv}
    local_hashes = {s['sha256'] for s in sources}
    peer_frames = {}

    def add_peer(path, raw, model):
        if model in local_models or hashlib.sha256(raw).hexdigest() in local_hashes:
            return
        d = read_medians(raw, Path(path).suffix, reference)
        if d.empty:
            return
        if model in peer_frames:
            raise ValueError(f'Multiple peer files for {model}; choose a Hub ref without that model or remove --peer-csv')
        d['model'], d['role'] = model, 'peer'
        peer_frames[model] = d
        record(path, raw, 'peer', model)

    paths = git('ls-tree', '-r', '--name-only', commit, 'model-output').decode().splitlines()
    for path in paths:
        f = Path(path)
        if f.suffix in ('.csv', '.parquet') and f.name.startswith(f'{reference.date()}-'):
            add_peer(path, git('show', f'{commit}:{path}'), f.parent.name)
    for path in args.peer_csv:
        if not path.name.startswith(f'{reference.date()}-'):
            raise ValueError(f'Additional peer filename must be {reference.date()}-TEAM-MODEL.csv or .parquet: {path}')
        add_peer(path.resolve(), path.read_bytes(), path.stem[11:])
    frames.extend(peer_frames.values())
    forecasts = pd.concat(frames, ignore_index=True)
    featured = [m for m in args.peer_models if m in peer_frames]
    missing_peers = [m for m in args.peer_models if m not in peer_frames]
    if not peer_frames:
        warnings.append('No same-week peer forecasts are available; peer bands are omitted. Refresh the Hub clone and rerun later.')
    if missing_peers:
        warnings.append('Requested peer curves unavailable: '+', '.join(missing_peers))

    def hub_table(path):
        raw = git('show', f'{commit}:{path}')
        record(path, raw, 'history' if path.startswith('target-data') else 'locations', None)
        return pd.read_csv(BytesIO(raw), dtype={'location': str})

    locations = hub_table('auxiliary-data/locations.csv')
    lookup = dict(zip(locations.abbreviation, locations.location))
    names = locations.set_index('location').location_name.to_dict()
    selected = sorted(frames[0].location.unique(), key=lambda x: (x != 'US', x)) if args.locations == ['all'] else list(dict.fromkeys(lookup.get(x, x) for x in args.locations))
    if set(selected)-set(names):
        raise ValueError(f'Unknown locations: {set(selected)-set(names)}')
    year = reference.year
    season_start = pd.Timestamp(week_one(year)+timedelta(weeks=30))
    if reference < season_start:
        year -= 1
        season_start = pd.Timestamp(week_one(year)+timedelta(weeks=30))
    previous_start = pd.Timestamp(week_one(year-1)+timedelta(weeks=30))
    season_end = pd.Timestamp(week_one(year+1)+timedelta(weeks=30))-pd.Timedelta(days=1)
    cutoff, issuance = reference-pd.Timedelta(weeks=1), reference-pd.Timedelta(days=3)
    zoom_start = max(season_start, reference-pd.Timedelta(weeks=args.zoom_weeks))
    zoom_end = forecasts.target_end_date.max()+pd.Timedelta(days=3)
    histories = []
    targets = [t for t in TARGETS if t in forecasts[forecasts.role == 'local'].target.unique()]
    for target in targets:
        d = hub_table('target-data/'+TARGETS[target][0])
        d['date'] = pd.to_datetime(d.date)
        d = d[d.location.isin(selected)].copy()
        d['target'] = target
        for kind, start, end in [('previous', previous_start, season_start-pd.Timedelta(days=1)), ('current', season_start, cutoff)]:
            h = d[d.date.between(start, end)].copy()
            h['season'] = kind
            h['plot_date'] = h.date.map(align_previous) if kind == 'previous' else h.date
            histories.append(h[['location', 'target', 'season', 'date', 'plot_date', 'value']])
    history = pd.concat(histories, ignore_index=True)
    forecasts.to_csv(out/'forecast-medians.csv', index=False)
    history.to_csv(out/'history.csv', index=False)
    positions = []
    for _, r in forecasts[(forecasts.role == 'local') & forecasts.location.isin(selected)].iterrows():
        peers = forecasts[(forecasts.role == 'peer') & (forecasts.target == r.target) & (forecasts.location == r.location) & (forecasts.horizon == r.horizon)].value
        positions.append(dict(model=r.model, target=r.target, location=r.location, horizon=int(r.horizon), median=r.value,
                              peers=len(peers), peer_median=peers.median(),
                              percentile=100*((peers < r.value).sum()+.5*(peers == r.value).sum())/len(peers) if len(peers) else None))
    pd.DataFrame(positions).to_csv(out/'positions.csv', index=False)

    model_styles = {label: (COLORS[i % len(COLORS)], '--' if i else '-', 2.8) for i, label in enumerate(labels)}
    palette = plt.get_cmap('tab10')
    model_styles.update({m: (palette(i), '-', 1.5) for i, m in enumerate(featured)})
    with PdfPages(out/'comparison.pdf') as pdf:
        for location in selected:
            fig, axes = plt.subplots(len(targets), 2, figsize=(16, 6+4*len(targets)), squeeze=False,
                                     gridspec_kw={'width_ratios':[1, 1.2]})
            legend = {}
            for row, target in enumerate(targets):
                _, ylabel, factor = TARGETS[target]
                h = history[(history.location == location) & (history.target == target)]
                z = forecasts[(forecasts.location == location) & (forecasts.target == target)]
                peers = z[z.role == 'peer']
                limits = peers.groupby('target_end_date').value.quantile([.1,.5,.9]).unstack() if len(peers) else pd.DataFrame()
                for col, ax in enumerate(axes[row]):
                    start, end = (season_start, season_end) if col == 0 else (zoom_start, zoom_end)
                    for kind, color, style, label in [('previous','#95613e',':',f'{year-1}–{str(year)[2:]} observed, aligned by CDC week'),
                                                      ('current','#222222','-',f'{year}–{str(year+1)[2:]} reported (preliminary)')]:
                        d = h[h.season == kind].dropna(subset=['plot_date']).set_index('plot_date').sort_index()
                        if len(d):
                            d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq='7D'))
                            ax.plot(d.index, d.value*factor, color=color, ls=style, lw=2.3, label=label)
                    if len(limits):
                        ax.fill_between(limits.index, limits[.1]*factor, limits[.9]*factor, color='#d5d9df', alpha=.7, label='10th–90th percentile of peer medians')
                        ax.plot(limits.index, limits[.5]*factor, color='#777777', ls='--', label='Median of peer medians')
                    for model, (color, style, width) in model_styles.items():
                        d = z[z.model == model].sort_values('target_end_date')
                        if len(d):
                            ax.plot(d.target_end_date, d.value*factor, color=color, ls=style, lw=width, marker='o', ms=4, label=model)
                    visible = [h[h.plot_date.between(start,end)].value,
                               z[z.model.isin(model_styles) & z.target_end_date.between(start,end)].value]
                    if len(limits):
                        visible.append(limits.loc[limits.index.to_series().between(start,end), .9])
                    ymax = pd.concat(visible).max()*factor
                    ax.set_ylim(0, 1.12*ymax if pd.notna(ymax) and ymax > 0 else 1)
                    ax.set_xlim(start, end)
                    ax.axvline(issuance, color='#999999', ls=':', lw=1)
                    if col == 0:
                        ax.axvspan(zoom_start, zoom_end, color='#e8f1fa', alpha=.4, zorder=-1)
                    if h[(h.season == 'current') & h.value.notna()].empty:
                        ax.text(.02,.95,'Current reported history unavailable', transform=ax.transAxes, va='top', fontsize=10)
                    if peers.empty:
                        ax.text(.02,.85,'No same-week peers for this target/location', transform=ax.transAxes, va='top', fontsize=9)
                    ax.set_title('Full season' if col == 0 else f'Zoom · {start:%b %d}–{end:%b %d}', loc='left', fontsize=13)
                    ax.set_ylabel(ylabel, fontsize=12)
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2) if col == 0 else mdates.WeekdayLocator(byweekday=mdates.SA,interval=max(1,args.zoom_weeks//4)))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b' if col == 0 else '%b %d'))
                    ax.grid(axis='y', alpha=.2)
                    handles, texts = ax.get_legend_handles_labels()
                    legend.update(zip(texts, handles))
            fig.suptitle(f'{names[location]} · peer comparison · reference {reference.date()}', fontsize=18, y=.985)
            fig.legend(legend.values(), legend.keys(), loc='upper center', bbox_to_anchor=(.5,.95), ncol=3, fontsize=9, frameon=False)
            fig.text(.5,.018,'Medians only; gray band is peer disagreement, not a prediction interval. Each panel has its own linear scale.\n'
                     f'History through {cutoff.date()}; prior season is aligned by CDC week, without rescaling (unmatched week 53 omitted).\n'
                     f'Hub snapshot {commit[:8]}; training/input descriptions and file hashes: manifest.json. Forecasts are unscored.',ha='center',fontsize=9)
            fig.subplots_adjust(left=.08,right=.985,bottom=.14,top=.78,wspace=.22,hspace=.35)
            fig.savefig(out/f'comparison-{location}.png',dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), reference_date=str(reference.date()),
                    hub_commit=commit, hub=str(args.hub.resolve()), local_labels=labels,
                    locations=selected, featured_peers=featured, missing_featured_peers=missing_peers,
                    peer_models=sorted(peer_frames), warnings=warnings, sources=sources,
                    history_cutoff=str(cutoff.date()), zoom_weeks=args.zoom_weeks,
                    assumptions=['Descriptive forecast comparison, not accuracy scoring.',
                                 'Latest target values at the pinned Hub commit, not reconstructed issuance-time truth.',
                                 'No history dated after reference minus seven days is shown as current observations.',
                                 'Prior season aligned by CDC week; unmatched week 53 omitted, no value rescaling.',
                                 'Peer bands weight each available model median equally; local candidates are excluded.',
                                 'Partial peer coverage is retained; no missing forecasts or history are imputed.',
                                 'Training details are copied only from matching local export metadata; peer training is not inferred.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines = ['# Submission comparison', '', f'Reference date: {reference.date()}. Hub commit: `{commit}`.', '',
             'Full-season and zoom panels use independent linear scales. Peer percentiles describe forecast magnitude, not accuracy.', '',
             'Files: comparison.pdf, comparison-<FIPS>.png, forecast-medians.csv, history.csv, positions.csv, manifest.json.', '',
             '## Model descriptions', '']
    for source in sources:
        if source['role'] == 'local':
            note = (source.get('definition') or {}).get('note')
            lines.append(f'- {source["model"]}: {note or "Training seasons, input treatments and prediction labels are not specified in the export note; consult the model record."}')
    lines += ['', '## Assumptions', ''] + ['- '+x for x in manifest['assumptions']]
    lines += ['', '## Availability', ''] + ['- '+x for x in warnings or ['All requested peer model names are available somewhere in this snapshot; target/location coverage can differ.']]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    for warning in warnings:
        print('Note:',warning)
    print(out/'comparison.pdf')


if __name__ == '__main__':
    main()
