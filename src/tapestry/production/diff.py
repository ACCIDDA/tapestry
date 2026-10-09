"""Compare two quantile files for the same reference date, task by task.

Usage: python -m tapestry.production diff SUBMITTED.csv NEW.csv OUTDIR [--labels A B]
Writes all-tasks.csv (medians and 50/95% intervals of both, ratios of new to submitted
median and 95% width), national.csv and summary.md. For the overlaid interval figure use
`python -m tapestry.production intervals SUBMITTED.csv NEW.csv --labels A B`.
History: scripts/compare_submissions.py (2026-10-07); its figure duplicated `intervals`.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('submitted', type=Path)
    p.add_argument('new', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--labels', nargs=2, default=['Submitted', 'New'])
    p.add_argument('--hub', type=Path, default=Path('production/hubs/FluSight-forecast-hub'))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    TARGETS = {'wk inc flu hosp': ('Flu admissions', 'target-hospital-admissions.csv', 1),
               'wk inc flu prop ed visits': ('Flu ED share (%)', 'target-ed-visits-prop.csv', 100)}
    names = pd.read_csv(args.hub / 'auxiliary-data/locations.csv', dtype={'location': str}).set_index('location').location_name


    def wide(path):
        d = pd.read_csv(path, dtype={'location': str, 'output_type_id': str})
        d = d[d.output_type == 'quantile']
        return d.pivot_table(index=['target', 'location', 'horizon', 'target_end_date'], columns='output_type_id', values='value')


    sub, new = wide(args.submitted), wide(args.new)
    if not sub.index.equals(new.index):
        raise ValueError('The two files cover different tasks')
    rows = []
    for (target, loc, h, end), a in sub.iterrows():
        b = new.loc[(target, loc, h, end)]
        rows.append(dict(target=target, location=loc, name=names.get(loc, loc), horizon=h, target_end_date=end,
                         sub_median=a['0.5'], new_median=b['0.5'], sub_lo95=a['0.025'], sub_hi95=a['0.975'],
                         new_lo95=b['0.025'], new_hi95=b['0.975'], sub_lo50=a['0.25'], sub_hi50=a['0.75'],
                         new_lo50=b['0.25'], new_hi50=b['0.75']))
    t = pd.DataFrame(rows)
    t['median_ratio'] = t.new_median / t.sub_median.where(t.sub_median > 0)
    t['width95_ratio'] = (t.new_hi95 - t.new_lo95) / (t.sub_hi95 - t.sub_lo95).where(lambda w: w > 0)
    t.to_csv(args.out / 'all-tasks.csv', index=False)
    t[t.location == 'US'].to_csv(args.out / 'national.csv', index=False)
    states = t[t.location != 'US']
    lines = [f'# {args.labels[1]} compared with {args.labels[0]}', '',
             'Ratios are new divided by submitted. Medians and 95% interval widths per location and horizon.', '']
    for target, (label, _, scale) in TARGETS.items():
        us = t[(t.location == 'US') & (t.target == target)].sort_values('horizon')
        lines += [f'## {label}', '', f'| Horizon | Week ending | {args.labels[0]} median (95%) | {args.labels[1]} median (95%) | Median ratio |',
                  '|---:|---|---:|---:|---:|']
        fmt = (lambda v: f'{v:,.0f}') if scale == 1 else (lambda v: f'{100 * v:.2f}')
        for _, r in us.iterrows():
            lines.append(f'| {r.horizon} | {r.target_end_date} | {fmt(r.sub_median)} ({fmt(r.sub_lo95)}–{fmt(r.sub_hi95)}) | '
                         f'{fmt(r.new_median)} ({fmt(r.new_lo95)}–{fmt(r.new_hi95)}) | {r.median_ratio:.3f} |')
        s = states[states.target == target]
        lines += ['', 'States/DC/PR, median ratio new/submitted by horizon (10th, 50th, 90th percentile across locations); '
                  '95% width ratio median:', '']
        for h, g in s.groupby('horizon'):
            q = g.median_ratio.quantile([.1, .5, .9])
            lines.append(f'- horizon {h}: {q.iloc[0]:.3f}, {q.iloc[1]:.3f}, {q.iloc[2]:.3f}; width {g.width95_ratio.median():.3f}')
        h3 = s[s.horizon == 3].sort_values('median_ratio')
        lines += ['', 'Largest horizon-3 differences: ' + ', '.join(f"{r['name']} {r.median_ratio:.2f}" for _, r in pd.concat([h3.head(3), h3.tail(3)]).iterrows()), '']
    (args.out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print((args.out / 'summary.md').read_text())


if __name__ == '__main__':
    main()
