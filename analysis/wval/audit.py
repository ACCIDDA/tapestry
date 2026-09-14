"""Bounded revision audit. Uses existing Sept 4 snapshots; fetches Pennsylvania WVAL.
Run from any directory with Python 3 (standard library only). Latest data will change.
The state medians are descriptive reconstructions, not dashboard parity assertions.
"""
import collections
import csv
import datetime
import decimal
import gzip
import json
import math
from pathlib import Path
import statistics
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVIDENCE = HERE / 'evidence'
EVIDENCE.mkdir(exist_ok=True)
OLD = ROOT / 'data/raw/cdc_nwss_wval/snapshots/20260904T224139.023349Z'
DELPHI = ROOT / 'data/raw/delphi_nwss/snapshots/20260904T231517.056225Z'


def number(row, name='site_wval'):
    try:
        x = float(row.get(name, ''))
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def key(row):
    return tuple(row.get(k) for k in ('state_territory', 'site', 'week_end', 'pathogen_target'))


def index(rows):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError(f'Duplicate key: {k}')
        result[k] = row
    return result


def medians(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        v = number(row)
        if v is not None:
            groups[(row['state_territory'], row['week_end'], row['pathogen_target'])].append(v)
    return {k: statistics.median(v) for k, v in groups.items()}


def main():
    query = {'$where': "state_territory='Pennsylvania'", '$limit': '50000',
             '$order': 'site,week_end,pathogen_target'}
    url = 'https://data.cdc.gov/resource/atcp-73re.json?' + urllib.parse.urlencode(query)
    data = urllib.request.urlopen(url, timeout=60).read()
    (EVIDENCE / 'pennsylvania_current.json').write_bytes(data)
    new_rows = json.loads(data)
    assert len(new_rows) < 50000, 'Pagination needed'
    with gzip.open(OLD / 'data.ndjson.gz', 'rt') as f:
        old_rows = [r for line in f if (r := json.loads(line))['state_territory'] == 'Pennsylvania']
    (EVIDENCE / 'pennsylvania_20260904.json').write_text(json.dumps(old_rows))
    old, new = index(old_rows), index(new_rows)
    common = old.keys() & new.keys()
    numeric = [k for k in common if number(old[k]) is not None and number(new[k]) is not None]
    changed = [k for k in numeric if number(old[k]) != number(new[k])]
    details = []
    for k in sorted(changed):
        details.append(dict(zip(('state', 'site', 'week_end', 'pathogen'), k),
                            before=number(old[k]), after=number(new[k])))
    (EVIDENCE / 'wval_changed_values.json').write_text(json.dumps(details, indent=2))
    by_pathogen = {}
    for pathogen in sorted({k[-1] for k in common}):
        ks = [k for k in numeric if k[-1] == pathogen]
        cs = [k for k in changed if k[-1] == pathogen]
        by_pathogen[pathogen] = {'comparable_numeric': len(ks), 'changed': len(cs),
                                'max_absolute_change': max((abs(number(new[k])-number(old[k])) for k in cs), default=0)}
    om, nm = medians(old_rows), medians(new_rows)
    sm = [{'state': k[0], 'week_end': k[1], 'pathogen': k[2], 'before': om[k], 'after': nm[k]}
          for k in sorted(om.keys() & nm.keys()) if om[k] != nm[k]]
    (EVIDENCE / 'state_median_changes.json').write_text(json.dumps(sm, indent=2))
    result = {'retrieved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'url': url,
              'old_snapshot': str(OLD.relative_to(ROOT)),
              'old_release': sorted({r.get('date_updated') for r in old_rows}),
              'new_release': sorted({r.get('date_updated') for r in new_rows}),
              'old_rows': len(old_rows), 'new_rows': len(new_rows), 'common_keys': len(common),
              'added_keys': len(new.keys()-old.keys()), 'removed_keys': len(old.keys()-new.keys()),
              'numeric_pairs': len(numeric), 'numeric_changed': len(changed),
              'missingness_changed': sum((number(old[k]) is None) != (number(new[k]) is None) for k in common),
              'category_changed': sum(old[k].get('site_wval_category') != new[k].get('site_wval_category') for k in common),
              'by_pathogen': by_pathogen, 'state_median_comparable': len(om.keys() & nm.keys()),
              'state_median_changed': len(sm), 'state_median_change_examples': sm[-9:],
              'site_change_examples': details[-9:]}
    (EVIDENCE / 'revision_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

    # Scan one local signal. Distinguish repeated archive records from actual value changes.
    keycols = ('signal', 'geo_type', 'geo_value', 'fill_method', 'reference_time', 'nwss_source', 'sample_index')
    seen = {}; changed_keys = set(); reports = collections.Counter(); examples = []
    path = DELPHI / 'signal=flu_avg_conc_lin/geo_type=sewershed/archive.csv.gz'
    with gzip.open(path, 'rt') as f:
        for row in csv.DictReader(f):
            reports[row['report_time']] += 1
            k = tuple(row[c] for c in keycols)
            # Decimal avoids counting representation-only changes such as 1 vs 1.0.
            try:
                val = decimal.Decimal(row['value'])
                val = str(val) if not val.is_finite() else val
            except decimal.InvalidOperation:
                val = row['value']
            if k in seen:
                pv, pr = seen[k]
                if val != pv:
                    changed_keys.add(k)
                    if len(examples) < 5:
                        examples.append({'key': dict(zip(keycols, k)), 'record_a': {'report_time': pr, 'value': str(pv)},
                                         'record_b': {'report_time': row['report_time'], 'value': str(val)}})
            else:
                seen[k] = (val, row['report_time'])
    out = {'file': str(path.relative_to(ROOT)), 'rows': sum(reports.values()), 'unique_keys': len(seen),
           'keys_with_multiple_values': len(changed_keys), 'report_times': dict(sorted(reports.items())),
           'examples': examples}
    (EVIDENCE / 'delphi_revision_summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
