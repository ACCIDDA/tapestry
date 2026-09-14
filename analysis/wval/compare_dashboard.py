"""Compare saved dashboard values with medians of saved PA site WVALs (no network)."""
import collections
import json
from pathlib import Path
from audit import medians, number

p = Path(__file__).resolve().parent / 'evidence'
rows = json.loads((p / 'state_activity_current.json').read_text())
site_rows = json.loads((p / 'pennsylvania_current.json').read_text())
m = medians(site_rows)
state = {}; duplicate_keys = 0
for r in rows:
    k = (r['State/Territory'], r['Week_End'], r['Pathogen_Target'])
    if k in state:
        duplicate_keys += 1
    state[k] = r
assert duplicate_keys == 0
pairs = []
for k in sorted(m.keys() & state.keys()):
    v = number(state[k], 'State/Territory_WVAL')
    if v is not None:
        pairs.append({'state': k[0], 'week_end': k[1], 'pathogen': k[2],
                      'site_median': m[k], 'dashboard': v, 'abs_difference': abs(m[k]-v)})
out = {'dashboard_rows': len(rows), 'dashboard_duplicate_keys': duplicate_keys,
       'dashboard_updated': sorted({r['Date_Updated'] for r in rows}),
       'dashboard_max_week': max(r['Week_End'] for r in rows),
       'pa_numeric_pairs': len(pairs), 'equal_within_1e_9': sum(x['abs_difference'] < 1e-9 for x in pairs),
       'within_0_01': sum(x['abs_difference'] <= 0.010000001 for x in pairs),
       'max_absolute_difference': max(x['abs_difference'] for x in pairs),
       'largest_differences': sorted(pairs, key=lambda x:x['abs_difference'], reverse=True)[:5],
       'latest_examples': pairs[-3:]}
(p / 'dashboard_comparison.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
