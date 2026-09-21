"""One streaming pass over the 10GB aux_data dump -> two compact tables."""
import csv, sys, pandas as pd

keys = set()
for p in ['covid.csv', 'flu.csv', 'rsv.csv']:
    cols = ['geo_value','nwss_source','reference_time','sample_index','pcr_target']
    d = pd.read_csv(p, dtype=str, usecols=cols)[cols]
    keys.update(map(tuple, d.itertuples(index=False, name=None)))
print("signal keys:", len(keys), file=sys.stderr, flush=True)

site = {}
out = open('aux_lab.csv', 'w', newline='')
w = csv.writer(out)
w.writerow(['geo_value','nwss_source','reference_time','sample_index','pcr_target',
            'major_lab_method','flow_rate','sample_matrix'])
seen = set()
n = 0
with open('aux_full.csv', newline='') as f:
    for r in csv.DictReader(f):
        n += 1
        if n % 5_000_000 == 0:
            print(f"  {n:,} rows, {len(seen):,} matched", file=sys.stderr, flush=True)
        gv = r['geo_value']
        if gv:
            st, ps = r['state_territory'], r['population_served']
            if st:
                prev = site.get(gv)
                site[gv] = (st, ps or (prev[1] if prev else ''))
        k = (gv, r['nwss_source'], r['reference_time'], r['sample_index'], r['pcr_target'])
        if k in keys and k not in seen:
            seen.add(k)
            w.writerow([*k, r['major_lab_method'], r['flow_rate'], r['sample_matrix']])
out.close()
print(f"done: {n:,} aux rows, {len(seen):,} matched, {len(site)} sewersheds", file=sys.stderr)
pd.DataFrame([(k, v[0], v[1]) for k, v in site.items()],
             columns=['geo_value','state','population_served']).to_csv('aux_site.csv', index=False)
