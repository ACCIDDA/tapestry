# State wastewater activity index from Delphi NWSS

Builds the candidate indices described in `docs/data/wastewater.md`. Standard
library plus pandas/numpy/matplotlib; no project imports, so it runs outside the
package environment.

## Acquisition

Delphi V5, `source=nwss`, `geo_type=sewershed`. Set `snapshot_date` to the
vintage you want; `2026-09-18` was the latest `report_time` when this was
written.

```sh
KEY=$(cat DELPHI_API_KEY)
for s in covid flu rsv; do
  for sig in ${s}_avg_conc_lin ${s}_flowpop_lin; do
    curl -H "Authorization: Bearer $KEY" \
      "https://delphi.cmu.edu/epidata/v5/snapshot/?source=nwss&signal=${sig}&geo_type=sewershed&snapshot_date=2026-09-18&limit=100000000&format=csv" \
      -o ${sig}.csv
  done
done
curl -H "Authorization: Bearer $KEY" \
  "https://delphi.cmu.edu/epidata/v5/aux_data/?source=nwss&limit=100000000" -o aux_full.csv
```

The signal files are 20-70 MB each. `aux_full.csv` is **~10 GB**: it is the
whole multi-vintage auxiliary table and the endpoint accepts no filter beyond
`source` and `limit`. Rename `covid_avg_conc_lin.csv` to `covid.csv` (and the
same for flu/rsv) to match the paths in `indices.py`.

## Running

```sh
python analysis/wval/index/extract_aux.py   # 10GB -> aux_site.csv, aux_lab.csv
python analysis/wval/index/indices.py       # -> indices_all.csv
python analysis/wval/index/plot_all.py      # -> indices_all.png
```

`extract_aux.py` makes one streaming pass over the dump and keeps only the rows
whose sample key appears in a signal file, plus a sewershed/state/population
table. It takes a few minutes and reduces 27M rows to 1.3M.

Outputs are written next to the inputs. The copies kept in this repository are
`analysis/wval/evidence/state_indices.csv.gz`,
`analysis/wval/evidence/index_diagnostics.txt`, and
`docs/data/wastewater/state-indices.png`.

## Vintages

Everything here is keyed on one `snapshot_date`. Re-running with an earlier one
produces the index as it was computable at that date, which is the reason for
using Delphi rather than CDC's published WVAL. The baselines are currently
estimated over each group's full history, so a vintage rebuild is *not* yet
origin-safe — see the caveat in `docs/data/wastewater.md`.
