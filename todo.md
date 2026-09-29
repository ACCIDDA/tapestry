# To do

## Published (GitHub Pages) explorer: decide how to cull the static export

Opened 2026-09-23. The online explorer is **not** updated for now; it still shows
the 2026-09-17 export (28 MB, before FluView/FluSurv). A fresh export of the
current index is 256 MB (52.4 M rows), too large to publish (user judgment).
Almost all of it is the daily Delphi claims revision history, kept in full since
the eight-week claims cap was removed.

Measured 2026-09-22 on the local ledger (simulation script, not the exporter):
keeping revisions released within W weeks after each week, plus the latest value.

| Window | Rows kept | Estimated export |
|---|---:|---:|
| none (now) | 57.6 M | 256 MB |
| 4 weeks | 6.7 M | ~33 MB |
| 8 weeks | 10.6 M | ~52 MB |

With 4 weeks, as-of views within 4 weeks of a week stay exact. Share of values
whose shown value is off by >5% at some later as-of date: claims 70%, FluView
11%, NWSS 5%, FluSurv 4%, Hub targets 4%, Delphi NHSN 3%, Delphi NSSP 2%.
Claims are 4.3 M of the 6.7 M rows kept at 4 weeks.

Open questions:
- Window: 4 weeks for everything?
- Claims: Saturday values only (as the panel uses, ~7x fewer rows, est. 15–20 MB
  total), or full history, or drop claims from the published copy?
- "Last 4 weeks" was read as revisions within 4 weeks after each week's date;
  the alternative (the 4 most recent revisions of each value) was not measured.

Then: implement the rule in `src/tapestry/explorer/export.py`, run
`scripts/update_published_explorer.sh`, check the real size, update
`docs/explorer/overview.md`.
