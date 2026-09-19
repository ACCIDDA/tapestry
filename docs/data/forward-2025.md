# Forward benchmark data: two recent weeks

The data preparation step completed without fitting or scoring. The user has
now authorized the [forward model benchmark](../workflows/forward-2025.md).

## Prepared information sets

- Historical cutoff: **July 26, 2025, end of day UTC**, before the project's
  2025–26 epiweek season begins.
- Test Wednesdays: **July 30, 2025–July 29, 2026**. Future forecast target range:
  **August 2, 2025–August 1, 2026**. This is forward development, not an untouched
  final test: 2025–26 was previously explored.
- Recent pairs cover **only the two most recent observation weeks**, ages
  **4 and 11 days** at each Wednesday. Each visible preliminary report is paired
  with that same observation week's cutoff reference value. Missing reports
  have missing revision values, never zero revisions.
- Training references use only revisions released by **July 26, 2025**.
  A provisional 28-day maturity rule makes weeks through **June 28, 2025**
  eligible as finalized training labels. Reference availability is recorded
  separately from that maturity/eligibility rule. “Final” means latest by the
  cutoff, not today's latest or certified eventual finality.
- Test references are pinned to **September 16, 2026** and used as labels only.
  Test inputs, including older context, reproduce Wednesday information states
  without filling missing reports with later finals.

The input NPZ retains a 12-week forecasting context for the future modeling step.
Older fitting context is a cutoff-final proxy shared across future candidates;
older test context is as of issuance. That train/deployment mismatch is explicit.
Recent preliminary inputs are never final-filled. Source-certified final status
is unavailable: supplied training reference inputs are flagged; test reports
are not certified final and their flags are false.

## Sources and scope

The existing acquisition tools and `VintageArchive.resolve/panel` are reused.
No separate replacement vintage-resolution layer is introduced.

- Delphi NHSN: **7,031,741 rows**, downloaded September 18, 2026 EDT.
- Delphi NSSP: **3,180,203 rows**, downloaded the same evening.
- Both acquisitions passed checksum verification for all eight files each.
- All three Hub raw acquisitions were restored from pinned Git mirrors, including
  Git-history artifacts where needed. Existing prepared retrospective datasets
  were preserved.
- The population file is pinned to the training cutoff, from FluSight commit
  `683cee50b236ed338467acfc4accbef5412a3e4e`. The ordinary current population file
  had a later update, so it is not used here.

Native Hub `as_of` is preferred; Git-history snapshots cover unversioned Hub
files; Delphi `report_time` supplies fallback outside established Hub coverage.
Holes/retractions within a covered Hub series stay missing. The existing resolver
preserves identical duplicates and quarantines conflicting cells. Each cell has
release provenance, and the dataset records acquisition-manifest hashes.

**Availability limitation:** native vintage timestamps are accepted as advertised
historical availability; Git timestamps are repository-publication proxies.
These are not independent proof of provider publication times or an intraday
submission deadline. After first source/channel/location coverage, retained
archives are assumed complete; an unrecorded interior archive outage cannot be
identified from file integrity alone. Before coverage, missing reports are
archive-unknown, not reconstruction examples.

NHSN recent supervision requires observation week-end **November 1, 2024 or
later** (first eligible Saturday November 2, a boundary week). Older epidemic
history remains available for future forecasting. NSSP uses its actual vintage
history and has no NHSN policy boundary. Measured earliest retained Delphi report
timestamps are April 18, 2024 for NSSP; November 19, 2024 for NHSN flu/COVID;
December 11, 2024 for NHSN RSV. Hub archives may add earlier support.

Visible revisions, covered missing-report reconstruction, and unknown archive
coverage are separate. Four-day and eleven-day ages are never pooled implicitly.

## Files and reproduction

- `data/processed/forward_2025.npz`: cutoff-pinned contexts, masks, references,
  eligibility and per-cell release provenance.
- `data/processed/forward_2025.recent.parquet`: **two-week revision pairs only**,
  with preliminary/reference availability, cutoffs, age, regime and supervision
  eligibility. `revision = reference − preliminary` only when both exist.
- `data/processed/forward_2025.support.csv`: target/age coverage and supervision
  counts, split into training and test.
- `data/processed/forward_2025.audit.json`: numerical alignment and leakage audit.
- `data/metadata/forward_2025_locations.csv`: historically available populations.

```bash
.venv/bin/python -m tapestry.model_data.forward
.venv/bin/python -m pytest -q tests/test_forward_science.py
```

The builder caches parsed source archives under `tmp/forward-benchmark/`, keyed
by acquisition manifests. Deleting this cache forces a fresh parse; it does not
change source acquisitions. Source downloads remain the canonical raw record.
The audit verifies unique aligned cells, exact 4/11-day ages, no future training
reference releases, no future test-input releases, and no invented revisions
for missing reports. Two focused tests cover the NHSN/NSSP boundary and exclusion
of a later final from a missing four-day input.

## Completed build

The build contains **94,848 target/location/age rows** across **152 Wednesdays**,
six targets and 52 geographies (states/DC/US). Rows include unavailable reports,
which remain explicitly missing. Units are admission counts and ED proportions
(0–1). All five raw acquisitions passed checksum verification. The provenance
and alignment audit passed; both focused scientific tests passed.

Eligible recent pairs (training labels use the provisional maturity rule):

| partition | target | age_days | revision | reconstruction | archive_unknown |
| --- | --- | --- | --- | --- | --- |
| train | nhsn_flu_admissions | 11 | 1716 | 104 | 156 |
| train | nhsn_flu_admissions | 4 | 1404 | 416 | 156 |
| train | nhsn_covid_admissions | 11 | 1768 | 0 | 3240 |
| train | nhsn_covid_admissions | 4 | 1560 | 156 | 3240 |
| train | nhsn_rsv_admissions | 11 | 1716 | 0 | 2002 |
| train | nhsn_rsv_admissions | 4 | 1508 | 156 | 2054 |
| train | nssp_flu_proportion | 11 | 2521 | 627 | 1746 |
| train | nssp_flu_proportion | 4 | 150 | 2947 | 1747 |
| train | nssp_covid_proportion | 11 | 2525 | 675 | 1747 |
| train | nssp_covid_proportion | 4 | 153 | 2996 | 1747 |
| train | nssp_rsv_proportion | 11 | 2523 | 677 | 1186 |
| train | nssp_rsv_proportion | 4 | 153 | 2996 | 1237 |
| test | nhsn_flu_admissions | 11 | 2092 | 560 | 0 |
| test | nhsn_flu_admissions | 4 | 1818 | 782 | 0 |
| test | nhsn_covid_admissions | 11 | 2444 | 312 | 0 |
| test | nhsn_covid_admissions | 4 | 2340 | 416 | 0 |
| test | nhsn_rsv_admissions | 11 | 2444 | 312 | 0 |
| test | nhsn_rsv_admissions | 4 | 2340 | 416 | 0 |
| test | nssp_flu_proportion | 11 | 2194 | 413 | 45 |
| test | nssp_flu_proportion | 4 | 1888 | 667 | 45 |
| test | nssp_covid_proportion | 11 | 2405 | 306 | 45 |
| test | nssp_covid_proportion | 4 | 2201 | 510 | 45 |
| test | nssp_rsv_proportion | 11 | 2405 | 306 | 45 |
| test | nssp_rsv_proportion | 4 | 2201 | 510 | 45 |

Training NSSP four-day visible revision support begins only on June 18, 2025;
earlier four-day cells are predominantly reconstruction or archive-unknown.
Eleven-day NSSP reports have earlier vintage support. The model step must not
interpret this sparse four-day correction support as evidence of zero revision.
