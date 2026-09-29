# FluView and FluSurv-NET (Delphi V5)

Added 2026-09-22 at the user's request, after Delphi announced (email to the
user, 2026-09-19) that FluView and FluSurv moved from the legacy V3 endpoints to
the V5 API. V3 `pub_fluview()` becomes the V5 source `fluview_ilinet`,
`pub_fluview_clinical()` is split into `fluview_resp_lab_clinical` and
`fluview_resp_lab_ph`, and `pub_flusurv()` becomes `flusurv`. The public health
labs (`fluview_resp_lab_ph`) are not acquired: pulled once on 2026-09-22, their
state rows turned out to be season totals (one row per state per season), not
weekly values, and the user decided to drop the source from the dataset and the
explorer. Each Delphi
source page has a "Relationship to V3" section describing schema changes. The
project's `epidatpy>=0.6.0,<0.7` pin already includes these sources; the
downloader is the shared `delphi_v5` fetcher (`data/sources/delphi.py`).

| Catalog key | V5 source | Signals | Geography pulled | Panel covariate |
|---|---|---|---|---|
| `delphi_fluview_ilinet` | `fluview_ilinet` | `ili`, `wili`, `num_ili`, `num_patients`, `num_providers` | state, nation, hhs, census division | `ilinet_ili` (group `ilinet`) |
| `delphi_fluview_clinical` | `fluview_resp_lab_clinical` | `pct_positive`, `pct_positive_a`, `pct_positive_b`, `positive_a`, `positive_b`, `total_specimens` | state, nation, hhs, census division | `clinical_lab_flu_pct_positive` (group `clinical_lab`) |
| `delphi_flusurv` | `flusurv` | 28 `rate_*` signals: overall, influenza A/B, age, race and ethnicity, sex | state, nation, msa, misc | `flusurv_flu_rate` (group `flusurv`) |

```bash
.venv/bin/python -m tapestry.data pull delphi_fluview_ilinet delphi_fluview_clinical delphi_flusurv
```

Full archives (every report time) are small: 1.47 M, 0.44 M and 0.45 M rows,
together about 16 MB compressed (first pull 2026-09-22).

## Schema

FluView rows carry an extra key column, `age_group`. ILINet visit counts
(`num_ili`) are stratified (`0-4`, `5-24`, `25-49`, `25-64`, `50-64`, `65+`,
`all`); every other signal pulled has only `all`. FluSurv has no `age_group`
column: its strata are separate signals. Reference times are Saturday week ends
throughout. Values keep Delphi's units: ILI and positivity in percent (0–100),
FluSurv rates per 100,000 residents of the catchment, the rest counts.

Geography, as published:

- ILINet reports 50 states plus DC, New York City and New York minus NYC, Puerto
  Rico and the US Virgin Islands. **Statewide New York exists only as a Delphi
  pool** of the two New York jurisdictions (`fill_method=nyc_plus_ny_minus_nyc`).
  Population-weighted ILI (`wili`) is suppressed at state level.
- Clinical laboratories report up to 50 jurisdictions; New York appears only as
  `ny_minus_nyc` (no statewide row), so the state panel has no New York value.
- FluSurv-NET covers catchment counties in 13 states for weeks since September
  2023 (CA, CO, CT, GA, MD, MI, MN, NC, NM, OH, OR, TN, UT; 18 states over its
  history). The national row is the network rate; `misc` holds the EIP
  and IHSP sub-network rates.

## Release timing and archive coverage (measured 2026-09-22)

From the `report_time` history of the downloaded archives, weeks ending from
2023-09-01 on:

| Source | First release after week end (p10 / median / p90) | Revisions per state-week (median / max) |
|---|---|---|
| ILINet `ili`, state | 6 / 6 / 9 days | 2 / 18 |
| ILINet `ili`, nation | 6 / 6 / 9 days | 16 / 28 |
| Clinical labs `pct_positive`, state | 6 / 6 / 13 days | 1 / 27 |
| FluSurv `rate_overall` | dominated by the archive gap below | 1 / 16 |

FluView is first released on the Friday after the week ends. A Wednesday
issuance's context end is the Saturday four days earlier, so **the context-end
week (j = 0) is never visible at the cutoff; the latest visible week is
j = 1**.

Gaps in the release history (no report time in between):

- **FluView** (ILINet and clinical labs): 2025-09-26 to 2025-11-14. This matches
  the October–November 2025 US federal government shutdown; that it was a
  publication pause rather than a Delphi acquisition gap is a hypothesis, not
  verified.
- **FluSurv**: 2020-11-06 to 2025-11-03, then 2025-11-03 to 2026-02-03. Delphi's
  FluSurv archive therefore holds no vintage from November 2020 to November
  2025. Weeks from those seasons appear only at their first archived vintage,
  up to 758 days after the week ended. **In vintaged mode FluSurv is
  unavailable at every issuance before 2025-11-03 and between then and
  2026-02-03**; final-mode inputs use values that were not available at the
  time. This is an archive limit, not FluSurv's operational reporting lag.

## Panel covariates

`dataset/extract.py` reads each covariate as a Delphi-only revision archive
(`DELPHI_COVARIATES`), resolved like the claims covariates: latest eligible
revision per cell, rows with `age_group` other than `all` dropped, and
non-native fill methods dropped except `nyc_plus_ny_minus_nyc` for ILINet (so
New York has an ILINet value). Choices and their reasons:

- `ilinet_ili` uses unweighted ILI because weighted ILI is not published for
  states. One definition therefore serves states and the US row.
- `clinical_lab_flu_pct_positive` is influenza A + B positivity. Positive
  counts depend on laboratory volume and are not comparable across states.
- `flusurv_flu_rate` is the overall rate. It exists only for catchment states
  and the US (network) row; elsewhere it is unavailable, never filled.

Scenarios select them with `covariate_set` groups `ilinet`, `clinical_lab` and
`flusurv`. No scenario uses them yet.

In the panel built 2026-09-22 (160 Wednesday issuances from 2023-09-06), share
of cells with a final value but nothing visible at the Wednesday cutoff, all
locations (full table in the generated [panel analysis](panel.md) §3):

| Covariate | Locations with a value | Issuances with any as-of value | j = 0 | j = 1 | j = 2 |
|---|---:|---:|---:|---:|---:|
| `ilinet_ili` | 52 | 158 | 99.3% | 6.5% | 5.0% |
| `clinical_lab_flu_pct_positive` | 45 | 158 | 99.4% | 14.0% | 7.3% |
| `flusurv_flu_rate` | 14 (13 states + US) | 46, first 2025-11-05 | 98.5% | 78.6% | 77.7% |

ILINet and clinical labs are usable one week behind the context end at almost
every issuance. FluSurv is usable in vintaged mode only from November 2025,
because of the archive gap above.
