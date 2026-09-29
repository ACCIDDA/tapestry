# Historical information cutoffs for the B2 research run

`build_deadline_panel.py` creates `data/processed/panel-b2-deadline.npz` from
`data/processed/panel.npz`. The finalized truth arrays are unchanged, including
finalized covariates used for complete-history training. The new `asof_*` arrays
contain actual historically available values, not finalized values under a
historical availability mask. The training/validation/forecast distinction is
controlled by the experiment, not by replacing truth arrays in this builder.

Each nominal Wednesday remains the origin ID. Its input history still ends on
the preceding Saturday even when a holiday delays submission. The additional
`forecast_cutoff_utc` array records the information cutoff: 23:00 Eastern on the
Wednesday, with these joint-forecast exceptions:

| Forecast reference Saturday | Common deadline, 23:00 Eastern |
| --- | --- |
| 2024-12-28 | 2024-12-26 |
| 2025-01-04 | 2025-01-02 |
| 2025-12-27 | 2025-12-29 |
| 2026-01-03 | 2026-01-04 |

The modeled calendar also contains pre-challenge and off-season origins. For
those with no active Hub round, Wednesday 23:00 Eastern is an explicit nominal
research cutoff, not a claim that a submission round existed. FluSight
`b450296f1` establishes the Wednesday schedule for the challenge starting
October 11, 2023; all later documented schedule changes are the four exceptions
above.

The 2024 holidays are established in the local Hub Git histories: FluSight
`6da755c738e2ac01c99f5d48c0a7a967c68a766e` changes `submissions_due.end` from -3 to
-2, and `b791af8e0817e2eebb8a39868ae47b267844fb01` restores it on January 7.
COVID's December 23 extension is `a79500e`; its January 3 change restores normal
deadlines. The 2025 holidays use the earlier COVID/RSV deadline for every input;
FluSight's extra day is excluded from the joint issuance. See the earlier audit
at `docs/results/b1-to-b0-chain/availability_inventory.md`.

Targets come from the exact Hub Git state at that deadline, using its native
history, direct target files, and raw NSSP CSV/Parquet files, plus native Delphi
reports already released. Actual values are used. Among public statements still
present in these files, the latest publication wins per observation. A missing
cell in one Hub does not retract an extant observation in another Hub; a later
explicit null Delphi statement remains null. The Git committer timestamp of the
latest first-parent file modification is the publication proxy. That is a
repository-publication assumption, not a claim about upstream release time.

Every covariate is independently resolved using its own archived releases.
Date-only or midnight release labels are placed conservatively at the end of
their UTC day; explicit non-midnight timestamps retain their time. Consequently,
a date-only report from Thursday is not assumed available in the first hours of
Thursday UTC corresponding to Wednesday 23:00 Eastern. No lag assumption, future
filling, revision substitution, or copying from finalized truth fills gaps.
Kinsa remains a native national series on disk; any model broadcast is explicit.

Public archives establish a lower bound on historical availability. In
particular, Kinsa first appears in April 2026, and NWSS vintage support also
begins in 2026. Finalized historical covariate training can learn their
relationships, but strict historical evaluation can only use them on origins
where the public archive has released them. Three-season average comparisons
therefore do not identify a three-season operational Kinsa or wastewater effect.
Report effects on supported origins as well as the identical full evaluation
calendar; do not reinterpret an absent-input arm as evidence that the underlying
signal has no predictive content.

The output policy JSON records source/output hashes, native first-release dates,
and the assertion that all finalized arrays are unchanged. `target-source-versions.csv`
records Git commits and file blobs for every historical origin.
`target-coverage.csv` and `covariate-coverage.csv` quantify actual 12-week context
support by origin and season.

```bash
.venv/bin/python analysis/b2-research/build_deadline_panel.py --workers 3
```

## Decision log

September 27, 2026: Reconstruct all panel origins, including earlier-season
holiday extensions, for the new B2 comparison. Preserve finalized target and
covariate training histories per the user's clarification, while evaluation
receives only actual reports available by its deadline. Keep national Kinsa's
historical absence explicit; do not infer earlier public availability from its
2026 backfill.

## Operational reporting hypothesis authorized for the new launch

The user subsequently clarified that forecasting should use what we expect to
have under the 2025–26 reporting regime, and should lean toward availability when
historical archive coverage is uncertain. `build_operational_panel.py` therefore
creates the separate main-experiment dataset
`data/processed/panel-b2-operational.npz`. The strict panel remains the evidence
baseline; it is never relabeled as an operational backcast.

Actual deadline vintages always win. A missing input can receive its frozen
finalized historical value only if (1) no explicit report or withdrawal exists
for that cell at the cutoff, (2) finalized truth exists at that same native
location and event week, (3) the event week is old enough under the source's
assumed release delay, and (4) it remains at or before the original context end.
Explicit null masks are retained from source resolution and are never refilled.
Missing Missouri ED inputs are also excluded from proxy filling. No state Kinsa
observations or missing native state/catchment coverage are manufactured.

The delay estimator takes each event-week/location's first finite publication,
selects first releases during 2025–26, excludes delays over 60 days as historical
archive backfills, and uses the floor of the remaining median delay in calendar
days. This 60-day restriction and optimistic median are assumptions; source
counts, excluded backfills, and p10/median/p90 are recorded. Targets use the
observed Wednesday schedule (four days after Saturday). ILINet and clinical labs
use their Friday schedule (six days). Kinsa uses its stabilized daily publication
regime (one day), so all seven days of the latest Saturday week can be available
by Wednesday. Remaining sources use their measured median. The generated
`operational-source-lags.csv` and `operational-policy.json` record every exact lag.
The latest-observation lag is reported separately and does not estimate release
delay: a single fresh location is not evidence of statewide completeness.

Assumed releases are placed at the end of the UTC release day and compared with
the actual holiday-aware cutoff. Thus the same six-day lag naturally makes the
newest weekly ILINet report unavailable on Wednesday and available after a Monday
holiday extension. A proxy never changes an actually observed vintage.

This proxy policy uses later finalized revisions when the historical report is
unknown. For wastewater it can also use an index whose finalized normalization
baseline incorporated later samples. The experiment therefore tests a reporting
hypothesis and the value of covariate information; it does not establish strict
historical real-time forecast performance. Proxy masks (`proxy_targets`,
`proxy_covariates`, `proxy_covariates_national`) and per-origin source counts make
that distinction inspectable.

```bash
.venv/bin/python analysis/b2-research/build_operational_panel.py
```

September 27, 2026, follow-up decision: use the operational reporting backcast
for the requested main launch following the user's explicit clarification.
Keep the strictly reconstructed panel and its coverage evidence separately.
