# CDC state WVAL, Delphi wastewater, and state-level forecasting

Research date: **September 13, 2026**. This analysis now lives under `analysis/wval/`; CDC's metric is **WVAL** (wastewater viral activity level).

## Findings

**The CDC state WVAL is recoverable directly from CDC. Delphi's wastewater endpoint does not supply that metric directly.** It supplies concentrations from which you could construct a useful state predictor, but exact reproduction of CDC WVAL requires more than taking a median of raw concentrations.

**Historical WVAL values can change. An unchanged state value does not establish that the underlying data were unchanged.** In a comparison of two actual releases, 15 Pennsylvania site-week/pathogen WVAL values changed, while all 665 comparable state-week/pathogen medians stayed the same. These are computations from the saved inputs described below, not estimates of the national revision rate.

My recommendation is to start with **CDC state WVAL as one additional forecasting covariate**, then test whether a carefully aggregated Delphi concentration signal adds predictive information. Preserve the vintage used at each forecast origin for both routes.

### Scope and assumptions

- **User request:** assess recovery of the linked state WVAL, state forecasting uses of Delphi wastewater, and revisions; deliver a Markdown report.
- **Assumption:** the primary forecasting application is weekly state respiratory admissions or ED visits, initially at 1–4-week horizons. A separate interpretation—forecasting WVAL itself—is addressed below.
- **Assumption:** Pennsylvania is a convenient bounded audit case, not a representative sample of states. The recommendations apply separately to influenza A, SARS-CoV-2, and RSV.
- **Observed project context:** this workspace already has CDC site WVAL and Delphi NWSS archive downloads from September 4. I used those existing data and added analysis artifacts here. I did not change the application or fit a forecasting model.

## 1. Getting the state series

### Best route: the JSON behind the current CDC chart

Inspection of the [user-supplied state page](https://www.cdc.gov/wastewater/respiratory-viruses/state.html) identified its [visualization configuration](https://www.cdc.gov/wastewater/modules/state-data/state-page-combined.json). That configuration points the trend chart to this [state WVAL JSON dataset](https://www.cdc.gov/wcms/vizdata/NCEZID_DIDRI/NWSS_WVAL_metric/NWSSWVALStateActivityLevel.json).

The downloaded feed contains **16,264 rows**, with no duplicate `(State/Territory, Week_End, Pathogen_Target)` keys. Its latest observation week ends **2026-09-05**; its embedded update stamp is **September 10, 2026 9:51 AM**. The webpage itself is dated September 11. Keep the observation date, embedded processing timestamp, and your retrieval timestamp separately.

Useful actual fields are:

| Field | Use |
|---|---|
| `State/Territory` | Map names to the project's state identifiers |
| `Week_End` | Observation week |
| `Pathogen_Target` | Keep the three pathogens separate |
| `State/Territory_WVAL` | Numeric state predictor or target; parse strings explicitly |
| `State/Territory_WVAL_Category` | Optional interpretation or categorical target |
| `Coverage` | Preserve the original coverage flag |
| `Date_Updated` | Publisher's processing/update stamp |

This is a **verified dashboard data URL**, not a promise of a permanent API contract. Save complete dated copies. The local copy is [state_activity_current.json](evidence/state_activity_current.json); [dashboard_comparison.json](evidence/dashboard_comparison.json) records validation results.

A useful caution from the endpoint checks: an older influenza CSV linked in a jurisdiction's WVAL script still returned HTTP 200, but was stamped **May 7, 2026**. Older COVID/RSV CSV URLs returned 404. A successful download alone does not establish freshness.

### Second route: aggregate CDC's already-calculated site WVALs

The current [CDC site WVAL dataset is `atcp-73re`](https://data.cdc.gov/Public-Health-Surveillance/CDC-Wastewater-Viral-Activity-Level-for-SARS-CoV-2/atcp-73re/about_data). The [JSON API](https://data.cdc.gov/resource/atcp-73re.json) exposes `site_wval`, `site_wval_category`, `site`, `state_territory`, `week_end`, `pathogen_target`, `population_served`, `date_included_in_wval`, and `date_updated`. Use pagination for a full download; the default response is not the entire dataset. The [saved metadata](evidence/atcp-73re.json) describes `date_updated` as the processing run's as-of stamp, not an append-only revision history.

For a given published vintage, the basic reconstruction is:

```text
state_wval[state, week, pathogen]
    = median(eligible numeric site_wval values for that state/week/pathogen)
```

Deduplicate by site/week/pathogen and check eligibility before treating this as an exact dashboard reproduction. Use an **unweighted median** for that reconstruction. Population weighting would define a different research signal.

**Empirical check:** I compared Pennsylvania's September 11 site release with the current dashboard feed. Of 312 numeric state-week/pathogen pairs, 270 matched within 1e-9 and all 312 differed by at most approximately **0.005**. The small differences are consistent with rounding, although I did not establish CDC's precise internal rounding procedure. Latest-week medians and chart values were identical: influenza A **1.00**, RSV **1.00**, SARS-CoV-2 **1.01**. This supports the aggregation route for this state and vintage; it is not an all-state parity guarantee.

The site's full history and the chart's displayed history need not have the same span. The chart comparison therefore has fewer pairs than the two-release site-history comparison.

### What Delphi gives you instead

[Delphi's NWSS documentation](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/nwss.html) describes sewershed-level concentration signals and auxiliary sample/laboratory metadata. It does not document a state WVAL signal. Public sample availability and changing laboratory methods are additional obstacles to exact CDC reconstruction.

The workspace's actual September 4 [Delphi metadata](evidence/delphi_metadata_20260904.json) confirms `sewershed` as its only geography and includes these influenza signals:

| Signal | Documented quantity |
|---|---|
| `flu_avg_conc` | Concentration on the reported/back-calculated basis |
| `flu_avg_conc_lin` | Concentration on a linear scale |
| `flu_flowpop_lin` | Flow/population-normalized concentration, copies/person/day |
| `flu_mic_lin` | Microbial-normalized concentration ratio |

The `covid_` and `rsv_` families have analogous signals. Normalizations have different meanings and should not be pooled as if their units were interchangeable. Provider, sample, and PCR-target dimensions matter; the auxiliary table carries state, population, and laboratory information. These details are documented in [Delphi NWSS](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/nwss.html).

**Conclusion about recoverability:** a state forecasting feature is feasible from Delphi; an exact CDC state WVAL is not established from Delphi alone. To reproduce it, you would need to match CDC's input sample set, site/laboratory grouping, normalization choices, censoring/outlier handling, baseline/scale estimates, eligibility, weekly aggregation, and publication vintage. Verify these against CDC outputs rather than assuming a shared upstream source guarantees equality.

A [jurisdiction-contributed WVAL implementation](https://github.com/CDCgov/NWSS/tree/master/jurisdiction-scripts/WVAL%20Collaborative%20Script) can help investigate the transformation. The downloaded script explicitly targets **August 2025** methods, and includes exponentiation after log-scale standardization. It should not be used unmodified as the specification for September 2026 WVAL. CDC's [repository](https://github.com/CDCgov/NWSS) identifies such scripts as jurisdiction-maintained contributions.

## 2. Are historical values revised regularly?

### CDC's stated policy

The [state page](https://www.cdc.gov/wastewater/respiratory-viruses/state.html) says updates occur Fridays and historical data may change as additional reports arrive. This allows revisions; it does not mean every historical value changes weekly.

CDC's [current WVAL methodology](https://www.cdc.gov/wastewater/about/wval.html) states that baselines use the preceding two years. COVID-19 baselines are reset on **April 1 and October 1**; influenza A and RSV on **August 1**. Historical WVALs are recalculated with the new baseline. Method updates are also applied retrospectively. August 2026 changes include outlier winsorization and laboratory-based initialization for newer sites. State WVAL is the median of site WVALs; sites need sufficient history, with additional seasonal inclusion rules for influenza A and RSV. Consequently, an old week's WVAL is not permanently fixed merely because its samples are old.

### Actual CDC revision check: September 4 versus September 11

Inputs: the project's stored September 4 `atcp-73re` snapshot and a fresh Pennsylvania query with a September 11 processing stamp. Matching keys were `(state_territory, site, week_end, pathogen_target)`. Duplicate keys were checked; numeric comparisons excluded timestamp-only changes.

| Result | Count |
|---|---:|
| Site-week/pathogen rows in September 4 release | 14,460 |
| Rows in September 11 release | 14,503 |
| Matched numeric pairs | 14,460 |
| Numeric WVAL changes | **15** |
| Added keys / removed keys | 43 / 0 |
| Category changes among matched rows | **0** |
| Missingness changes among matched rows | 0 |
| Comparable reconstructed state-week/pathogen medians | 665 |
| Changed reconstructed state medians | **0** |

| Pathogen | Matched numeric pairs | Changed site values | Largest absolute change |
|---|---:|---:|---:|
| Influenza A | 3,645 | 7 | 2.44 |
| RSV | 3,044 | 5 | 0.46 |
| SARS-CoV-2 | 7,771 | 3 | 0.75 |

Examples at Pennsylvania site `ID:2690`:

| Week ending | Pathogen | September 4 value | September 11 value |
|---|---|---:|---:|
| 2026-02-21 | Influenza A | 17.13 | 17.61 |
| 2026-02-14 | RSV | 13.87 | 14.33 |
| 2026-08-22 | SARS-CoV-2 | 2.50 | 1.75 |

These examples prove that published numeric WVALs changed, including months-old observations. They do **not** identify the cause of each revision. See [summary](evidence/revision_summary.json) and [all 15 changes](evidence/wval_changed_values.json).

An unchanged value can therefore mean several things: the median did not move, a category threshold was not crossed, display rounding hid a small change, or no relevant measurements changed. Our Pennsylvania audit directly demonstrates the first two. Equal values on repeat downloads do not establish a no-revision policy.

### Delphi also has actual revisions

The version of the [Delphi NWSS page](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/nwss.html) retrieved during the initial investigation had a top-level revision label of “Never.” The later retrieval no longer displayed that label. Neither establishes that individual observations cannot change; the archive measurements below are the relevant evidence.

I scanned the existing `flu_avg_conc_lin` archive, keeping the full observation key and comparing values across `report_time`:

- **4,375,142 archive rows**, **326,930 distinct observation keys**.
- **5,091 keys** had differing stored values across report times. This count allows changes involving missing-value markers as well as numeric changes; it is not a count of weekly numeric revision events.
- One definite numeric example: sewershed `1128`, collection date `2026-01-26`, source `State_Territory`, sample index `5618015`: **859.65545** in report `2026-06-26`, then **600** in report `2026-07-06`.

The scan compares numeric representations using `Decimal`, so `1` versus `1.0` does not create a false change. Archive records are not assumed to arrive chronologically. See [Delphi audit output](evidence/delphi_revision_summary.json).

[Delphi V5](https://cmu-delphi.github.io/delphi-epidata/api/v5.html) supports snapshots and archive history. However, the locally downloaded metadata has an earliest report time of **2026-02-25**, despite observations reaching back to 2020. A sample dated 2023 in a February 2026 snapshot is not evidence of what a forecaster knew in 2023. Earlier real-time vintages cannot be recovered simply by filtering observation dates.

## 3. How I would use these data for state forecasting

The following is a proposed research design, not an empirically validated forecasting result.

### Experiment A: add CDC WVAL to the existing clinical model

Start with the model already used for the clinical outcome, or a small pooled autoregression if no baseline exists. Compare it with the same model plus numeric WVAL and its recent change. Use separate pathogen-specific fits initially.

At each weekly forecast origin:

1. Load only the CDC release actually available before the operational cutoff. Save the full state feed and its metadata.
2. Extract the most recently available state WVAL, one- and two-week lags, and a one-week change on a log scale for positive values. Preserve missingness; do not turn “Limited/No Data” into zero.
3. Include coverage and observation age where available. A missing week's change is missing, not a zero change. Do not repeatedly carry a value forward without indicating its age.
4. Forecast the clinical outcome at each requested horizon using clinical lags, seasonality, and the wastewater features. Do not assume wastewater always provides a fixed lead.
5. Compare with the identical model without wastewater using rolling forecast origins and held-out seasons.

For influenza, remember the wastewater signal here is **influenza A**; it is not a measurement of influenza B. A mismatch with an all-influenza outcome should be an explicit modeling assumption.

### Experiment B: a custom state feature from Delphi concentrations

This is worthwhile if it provides better coverage, useful growth information, or more control over preprocessing than WVAL. It is **not CDC WVAL** and should be named accordingly in results.

1. **Retrieve the correct vintage.** Use a Delphi snapshot at the forecast cutoff, or reconstruct from archive records with `report_time` no later than the cutoff. Resolve versions before aggregation; never average archive rows across vintages. Prefer the snapshot service over a hand-written reconstruction until deletion/missingness semantics have been checked.
2. **Join metadata with version awareness.** Obtain the `nwss` auxiliary table and verify its returned schema. Align report time, sewershed, observation time, source, sample index, and PCR target. The NWSS prose documentation calls the observation field `time_value`, while the downloaded signal data use `reference_time`; inspect actual headers. Check join cardinality and unmatched records. Do not assign a whole sewershed observation to every listed county and then sum it back into a state.
3. **Keep measurement regimes distinct.** Define series by site, pathogen, provider/laboratory method, and normalization. Choose a consistent signal within each regime. Start with linearized concentration where comparable metadata are available; compare normalized variants separately. Do not infer geographic identifiers merely by stripping `ID:` from CDC site names without checking correspondence.
4. **Treat censoring and missingness explicitly.** Use an assay-specific detection limit where available. A simple candidate is half the detection limit for nondetects, with sensitivity analysis. If a defensible limit is unavailable, keep a detection indicator and analyze positive measurements separately. Avoid a universal pseudocount across incompatible units.
5. **Standardize using past information only.** For positive/censoring-adjusted concentrations, a candidate within-regime score is:

   ```text
   x[i,t] = log(concentration[i,t])
   z[i,t] = (x[i,t] - training_median[i]) / training_IQR[i]
   ```

   Estimate center and scale exclusively from data available at that forecast origin. Flag or exclude regimes with too little history or zero IQR. This is a research normalization, not the CDC transformation. Test whether a common trailing training window improves comparability across established sites.

6. **Aggregate in stages.** Combine technical replicates within a sample/day as appropriate, then make one robust site-week estimate. Compute an unweighted median of site-week scores within each state. This gives sites, rather than sample counts, equal influence.
7. **Measure growth on matched sites.** Compute the median within-site week-to-week change using sites observed in both weeks. This reduces apparent growth caused solely by changes in site composition. Record reporting-site count, matched-site count, dispersion, missingness, and sample age.
8. **Test population weights as a sensitivity analysis.** Use reliable, non-overlapping population estimates; cap dominant weights if justified and documented. Do not call the sum of site populations statewide coverage unless overlap has been resolved. Compare a stable panel with the changing panel before attributing improvements to epidemiology.

[Delphi's query documentation](https://cmu-delphi.github.io/delphi-epidata/api/v5_api_queries.html) specifies `source`, `signal`, `geo_type`, and optional `snapshot_date` for snapshots, and a separate auxiliary-data endpoint. Conceptually:

```text
snapshot(source="nwss", signal="flu_avg_conc_lin",
         geo_type="sewershed", snapshot_date=forecast_cutoff)
aux_data(source="nwss", report_time <= forecast_cutoff)
```

This is pseudocode, not a tested client invocation. The existing project fetcher and installed client are preferable to building a second downloader. Confirm that auxiliary vintages needed by the join exist; they were not downloaded in this analysis.

### What would count as useful predictive improvement?

Use the same states, horizons, information cutoffs, and target definition for:

| Model | Additional wastewater information |
|---|---|
| Clinical baseline | None |
| Baseline + CDC WVAL | State WVAL level/change, coverage, age |
| Baseline + Delphi feature | Standardized site aggregation, matched-site growth, coverage diagnostics |

Evaluate MAE for point forecasts and weighted interval score plus interval coverage for probabilistic forecasts. Summarize by horizon, state, epidemic phase, and reporting coverage. Report whether any average benefit comes only from a few well-covered states. Tune preprocessing inside training folds, not using the test season.

For periods without authentic historical wastewater releases, label the exercise **retrospective analysis using revised data**. A fixed artificial delay can test sensitivity to latency, but cannot recreate missing historical revisions. Compare latest-vintage and actual-vintage results wherever both exist to quantify revision sensitivity.

### If the intended target is future WVAL itself

Forecast the numeric state series directly, beginning with persistence and a small autoregressive model. Predefine whether truth means the **first published WVAL for the future week** or its **value after a fixed maturation interval**, such as four weeks. Those are different targets. Retain both scores if useful, and evaluate recalibration periods separately. Do not let an arbitrary later annual reprocessing silently redefine the evaluation target.

## 4. Project-specific next steps and limitations

The existing [catalog](../../src/tapestry/data/catalog.py) already has `cdc_nwss_wval` (`atcp-73re`) and `delphi_nwss`. The CDC catalog's measure labels are generic `wval`/`wval_category`; the actual downloaded fields are **`site_wval`/`site_wval_category`**. Check that distinction in any new transform.

The smallest next implementation would be a state-feed snapshot source plus a state-week WVAL feature table. Keep the existing site data for diagnostics. Add the custom Delphi aggregation only as the next controlled experiment, including its auxiliary metadata acquisition.

No nationwide revision frequency, baseline-reset effect size, forecasting accuracy improvement, or exact raw-Delphi-to-CDC equivalence was estimated. The two-release WVAL test is one state over one weekly update. The Delphi scan covers one signal in the existing archive. The state dashboard parity check covers one state and the overlapping displayed dates. These bounds are intentional.

**Correction after follow-up:** `x5bf-y9if` is a HealthData.gov catalog/mirror identifier referencing CDC's `atcp-73re`, not evidence of a discontinued CDC WVAL dataset. Its failure under the CDC domain was not diagnostic. The [HealthData.gov archive](https://healthdata.gov/dataset/CDC-Wastewater-Viral-Activity-Level-for-SARS-CoV-2/nsxq-m59y) returned 14 metadata-history entries. The first linked archived text payload contains links to CDC downloads, not historical WVAL rows. It does not supply a verified numeric vintage archive. See the follow-up below for successful Internet Archive recovery.

## 5. Reproducibility and saved evidence

- [audit.py](audit.py) fetches Pennsylvania's current site WVALs, compares them with the fixed September 4 local snapshot, and scans the fixed local Delphi influenza archive. It uses Python's standard library. Running it later will compare against a newer CDC release.
- [compare_dashboard.py](compare_dashboard.py) compares the saved current state feed and Pennsylvania site values without network calls.
- [revision_summary.json](evidence/revision_summary.json), [delphi_revision_summary.json](evidence/delphi_revision_summary.json), and [dashboard_comparison.json](evidence/dashboard_comparison.json) contain the computed counts and examples.
- The evidence folder preserves both Pennsylvania inputs, the dashboard feed/configuration, CDC metadata, the legacy script, and the local Delphi metadata. [provenance.json](evidence/provenance.json) records inputs and checksums.

From the project root:

```sh
python 'analysis/wval/audit.py'
python 'analysis/wval/compare_dashboard.py'
```

The second command deliberately uses the saved dashboard copy; refresh that copy with provenance if comparing a later site's release. The first command requires the two existing project snapshots at the paths recorded in the script.


## 6. Follow-up: operational availability and Internet Archive recovery

### What is clear, and what remains unknown?

CDC explicitly schedules WVAL updates on Fridays, reporting the previous week. I found no guaranteed publication hour in the cited documentation. The JSON processing stamp is not proof of the exact moment the file first became publicly available. A file may be generated Thursday and exposed Friday; the saved September files illustrate different processing stamps, but do not prove that particular deployment sequence.

For prospective operations, my recommendation is to poll at a chosen interval (for example hourly Thursday–Saturday, daily otherwise), save successful retrieval timestamps in UTC, HTTP validators when supplied, and complete immutable payloads whenever they change. Record the latest successful observation of the old content and first observation of the new content. These bound the observed transition, subject to caching and failed requests. Compare keyed numeric rows as well as whole-file hashes so timestamp-only updates are not counted as value revisions. Use only the latest successfully retrieved data before each forecasting cutoff and flag stale inputs. This is a proposed acquisition design; no scheduled monitor was created.

Revision magnitude remains incompletely characterized. The Pennsylvania experiment establishes that revisions occur and can leave state medians unchanged; it does not estimate nationwide revision distributions. A proper audit would compare successive vintages by pathogen, state, observation age, numeric change, missingness, and category change, and separate routine weekly updates from baseline/method resets.

### Exactly what February 25 means for Delphi

Delphi obtains concentrations from CDC's public Socrata NWSS datasets: [influenza A `ymmh-divb`](https://data.cdc.gov/d/ymmh-divb), [SARS-CoV-2 `j9g8-acpt`](https://data.cdc.gov/d/j9g8-acpt), and [RSV `45cq-cw4i`](https://data.cdc.gov/d/45cq-cw4i). CDC collects upstream reports from public health agencies and testing networks. Delphi documents its ingestion of unversioned CDC data in its [NWSS source description](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/nwss.html).

The checked Delphi metadata starts report-time history on **2026-02-25**. That is a limitation of the checked **concentration vintage archive**, not a start date for WVAL observations. Delphi does not expose WVAL in this source. Older samples exist, but their originally published values before that date are not supplied by the archive we examined. Concentration revisions observed since February do not reconstruct CDC's historical WVAL baselines or its earlier published state series.

### Internet Archive: numeric payloads recovered successfully

I checked the Wayback CDX index and downloaded actual archived payloads, rather than relying on page screenshots:

| Archived resource | Capture time (UTC) | Verification |
|---|---|---|
| Current state page URL | 2026-04-10 19:48:48, among other captures | HTML capture indexed; page alone does not guarantee chart preservation |
| Combined state JSON | 2026-04-17 23:35:01 | Successfully downloaded and parsed 16,370 rows; sample embedded stamp April 16, 2026 |
| Legacy COVID state CSV | **2025-07-21 23:22:02** | Successfully downloaded numeric state WVAL rows; sample embedded stamp July 17, 2025 |

Links: [April 2026 JSON capture](https://web.archive.org/web/20260417233501id_/https://www.cdc.gov/wcms/vizdata/NCEZID_DIDRI/NWSS_WVAL_metric/NWSSWVALStateActivityLevel.json), [July 2025 COVID CSV capture](https://web.archive.org/web/20250721232202id_/https://www.cdc.gov/wcms/vizdata/NCEZID_DIDRI/SC2/nwsssc2stateactivitylevelDL.csv).

The combined JSON index returned five distinct content digests across April–August 2026. This is sparse recovery, not a demonstrated weekly archive. The legacy COVID CSV repeats observations across display-period selections (`1 Year`, `All Results`); deduplicate consistent repeated observations before analysis.

**Operational interpretation:** a successful July 21 capture establishes availability by July 21. Its July 17 processing stamp does not prove public availability at a July 18 forecast cutoff. Use capture timestamps conservatively unless separate evidence establishes an earlier release. Archived HTML may omit separately fetched JSON, so recover each data asset and verify its actual timestamp, response body, schema, and pathogen. Other historical URLs may yield more vintages; exhaustive archive discovery and a national revision audit remain unperformed.

Saved additions include the three CDX index responses, recovered [April JSON](evidence/wayback_state_20260417.json), recovered [July COVID CSV](evidence/wayback_covid_20250721.csv), [HealthData.gov metadata history](evidence/healthdata_archive_rows.json), and its [sample archive payload](evidence/healthdata_archive_sample.txt).
