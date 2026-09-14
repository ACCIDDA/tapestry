# Local measure inventory by pathogen

Updated 2026-09-13 after selection policy 4. This supersedes the earlier inventory. Frequency means observation cadence. Publisher display names come from the saved CDC API metadata. Columns below are selected measures; Delphi signals and Hub targets identify measures stored in long-form tables.

25 acquisition datasets; 23 downloaded; 17 with indexed series; 12 logical origin groups (including groups not currently available). NHSN now has 14 all-age measures. Adult, pediatric, individual age-band, and unknown-age NHSN fields are excluded from selection. Raw downloads retain their native schemas.

Delphi is orange and Hub green in the explorer’s signal list; plotted lines each have a distinct color. Current Hub hospitalization and ED targets join NHSN and NSSP respectively. Hub ED values are 0–1 proportions; CDC/Delphi ED values are percentages. Grouping preserves values and provider variants. Smoothed and reported columns remain separate.

CDC publishes separate weekly [state WVAL](https://www.cdc.gov/wastewater/respiratory-viruses/state.html) for influenza A, COVID-19 and RSV, but this project currently downloads only site WVAL and raw wastewater concentrations. State WVAL is not yet cataloged.

## Influenza

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| National respiratory-virus percent positivity | Influenza · percent_test_positivity | `percent_test_positivity` | `cdc_nrevss_national`: `percent_test_positivity` (weekly) |
| Legacy FluSight hospitalization truth | Incident Hospitalizations | `value` | `hub_flusight_legacy`: `value` (weekly) |
| NHSN · Hospital surveillance | Total Patients Hospitalized with Influenza | `totalconffluhosppats` | `cdc_nhsn_final`: `totalconffluhosppats` (weekly)<br>`cdc_nhsn_initial_release`: `totalconffluhosppats` (weekly)<br>`cdc_nhsn_preliminary`: `totalconffluhosppats` (weekly) |
| NHSN · Hospital surveillance | Total ICU Patients Hospitalized with Influenza | `totalconffluicupats` | `cdc_nhsn_final`: `totalconffluicupats` (weekly)<br>`cdc_nhsn_initial_release`: `totalconffluicupats` (weekly)<br>`cdc_nhsn_preliminary`: `totalconffluicupats` (weekly) |
| NHSN · Hospital surveillance | Total Influenza Admissions | `totalconfflunewadm` | `cdc_nhsn_final`: `totalconfflunewadm` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfflunewadm` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfflunewadm` (weekly)<br>`delphi_nhsn`: `confirmed_admissions_flu_ew` (weekly)<br>`hub_flusight_current`: `wk inc flu hosp` (weekly) |
| NHSN · Hospital surveillance | Number Hospitals Reporting Influenza Admissions | `totalconfflunewadmhosprep` | `cdc_nhsn_final`: `totalconfflunewadmhosprep` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfflunewadmhosprep` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfflunewadmhosprep` (weekly)<br>`delphi_nhsn`: `hosprep_confirmed_admissions_flu_ew` (weekly) |
| NSSP · Emergency department surveillance | percent_visits_influenza | `percent_visits_influenza` | `cdc_nssp_daily`: `percent_visits` (daily)<br>`cdc_nssp_demographics`: `percent_visits` (weekly)<br>`cdc_nssp_trajectories`: `percent_visits_influenza` (weekly)<br>`delphi_nssp`: `pct_ed_visits_influenza` (weekly)<br>`hub_flusight_current`: `wk inc flu prop ed visits` (weekly) |
| NSSP · Emergency department surveillance | percent_visits_smoothed_influenza | `percent_visits_smoothed_1` | `cdc_nssp_trajectories`: `percent_visits_smoothed_1` (weekly)<br>`delphi_nssp`: `smoothed_pct_ed_visits_influenza` (weekly) |

## COVID-19

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NREVSS comprehensive respiratory NAAT surveillance | SARS-COV-2 · Detections | `detections` | `cdc_nrevss_comprehensive`: `detections` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | SARS-COV-2 · Detections_5wma | `detections_5wma` | `cdc_nrevss_comprehensive`: `detections_5wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | SARS-COV-2 · Percent_pos | `percent_pos` | `cdc_nrevss_comprehensive`: `percent_pos` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | SARS-COV-2 · Percent_Pos_3wma | `percent_pos_3wma` | `cdc_nrevss_comprehensive`: `percent_pos_3wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | SARS-COV-2 · Tests | `tests` | `cdc_nrevss_comprehensive`: `tests` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | SARS-COV-2 · Tests_3wma | `tests_3wma` | `cdc_nrevss_comprehensive`: `tests_3wma` (weekly) |
| NREVSS COVID-19 NAAT posted vintages | number_tested | `number_tested` | `cdc_nrevss_covid_vintages`: `number_tested` (weekly) |
| NREVSS COVID-19 NAAT posted vintages | number_tested_2_week | `number_tested_2_week` | `cdc_nrevss_covid_vintages`: `number_tested_2_week` (weekly) |
| NREVSS COVID-19 NAAT posted vintages | number_tested_4_week | `number_tested_4_week` | `cdc_nrevss_covid_vintages`: `number_tested_4_week` (weekly) |
| NREVSS COVID-19 NAAT posted vintages | perc_diff | `perc_diff` | `cdc_nrevss_covid_vintages`: `perc_diff` (weekly) |
| NREVSS COVID-19 NAAT posted vintages | percent_pos | `percent_pos` | `cdc_nrevss_covid_vintages`: `percent_pos` (weekly) |
| NREVSS COVID-19 NAAT posted vintages | percent_pos_2_week | `percent_pos_2_week` | `cdc_nrevss_covid_vintages`: `percent_pos_2_week` (weekly) |
| NREVSS COVID-19 NAAT posted vintages | percent_pos_4_week | `percent_pos_4_week` | `cdc_nrevss_covid_vintages`: `percent_pos_4_week` (weekly) |
| National respiratory-virus percent positivity | COVID-19 · percent_test_positivity | `percent_test_positivity` | `cdc_nrevss_national`: `percent_test_positivity` (weekly) |
| NHSN · Hospital surveillance | Total Patients Hospitalized with COVID-19 | `totalconfc19hosppats` | `cdc_nhsn_final`: `totalconfc19hosppats` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfc19hosppats` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfc19hosppats` (weekly) |
| NHSN · Hospital surveillance | Total ICU Patients Hospitalized with COVID-19  | `totalconfc19icupats` | `cdc_nhsn_final`: `totalconfc19icupats` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfc19icupats` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfc19icupats` (weekly) |
| NHSN · Hospital surveillance | Total COVID-19 Admissions | `totalconfc19newadm` | `cdc_nhsn_final`: `totalconfc19newadm` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfc19newadm` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfc19newadm` (weekly)<br>`delphi_nhsn`: `confirmed_admissions_covid_ew` (weekly)<br>`hub_covid_current`: `wk inc covid hosp` (weekly) |
| NHSN · Hospital surveillance | Number Hospitals Reporting COVID-19 Admissions | `totalconfc19newadmhosprep` | `cdc_nhsn_final`: `totalconfc19newadmhosprep` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfc19newadmhosprep` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfc19newadmhosprep` (weekly)<br>`delphi_nhsn`: `hosprep_confirmed_admissions_covid_ew` (weekly) |
| NSSP · Emergency department surveillance | percent_visits_covid | `percent_visits_covid` | `cdc_nssp_daily`: `percent_visits` (daily)<br>`cdc_nssp_demographics`: `percent_visits` (weekly)<br>`cdc_nssp_trajectories`: `percent_visits_covid` (weekly)<br>`delphi_nssp`: `pct_ed_visits_covid` (weekly)<br>`hub_covid_current`: `wk inc covid prop ed visits` (weekly) |
| NSSP · Emergency department surveillance | percent_visits_smoothed_covid | `percent_visits_smoothed_covid` | `cdc_nssp_trajectories`: `percent_visits_smoothed_covid` (weekly)<br>`delphi_nssp`: `smoothed_pct_ed_visits_covid` (weekly) |

## RSV

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NREVSS comprehensive respiratory NAAT surveillance | RSV · Detections | `detections` | `cdc_nrevss_comprehensive`: `detections` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RSV · Detections_5wma | `detections_5wma` | `cdc_nrevss_comprehensive`: `detections_5wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RSV · Percent_pos | `percent_pos` | `cdc_nrevss_comprehensive`: `percent_pos` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RSV · Percent_Pos_3wma | `percent_pos_3wma` | `cdc_nrevss_comprehensive`: `percent_pos_3wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RSV · Tests | `tests` | `cdc_nrevss_comprehensive`: `tests` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RSV · Tests_3wma | `tests_3wma` | `cdc_nrevss_comprehensive`: `tests_3wma` (weekly) |
| National respiratory-virus percent positivity | RSV · percent_test_positivity | `percent_test_positivity` | `cdc_nrevss_national`: `percent_test_positivity` (weekly) |
| NREVSS RSV NAAT posted vintages | detections_2_week | `detections_2_week` | `cdc_nrevss_rsv_vintages`: `detections_2_week` (weekly) |
| NREVSS RSV NAAT posted vintages | detections_4_week | `detections_4_week` | `cdc_nrevss_rsv_vintages`: `detections_4_week` (weekly) |
| NREVSS RSV NAAT posted vintages | pcr_detections | `pcr_detections` | `cdc_nrevss_rsv_vintages`: `pcr_detections` (weekly) |
| NREVSS RSV NAAT posted vintages | pcr_percent_positive | `pcr_percent_positive` | `cdc_nrevss_rsv_vintages`: `pcr_percent_positive` (weekly) |
| NREVSS RSV NAAT posted vintages | perc_diff | `perc_diff` | `cdc_nrevss_rsv_vintages`: `perc_diff` (weekly) |
| NREVSS RSV NAAT posted vintages | percent_pos_2_week | `percent_pos_2_week` | `cdc_nrevss_rsv_vintages`: `percent_pos_2_week` (weekly) |
| NREVSS RSV NAAT posted vintages | percent_pos_4_week | `percent_pos_4_week` | `cdc_nrevss_rsv_vintages`: `percent_pos_4_week` (weekly) |
| Johns Hopkins RSV Forecast Hub target data | rate hosp | `value` | `hub_rsvnet`: `inc hosp` (weekly)<br>`hub_rsvnet`: `rate hosp` (weekly) |
| NHSN · Hospital surveillance | Total Patients Hospitalized with RSV | `totalconfrsvhosppats` | `cdc_nhsn_final`: `totalconfrsvhosppats` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfrsvhosppats` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfrsvhosppats` (weekly) |
| NHSN · Hospital surveillance | Total ICU Patients Hospitalized with RSV | `totalconfrsvicupats` | `cdc_nhsn_final`: `totalconfrsvicupats` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfrsvicupats` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfrsvicupats` (weekly) |
| NHSN · Hospital surveillance | Total RSV Admissions | `totalconfrsvnewadm` | `cdc_nhsn_final`: `totalconfrsvnewadm` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfrsvnewadm` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfrsvnewadm` (weekly)<br>`delphi_nhsn`: `confirmed_admissions_rsv_ew` (weekly)<br>`hub_rsv_current`: `wk inc rsv hosp` (weekly) |
| NHSN · Hospital surveillance | Number Hospitals Reporting RSV Admissions | `totalconfrsvnewadmhosprep` | `cdc_nhsn_final`: `totalconfrsvnewadmhosprep` (weekly)<br>`cdc_nhsn_initial_release`: `totalconfrsvnewadmhosprep` (weekly)<br>`cdc_nhsn_preliminary`: `totalconfrsvnewadmhosprep` (weekly)<br>`delphi_nhsn`: `hosprep_confirmed_admissions_rsv_ew` (weekly) |
| NSSP · Emergency department surveillance | percent_visits_rsv | `percent_visits_rsv` | `cdc_nssp_daily`: `percent_visits` (daily)<br>`cdc_nssp_demographics`: `percent_visits` (weekly)<br>`cdc_nssp_trajectories`: `percent_visits_rsv` (weekly)<br>`delphi_nssp`: `pct_ed_visits_rsv` (weekly)<br>`hub_rsv_current`: `wk inc rsv prop ed visits` (weekly) |
| NSSP · Emergency department surveillance | percent_visits_smoothed_rsv | `percent_visits_smoothed_rsv` | `cdc_nssp_trajectories`: `percent_visits_smoothed_rsv` (weekly)<br>`delphi_nssp`: `smoothed_pct_ed_visits_rsv` (weekly) |

## ARI

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NSSP · Emergency department surveillance | ARI · percent_visits | `percent_visits_ari` | `cdc_nssp_daily`: `percent_visits` (daily)<br>`delphi_nssp`: `pct_ed_visits_ari` (weekly) |

## Combined respiratory

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NSSP · Emergency department surveillance | percent_visits_combined | `percent_visits_combined` | `cdc_nssp_demographics`: `percent_visits` (weekly)<br>`cdc_nssp_trajectories`: `percent_visits_combined` (weekly)<br>`delphi_nssp`: `pct_ed_visits_combined` (weekly) |
| NSSP · Emergency department surveillance | percent_visits_smoothed_combined | `percent_visits_smoothed` | `cdc_nssp_trajectories`: `percent_visits_smoothed` (weekly)<br>`delphi_nssp`: `smoothed_pct_ed_visits_combined` (weekly) |

## Adenovirus

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NREVSS comprehensive respiratory NAAT surveillance | Adenovirus · Detections | `detections` | `cdc_nrevss_comprehensive`: `detections` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | Adenovirus · Detections_5wma | `detections_5wma` | `cdc_nrevss_comprehensive`: `detections_5wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | Adenovirus · Percent_pos | `percent_pos` | `cdc_nrevss_comprehensive`: `percent_pos` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | Adenovirus · Percent_Pos_3wma | `percent_pos_3wma` | `cdc_nrevss_comprehensive`: `percent_pos_3wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | Adenovirus · Tests | `tests` | `cdc_nrevss_comprehensive`: `tests` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | Adenovirus · Tests_3wma | `tests_3wma` | `cdc_nrevss_comprehensive`: `tests_3wma` (weekly) |

## Seasonal coronaviruses (HCOV)

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NREVSS comprehensive respiratory NAAT surveillance | HCOV · Detections | `detections` | `cdc_nrevss_comprehensive`: `detections` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HCOV · Detections_5wma | `detections_5wma` | `cdc_nrevss_comprehensive`: `detections_5wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HCOV · Percent_pos | `percent_pos` | `cdc_nrevss_comprehensive`: `percent_pos` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HCOV · Percent_Pos_3wma | `percent_pos_3wma` | `cdc_nrevss_comprehensive`: `percent_pos_3wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HCOV · Tests | `tests` | `cdc_nrevss_comprehensive`: `tests` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HCOV · Tests_3wma | `tests_3wma` | `cdc_nrevss_comprehensive`: `tests_3wma` (weekly) |

## Human metapneumovirus (HMPV)

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NREVSS comprehensive respiratory NAAT surveillance | HMPV · Detections | `detections` | `cdc_nrevss_comprehensive`: `detections` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HMPV · Detections_5wma | `detections_5wma` | `cdc_nrevss_comprehensive`: `detections_5wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HMPV · Percent_pos | `percent_pos` | `cdc_nrevss_comprehensive`: `percent_pos` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HMPV · Percent_Pos_3wma | `percent_pos_3wma` | `cdc_nrevss_comprehensive`: `percent_pos_3wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HMPV · Tests | `tests` | `cdc_nrevss_comprehensive`: `tests` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | HMPV · Tests_3wma | `tests_3wma` | `cdc_nrevss_comprehensive`: `tests_3wma` (weekly) |

## Parainfluenza (PIV)

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NREVSS comprehensive respiratory NAAT surveillance | PIV · Detections | `detections` | `cdc_nrevss_comprehensive`: `detections` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | PIV · Detections_5wma | `detections_5wma` | `cdc_nrevss_comprehensive`: `detections_5wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | PIV · Percent_pos | `percent_pos` | `cdc_nrevss_comprehensive`: `percent_pos` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | PIV · Percent_Pos_3wma | `percent_pos_3wma` | `cdc_nrevss_comprehensive`: `percent_pos_3wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | PIV · Tests | `tests` | `cdc_nrevss_comprehensive`: `tests` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | PIV · Tests_3wma | `tests_3wma` | `cdc_nrevss_comprehensive`: `tests_3wma` (weekly) |

## Rhinovirus/enterovirus (RV/EV)

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NREVSS comprehensive respiratory NAAT surveillance | RV/EV · Detections | `detections` | `cdc_nrevss_comprehensive`: `detections` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RV/EV · Detections_5wma | `detections_5wma` | `cdc_nrevss_comprehensive`: `detections_5wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RV/EV · Percent_pos | `percent_pos` | `cdc_nrevss_comprehensive`: `percent_pos` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RV/EV · Percent_Pos_3wma | `percent_pos_3wma` | `cdc_nrevss_comprehensive`: `percent_pos_3wma` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RV/EV · Tests | `tests` | `cdc_nrevss_comprehensive`: `tests` (weekly) |
| NREVSS comprehensive respiratory NAAT surveillance | RV/EV · Tests_3wma | `tests_3wma` | `cdc_nrevss_comprehensive`: `tests_3wma` (weekly) |

## Shared hospital capacity

| Origin | CDC display name / measure | Origin field | Available acquisition / signal / frequency |
|---|---|---|---|
| NHSN · Hospital surveillance | Number of Inpatient Beds | `numinptbeds` | `cdc_nhsn_final`: `numinptbeds` (weekly)<br>`cdc_nhsn_initial_release`: `numinptbeds` (weekly)<br>`cdc_nhsn_preliminary`: `numinptbeds` (weekly)<br>`delphi_nhsn`: `inpatient_beds_ew` (weekly) |
| NHSN · Hospital surveillance | Number of Inpatient Beds Occupied | `numinptbedsocc` | `cdc_nhsn_final`: `numinptbedsocc` (weekly)<br>`cdc_nhsn_initial_release`: `numinptbedsocc` (weekly)<br>`cdc_nhsn_preliminary`: `numinptbedsocc` (weekly)<br>`delphi_nhsn`: `inpatient_beds_occupied_pct_ew` (weekly) |

## All acquisition datasets

| Dataset | Observation cadence | Availability / revisions |
|---|---|---|
| `cdc_nhsn_final` | weekly | Current snapshot |
| `cdc_nhsn_initial_release` | weekly | Frozen first publication |
| `cdc_nhsn_preliminary` | weekly | Current snapshot |
| `cdc_nrevss_comprehensive` | weekly | Current snapshot |
| `cdc_nrevss_covid_vintages` | weekly | 8 indexed release timestamps |
| `cdc_nrevss_national` | weekly | Current snapshot |
| `cdc_nrevss_rsv_vintages` | weekly | 8 indexed release timestamps |
| `cdc_nssp_daily` | daily | Current snapshot |
| `cdc_nssp_demographics` | weekly | Current snapshot |
| `cdc_nssp_trajectories` | weekly | Current snapshot |
| `cdc_nwss_covid_raw` | sample | Downloaded; site/sewershed observations excluded from explorer |
| `cdc_nwss_influenza_raw` | sample | Downloaded; site/sewershed observations excluded from explorer |
| `cdc_nwss_rsv_raw` | sample | Downloaded; site/sewershed observations excluded from explorer |
| `cdc_nwss_wval` | weekly | Downloaded; site/sewershed observations excluded from explorer |
| `delphi_claims_inpatient` | daily | Not downloaded; report_time archive supported |
| `delphi_claims_outpatient` | daily | Not downloaded; report_time archive supported |
| `delphi_nhsn` | weekly | 178 indexed release timestamps |
| `delphi_nssp` | weekly | 155 indexed release timestamps |
| `delphi_nwss` | sample | Downloaded; site/sewershed observations excluded from explorer |
| `hub_covid_current` | weekly | 86 indexed release timestamps |
| `hub_covid_legacy` | weekly | Canonical truth payloads unavailable (Git LFS pointers) |
| `hub_flusight_current` | weekly | 96 indexed release timestamps |
| `hub_flusight_legacy` | weekly | One exported commit; Git history not indexed |
| `hub_rsv_current` | weekly | 86 indexed release timestamps |
| `hub_rsvnet` | weekly | One exported commit; Git history not indexed |

## Delphi measure lists

These are Delphi signal identifiers, not CSV storage columns. `value` holds each signal’s observations; dates and geography describe the observations.

### delphi_claims_inpatient

Configured signals; not downloaded.

| Delphi measure | Origin group | CDC field | CDC display name / measure |
|---|---|---|---|
| `claims_inpatient_adm_pct_claims_covid` | Delphi inpatient claims revision archive | Unmapped | claims_inpatient_adm_pct_claims_covid |
| `claims_inpatient_adm_pct_claims_flu` | Delphi inpatient claims revision archive | Unmapped | claims_inpatient_adm_pct_claims_flu |
| `claims_inpatient_adm_pct_ari_other` | Delphi inpatient claims revision archive | Unmapped | claims_inpatient_adm_pct_ari_other |

### delphi_claims_outpatient

Configured signals; not downloaded.

| Delphi measure | Origin group | CDC field | CDC display name / measure |
|---|---|---|---|
| `claims_outpatient_ov_pct_claims_covid` | Delphi outpatient claims revision archive | Unmapped | claims_outpatient_ov_pct_claims_covid |
| `claims_outpatient_ov_pct_claims_flu` | Delphi outpatient claims revision archive | Unmapped | claims_outpatient_ov_pct_claims_flu |
| `claims_outpatient_ov_pct_ari_other` | Delphi outpatient claims revision archive | Unmapped | claims_outpatient_ov_pct_ari_other |

### delphi_nhsn

Downloaded signals.

| Delphi measure | Origin group | CDC field | CDC display name / measure |
|---|---|---|---|
| `confirmed_admissions_covid_ew` | NHSN · Hospital surveillance | `totalconfc19newadm` | Total COVID-19 Admissions |
| `confirmed_admissions_flu_ew` | NHSN · Hospital surveillance | `totalconfflunewadm` | Total Influenza Admissions |
| `confirmed_admissions_rsv_ew` | NHSN · Hospital surveillance | `totalconfrsvnewadm` | Total RSV Admissions |
| `hosprep_confirmed_admissions_covid_ew` | NHSN · Hospital surveillance | `totalconfc19newadmhosprep` | Number Hospitals Reporting COVID-19 Admissions |
| `hosprep_confirmed_admissions_flu_ew` | NHSN · Hospital surveillance | `totalconfflunewadmhosprep` | Number Hospitals Reporting Influenza Admissions |
| `hosprep_confirmed_admissions_rsv_ew` | NHSN · Hospital surveillance | `totalconfrsvnewadmhosprep` | Number Hospitals Reporting RSV Admissions |
| `inpatient_beds_ew` | NHSN · Hospital surveillance | `numinptbeds` | Number of Inpatient Beds |
| `inpatient_beds_occupied_pct_ew` | NHSN · Hospital surveillance | `numinptbedsocc` | Number of Inpatient Beds Occupied |

### delphi_nssp

Downloaded signals.

| Delphi measure | Origin group | CDC field | CDC display name / measure |
|---|---|---|---|
| `pct_ed_visits_ari` | NSSP · Emergency department surveillance | `percent_visits_ari` | ARI · ED visit percentage |
| `pct_ed_visits_combined` | NSSP · Emergency department surveillance | `percent_visits_combined` | percent_visits_combined |
| `pct_ed_visits_covid` | NSSP · Emergency department surveillance | `percent_visits_covid` | percent_visits_covid |
| `pct_ed_visits_influenza` | NSSP · Emergency department surveillance | `percent_visits_influenza` | percent_visits_influenza |
| `pct_ed_visits_rsv` | NSSP · Emergency department surveillance | `percent_visits_rsv` | percent_visits_rsv |
| `smoothed_pct_ed_visits_combined` | NSSP · Emergency department surveillance | `percent_visits_smoothed` | percent_visits_smoothed_combined |
| `smoothed_pct_ed_visits_covid` | NSSP · Emergency department surveillance | `percent_visits_smoothed_covid` | percent_visits_smoothed_covid |
| `smoothed_pct_ed_visits_influenza` | NSSP · Emergency department surveillance | `percent_visits_smoothed_1` | percent_visits_smoothed_influenza |
| `smoothed_pct_ed_visits_rsv` | NSSP · Emergency department surveillance | `percent_visits_smoothed_rsv` | percent_visits_smoothed_rsv |

### delphi_nwss

Downloaded signals.

| Delphi measure | Origin group | CDC field | CDC display name / measure |
|---|---|---|---|
| `covid_avg_conc` | NWSS · Wastewater surveillance | `pcr_target_avg_conc` | pcr_target_avg_conc |
| `covid_avg_conc_lin` | NWSS · Wastewater surveillance | `pcr_target_avg_conc_lin` | pcr_target_avg_conc_lin |
| `covid_flowpop_lin` | NWSS · Wastewater surveillance | `pcr_target_flowpop_lin` | pcr_target_flowpop_lin |
| `covid_mic_lin` | NWSS · Wastewater surveillance | `pcr_target_mic_lin` | pcr_target_mic_lin |
| `flu_avg_conc` | NWSS · Wastewater surveillance | `pcr_target_avg_conc` | pcr_target_avg_conc |
| `flu_avg_conc_lin` | NWSS · Wastewater surveillance | `pcr_target_avg_conc_lin` | pcr_target_avg_conc_lin |
| `flu_flowpop_lin` | NWSS · Wastewater surveillance | `pcr_target_flowpop_lin` | pcr_target_flowpop_lin |
| `flu_mic_lin` | NWSS · Wastewater surveillance | `pcr_target_mic_lin` | pcr_target_mic_lin |
| `rsv_avg_conc` | NWSS · Wastewater surveillance | `pcr_target_avg_conc` | pcr_target_avg_conc |
| `rsv_avg_conc_lin` | NWSS · Wastewater surveillance | `pcr_target_avg_conc_lin` | pcr_target_avg_conc_lin |
| `rsv_flowpop_lin` | NWSS · Wastewater surveillance | `pcr_target_flowpop_lin` | pcr_target_flowpop_lin |
| `rsv_mic_lin` | NWSS · Wastewater surveillance | `pcr_target_mic_lin` | pcr_target_mic_lin |

## Scope and limitations

Availability is local and varies by location, age/demographic/subtype facet, and date. Only native state and national support is indexed. Release counts are dataset-wide distinct timestamps, not the number of revisions for every observation. The explorer does not traverse Git history. Current FluSight retains its existing quarantine of 255 conflicting observation/release keys.

CDC API field identifiers are retained in raw rows; publisher display names and descriptions are saved by new downloads in `columns.json`. Existing snapshots use their saved `metadata.json` directly. The explorer’s “displayed versions” count refers to curves currently selected, not all stored release dates.
