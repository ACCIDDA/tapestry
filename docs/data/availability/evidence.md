# Availability evidence: unknown timing, later arrival, and missing values

**[Open the evidence explorer](timeline/evidence.html)**. Select a covariate/target and then a state. The aggregate view can mix statuses across states; hover lists the states in each category.

| Mark | What the archive establishes | Use for next season |
| --- | --- | --- |
| Green | A finite value was reported by this deadline | Observed availability |
| Gray | No eligible report; a value exists elsewhere in the inspected history or final panel | Timing unknown. Infer timing from documented periods, not this gap |
| Blue | An explicit missing statement at this deadline, followed by a later finite report | Documented missing-then-present transition; not necessarily first-ever publication |
| Red × | Explicit missing statement, no later finite report observed in our archive | Missing/withdrawn at that deadline; permanent absence is not proven |
| Beige × | No finite value anywhere in the inspected archive or final panel for that location/week | Never observed in this dataset; may be unsupported or an unresolved archive gap |
| Yellow | Different locations have different evidence | Hover or select a state |
| White | Observation week is still in the future | Not an availability failure |

A value appearing in a later archive **does not by itself prove late publication**. It could be an old value first collected later. We therefore require a preceding explicit missing statement for blue; otherwise timing stays gray. An explicit null can also mean suppression or withdrawal and is not evidence of “never ever.” Crosses are separated by color for that reason. Native unsupported states can be selected individually; aggregate views exclude them.

The classification uses known-report masks in the frozen deadline panel, plus raw claims/ILI/lab/FluSurv statement histories before conflict rejection. Finite conflicting claims count as reported. For those raw sources, later reports are checked through the pinned snapshot; for other sources, through the panel's last available issuance. Final-only values do not prove when a later report occurred. No speculative release delays are filled into this view.

For next-season assumptions, use stable observed release periods to estimate delay, and label any extension over gray periods as inferred. This page does not convert those assumptions into historical observations. The prior coverage heatmaps still answer how many values the archive supplies; this evidence view answers what we can conclude about missing cells.

[Coverage heatmaps](timeline.md) · [Evidence data](timeline/evidence.json) · [Evidence counts](timeline/evidence-counts.csv).

## nhsn_flu_admissions

![Evidence for nhsn_flu_admissions](timeline/nhsn_flu_admissions-evidence.png)

## nhsn_covid_admissions

![Evidence for nhsn_covid_admissions](timeline/nhsn_covid_admissions-evidence.png)

## nhsn_rsv_admissions

![Evidence for nhsn_rsv_admissions](timeline/nhsn_rsv_admissions-evidence.png)

## nssp_flu_proportion

![Evidence for nssp_flu_proportion](timeline/nssp_flu_proportion-evidence.png)

## nssp_covid_proportion

![Evidence for nssp_covid_proportion](timeline/nssp_covid_proportion-evidence.png)

## nssp_rsv_proportion

![Evidence for nssp_rsv_proportion](timeline/nssp_rsv_proportion-evidence.png)

## inpatient_flu

![Evidence for inpatient_flu](timeline/inpatient_flu-evidence.png)

## inpatient_covid

![Evidence for inpatient_covid](timeline/inpatient_covid-evidence.png)

## outpatient_flu

![Evidence for outpatient_flu](timeline/outpatient_flu-evidence.png)

## outpatient_covid

![Evidence for outpatient_covid](timeline/outpatient_covid-evidence.png)

## nwss_flu_wval_like

![Evidence for nwss_flu_wval_like](timeline/nwss_flu_wval_like-evidence.png)

## nwss_covid_wval_like

![Evidence for nwss_covid_wval_like](timeline/nwss_covid_wval_like-evidence.png)

## nwss_rsv_wval_like

![Evidence for nwss_rsv_wval_like](timeline/nwss_rsv_wval_like-evidence.png)

## nwss_flu_pct_rank

![Evidence for nwss_flu_pct_rank](timeline/nwss_flu_pct_rank-evidence.png)

## nwss_covid_pct_rank

![Evidence for nwss_covid_pct_rank](timeline/nwss_covid_pct_rank-evidence.png)

## nwss_rsv_pct_rank

![Evidence for nwss_rsv_pct_rank](timeline/nwss_rsv_pct_rank-evidence.png)

## ilinet_ili

![Evidence for ilinet_ili](timeline/ilinet_ili-evidence.png)

## clinical_lab_flu_pct_positive

![Evidence for clinical_lab_flu_pct_positive](timeline/clinical_lab_flu_pct_positive-evidence.png)

## flusurv_flu_rate

![Evidence for flusurv_flu_rate](timeline/flusurv_flu_rate-evidence.png)

## kinsa_ili

![Evidence for kinsa_ili](timeline/kinsa_ili-evidence.png)
