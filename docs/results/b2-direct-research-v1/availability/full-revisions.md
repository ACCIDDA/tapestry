# All covariates: revisions at every lag

These are **conditional on the old pipeline retaining both an unambiguous deadline value and final value**. The conflict correction improves availability counts but does not pick an arbitrary value to recompute revisions. Percentages are sum of absolute revisions divided by sum of absolute final values, ×100; n is the number of finite pairs. Missing pairs do not count as zero change. Sample composition varies by lag. Frozen final values can themselves have missing cells from conflict rejection.

[Original-unit bias, mean and 90th percentile absolute changes](summary.csv) · [Availability and missing states](full-tables.md) · [Conflict diagnosis](conflict-audit.md).

| Covariate | Frequency / representation | Lag 0 | Lag 1 | Lag 2 | Lag 3 | Lag 4 | Lag 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inpatient_flu | Daily; Saturday trailing-7-day value | 59.1% (n=1610) | 39.4% (n=1771) | 29.8% (n=1829) | 23.5% (n=1881) | 18.1% (n=1934) | 14.7% (n=1990) |
| inpatient_covid | Daily; Saturday trailing-7-day value | 67.1% (n=1407) | 43.1% (n=1552) | 31.6% (n=1623) | 24.6% (n=1690) | 20.0% (n=1740) | 17.3% (n=1807) |
| outpatient_flu | Daily; Saturday trailing-7-day value | 26.7% (n=768) | 17.3% (n=804) | 13.8% (n=854) | 12.2% (n=884) | 11.0% (n=926) | 8.8% (n=972) |
| outpatient_covid | Daily; Saturday trailing-7-day value | 38.6% (n=704) | 29.4% (n=744) | 25.4% (n=800) | 24.4% (n=822) | 22.6% (n=864) | 19.2% (n=909) |
| nwss_flu_wval_like | Weekly derived index; irregular samples | 31.6% (n=339) | 14.0% (n=805) | 11.6% (n=1012) | 10.7% (n=1039) | 10.6% (n=1056) | 10.4% (n=1070) |
| nwss_covid_wval_like | Weekly derived index; irregular samples | 38.7% (n=326) | 26.1% (n=774) | 23.0% (n=1059) | 21.4% (n=1093) | 20.7% (n=1107) | 20.5% (n=1120) |
| nwss_rsv_wval_like | Weekly derived index; irregular samples | 50.7% (n=443) | 15.8% (n=948) | 12.2% (n=1021) | 10.4% (n=1037) | 10.2% (n=1046) | 10.0% (n=1054) |
| nwss_flu_pct_rank | Weekly derived index; irregular samples | 17.7% (n=339) | 8.8% (n=805) | 6.5% (n=1012) | 5.6% (n=1039) | 5.2% (n=1056) | 4.8% (n=1070) |
| nwss_covid_pct_rank | Weekly derived index; irregular samples | 38.6% (n=326) | 26.4% (n=774) | 21.1% (n=1059) | 19.1% (n=1093) | 17.7% (n=1107) | 16.8% (n=1120) |
| nwss_rsv_pct_rank | Weekly derived index; irregular samples | 19.9% (n=443) | 6.1% (n=948) | 4.4% (n=1021) | 4.0% (n=1037) | 3.9% (n=1046) | 3.8% (n=1054) |
| ilinet_ili | Weekly | 6.7% (n=52) | 2.5% (n=2001) | 1.2% (n=2054) | 0.9% (n=2107) | 0.8% (n=2160) | 0.7% (n=2213) |
| clinical_lab_flu_pct_positive | Weekly | 13.8% (n=38) | 8.2% (n=1476) | 2.7% (n=1607) | 1.7% (n=1669) | 1.0% (n=1714) | 0.8% (n=1759) |
| flusurv_flu_rate | Weekly | 30.4% (n=26) | 29.2% (n=351) | 10.1% (n=351) | 6.6% (n=351) | 5.4% (n=351) | 4.7% (n=351) |
| kinsa_ili | Daily; complete weekly mean | 0.0% (n=14) | 0.0% (n=15) | 0.0% (n=16) | 0.0% (n=17) | 0.0% (n=18) | 0.0% (n=18) |

| Covariate | Frequency / representation | Lag 6 | Lag 7 | Lag 8 | Lag 9 | Lag 10 | Lag 11 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inpatient_flu | Daily; Saturday trailing-7-day value | 12.1% (n=2001) | 9.8% (n=2030) | 8.9% (n=2037) | 8.1% (n=2071) | 7.7% (n=2108) | 7.5% (n=2128) |
| inpatient_covid | Daily; Saturday trailing-7-day value | 15.1% (n=1826) | 13.9% (n=1846) | 12.2% (n=1879) | 11.0% (n=1901) | 10.2% (n=1938) | 9.2% (n=1958) |
| outpatient_flu | Daily; Saturday trailing-7-day value | 8.2% (n=987) | 6.9% (n=1034) | 6.8% (n=1078) | 6.3% (n=1128) | 5.8% (n=1151) | 5.5% (n=1197) |
| outpatient_covid | Daily; Saturday trailing-7-day value | 16.7% (n=928) | 15.8% (n=971) | 14.8% (n=1025) | 14.0% (n=1077) | 13.3% (n=1102) | 12.5% (n=1151) |
| nwss_flu_wval_like | Weekly derived index; irregular samples | 10.7% (n=1074) | 14.5% (n=1078) | 15.0% (n=1081) | 15.2% (n=1082) | 15.0% (n=1082) | 14.6% (n=1082) |
| nwss_covid_wval_like | Weekly derived index; irregular samples | 20.2% (n=1125) | 19.7% (n=1129) | 19.4% (n=1131) | 19.1% (n=1131) | 18.7% (n=1131) | 18.3% (n=1131) |
| nwss_rsv_wval_like | Weekly derived index; irregular samples | 10.3% (n=1058) | 10.0% (n=1063) | 9.8% (n=1068) | 9.7% (n=1070) | 9.4% (n=1072) | 9.0% (n=1073) |
| nwss_flu_pct_rank | Weekly derived index; irregular samples | 4.5% (n=1074) | 4.2% (n=1078) | 4.0% (n=1081) | 3.9% (n=1082) | 3.9% (n=1082) | 3.9% (n=1082) |
| nwss_covid_pct_rank | Weekly derived index; irregular samples | 15.8% (n=1125) | 14.8% (n=1129) | 14.0% (n=1131) | 13.4% (n=1131) | 13.0% (n=1131) | 12.6% (n=1131) |
| nwss_rsv_pct_rank | Weekly derived index; irregular samples | 3.8% (n=1058) | 3.6% (n=1063) | 3.4% (n=1068) | 3.4% (n=1070) | 3.2% (n=1072) | 3.1% (n=1073) |
| ilinet_ili | Weekly | 0.7% (n=2266) | 0.6% (n=2341) | 0.5% (n=2363) | 0.4% (n=2385) | 0.3% (n=2406) | 0.3% (n=2427) |
| clinical_lab_flu_pct_positive | Weekly | 0.6% (n=1800) | 0.5% (n=1858) | 0.5% (n=1880) | 0.4% (n=1898) | 0.3% (n=1914) | 0.3% (n=1930) |
| flusurv_flu_rate | Weekly | 4.3% (n=351) | 3.8% (n=351) | 3.5% (n=351) | 3.3% (n=351) | 3.2% (n=351) | 3.1% (n=351) |
| kinsa_ili | Daily; complete weekly mean | 0.0% (n=18) | 0.0% (n=18) | 0.0% (n=18) | 0.0% (n=18) | 0.0% (n=18) | 0.0% (n=18) |

