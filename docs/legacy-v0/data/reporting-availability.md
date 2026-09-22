# Data exploration: what is available when we forecast?

A Wednesday forecast uses surveillance weeks ending on Saturdays. The latest
completed week is **4 days old**; the preceding week is **11 days old**. A report
can be missing for the latest week while already available for the preceding
week. That is a question of availability, separate from how much an available
report will later revise.

## Read the dates before reading the curves

This illustrative calendar uses March 2025:

```text
Sat Mar 1              Sat Mar 8              Wed Mar 12
older week ends        latest week ends       forecast cutoff
     |                      |                       |
     |<------------------ 11 days ----------------->|
                            |<------ 4 days ------->|

At this Wednesday:
  Older week:  preliminary report may already be available.
  Latest week: preliminary report may not be available yet.
```

For the same observation week ending March 8, March 12 is its four-day cutoff
and March 19 its eleven-day cutoff. The ages describe time since the observation
week ended, not time since a file was downloaded. In our inputs, unavailable
reports may be replaced by flagged later finals; those replacements are **not
historically available reports** and do not enter these availability lines.

## Availability on the actual forecast date

![Archived reports available on each Wednesday](../results/b1-reporting-regimes/wednesday-availability.png)

**Blue:** latest Saturday, four days earlier. **Orange:** preceding Saturday,
eleven days earlier. Both are solid weekly step lines. Yellow marks Wednesdays
where at least one location has an older-week report but no latest-week report.
When orange is high and blue is zero, the archive supports estimating revisions
to the older report, but not revisions to an observed newest-week report.
Estimating that missing newest value is a different task.

The denominator is all 51 states/DC. Availability means an eligible report in
Hub, Git or Delphi archives by the Wednesday cutoff, before B1 provider
precedence and without requiring a reference-final label. US is excluded.
Dates without both ages in the provider audit are omitted, rather than treated
as zero availability. Steps carry the status to the next plotted Wednesday;
they do not determine when releases occurred between Wednesdays.

**Archive absence does not prove publication absence.** Missing vintages may
reflect collection or provider-selection limitations. Establishing the actual
publication schedule requires a release-timestamp/source audit, which these
plots do not complete. Similarly, the fraction of locations with records is
not the fraction of hospitals reporting.

[Weekly availability counts](../results/b1-reporting-regimes/wednesday-availability.csv) ·
[Source hash and definitions](../results/b1-reporting-regimes/wednesday-availability-manifest.json).

## NSSP: missing newest-week reports are not zero revisions

In 2024–25, the provider audit finds Delphi NSSP vintages for each pathogen at
11 days across **43 weeks**, but at four days across only **seven weeks**, June
14–July 26, 2025. Delphi therefore does contain older-week revision information
when the newest-week history is unavailable.

The initial revision comparison required both ages for the same observation,
location and provider. This dropped much of the older-week NSSP history. The
following view shows each age on its own available support:

![NSSP and NHSN revisions on all available report support](../results/b1-reporting-regimes/net-revisions-all-reports.png)

A gap is missing report support, not a zero correction. Net revision is
`100 × sum(final − report) / sum(final)`; positive means upward correction.
States/DC are pooled within target/week. Counts weight admissions by count mass;
ED proportions are summed across equally weighted locations, not weighted by
national visit volume. These are descriptive revision percentages, not WIS.
The extreme COVID admission point is marked off scale and retained in summaries.

Comparing different ages on different available weeks does not isolate maturity.
On the **same** observation/location/provider support in 2025–26, NSSP behaves
as expected: newer reports require larger net upward corrections.

| NSSP target | At 4 days | At 11 days |
|---|---:|---:|
| Influenza ED | 4.61% | 1.45% |
| COVID ED | 6.23% | 3.06% |
| RSV ED | 8.72% | 3.03% |

These are aggregate patterns, not a guarantee of monotonic revision for every
individual report. [Matched results, absolute revisions and provider tables](../results/b1-reporting-regimes/index.md)
show the complementary comparisons.

## NHSN: the relevant regime is November 2024 onward

The user supplied the following reporting context, recorded in the repository's
`icare.md`:

| Observation period | Interpretation for this project |
|---|---|
| Before May 1, 2024 | Previous NHSN reporting regime. |
| May 1–October 31, 2024 | Voluntary reporting after the mandate paused; poor coverage, not representative of the intended reporting process. |
| From November 1, 2024 | New mandate and the relevant NHSN regime for next-season nowcasting. |

These are user-provided premises, not independently verified policy findings in
this analysis. The working assumption is that the post-November-2024 process
remains relevant next season. It does **not** mean reports are revision-free.
NSSP has its own availability history; the NHSN mandate is not an NSSP boundary.
Week-ending dates assign regime in the descriptive tables, so transition weeks
can span a policy boundary.

Current B1 inputs contain no genuine recent NHSN reports during May–October
2024, and no genuine recent COVID/RSV admission reports before May 2024. Supplied
finals in those periods do not teach correction of a visible preliminary report.
Within the new regime, matched newest-week NHSN net revisions remain around
7–11%, declining to roughly 4–6% for the preceding week. Earlier reporting
examples therefore cannot be assumed to teach the correction needed next season.

## What this does and does not say about joint models

Here **joint** means learning forecasts and nowcasts together. It is distinct
from the “joint MLP” backbone that pools all six output targets in one fit.

**These data do not establish that joint forecast/nowcast training is inadvisable.**
They show that forecast history and revision supervision have different coverage
and relevance. A joint model can still use older epidemic histories for its
forecast objective while applying genuine-report revision loss only to eligible
source/age/regime examples. Missing-report reconstruction should remain a
separate objective rather than silently stand in for revision learning.

| Evidence from B1 | Supported conclusion |
|---|---|
| Current two-stage variants have worse natural forecast means than the ensemble. | Do not adopt that tested recipe as the default; this does not reject every joint model. |
| Target gated-20%, without revision augmentation, improves matched mixed-mask B from 0.997 to 0.948 in all five seeds. | Joint learning can be competitive and deserves a fair comparison. |
| Target B gap-only scores 0.938; pathogen B no-mask scores 0.943. | Direct B remains a strong control; gated has not displaced the best tested forecast recipes. |
| Genuine report coverage and reporting regimes differ across seasons. | The current cross-validation does not settle which design best serves the next season. |

The experiment has **not** yet compared separately fitted nowcasting against a
joint model under matched current-regime supervision and identical evaluation.
Separate fitting is a useful diagnostic/control, not a demonstrated winner.
Nor have we established that reporting-regime mixing caused the earlier failures.

A focused next comparison would retain direct B, add a jointly trained gated
model with eligible revision supervision, and include a separately fitted recent
model. Keep natural forecast inputs, backbone, masking and evaluation cells
matched. Evaluate forward on the later season, with NHSN nowcast results focused
on the current regime and NSSP results split by actual vintage availability.
A joint model need not lose access to older forecast-training examples just
because their revision labels are ineligible.

For chronological validation, fitting and any updates must use only labels and
revisions available at the historical fit date. Later finals can score the
predictions but cannot be smuggled into training. Adjacent overlapping episodes
must not expose held-out target labels. Report limited seasonal replication
rather than treating many locations or random seeds as many independent seasons.
If separately trained nowcasts feed forecasting, generate its training inputs
out of fold or chronologically and test the complete uncertainty-propagating
pipeline against direct B. None of these comparisons has been launched here.

## Reproduction and limits

The figures reuse the current B1 archive and pinned reference finals (September
16, 2026). Reference finals can still revise. The audit is descriptive, uses the
full model calendar rather than only Hub-scored tasks, and does not fit models
or measure a causal regime effect. No figures were visually inspected.

```bash
.venv/bin/python scripts/plot_b1_availability.py
.venv/bin/python scripts/plot_b1_reporting_regimes.py
```

[Detailed revision analysis](../results/b1-reporting-regimes/index.md) ·
[B1 experiment overview](../results/b1-conclusions.md) ·
[Vintage/source policy](../design/b1.md#calendar-and-source-assumptions).

## Log

- 2026-09-18: added a data-exploration entry explaining Wednesday availability,
  Saturday observation weeks, NSSP support, NHSN regimes and implications for
  validation. Clarified that separate fitting is a proposed comparison, not
  evidence against jointly trained forecasting and nowcasting.
