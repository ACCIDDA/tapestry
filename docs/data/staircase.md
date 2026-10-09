# Interactive availability staircase

Choose a target or covariate and a location below. Select a covariate/target and then a state. The aggregate view can mix statuses across states; hover lists the states in each category.

<div class="staircase-comparison">
<div><h2>Availability</h2><iframe class="availability-frame" src="../availability/timeline/evidence.html" title="Interactive availability staircase" loading="lazy"></iframe></div>
<div><h2>Revisions</h2><iframe class="availability-frame" src="../revisions/index.html?view=revisions" title="Separate signed revision explorer" loading="lazy"></iframe></div>
</div>

The separate revision graph shows one time-series line per Wednesday over a selectable 2–12 trailing weeks, with a dashed reference curve on the right axis. Positive means an upward revision; 0% means unchanged. Select the same signal and state in each panel to compare them. [Revision definitions and assumptions](revision-staircase.md) · [Open both graphs full-size](revisions/comparison.html).


[Open the full-size interactive view](availability/timeline/evidence.html). Its return link leads back to this documentation page.

## Read the staircase

The horizontal axis is the **Saturday observation week**. The vertical axis is the **nominal Wednesday forecast round**, with later rounds higher. **T−0 lies on the diagonal boundary:** the Saturday four days before Wednesday. One cell left is T−1; two left is T−2. Follow a column upward to see the evidence for that same observation change over time.

For Wednesday January 14, 2026, T−0 is January 10 and T−1 is January 3. Holiday extensions change the actual cutoff, but retain the nominal Wednesday row and its original Saturday. Hover shows the actual cutoff.

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

For next-season assumptions, use stable observed release periods to estimate delay, and label any extension over gray periods as inferred. This page does not convert those assumptions into historical observations. The evidence view distinguishes reported values, explicit missing statements and unknown timing.

[Evidence data](availability/timeline/evidence.json) · [Evidence counts](availability/timeline/evidence-counts.csv).

