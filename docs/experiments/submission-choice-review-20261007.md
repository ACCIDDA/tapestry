# Submission choice after reading both audit conversations

Reviewed the Claude conversation **Model run audit and submission prep** and the
Codex conversation **Review Audit training run**, including their later messages,
and checked the saved score table and production definitions. No model was trained,
forecast regenerated, or submission changed in this review.

## Choice and objective

Prefer the B7 three-recipe predictive-distribution mixture if the primary objective
is season-long state/DC log-admission accuracy. This is a modest preference, not
evidence of a reliable probability of winning. If admission-count accuracy is the
primary objective, the existing eight-recipe System2 mixture has the stronger
record in this comparison. The ranking proxy here is pooled WIS on log(1 + count),
not the official pairwise relative-WIS calculation.

The B7 mixture combines a width-96 sampled MLP with a three-block decoder
(`A_blocks3`), a width-96 sampled MLP pretrained on pre-August-2022 state ILI
(`B5_confirmed_candidate_1`), and a width-256 neighbor-sharing direct-quantile MLP
(`B_width256`). The sampled recipes use Kinsa. No recipe uses FluSurv.

For B7, training admission and ED histories are artificially revised using
2025–26 reporting errors. The sampled models learn from corrected artificial
histories; the direct-quantile model learns from preliminary artificial histories
with a recent-history reconstruction objective. Both learn latest future flu
admission and ED labels. System2 retains eight B6 recipes after removing the
FluSurv recipe; its earlier recipe-specific treatments primarily revise admission
histories, with ED treatments in selected recipes. Both systems use their own
saved input corrections at evaluation.

## Evidence on the actual eight-recipe alternative

Source: `b7-folds-20261007/vs-system2/season-scores.csv`.

The comparison uses two seeds per recipe on both sides. Models evaluated on
2024–25 were trained on 2022–23, 2023–24 and 2025–26. Models evaluated on 2025–26
were trained on 2022–23 through 2024–25. Labels are latest future admissions and
ED. Existing checkpoints were replayed on shared archived Wednesday reports,
retaining finalized-value substitutions and each model's own correction pipeline;
they were not retrained for this replay. Evaluation uses frozen Hub tasks and
truth, horizons 0–3, late November–May. Admission seasons have equal weight; ED
has only the 2025–26 matched Hub comparison. States/DC exclude US and Puerto Rico.

| Measure | B7 three recipes | B6 System2 eight recipes, no FluSurv |
|---|---:|---:|
| State/DC log-admission WIS, 2024–25 | 0.249855 | 0.261531 |
| State/DC log-admission WIS, 2025–26 | 0.278462 | 0.285903 |
| State/DC log-admission WIS, equal-season mean | 0.264159 | 0.273717 |
| State/DC admission-count WIS, equal-season mean | 62.1414 | 59.5928 |
| State/DC ED-proportion WIS, 2025–26 | 0.0050412 | 0.0050901 |

Lower WIS is better. B7 has 3.49% lower log-admission WIS, 4.28% higher
admission-count WIS, and 0.96% lower ED WIS. Against the five-seed eight-recipe
System2 comparison, B7 has 4.05% lower log WIS and 2.48% higher count WIS.
The often-quoted 2.9%/3.7% gains instead compare against the nine-recipe mixture
that still includes FluSurv.

The earlier Codex recommendation compared the wider sampled model and direct-
quantile model, individually and paired, and balanced count/ED accuracy with
readiness. Claude subsequently evaluated different sampled partners, including
the ILI-pretrained model, then fitted the three-recipe production mixture.
Therefore the recommendations differ in candidate set, objective, and readiness
at the time of the decision.

## Production and limitations

The saved B7 production definition has ten seeds per recipe, 30 fits, trained on
all four completed seasons 2022–23 through 2025–26 with the same B7 treatments
and latest future labels. The saved System2 definition has ten seeds per recipe,
80 fits, also trained on all four seasons with its B6 treatments. Both were
replayed on the same pinned 7 October 2026 operational panel. The retrospective
numbers above are not validation of these all-season production fits.

Material assumptions and limits:

- The priority is season-long log-admission performance, rather than count WIS
  or only the next October forecast.
- The 2025–26 reporting-error process is assumed useful for the coming season.
  Its use in training the B7 correction also contaminates the 2025–26 retrospective
  evaluation; that season is not independent confirmation.
- The 2024–25 comparison avoids that same-season reporting-error reuse, but both
  systems trained on the later 2025–26 season. It is not a prospective backtest.
- Both evaluated seasons were repeatedly used for model selection. B7 also inherits
  earlier recipe development; fewer final combinations alone do not establish
  less selection bias.
- Two seeds per recipe do not establish seed robustness, and ten production seeds
  do not remove validation bias.
- October and early November are absent from the matched Hub comparison. Neither
  model has demonstrated superiority for the unusually early current wave.
- Calendar dependence and training on 2024 reporting-lapse labels remain concerns
  for both systems. Narrower national intervals alone are not evidence of better
  forecasts, and a faster observed growth rate alone does not determine WIS.

This supports choosing the B7 three-recipe model with limited confidence. It does
not support attaching a measured 60–65% probability to that choice.
