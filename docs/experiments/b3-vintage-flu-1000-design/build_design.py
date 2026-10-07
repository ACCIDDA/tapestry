"""Write a proposed design, not runnable Scenario strings or an experiment plan."""
import csv
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
BASE = dict(
    architecture="independent_pathogen_mlp", output="sampled_crps", lookback=12,
    width=64, lr=.001, weight_decay=0., spatial="distance", mask_rate=0.,
    covariates="none", target_inputs="all_six", labels="all_six",
    training_inputs="finalized", evaluation_history="raw_reports",
    loss="native_q95", adapter_rank=4, tree_depth=0, tree_min_leaf=0,
)


def add(block, **changes):
    rows.append(dict(configuration=f"B3-{len(rows)+1:04d}", block=block,
                     seeds="42 43", **(BASE | changes)))


# Include simple matched width, learning-rate and weight-decay contrasts.
recipes = [(32, .0003, 0.), (32, .001, 0.), (64, .0003, 0.),
           (64, .001, 0.), (128, .001, 0.), (64, .003, 0.),
           (64, .001, .0001), (64, .001, .001),
           (32, .001, .001), (128, .001, .001)]
for history, spatial, partition, mask, recipe in itertools.product(
        [8, 10, 12], ["none", "distance", "neighbors"],
        ["independent_pathogen_mlp", "independent_target_mlp"], [0., .2], recipes):
    width, lr, decay = recipe
    add("existing_mlp_360", lookback=history, spatial=spatial,
        architecture=partition, mask_rate=mask, width=width, lr=lr,
        weight_decay=decay)

architectures = ["independent_pathogen_mlp", "shared_series_mlp_adapter",
                 "residual_time_mixer", "residual_basis_mlp", "damped_growth_mlp"]
for architecture, history, width, spatial, lr, covariates in itertools.product(
        architectures, [8, 12], [32, 64], ["none", "distance"],
        [.0003, .001], ["none", "kinsa", "ilinet"]):
    add("quantile_architectures_240", architecture=architecture,
        output="ordered_quantiles", lookback=history, width=width,
        spatial=spatial, lr=lr, covariates=covariates)

# Twenty fixed anchors, not winners selected after seeing evaluation results.
anchors = [dict(architecture=a, output="ordered_quantiles", lookback=h,
                width=w, spatial="none")
           for a, h, w in itertools.product(architectures, [8, 12], [32, 64])]
for anchor, labels, inputs in itertools.product(
        anchors, ["flu_admissions", "flu_admissions_and_ed"],
        ["flu_admissions", "flu_admissions_and_ed", "all_admissions", "all_six"]):
    add("flu_scope_160", **anchor, labels=labels, target_inputs=inputs)

treatments = [
    dict(training_inputs="finalized_early_archived_recent"),
    dict(training_inputs="half_strength_empirical_admission_errors"),
    dict(evaluation_history="half_raw_half_seasonal_correction"),
    dict(evaluation_history="half_raw_half_synthetic_neural_correction"),
    dict(evaluation_history="half_raw_half_real_pair_neural_correction"),
    dict(evaluation_history="half_raw_half_pretrained_real_pair_neural_correction"),
]
for anchor, treatment in itertools.product(anchors, treatments):
    add("reporting_treatments_120", **anchor, labels="flu_admissions", **treatment)

for anchor, loss in itertools.product(anchors, [
        "native_baseline_error", "log1p_unscaled", "log1p_baseline_error",
        "half_native_q95_half_log1p_q95"]):
    add("loss_alignment_80", **anchor, labels="flu_admissions", loss=loss)

for history, inputs, depth, min_leaf in itertools.product(
        [8, 12], ["flu_admissions_and_ed", "all_six"], [2, 3], [10, 20, 40, 80, 160]):
    add("boosted_quantile_trees_40", architecture="pooled_quantile_trees",
        output="ordered_quantiles", lookback=history, width=0, lr=.05,
        spatial="none", labels="flu_admissions", target_inputs=inputs,
        loss="log1p_unscaled", tree_depth=depth, tree_min_leaf=min_leaf)

semantic = [{k: v for k, v in r.items() if k not in ("configuration", "block", "seeds")}
            for r in rows]
assert len(rows) == 1000
assert len({json.dumps(r, sort_keys=True) for r in semantic}) == 1000
with (ROOT / "configurations.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(ROOT / "design.json").write_text(json.dumps(dict(
    status="proposal_only_not_runnable", configurations=1000, seeds=[42, 43],
    seed_configurations=2000, seasonal_evaluations=4000,
    primary="2025-26 flu admission mean log1p WIS on common frozen state/DC tasks",
    folds=[dict(train=["2022-23", "2023-24", "2024-25"], evaluate="2025-26",
                reporting_donor="2024-25", role="primary_forward_development"),
           dict(train=["2022-23", "2023-24", "2025-26"], evaluate="2024-25",
                reporting_donor="2025-26", role="secondary_retrospective")],
    blocks={b: sum(r["block"] == b for r in rows) for b in dict.fromkeys(r["block"] for r in rows)},
    configurations_file="configurations.csv",
), indent=2) + "\n")
print(f"Wrote {len(rows)} distinct proposed configurations; no jobs planned or launched.")
