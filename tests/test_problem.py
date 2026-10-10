import json

import pytest

from chromantis.model.scenario import Scenario
from chromantis.problem import Problem


def changed_problem(tmp_path, **changes):
    source = Problem.load("problems/us-flu-short-term.json")
    value = json.loads(source.path.read_text())
    value["dataset"] = str(source.dataset.path)
    for path, replacement in changes.items():
        target = value
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = replacement
    destination = tmp_path / "problem.json"
    destination.write_text(json.dumps(value))
    return destination


def test_problem_rejects_horizons_the_weekly_evaluator_would_mislabel(tmp_path):
    with pytest.raises(ValueError, match="consecutive horizons"):
        Problem.load(changed_problem(tmp_path, **{"forecast.horizons": [1, 3, 4, 6]}))


def test_problem_rejects_a_comparison_for_the_wrong_hub(tmp_path):
    with pytest.raises(ValueError, match="headline target's Hub"):
        Problem.load(changed_problem(tmp_path, **{"comparison.hub": "covid"}))


def test_removed_task_fields_do_not_enter_through_scenario_strings():
    for field in ("forecast_targets", "pathogen_inputs", "forecast_weeks", "evaluation_seasons"):
        with pytest.raises(ValueError, match="Unknown scenario field"):
            Scenario.from_string(f"{field}=flu")


def test_rolling_origin_trains_only_before_the_window_and_scores_only_inside_it():
    from chromantis.dataset.cv import week_roles
    import numpy as np

    problem = Problem.load("problems/us-flu-rolling-origin.json")
    first, last = problem.periods["2025-26-winter"]
    dates = (np.datetime64("2024-08-03") + 7 * np.arange(120)).astype(str)
    roles = week_roles(dates, problem, Scenario(patience=25, epochs=160, forecast_view="raw"), "2025-26-winter")
    assert set(dates[np.isin(roles, ["fit", "validation"])]) == {d for d in dates if d < first}
    assert set(dates[roles == "score"]) == {d for d in dates if first <= d <= last}
    assert (roles == "validation").any()


def test_period_folds_never_train_on_the_held_out_period(tmp_path):
    from chromantis.dataset.cv import week_roles
    import numpy as np

    periods = {"a": ["2023-01-07", "2023-06-24"], "b": ["2024-01-06", "2024-06-29"], "c": ["2025-01-04", "2025-06-28"]}
    path = changed_problem(tmp_path, folds={"kind": "leave_one_period_out", "ids": ["b", "c"],
                                           "training_ids": ["a", "b", "c"], "periods": periods},
                           **{"evaluation.inputs": "finalized", "comparison": {"kind": "none"}})
    problem = Problem.load(path)
    dates = (np.datetime64("2022-12-03") + 7 * np.arange(140)).astype(str)
    roles = week_roles(dates, problem, Scenario(forecast_view="raw"), "b")
    inside = lambda p: {d for d in dates if periods[p][0] <= d <= periods[p][1]}
    assert set(dates[roles == "score"]) == inside("b")
    assert set(dates[roles == "fit"]) == inside("a") | inside("c")


def test_no_reporting_stage_rejects_recipes_that_need_it():
    problem = Problem.load("problems/us-flu-rolling-origin.json")
    assert not problem.reporting_errors
    problem.validate_scenario(Scenario(forecast_view="raw"))
    with pytest.raises(ValueError, match="no reporting-error stage"):
        problem.validate_scenario(Scenario(history_source="artificial", history_correction=True))
