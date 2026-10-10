"""Dataset and forecasting-problem contracts.

A dataset describes the signals that can be built.  A problem selects targets,
horizons, folds and one evaluation from that dataset.  Model recipes may choose
only among the named input and covariate sets exposed by the dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


UNITS = {"count", "proportion", "rate", "continuous"}
FOLD_KINDS = {"leave_one_season_out", "leave_one_period_out", "rolling_origin"}
EVALUATION_INPUTS = {"reported", "prescribed", "finalized"}


def _read(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    return resolved, json.loads(resolved.read_text())


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Signal:
    name: str
    unit: str
    group: str
    hub: str | None
    hub_target: str | None
    weight: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Signal":
        signal = cls(
            name=str(value["name"]),
            unit=str(value["unit"]),
            group=str(value.get("group", value["name"])),
            hub=None if value.get("hub") is None else str(value["hub"]),
            hub_target=None if value.get("hub_target") is None else str(value["hub_target"]),
            weight=float(value.get("weight", 1.0)),
        )
        if signal.unit not in UNITS:
            raise ValueError(f"Unknown unit for {signal.name}: {signal.unit}")
        if signal.weight < 0:
            raise ValueError(f"Negative objective weight for {signal.name}")
        return signal


@dataclass(frozen=True)
class Dataset:
    id: str
    path: Path
    signals: tuple[Signal, ...]
    input_sets: dict[str, tuple[str, ...]]
    covariate_sets: dict[str, tuple[str, ...]]
    report_vintages: bool

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        resolved, value = _read(path)
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported dataset schema in {resolved}")
        signals = tuple(Signal.from_dict(item) for item in value["signals"])
        names = [signal.name for signal in signals]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate signal names in {resolved}")
        input_sets = {str(k): tuple(map(str, v)) for k, v in value["input_sets"].items()}
        covariate_sets = {str(k): tuple(map(str, v)) for k, v in value.get("covariate_sets", {}).items()}
        covariate_sets.setdefault("none", ())
        for name, members in input_sets.items():
            unknown = set(members) - set(names)
            if unknown:
                raise ValueError(f"Input set {name!r} contains unknown signals: {sorted(unknown)}")
        # Whether the built panel holds archived report vintages (`asof_*` arrays and per-Hub
        # deadlines). Without them, the reporting-error stage cannot run (see Problem.reporting_errors).
        report_vintages = bool(value.get("report_vintages", False))
        return cls(str(value["id"]), resolved, signals, input_sets, covariate_sets, report_vintages)

    @property
    def by_name(self) -> dict[str, Signal]:
        return {signal.name: signal for signal in self.signals}

    @property
    def reference(self) -> str:
        try:
            return self.path.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return str(self.path)


@dataclass(frozen=True)
class Problem:
    id: str
    path: Path
    dataset: Dataset
    targets: tuple[str, ...]
    horizons: tuple[int, ...]
    fold_kind: str
    folds: tuple[str, ...]
    training_folds: tuple[str, ...]
    periods: dict[str, tuple[str, str]]
    evaluation_kind: str
    evaluation_inputs: str
    evaluation_draws: int
    headline_target: str
    headline_scale: str
    headline_geography: str
    comparison: dict[str, Any]
    production: bool
    panel: str
    locations: str

    @classmethod
    def load(cls, path: str | Path) -> "Problem":
        resolved, value = _read(path)
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported problem schema in {resolved}")
        dataset_path = (resolved.parent / value["dataset"]).resolve()
        dataset = Dataset.load(dataset_path)
        targets = tuple(map(str, value["targets"]))
        unknown = set(targets) - set(dataset.by_name)
        if unknown or not targets:
            raise ValueError(f"Problem targets are empty or unknown: {sorted(unknown)}")
        if len(targets) != len(set(targets)):
            raise ValueError("Problem targets must be unique")
        forecast = value["forecast"]
        horizons = tuple(int(h) for h in forecast["horizons"])
        if not horizons or min(horizons) < 1 or tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
            raise ValueError("Forecast horizons must be sorted unique positive integers")
        folds = value["folds"]
        evaluation = value["evaluation"]
        headline = evaluation["primary"]
        problem = cls(
            id=str(value["id"]),
            path=resolved,
            dataset=dataset,
            targets=targets,
            horizons=horizons,
            fold_kind=str(folds["kind"]),
            folds=tuple(map(str, folds["ids"])),
            training_folds=tuple(map(str, folds.get("training_ids", ()))),
            periods={str(k): (str(v[0]), str(v[1])) for k, v in folds.get("periods", {}).items()},
            evaluation_kind=str(evaluation["kind"]),
            evaluation_inputs=str(evaluation.get("inputs", "reported")),
            evaluation_draws=int(evaluation.get("draws", 1)),
            headline_target=str(headline["target"]),
            headline_scale=str(headline["scale"]),
            headline_geography=str(headline["geography"]),
            comparison=dict(value.get("comparison", {"kind": "none"})),
            production=bool(value.get("production", False)),
            # Repository-relative data files (git-ignored, copied to the cluster). Several problems
            # may share one panel; each experiment pins the hashes of both at plan time.
            panel=str(value["panel"]),
            locations=str(value["locations"]),
        )
        if problem.fold_kind not in FOLD_KINDS:
            raise ValueError(f"Unsupported fold kind: {problem.fold_kind}")
        if not problem.folds or len(problem.folds) != len(set(problem.folds)):
            raise ValueError("Evaluation fold IDs must be nonempty and unique")
        if problem.fold_kind == "rolling_origin":
            # Training is every week before the held-out window, so no training IDs are declared.
            if problem.training_folds:
                raise ValueError("rolling_origin trains on every week before each window; omit training_ids")
        elif not problem.training_folds or len(problem.training_folds) != len(set(problem.training_folds)):
            raise ValueError("Training fold IDs must be nonempty and unique")
        if problem.fold_kind == "leave_one_season_out":
            if problem.periods:
                raise ValueError("leave_one_season_out folds are CDC seasons; periods are not allowed")
        else:
            if set(problem.periods) != set(problem.folds) | set(problem.training_folds):
                raise ValueError("Every fold and training ID needs exactly one period [first, last]")
            spans = sorted(problem.periods.values())
            if any(first > last for first, last in spans) or any(a[1] >= b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError("Fold periods must be ordered [first, last] date pairs that do not overlap")
        if not problem.production and problem.fold_kind != "rolling_origin" \
                and not set(problem.folds).issubset(problem.training_folds):
            raise ValueError("Held-out folds must be among the training folds for a retrospective problem")
        if problem.evaluation_kind != "weekly_quantile_wis":
            raise ValueError(f"Unsupported evaluation kind: {problem.evaluation_kind}")
        # The current evaluator exports array position 0 as Hub horizon 0, position 1
        # as horizon 1, and so on.  Requiring consecutive problem weeks prevents a
        # sparse horizon list from being scored under the wrong labels.  Its standard
        # headline and distribution diagnostics use the first four future weeks.
        if problem.horizons != tuple(range(1, max(problem.horizons) + 1)) or len(problem.horizons) < 4:
            raise ValueError("weekly_quantile_wis requires consecutive horizons starting at 1, including weeks 1-4")
        if problem.evaluation_inputs not in EVALUATION_INPUTS:
            raise ValueError(f"Unknown evaluation inputs: {problem.evaluation_inputs}")
        if problem.reporting_errors and not dataset.report_vintages:
            raise ValueError("Reported or prescribed evaluation inputs need a dataset with report_vintages")
        if problem.production and (problem.fold_kind != "leave_one_season_out" or not problem.reporting_errors):
            raise ValueError("Production replays Wednesday reports: season folds and reported/prescribed inputs")
        if problem.reporting_errors and problem.fold_kind != "leave_one_season_out":
            # Donor reporting errors, error_reference and correction pairs are defined per CDC season.
            raise ValueError("The reporting-error stage is defined for leave_one_season_out folds only")
        if problem.evaluation_draws not in {1, 3, 5}:
            raise ValueError("Evaluation draws must be 1, 3 or 5")
        if problem.headline_target not in targets:
            raise ValueError("The headline target must be one of the problem targets")
        if problem.headline_scale not in {"natural", "log"}:
            raise ValueError("The headline scale must be natural or log")
        if problem.headline_scale == "log" and dataset.by_name[problem.headline_target].unit != "count":
            raise ValueError("Log headline scoring is defined only for count targets")
        if problem.headline_geography not in {"states_dc", "US"}:
            raise ValueError("The headline geography must be states_dc or US")
        comparison_kind = problem.comparison.get("kind", "none")
        if comparison_kind not in {"none", "hub"}:
            raise ValueError(f"Unsupported comparison kind: {comparison_kind}")
        if comparison_kind == "hub" and problem.fold_kind != "leave_one_season_out":
            raise ValueError("Hub comparisons are frozen per CDC season; use leave_one_season_out folds")
        if comparison_kind == "hub":
            comparison_hub = problem.comparison.get("hub")
            headline_hub = dataset.by_name[problem.headline_target].hub
            if comparison_hub != headline_hub:
                raise ValueError("Hub comparison must match the headline target's Hub")
        return problem

    @property
    def reporting_errors(self) -> bool:
        """Whether the reporting-error stage runs: artificial errors, correction trees, Wednesday-report inputs.

        Off when the problem is evaluated on finalized histories (`evaluation.inputs = finalized`),
        for example a dataset without archived report vintages. Recipes must then train on
        finalized histories and forecast from the raw inputs (`validate_scenario`)."""
        return self.evaluation_inputs != "finalized"

    def fold_labels(self, dates) -> "np.ndarray":
        """Fold ID of each calendar week ('' outside every declared fold)."""
        import numpy as np
        from chromantis.dataset.cv import season

        dates = [str(d)[:10] for d in dates]
        if self.fold_kind == "leave_one_season_out":
            return np.array([season(d) for d in dates]).astype("U32")
        labels = np.full(len(dates), "", dtype="U32")
        for fold, (first, last) in self.periods.items():
            labels[[first <= d <= last for d in dates]] = fold
        return labels

    def training_weeks(self, dates, held_out: str, window: str = "all") -> "np.ndarray":
        """[T] bool: the weeks a model held out on `held_out` may train on."""
        import numpy as np

        if window != "all" and self.fold_kind != "leave_one_season_out":
            raise ValueError("training_window recent2/last2 is defined for season folds only")
        if self.fold_kind == "rolling_origin":
            first = self.periods[held_out][0]
            return np.array([str(d)[:10] < first for d in dates])
        labels = self.fold_labels(dates)
        if window == "recent2":
            i = self.training_folds.index(held_out)
            permitted = self.training_folds[max(0, i - 2):i]
        else:
            permitted = tuple(s for s in self.training_folds if s != held_out)
            permitted = permitted[-2:] if window == "last2" else permitted
        return np.isin(labels, permitted)

    def fold_start(self, label: str):
        """First day of a fold; validation-week positions count from it."""
        from datetime import date, timedelta
        from chromantis.dataset.cv import season_start

        if self.fold_kind == "leave_one_season_out":
            return season_start(int(label[:4]))
        # Periods name week-ending Saturdays, like the panel calendar; the week starts on Sunday.
        return date.fromisoformat(self.periods[label][0]) - timedelta(days=6)

    @property
    def hash(self) -> str:
        problem = json.loads(self.path.read_text())
        problem["dataset"] = self.dataset.id
        dataset = json.loads(self.dataset.path.read_text())
        digest = hashlib.sha256()
        digest.update(json.dumps(problem, sort_keys=True, separators=(",", ":")).encode())
        digest.update(json.dumps(dataset, sort_keys=True, separators=(",", ":")).encode())
        return digest.hexdigest()

    @property
    def reference(self) -> str:
        try:
            return self.path.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return str(self.path)

    @property
    def target_signals(self) -> tuple[Signal, ...]:
        return tuple(self.dataset.by_name[name] for name in self.targets)

    @property
    def target_units(self) -> tuple[str, ...]:
        return tuple(signal.unit for signal in self.target_signals)

    @property
    def target_groups(self) -> tuple[str, ...]:
        return tuple(signal.group for signal in self.target_signals)

    @property
    def target_hubs(self) -> tuple[str | None, ...]:
        return tuple(signal.hub for signal in self.target_signals)

    def input_names(self, name: str) -> tuple[str, ...]:
        if name == "target":
            return self.targets
        if name not in self.dataset.input_sets:
            raise ValueError(f"Unknown input_set={name!r}; choose from {sorted(self.dataset.input_sets)}")
        inputs = self.dataset.input_sets[name]
        missing = set(self.targets) - set(inputs)
        if missing:
            raise ValueError(f"Input set {name!r} lacks target histories: {sorted(missing)}")
        return inputs

    def covariate_names(self, name: str) -> tuple[str, ...]:
        selected: list[str] = []
        for part in name.split("+"):
            if part not in self.dataset.covariate_sets:
                raise ValueError(
                    f"Unknown covariate_set={name!r}; choose from {sorted(self.dataset.covariate_sets)}"
                )
            selected.extend(self.dataset.covariate_sets[part])
        return tuple(dict.fromkeys(selected))

    def target_input_indices(self, input_set: str) -> tuple[int, ...]:
        inputs = self.input_names(input_set)
        return tuple(inputs.index(name) for name in self.targets)

    def target_weights(self, loss_weights: str) -> tuple[float, ...]:
        signals = self.target_signals
        if loss_weights == "objective":
            weights = [signal.weight for signal in signals]
        elif loss_weights == "admissions_only":
            weights = [float(signal.unit == "count") for signal in signals]
        elif loss_weights == "ed_only":
            weights = [float(signal.unit == "proportion") for signal in signals]
        else:
            raise ValueError(f"Unknown loss_weights={loss_weights!r}")
        if not any(weights):
            raise ValueError(f"loss_weights={loss_weights!r} gives no weight to {self.id}")
        return tuple(weights)

    def model_horizons(self, scenario) -> tuple[int, ...]:
        if not scenario.reconstruction_labels:
            return self.horizons
        return tuple(range(1 - scenario.reconstruction_weeks, 1)) + self.horizons

    def future_indices(self, scenario) -> tuple[int, ...]:
        return tuple(i for i, h in enumerate(self.model_horizons(scenario)) if h > 0)

    def validate_panel(self, panel: dict[str, Any]) -> None:
        names = tuple(map(str, panel["target_names"]))
        unknown = set(self.dataset.by_name) - set(names)
        if unknown:
            raise ValueError(f"Panel lacks dataset signals: {sorted(unknown)}")

    def populations(self, locations) -> dict[str, float]:
        """Population of each panel location, from the problem's location file."""
        import csv
        import math

        values = {}
        with open(self.locations) as stream:
            for row in csv.DictReader(stream):
                loc, value = row.get("abbreviation") or row["location"], float(row["population"])
                if loc in values or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid or duplicate population for {loc} in {self.locations}")
                values[loc] = value
        missing = [loc for loc in locations if loc not in values]
        if missing:
            raise ValueError(f"{self.locations} lacks populations for {missing}")
        return {loc: values[loc] for loc in locations}

    def validate_scenario(self, scenario) -> None:
        inputs = self.input_names(scenario.input_set)
        self.covariate_names(scenario.covariate_set)
        self.target_weights(scenario.loss_weights)
        if scenario.sum_wis_weight and (len(self.horizons) != 4 or self.target_units[0] != "count"):
            raise ValueError("Sum WIS needs a four-horizon problem whose first target is a count")
        if self.evaluation_inputs == "prescribed" and not scenario.error_reference:
            raise ValueError("Prescribed evaluation inputs require an error_reference in the recipe")
        if not self.reporting_errors:
            used = [name for name, on in (
                ("history_source != finalized", scenario.history_source != "finalized"),
                ("forecast_view != raw", scenario.forecast_view != "raw"),
                ("error_reference", bool(scenario.error_reference)),
                ("correction_noise", bool(scenario.correction_noise)),
                ("stress_views", scenario.stress_views)) if on]
            if used:
                raise ValueError(f"{self.id} has no reporting-error stage (finalized evaluation inputs); "
                                 f"the recipe uses it: {', '.join(used)}")
        if self.production and scenario.training_window != "all":
            raise ValueError("Production problems train on every declared training fold")
        if scenario.ili_training != "none" and self.target_groups[0] != "flu":
            raise ValueError("Historical ILI transfer is only defined for a flu target")
        if scenario.encoder in ("series_mlp", "series_mixer") and scenario.covariate_set != "none":
            raise ValueError("Series models do not use covariates")
        if not inputs:
            raise ValueError("A problem must expose at least one model input")

