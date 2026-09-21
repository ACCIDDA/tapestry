"""B2: B1 outcomes and histories with separately masked covariate histories.

Claims are read from their native Delphi revision archives.  Wastewater is
read from an origin-safe derived artifact; this module never assigns a modelled
release date to wastewater observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .finalized import season
from .wednesday import WednesdayDataset

DEFAULT_DATASET = "data/processed/build_b2.npz"
NWSS_DERIVED_DATASET = "derived_nwss_state_indices"
KINSA_DATASET = "pophive_kinsa_ili"
KINSA_SIGNAL = "kinsa_cough_cold_flu"
NWSS_ARCHIVE_SERVICE_START = "2026-02-25"

COVARIATES = (
    "inpatient_flu", "inpatient_covid",
    "outpatient_flu", "outpatient_covid",
    "nwss_flu_wval_like", "nwss_covid_wval_like", "nwss_rsv_wval_like",
    "nwss_flu_pct_rank", "nwss_covid_pct_rank", "nwss_rsv_pct_rank",
    "kinsa_ili",
)
COVARIATE_GROUPS = {
    "inpatient": (0, 1),
    "outpatient": (2, 3),
    "ww_wval_like": (4, 5, 6),
    "ww_pct_rank": (7, 8, 9),
    "kinsa": (10,),
}
# A national-only source has values for US alone; every state cell stays
# unavailable by native geography and is masked, never broadcast or filled.
NATIONAL_ONLY = ("kinsa_ili",)
CLAIMS = (
    ("delphi_claims_inpatient", "claims_inpatient_adm_pct_claims_flu", 0),
    ("delphi_claims_inpatient", "claims_inpatient_adm_pct_claims_covid", 1),
    ("delphi_claims_outpatient", "claims_outpatient_ov_pct_claims_flu", 2),
    ("delphi_claims_outpatient", "claims_outpatient_ov_pct_claims_covid", 3),
)


def covariate_indices(groups):
    """Expand canonical B2 group names without changing their fixed order."""
    selected = set(groups)
    unknown = selected - COVARIATE_GROUPS.keys()
    if unknown:
        raise ValueError(f"Unknown B2 covariate groups: {sorted(unknown)}")
    return tuple(i for group in COVARIATE_GROUPS for i in COVARIATE_GROUPS[group]
                 if group in selected)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_snapshot(data_root, dataset):
    root = Path(data_root) / "raw" / dataset
    latest = root / "latest.json"
    if not latest.is_file():
        raise FileNotFoundError(f"Missing selected raw dataset: {latest}")
    record = json.loads(latest.read_text())
    snapshot_id = record.get("snapshot_id") or record.get("id")
    if not snapshot_id:
        # Older repositories stored a relative manifest path only.
        manifest = record.get("manifest") or record.get("path")
        if manifest:
            snapshot_id = Path(manifest).parent.name
    if not snapshot_id:
        raise ValueError(f"Cannot resolve snapshot id from {latest}")
    return root / "snapshots" / snapshot_id, snapshot_id


def _claim_paths(data_root, dataset, signal):
    snapshot, snapshot_id = _latest_snapshot(data_root, dataset)
    paths = []
    for geography in ("state", "nation"):
        path = snapshot / f"signal={signal}" / f"geo_type={geography}" / "archive.csv.gz"
        if path.is_file():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No state/nation archive for {dataset}:{signal}")
    return paths, snapshot_id


def _claims_frame(paths, reference_dates, truth_cutoff):
    """Stream large claims archives, retaining only B2 Saturday/state cells."""
    frames = []
    starts = []
    wanted = set(reference_dates)
    usecols = ["report_time", "geo_type", "geo_value", "fill_method", "reference_time", "value"]
    for path in paths:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=750_000,
                                 dtype={"geo_value": str, "fill_method": str}):
            chunk["reference_time"] = chunk["reference_time"].astype(str).str[:10]
            chunk["report_time"] = chunk["report_time"].astype(str).str[:10]
            starts.append(chunk["report_time"].min())
            chunk = chunk[(chunk["reference_time"].isin(wanted)) &
                          (chunk["report_time"] <= truth_cutoff)]
            # Acquisition requests native values, but defend against filled rows.
            fill = chunk["fill_method"].fillna("").str.lower()
            chunk = chunk[fill.isin(("", "none", "source", "nan"))]
            if chunk.empty:
                continue
            chunk["geo_value"] = np.where(
                chunk["geo_type"].eq("nation"), "US", chunk["geo_value"].str.upper())
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk.loc[~np.isfinite(chunk["value"]) | (chunk["value"] < 0), "value"] = np.nan
            frames.append(chunk[["report_time", "geo_value", "reference_time", "value"]])
    frame = (pd.concat(frames, ignore_index=True) if frames else
             pd.DataFrame(columns=["report_time", "geo_value", "reference_time", "value"]))
    # Same-release conflicts are unavailable, rather than resolved by row order.
    key = ["report_time", "geo_value", "reference_time"]
    conflicts = frame.groupby(key, sort=False)["value"].nunique(dropna=False)
    bad = conflicts[conflicts > 1].index
    if len(bad):
        indexed = frame.set_index(key)
        indexed.loc[bad, "value"] = np.nan
        frame = indexed.reset_index()
    frame = frame.sort_values("report_time").drop_duplicates(key, keep="last")
    return frame, min(starts) if starts else None


def _fill_revisions(frame, issuances, context_dates, locations, final, wednesday, c):
    loc_index = {loc: i for i, loc in enumerate(locations)}
    requested = {}
    for i, dates in enumerate(context_dates):
        for t, day in enumerate(dates):
            requested.setdefault(day, []).append((i, t))
    grouped = frame.groupby(["reference_time", "geo_value"], sort=False)
    for (day, loc), revisions in grouped:
        if day not in requested or loc not in loc_index:
            continue
        releases = revisions["report_time"].to_numpy(str)
        values = revisions["value"].to_numpy(np.float32)
        l = loc_index[loc]
        for i, t in requested[day]:
            eligible = np.searchsorted(releases, issuances[i], side="right") - 1
            if eligible >= 0:
                wednesday[i, t, c, l] = values[eligible]
            final[i, t, c, l] = values[-1]


def _kinsa_weekly(data_root, truth_cutoff):
    """Saturday-ending weekly means of daily Kinsa values at every real report date.

    A week (Sunday to Saturday) has a value only once all seven days are visible.
    Each day contributes its latest report on or before that date, so the mean at
    a Wednesday cutoff is what PopHIVE held then. Report times are PopHIVE Git
    commit dates; no availability is modelled.
    """
    snapshot, snapshot_id = _latest_snapshot(data_root, KINSA_DATASET)
    path = snapshot / "archive.csv.gz"
    frame = pd.read_csv(path, dtype={"geo_type": str, "geo_value": str})
    frame = frame[frame["geo_value"].eq("US")].copy()
    frame["report_time"] = frame["report_time"].astype(str).str[:10]
    frame["reference_time"] = frame["reference_time"].astype(str).str[:10]
    frame = frame[frame["report_time"] <= truth_cutoff]
    frame["value"] = pd.to_numeric(frame[KINSA_SIGNAL], errors="coerce")
    frame.loc[~np.isfinite(frame["value"]) | (frame["value"] < 0), "value"] = np.nan
    if frame.empty:
        raise ValueError(f"Kinsa archive has no vintages through {truth_cutoff}")
    # Several commits on one date collapse to the last, as for every covariate.
    frame = frame.sort_values(["reference_time", "report_time"]).drop_duplicates(
        ["reference_time", "report_time"], keep="last")
    days = {day: (g["report_time"].to_numpy(str), g["value"].to_numpy(float))
            for day, g in frame.groupby("reference_time")}
    saturdays = sorted({(pd.Timestamp(day) + pd.Timedelta(days=(5 - pd.Timestamp(day).weekday()) % 7)).strftime("%Y-%m-%d")
                        for day in days})
    rows = []
    for saturday in saturdays:
        week = [(pd.Timestamp(saturday) - pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(6, -1, -1)]
        if not all(day in days for day in week):
            continue
        previous = np.nan
        for release in sorted({r for day in week for r in days[day][0]}):
            values = []
            for day in week:
                reports, daily = days[day]
                index = np.searchsorted(reports, release, side="right") - 1
                values.append(daily[index] if index >= 0 else np.nan)
            value = float(np.mean(values)) if np.isfinite(values).all() else np.nan
            if not (value == previous or (np.isnan(value) and np.isnan(previous))):
                rows.append((release, "US", saturday, value))
            previous = value
    weekly = pd.DataFrame(rows, columns=["report_time", "geo_value", "reference_time", "value"])
    return weekly, dict(dataset=KINSA_DATASET, snapshot_id=snapshot_id, path=str(path), sha256=_sha256(path))


def _read_nwss(path, truth_cutoff):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing real-vintage wastewater covariates: {path}. Build the origin-safe "
            "NWSS state artifact; B2 does not synthesize historical release dates.")
    frame = pd.read_csv(path)
    required = {"report_time", "geo_value", "reference_time", "pathogen",
                "wval_like", "pct_rank", "n_sites"}
    if missing := required - set(frame):
        raise ValueError(f"NWSS artifact lacks columns: {sorted(missing)}")
    for column in ("report_time", "reference_time"):
        frame[column] = frame[column].astype(str).str[:10]
    frame["geo_value"] = frame["geo_value"].astype(str).str.upper()
    frame["pathogen"] = frame["pathogen"].astype(str).str.lower()
    frame = frame[frame["report_time"] <= truth_cutoff].copy()
    if frame.empty:
        raise ValueError(f"NWSS artifact has no vintages through {truth_cutoff}")
    return frame


def build_nwss_covariates(*, data_root="data", output=None):
    """Build origin-safe indices from registered signal and auxiliary archives."""
    signal_root, signal_snapshot = _latest_snapshot(data_root, "delphi_nwss")
    aux_root, aux_snapshot = _latest_snapshot(data_root, "delphi_nwss_aux")
    aux_candidates = list(aux_root.glob("*.csv.gz")) + list(aux_root.glob("*.csv"))
    if not aux_candidates:
        raise FileNotFoundError(f"No NWSS auxiliary payload in {aux_root}")
    aux_path = aux_candidates[0]
    join = ["geo_value", "nwss_source", "reference_time", "sample_index", "pcr_target"]
    signals = {}
    signal_paths = []
    keys = []
    for pathogen in ("flu", "covid", "rsv"):
        matches = list(signal_root.glob(
            f"signal={pathogen}_avg_conc_lin/geo_type=sewershed/archive.csv.gz"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one registered {pathogen} NWSS archive in {signal_root}")
        path = matches[0]
        signal_paths.append(path)
        parts = []
        for chunk in pd.read_csv(path, chunksize=750_000, dtype_backend="pyarrow"):
            for column in join:
                chunk[column] = chunk[column].astype("string[pyarrow]")
            chunk["report_time"] = chunk["report_time"].astype(str).str[:10]
            chunk["reference_time"] = chunk["reference_time"].astype(str).str[:10]
            chunk = chunk[chunk["report_time"] >= NWSS_ARCHIVE_SERVICE_START]
            if len(chunk):
                parts.append(chunk[["report_time", *join, "value"]])
        d = pd.concat(parts, ignore_index=True).sort_values("report_time")
        revision_key = ["report_time", *join]
        duplicates = d[d.duplicated(revision_key, keep=False)]
        if len(duplicates):
            conflicts = duplicates.groupby(revision_key, sort=False)["value"].nunique(dropna=False)
            bad = conflicts[conflicts > 1].index
            if len(bad):
                indexed = d.set_index(revision_key)
                indexed.loc[bad, "value"] = np.nan
                d = indexed.reset_index().sort_values("report_time")
            d = d.drop_duplicates(revision_key, keep="last")
        signals[pathogen] = d
        keys.append(d[join].drop_duplicates())
    wanted = pd.MultiIndex.from_frame(pd.concat(keys, ignore_index=True).drop_duplicates())
    aux_parts = []
    aux_columns = ["report_time", *join, "state_territory", "major_lab_method"]
    for chunk in pd.read_csv(aux_path, usecols=aux_columns,
                             chunksize=750_000, dtype_backend="pyarrow"):
        for column in join:
            chunk[column] = chunk[column].astype("string[pyarrow]")
        chunk["report_time"] = chunk["report_time"].astype(str).str[:10]
        chunk["reference_time"] = chunk["reference_time"].astype(str).str[:10]
        chunk = chunk[(chunk["report_time"] >= NWSS_ARCHIVE_SERVICE_START) &
                      pd.MultiIndex.from_frame(chunk[join]).isin(wanted)]
        if len(chunk):
            aux_parts.append(chunk)
    aux = pd.concat(aux_parts, ignore_index=True).sort_values("report_time")
    aux_key = ["report_time", *join]
    duplicates = aux[aux.duplicated(aux_key, keep=False)].copy()
    if len(duplicates):
        signature = (duplicates["state_territory"].fillna("").astype(str) + "|" +
                     duplicates["major_lab_method"].fillna("").astype(str))
        conflicts = duplicates.assign(_signature=signature).groupby(
            aux_key, sort=False)["_signature"].nunique()
        bad = conflicts[conflicts > 1].index
        if len(bad):
            indexed = aux.set_index(aux_key)
            indexed.loc[bad, ["state_territory", "major_lab_method"]] = pd.NA
            aux = indexed.reset_index().sort_values("report_time")
        aux = aux.drop_duplicates(aux_key, keep="last")
    outputs = []
    for pathogen in ("flu", "covid", "rsv"):
        d = signals[pathogen]
        releases = sorted(set(d["report_time"]) | set(aux["report_time"]))
        for release in releases:
            # Resolve the complete publisher state at the origin, including
            # retractions/null revisions, before filtering invalid values.
            visible = d[d["report_time"] <= release].drop_duplicates(join, keep="last")
            visible["value"] = pd.to_numeric(visible["value"], errors="coerce")
            visible = visible[np.isfinite(visible["value"]) & (visible["value"] > 0) &
                              (visible["reference_time"] <= release)].copy()
            aux_visible = aux[aux["report_time"] <= release].drop_duplicates(join, keep="last")
            visible = visible.merge(aux_visible[join + ["state_territory", "major_lab_method"]],
                                    on=join, how="inner", validate="one_to_one")
            visible = visible[visible["state_territory"].notna()].copy()
            visible["state"] = visible["state_territory"].astype(str).str.upper()
            visible["major_lab_method"] = visible["major_lab_method"].fillna("").astype(str)
            visible["reference_time"] = pd.to_datetime(visible["reference_time"])
            visible["week_end"] = visible["reference_time"] + pd.to_timedelta(
                (5 - visible["reference_time"].dt.weekday) % 7, unit="D")
            visible["g"] = (visible["geo_value"].astype(str) + "|" +
                            visible["nwss_source"].astype(str) + "|" +
                            visible["pcr_target"].astype(str) + "|" +
                            visible["major_lab_method"])
            visible["x"] = np.log(visible["value"])
            groups = visible.groupby("g", sort=False)
            weeks = groups["week_end"].nunique()
            spread = groups["x"].std()
            eligible = weeks[weeks >= 26].index.intersection(spread[spread > 0].index)
            visible = visible[visible["g"].isin(eligible)].copy()
            if visible.empty:
                continue
            groups = visible.groupby("g", sort=False)["x"]
            p10 = visible["g"].map(groups.quantile(.10))
            sd = visible["g"].map(groups.std())
            visible["wval_like"] = np.exp((visible["x"] - p10) / sd)
            visible["pct_rank"] = groups.rank(pct=True)
            # Match analysis/wval/index/indices.py exactly: mean samples within
            # group-week, then median groups within site-week, then median sites.
            group_week = visible.groupby(
                ["state", "geo_value", "g", "week_end"], as_index=False).agg(
                    wval_like=("wval_like", "mean"), pct_rank=("pct_rank", "mean"))
            site_week = group_week.groupby(
                ["state", "geo_value", "week_end"], as_index=False).agg(
                    wval_like=("wval_like", "median"), pct_rank=("pct_rank", "median"))
            state = site_week.groupby(["state", "week_end"], as_index=False).agg(
                wval_like=("wval_like", "median"), pct_rank=("pct_rank", "median"),
                n_sites=("geo_value", "nunique"))
            state = state[state["n_sites"] >= 3].copy()
            nation = site_week.groupby("week_end", as_index=False).agg(
                wval_like=("wval_like", "median"), pct_rank=("pct_rank", "median"),
                n_sites=("geo_value", "nunique"))
            nation = nation[nation["n_sites"] >= 3].copy()
            nation["state"] = "US"
            state = pd.concat([state, nation], ignore_index=True)
            state["report_time"] = release
            state["pathogen"] = pathogen
            outputs.append(state.rename(columns={"state": "geo_value",
                                                  "week_end": "reference_time"}))
    if not outputs:
        raise ValueError("No eligible origin-safe NWSS state covariates were produced")
    result = pd.concat(outputs, ignore_index=True)[
        ["report_time", "geo_value", "reference_time", "pathogen",
         "wval_like", "pct_rank", "n_sites"]]
    result["reference_time"] = result["reference_time"].dt.strftime("%Y-%m-%d")
    # Each report_time is a complete derived state. Emit explicit null rows when
    # a previously present state-week disappears, so consumers do not carry an
    # obsolete index through a retraction or loss of eligible sites.
    complete = []
    for pathogen, values in result.groupby("pathogen"):
        cells = values[["geo_value", "reference_time"]].drop_duplicates()
        for release in sorted(values["report_time"].unique()):
            state = values[values["report_time"].eq(release)]
            expanded = cells.merge(state, on=["geo_value", "reference_time"], how="left")
            expanded["report_time"] = release
            expanded["pathogen"] = pathogen
            complete.append(expanded)
    result = pd.concat(complete, ignore_index=True)[
        ["report_time", "geo_value", "reference_time", "pathogen",
         "wval_like", "pct_rank", "n_sites"]]
    result.insert(1, "geo_type", np.where(result["geo_value"].eq("US"), "nation", "state"))
    metadata = dict(version=1, rows=len(result),
                    report_time_range=[result.report_time.min(), result.report_time.max()],
                    archive_service_start=NWSS_ARCHIVE_SERVICE_START,
                    observed_archive_report_time_start=result.report_time.min(),
                    availability_policy=(
                        "Native signal and auxiliary archive states resolved at each report_time; "
                        "no estimated availability or backfilled vintage."),
                    baseline_policy=(
                        "At each report_time, group p10, sd, empirical rank and 26-week eligibility "
                        "use only samples with report_time and reference_time no later than that origin."),
                    aggregation_policy=(
                        "Mean within group-week; median groups within site; median sites within "
                        "state/nation; at least 3 sites."),
                    inputs=[dict(dataset="delphi_nwss", snapshot_id=signal_snapshot,
                                 files=[dict(path=str(p), sha256=_sha256(p)) for p in signal_paths]),
                            dict(dataset="delphi_nwss_aux", snapshot_id=aux_snapshot,
                                 files=[dict(path=str(aux_path), sha256=_sha256(aux_path))])])
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, compression="gzip")
        output.with_suffix("").with_suffix(".json").write_text(
            json.dumps(dict(metadata, output=str(output)), indent=2) + "\n")
    from tapestry.data.catalog import CATALOG
    from tapestry.data.repository import RawDataRepository
    repository = RawDataRepository(data_root)
    repository.initialize(CATALOG)
    with repository.begin_snapshot(CATALOG[NWSS_DERIVED_DATASET]) as snapshot:
        destination = snapshot.path("data.csv.gz", rows=len(result), media_type="text/csv+gzip")
        result.to_csv(destination, index=False, compression="gzip")
        snapshot.write_json("derivation.json", metadata)
        manifest = snapshot.commit(
            selector={"pathogens": ["flu", "covid", "rsv"],
                      "indices": ["wval_like", "pct_rank"]},
            source_state={"inputs": metadata["inputs"],
                          "origin_safe": True})
    return dict(metadata, dataset_key=manifest.dataset_key, snapshot_id=manifest.snapshot_id)


def _nwss_artifact(data_root):
    snapshot, _ = _latest_snapshot(data_root, NWSS_DERIVED_DATASET)
    path = snapshot / "data.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Registered NWSS state index payload is missing: {path}")
    return path


@dataclass
class B2Dataset(WednesdayDataset):
    @classmethod
    def load(cls, path, *, archive=False):
        with np.load(path, allow_pickle=False) as data:
            required = {"C_final_values", "C_final_available",
                        "C_wednesday_values", "C_wednesday_available",
                        "X_finalized_values", "X_finalized_available"}
            if missing := required - set(data.files):
                raise ValueError(f"B2 dataset lacks covariate arrays: {sorted(missing)}")
            metadata = json.loads(str(data["metadata"]))
            if tuple(metadata.get("covariate_names", ())) != COVARIATES:
                raise ValueError("B2 dataset covariate order differs from the current contract; rebuild it")
            arrays = {k: data[k] for k in data.files if k != "metadata"}
            expected = (*arrays["context_dates"].shape, len(COVARIATES), len(arrays["locations"]))
            for key in ("C_final_values", "C_final_available",
                        "C_wednesday_values", "C_wednesday_available"):
                if arrays[key].shape != expected:
                    raise ValueError(f"B2 {key} shape {arrays[key].shape} != {expected}")
            if arrays["X_finalized_values"].shape != arrays["X_values"].shape:
                raise ValueError("B2 finalized and Wednesday history shapes differ")
            result = cls(arrays, metadata)
        return result if archive else result.model_view()

    def model_view(self):
        if self.metadata.get("view") == "model_calendar":
            return self
        result = super().model_view()
        arrays = result.arrays
        permitted = np.isin(arrays["context_dates"], result.calendar_weeks)
        for key in ("C_final_values", "C_final_available",
                    "C_wednesday_values", "C_wednesday_available",
                    "X_finalized_values", "X_finalized_available"):
            arrays[key][~permitted] = 0
        return B2Dataset(arrays, result.metadata)

    def episodes(self, *, input_mode="wednesday", covariate_groups=None, **kwargs):
        if input_mode not in ("finalized", "wednesday"):
            raise ValueError("input_mode must be 'finalized' or 'wednesday'")
        indices = (tuple(range(len(COVARIATES))) if covariate_groups is None
                   else covariate_indices(covariate_groups))
        view = self.model_view()
        kwargs.setdefault("allow_empty_context", True)
        array_mode = "final" if input_mode == "finalized" else "wednesday"
        values = view.arrays[f"C_{array_mode}_values"]
        available = view.arrays[f"C_{array_mode}_available"]
        for episode in WednesdayDataset.episodes(view, **kwargs):
            i = episode["index"]
            if input_mode == "finalized":
                x_available = view.arrays["X_finalized_available"][i]
                episode["X"] = np.stack((view.arrays["X_finalized_values"][i],
                                         x_available, x_available), axis=2)
            selected_values = np.take(values[i], indices, axis=1)
            selected_available = np.take(available[i], indices, axis=1)
            episode["C"] = np.stack((selected_values, selected_available), axis=2)
            episode["covariate_names"] = tuple(COVARIATES[j] for j in indices)
            episode["input_mode"] = input_mode
            yield episode


def build_b2(base_dataset, *, data_root="data", nwss_path=None):
    """Attach finalized and real Wednesday-as-of covariates to a B1 archive."""
    base_path = Path(base_dataset)
    base = WednesdayDataset.load(base_path, archive=True)
    a = base.arrays
    issuances = a["issuance_dates"].astype(str)
    context_dates = a["context_dates"].astype(str)
    locations = tuple(a["locations"].tolist())
    truth_cutoff = str(base.metadata["truth_cutoff"])
    shape = (*context_dates.shape, len(COVARIATES), len(locations))
    final = np.full(shape, np.nan, np.float32)
    wednesday = np.full(shape, np.nan, np.float32)
    evidence = []
    starts = {}

    # Reconstruct the pinned final six-channel history from B1's own final
    # outcomes and explicitly final-filled history. This keeps the truth source
    # identical to B1 rather than silently substituting the latest CDC panels.
    x_finalized = np.zeros_like(a["X_values"])
    x_finalized_available = np.zeros_like(a["X_available"])
    truth = {}
    for i in range(len(issuances)):
        for t, day in enumerate(context_dates[i]):
            for c in range(a["X_values"].shape[2]):
                for l in range(len(locations)):
                    if a["X_final"][i, t, c, l] and a["X_available"][i, t, c, l]:
                        key = str(day), c, l
                        value = a["X_values"][i, t, c, l]
                        if key in truth and not np.isclose(truth[key], value):
                            raise ValueError(f"Conflicting B1 final history values: {key}")
                        truth[key] = value
        targets = np.concatenate((a["Y_recent"][i], a["Y_future"][i]))
        valid = np.concatenate((a["Y_recent_valid"][i], a["Y_future_valid"][i]))
        for h, day in enumerate(a["target_dates"][i]):
            for c in range(targets.shape[1]):
                for l in range(len(locations)):
                    if valid[h, c, l]:
                        key = str(day), c, l
                        value = targets[h, c, l]
                        if key in truth and not np.isclose(truth[key], value):
                            raise ValueError(f"Conflicting B1 final target values: {key}")
                        truth[key] = value
    for i, dates in enumerate(context_dates):
        for t, day in enumerate(dates):
            for c in range(x_finalized.shape[2]):
                for l in range(len(locations)):
                    key = str(day), c, l
                    if key in truth:
                        x_finalized[i, t, c, l] = truth[key]
                        x_finalized_available[i, t, c, l] = True

    reference_dates = sorted(set(context_dates.flat))
    for dataset, signal, c in CLAIMS:
        paths, snapshot_id = _claim_paths(data_root, dataset, signal)
        frame, start = _claims_frame(paths, reference_dates, truth_cutoff)
        _fill_revisions(frame, issuances, context_dates, locations, final, wednesday, c)
        starts[COVARIATES[c]] = start
        evidence.append(dict(dataset=dataset, signal=signal, snapshot_id=snapshot_id,
                             files=[dict(path=str(p), sha256=_sha256(p)) for p in paths]))

    nwss_path = Path(nwss_path) if nwss_path else _nwss_artifact(data_root)
    nwss = _read_nwss(nwss_path, truth_cutoff)
    pathogen_index = {"flu": 0, "covid": 1, "rsv": 2}
    for metric, offset in (("wval_like", 4), ("pct_rank", 7)):
        for pathogen, j in pathogen_index.items():
            c = offset + j
            frame = nwss.loc[nwss["pathogen"].eq(pathogen),
                             ["report_time", "geo_value", "reference_time", metric]].rename(
                                 columns={metric: "value"})
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.sort_values("report_time").drop_duplicates(
                ["report_time", "geo_value", "reference_time"], keep="last")
            _fill_revisions(frame, issuances, context_dates, locations, final, wednesday, c)
            valid_rows = frame[np.isfinite(frame["value"])]
            starts[COVARIATES[c]] = valid_rows["report_time"].min() if len(valid_rows) else None
    evidence.append(dict(dataset=NWSS_DERIVED_DATASET, snapshot_id=nwss_path.parent.name,
                         path=str(nwss_path), sha256=_sha256(nwss_path)))

    kinsa, kinsa_evidence = _kinsa_weekly(data_root, truth_cutoff)
    c = COVARIATES.index("kinsa_ili")
    _fill_revisions(kinsa, issuances, context_dates, locations, final, wednesday, c)
    starts[COVARIATES[c]] = kinsa["report_time"].min()
    evidence.append(kinsa_evidence)

    arrays = {key: value.copy() for key, value in a.items()}
    arrays.update(X_finalized_values=x_finalized,
                  X_finalized_available=x_finalized_available,
                  C_final_values=np.nan_to_num(final), C_final_available=np.isfinite(final),
                  C_wednesday_values=np.nan_to_num(wednesday),
                  C_wednesday_available=np.isfinite(wednesday))
    coverage = {
        mode: {name: int(arrays[f"C_{mode}_available"][:, :, c, :].sum())
               for c, name in enumerate(COVARIATES)}
        for mode in ("final", "wednesday")
    }
    zero_coverage_locations = {
        mode: {name: [locations[l] for l in range(len(locations))
                      if not arrays[f"C_{mode}_available"][:, :, c, l].any()]
               for c, name in enumerate(COVARIATES)}
        for mode in ("final", "wednesday")
    }
    metadata = dict(base.metadata, version=1, model="B2", covariates=COVARIATES,
                    covariate_names=COVARIATES,
                    covariate_units=["native_percent_claims"] * 4 +
                                    ["dimensionless_positive_unbounded"] * 3 +
                                    ["unit_interval"] * 3 +
                                    ["weekly_mean_percent_of_kinsa_users_ill"],
                    covariate_native_geography={name: "nation" if name in NATIONAL_ONLY else "state+nation"
                                                for name in COVARIATES},
                    covariate_groups={k: [COVARIATES[i] for i in v]
                                      for k, v in COVARIATE_GROUPS.items()},
                    covariate_axes=["issuance", "history_week", "covariate", "location"],
                    covariate_input_modes=["finalized", "wednesday"],
                    history_input_modes=dict(
                        finalized="Pinned B1 reference-final six-channel history",
                        wednesday="B1 history: reference finals except the latest two weeks, which use eligible Wednesday reports with flagged final fallback"),
                    covariate_vintage_policy=(
                        "Wednesday mode uses the latest native report_time on or before the "
                        "Wednesday cutoff. Finalized mode uses the latest real revision through "
                        "the B1 truth cutoff. No modeled latency or backfilled vintage is used."),
                    covariate_vintage_starts=starts,
                    covariate_coverage=coverage,
                    covariate_zero_coverage_locations=zero_coverage_locations,
                    covariate_evidence=evidence,
                    nwss_diagnostic=(
                        "n_sites remains in the derived NWSS artifact for audit and is not a model covariate."),
                    outcome_policy="B1 final outcomes in both covariate input modes",
                    base_dataset=dict(path=str(base_path), sha256=_sha256(base_path)))
    return B2Dataset(arrays, metadata)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", default="data/processed/build_b1_wednesday_calendar.npz")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--nwss", help="Override the registered derived NWSS artifact (diagnostics only)")
    parser.add_argument("--output", default=DEFAULT_DATASET)
    args = parser.parse_args(argv)
    try:
        dataset = build_b2(args.base_dataset, data_root=args.data_root, nwss_path=args.nwss)
        dataset.save(args.output)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=args.output,
                          shape=list(dataset.arrays["C_final_values"].shape),
                          covariates=list(COVARIATES),
                          vintage_starts=dataset.metadata["covariate_vintage_starts"]), indent=2))


def main_nwss(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Build registered origin-safe B2 NWSS indices")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", help="Optional diagnostic copy outside the registered snapshot")
    args = parser.parse_args(argv)
    try:
        result = build_nwss_covariates(data_root=args.data_root, output=args.output)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
