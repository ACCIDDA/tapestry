"""Versioned post-intake selection policy, independent of the explorer.

Selection preserves native rows and releases. It never averages providers,
resamples time, broadcasts geography, or fills absent observations.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import tempfile
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .catalog import CATALOG
from .lineage import NHSN_DELPHI, series_lineage
from .geography import observation_geography
from .tables import Artifact, RawTables, TableSource

POLICY_VERSION = "4"
FAMILY_TITLES = {
    "nhsn": "NHSN · Hospital surveillance",
    "nssp": "NSSP · Emergency department surveillance",
    "nwss": "NWSS · Wastewater surveillance",
}
PATHOGENS = {"c19": "COVID-19", "flu": "Influenza", "rsv": "RSV"}
NHSN_MEASURES: dict[str, str] = {}
for pathogen, label in PATHOGENS.items():
    for suffix, title in (
        ("newadm", "Total admissions"), ("hosppats", "Total hospitalized"),
        ("icupats", "Total ICU patients"),
    ):
        NHSN_MEASURES[f"totalconf{pathogen}{suffix}"] = f"{label} · {title}"
    # Current research selection is all ages only; unknown-age strata are excluded too.

# Delphi V5 NHSN signals mirror explicit CDC HRD fields. Keep those fields in
# the NHSN menu too, so Delphi values can be selected as variants of the same
# measure rather than appearing as unrelated provider-only series.
for pathogen, label in PATHOGENS.items():
    NHSN_MEASURES[f"totalconf{pathogen}newadmhosprep"] = f"{label} · Hospitals reporting admissions"
NHSN_MEASURES.update({
    "numinptbeds": "Inpatient beds",
    "numinptbedsocc": "Inpatient beds occupied",
})


NSSP_COLUMNS = {
    "percent_visits_combined": "combined",
    "percent_visits_covid": "covid",
    "percent_visits_influenza": "influenza",
    "percent_visits_rsv": "rsv",
    "percent_visits_smoothed": "combined",
    "percent_visits_smoothed_1": "influenza",
    "percent_visits_smoothed_covid": "covid",
    "percent_visits_smoothed_rsv": "rsv",
}
NWSS_COLUMNS = {
    "pcr_target_avg_conc": "concentration",
    "pcr_target_avg_conc_lin": "concentration_linear",
    "pcr_target_flowpop_lin": "flow_population_normalized",
    "pcr_target_mic_lin": "microbial_normalized",
}
NWSS_LABELS = {
    "concentration": "Measured concentration",
    "concentration_linear": "Concentration with below-detection substitution",
    "flow_population_normalized": "Flow/population normalized concentration",
    "microbial_normalized": "Microbial normalized concentration",
    "activity_level": "Wastewater activity level",
}
HUB_FILES = {
    "hub_flusight_current": ("target-data/time-series.csv",),
    "hub_covid_current": ("target-data/time-series.parquet",),
    "hub_rsv_current": ("target-data/time-series.parquet",),
    "hub_flusight_legacy": ("data-truth/truth-Incident Hospitalizations.csv",),
    "hub_covid_legacy": tuple(
        f"data-truth/truth-{kind} {measure}.csv"
        for kind in ("Incident", "Cumulative")
        for measure in ("Cases", "Deaths", "Hospitalizations")
    ),
}
RELEASE_LABELS = {
    "cdc_nhsn_final": "Finalized", "cdc_nhsn_preliminary": "Preliminary",
    "cdc_nhsn_initial_release": "First publication", "delphi_nhsn": "Delphi archive",
    "cdc_nssp_trajectories": "CDC trajectories", "cdc_nssp_daily": "CDC daily",
    "cdc_nssp_demographics": "CDC demographics", "delphi_nssp": "Delphi archive",
    "hub_covid_current": "COVID-19 Hub target data",
    "hub_flusight_current": "FluSight Hub target data",
    "hub_rsv_current": "RSV Hub target data",
}
# Menu order; unlisted pathogens follow alphabetically, then non-pathogen measures.
PATHOGEN_TITLES = {
    "covid": "COVID-19", "influenza": "Influenza", "rsv": "RSV",
    "combined": "Combined COVID-19, influenza, and RSV",
    "ari": "Acute respiratory illness",
    "adenovirus": "Adenovirus", "hcov": "Seasonal coronaviruses (HCoV)",
    "hmpv": "Human metapneumovirus (HMPV)", "piv": "Parainfluenza (PIV)",
    "rv/ev": "Rhinovirus/enterovirus (RV/EV)",
}
PATHOGEN_LAST = {"all_cause": "All-cause hospital capacity", "unspecified": "Unspecified pathogen"}


def pathogen_of(key: str, name: str, pathogen: str = "") -> tuple[str, str, int]:
    """Canonical pathogen key, title, and menu rank; ``unspecified`` rather than a guess."""
    explicit = pathogen.lower()
    text = " ".join([explicit, name.lower(), key.lower()])
    if explicit in {"sars-cov-2", "covid-19", "c19"} or re.search(r"covid|c19|sars", text):
        canonical = "covid"
    elif explicit == "flu" or re.search(r"flu", text):
        canonical = "influenza"
    elif explicit in PATHOGEN_TITLES:
        canonical = explicit
    elif "rsv" in text:
        canonical = "rsv"
    elif re.search(r"combined|percent_visits_smoothed$", text):
        canonical = "combined"
    elif re.search(r"(?:^|_)ari(?:_|$)", text):
        canonical = "ari"
    elif "inptbeds" in text:
        canonical = "all_cause"
    else:
        canonical = explicit or "unspecified"
    order = [*PATHOGEN_TITLES, "", *PATHOGEN_LAST]  # "" ranks other named pathogens
    title = PATHOGEN_TITLES.get(canonical) or PATHOGEN_LAST.get(canonical) or pathogen
    return canonical, title, order.index(canonical if canonical in order else "")


def family(key: str) -> str:
    for name in FAMILY_TITLES:
        if key.startswith((f"cdc_{name}_", f"delphi_{name}")):
            return name
    return key


def source_groups(key: str) -> set[str]:
    if key.startswith("hub_") and key.endswith("_current"):
        return {"nhsn", "nssp"}
    return {family(key)}


def source_signal(path: str, dimensions: Mapping[str, Any] | None = None) -> str:
    match = re.search(r"(?:^|/)signal=([^/]+)", path)
    return str((dimensions or {}).get("signal") or (match.group(1) if match else ""))


def measure_columns(key: str, columns: Sequence[str], path: str = "") -> tuple[str, ...]:
    """Explicit outcome allowlists for the selected families; other families pass through.

    Consumers still distinguish numeric measures from coordinates/dimensions for
    passthrough families. Policy families never discover extra numeric outcomes.
    """
    signal = source_signal(path)
    if key.startswith("cdc_nhsn_"):
        allowed = NHSN_MEASURES
    elif key == "delphi_nhsn":
        allowed = {"value"} if not signal or signal in NHSN_DELPHI else set()
    elif key == "cdc_nssp_trajectories":
        allowed = NSSP_COLUMNS
    elif key in {"cdc_nssp_daily", "cdc_nssp_demographics"}:
        allowed = {"percent_visits"}
    elif key == "delphi_nssp":
        allowed = {"value"}
    elif key.startswith("cdc_nwss_"):
        allowed = {"site_wval"} if key == "cdc_nwss_wval" else NWSS_COLUMNS
    elif key == "delphi_nwss":
        allowed = {"value"}
    elif key in HUB_FILES or key == "hub_rsvnet":
        allowed = {"observation"} if key.endswith("_current") else {"value"}
    else:
        return tuple(columns)
    return tuple(column for column in columns if column.lower() in allowed)


def selected_row(key: str, path: str, row: Mapping[str, Any]) -> bool:
    if key == "delphi_nhsn":
        return source_signal(path, row) in NHSN_DELPHI
    if key == "delphi_nssp":
        return source_signal(path, row) in CATALOG[key].config["signals"]
    if key == "delphi_nwss":
        return source_signal(path, row) in CATALOG[key].config["signals"]
    return True


def describe(key: str, column: str, path: str, dimensions: Mapping[str, Any], *,
             catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Canonical menu identity and explicit variant provenance for one outcome."""
    group = family(key)
    spec = CATALOG.get(key)
    metadata = spec.to_dict() if spec else {"key": key}
    lineage = series_lineage(metadata, column, path, dimensions)
    origin_column = lineage["parent_column"]
    if key.startswith("hub_") and lineage["lineage_status"] == "mapped":
        group = lineage["parent_dataset"].lower()
    signal = source_signal(path, dimensions)
    name = column
    smoothing = ""
    pathogen = str(dimensions.get("pathogen", dimensions.get("pathogen_target", "")))
    if group == "nhsn":
        name = origin_column or NHSN_DELPHI.get(signal, column)
        title = NHSN_MEASURES.get(name, name)
    elif group == "nssp":
        mapped_column = origin_column or column
        pathogen = NSSP_COLUMNS.get(mapped_column, pathogen)
        if signal:
            pathogen = signal.split("pct_ed_visits_", 1)[-1]
        pathogen = {"covid-19": "covid", "flu": "influenza"}.get(pathogen.lower(), pathogen.lower())
        smoothing = "Smoothed" if "smoothed" in mapped_column or signal.startswith("smoothed_") else "Reported"
        # Long-form daily/demographic rows join the corresponding weekly CDC field.
        name = mapped_column if mapped_column in NSSP_COLUMNS else f"percent_visits_{pathogen}"
        title = f"{pathogen.upper() if pathogen in {'rsv','ari'} else pathogen.title()} · ED visit percentage"
    elif group == "nwss":
        pathogen = {"cdc_nwss_covid_raw": "covid", "cdc_nwss_influenza_raw": "flu",
                    "cdc_nwss_rsv_raw": "rsv"}.get(key, pathogen)
        name = "activity_level" if key == "cdc_nwss_wval" else NWSS_COLUMNS.get(column, column)
        if signal:
            pathogen, _, suffix = signal.partition("_")
            name = {"avg_conc": "concentration", "avg_conc_lin": "concentration_linear",
                    "flowpop_lin": "flow_population_normalized", "mic_lin": "microbial_normalized"}.get(suffix, suffix)
        pathogen = {"sars-cov-2": "covid", "covid-19": "covid", "influenza a": "flu"}.get(pathogen.lower(), pathogen.lower())
        title = f"{pathogen.upper()} · {NWSS_LABELS.get(name, name)}"
        name = f"{pathogen}_{name}"
    elif key.startswith("hub_"):
        name = str(dimensions.get("target", dimensions.get("target_variable", "")))
        if not name:
            name = Path(path).stem.removeprefix("truth-")
        title = name
    else:
        name = signal or column
        # Preserve unrelated family's existing signal facets.
        facets = {k: v for k, v in dimensions.items() if k != "spatial_support"}
        name += ":" + json.dumps(facets, sort_keys=True) if facets else ""
        title = signal or column
    publisher_column = None
    publisher_dataset = None
    if catalog:
        # Prefer the shared originating CDC product so mirrors use the same label.
        preferred = {"nhsn": "cdc_nhsn_final", "nssp": "cdc_nssp_trajectories"}.get(group)
        candidates = list(dict.fromkeys(filter(None, [preferred, key] + sorted(catalog))))
        for candidate in candidates:
            if not candidate.startswith(f"cdc_{group}_") and candidate != key:
                continue
            fields = catalog.get(candidate, {}).get("column_metadata", {})
            field_name = name if group in {"nhsn", "nssp"} else (origin_column or column)
            info = fields.get(field_name)
            if info is None and candidate == key and key.startswith("cdc_"):
                info = fields.get(column)
            if info:
                publisher_column, publisher_dataset = info, candidate
                title = info["name"]
                # Generic long-form fields need their pathogen facet in the label.
                if dimensions.get("pathogen") and field_name not in NSSP_COLUMNS and group != "nhsn":
                    title = f"{dimensions['pathogen']} · {title}"
                break
    cadence = spec.temporal_resolution if spec else ""
    support = str(dimensions.get("spatial_support", "native state"))
    if "spatial_support" not in dimensions:
        geo_type = str(dimensions.get("geo_type", "")).lower()
        geography = str(dimensions.get("jurisdiction", dimensions.get("geography", dimensions.get("level", "")))).lower()
        if not geo_type:
            if re.fullmatch(r"(?:hhs(?:\s*region)?|region)\s*0*(?:10|[1-9])", geography):
                geo_type = "hhs"
            elif geography in {"us", "usa", "united states", "nation", "national"}:
                geo_type = "nation"
        if geo_type in {"nation", "national", "hhs", "sewershed", "county", "hsa", "hsa_nci"}:
            support = f"native {geo_type}"
    if group == "nwss" and support == "native state":
        support = "native site/sewershed" if "site" in dimensions else "site/sewershed state summary"
    details = [RELEASE_LABELS.get(key, key), cadence, smoothing, support]
    if group == "nssp":
        details.append("Proportion (0–1)" if key.startswith("hub_") else "Percent (0–100)")
    for k, v in sorted(dimensions.items()):
        if k not in {"signal", "pathogen", "pathogen_target", "target", "target_variable", "spatial_support"}:
            details.append(f"{k}={v}")
    rank = {"cdc_nhsn_final": 0, "cdc_nhsn_preliminary": 1,
            "cdc_nhsn_initial_release": 2, "delphi_nhsn": 3,
            "cdc_nssp_trajectories": 0, "cdc_nssp_daily": 1,
            "cdc_nssp_demographics": 2, "delphi_nssp": 3}.get(key, int(key.startswith("delphi_")))
    pathogen, pathogen_title, pathogen_rank = pathogen_of(key, name, pathogen)
    return {
        "pathogen": pathogen, "pathogen_title": pathogen_title, "pathogen_rank": pathogen_rank,
        "source_group": group,
        "source_group_title": FAMILY_TITLES.get(group, spec.title if spec else key),
        "signal_key": f"{group}:{name}", "signal_title": title,
        "measure_id": signal or str(dimensions.get("target", "")) or column,
        "origin_column": name if group in {"nhsn", "nssp"} else origin_column,
        "column_name": publisher_column["name"] if publisher_column else title,
        "column_description": publisher_column.get("description", "") if publisher_column else "",
        "column_metadata_dataset": publisher_dataset,
        "provider_kind": "delphi" if key.startswith("delphi_") else "hub" if key.startswith("hub_") else "cdc",
        "variant_label": " · ".join(filter(None, details)),
        "variant_rank": (int(support != "native state"), rank, int(smoothing == "Smoothed")),
        "selection_policy": POLICY_VERSION,
        "spatial_support": support,
    }


@dataclass(frozen=True)
class SelectedRecord:
    """Selected raw values, native metadata, and acquisition provenance.

    ``metadata`` is context only, never an automatically discovered feature set.
    No release resolution, aggregation, or provider substitution is performed.
    """
    source_group: str
    dataset_key: str
    snapshot_id: str
    source_path: str
    values: Mapping[str, Any]
    metadata: Mapping[str, Any]
    available_at: str | None
    column_catalog: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def signals(self) -> dict[str, dict[str, Any]]:
        """Canonical descriptions for each selected value, including variant provenance."""
        return {column: describe(self.dataset_key, column, self.source_path, self.metadata, catalog=self.column_catalog)
                for column in self.values}


class SelectedData(RawTables):
    """Post-intake table/measure selection shared by the explorer and downstream."""
    def __init__(self, data_root: str | Path):
        super().__init__(data_root)
        self.audit: list[dict[str, Any]] = []

    def _catalog(self):
        catalog = super()._catalog()
        # Old migrated Delphi feeds stay on disk, but are not selected anymore.
        return {k: v for k, v in catalog.items()
                if k not in {"delphi_fluview", "delphi_fluview_clinical", "delphi_flusurv"}}

    def table_decision(self, key: str, name: str, names: Sequence[str]) -> tuple[bool, str]:
        if key in HUB_FILES:
            accepted = name in HUB_FILES[key]
            return accepted, "canonical hub truth" if accepted else "duplicate, auxiliary, or derived hub output"
        if key == "hub_rsvnet":
            candidates = sorted(n for n in names if re.fullmatch(r"target-data/\d{4}-\d{2}-\d{2}_rsvnet_hospitalization.csv", n))
            return bool(candidates and name == candidates[-1]), "latest RSV-NET file; historical releases remain in Git/archive"
        return self._is_primary_hub_member(name), "raw table"

    def _tar_sources(self, artifact: Artifact) -> Iterator[TableSource]:
        import tarfile
        import tempfile
        import shutil
        with tarfile.open(artifact.path, "r:gz") as archive:
            members = [m for m in archive if m.isfile()]
            names = [m.name for m in members]
            for required in HUB_FILES.get(str(artifact.dataset["key"]), ()):
                if required not in names:
                    self.audit.append({"dataset_key": artifact.dataset["key"], "source_path": required,
                                       "status": "unavailable", "message": "Canonical truth file absent; no provider substitution"})
            for member in members:
                if not self._is_tabular_member(member.name):
                    continue
                accepted, reason = self.table_decision(str(artifact.dataset["key"]), member.name, names)
                if not accepted:
                    if not str(artifact.dataset["key"]).startswith("hub_"):
                        continue
                    self.audit.append({"dataset_key": artifact.dataset["key"], "source_path": member.name,
                                       "status": "excluded", "message": reason})
                    continue
                with tempfile.TemporaryDirectory(prefix="tapestry-selected-") as directory:
                    path = Path(directory) / Path(member.name).name
                    with archive.extractfile(member) as source, path.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    yield self._source_for_path(artifact, path, f"{artifact.relative_path}!/{member.name}", member.name)

    def _validated_hub_rows(self, table: TableSource):
        """Choose unique rows only after checking the entire canonical table.

        Spill keys to SQLite so cross-batch conflicts cannot be averaged or
        resolved by file order. Retain compact row offsets for repeat reads.
        """
        original = table.iter_rows
        offsets = None
        conflicts = set()

        def rows():
            nonlocal offsets, conflicts
            if offsets is None:
                with tempfile.TemporaryDirectory(prefix="tapestry-truth-check-") as directory:
                    db = sqlite3.connect(Path(directory) / "keys.sqlite3")
                    try:
                        db.execute("CREATE TABLE keys (identity TEXT PRIMARY KEY, value TEXT, row_number INTEGER, conflict INTEGER DEFAULT 0)")
                        for number, row in enumerate(original()):
                            day = row.get(table.event_date_column) if table.event_date_column else None
                            day = day or row.get("date") or row.get("target_end_date")
                            vintage = row.get(table.vintage_column) if table.vintage_column else table.fallback_vintage
                            identity = json.dumps([
                                row.get("location", row.get("state")), day, vintage,
                                row.get("target", row.get("target_variable", "")),
                                row.get("age_group", row.get("age", "")),
                            ], default=str, separators=(",", ":"))
                            value = row.get("observation", row.get("value"))
                            if value is not None and str(value).strip().lower() not in {"", "na", "nan", "null", "none", "*", "."}:
                                value = float(value)
                            else:
                                value = None
                            encoded = json.dumps(value, allow_nan=False)
                            db.execute(
                                "INSERT INTO keys(identity,value,row_number) VALUES(?,?,?) "
                                "ON CONFLICT(identity) DO UPDATE SET conflict=MAX(keys.conflict, keys.value != excluded.value)",
                                (identity, encoded, number))
                        count = db.execute("SELECT COUNT(*) FROM keys WHERE conflict=1").fetchone()[0]
                        if count:
                            self.audit.append({"dataset_key": table.dataset_key, "source_path": table.source_path,
                                               "status": "quarantined",
                                               "message": f"{count} conflicting location/target/event/release keys quarantined as missing; remaining truth retained"})
                        offsets = array("Q", (r[0] for r in db.execute("SELECT row_number FROM keys ORDER BY row_number")))
                        conflicts = {r[0] for r in db.execute("SELECT row_number FROM keys WHERE conflict=1")}
                    finally:
                        db.close()
            wanted = iter(offsets)
            next_row = next(wanted, None)
            for number, row in enumerate(original()):
                if next_row is None:
                    break
                if number == next_row:
                    if number in conflicts:
                        row = dict(row)
                        row["observation" if "observation" in row else "value"] = None
                        row["_selection_conflict"] = True
                    yield row
                    next_row = next(wanted, None)
        return rows

    def _table_sources(self, artifact: Artifact) -> Iterator[TableSource]:
        from dataclasses import replace
        key = str(artifact.dataset["key"])
        signal = source_signal(artifact.relative_path)
        if key == "delphi_nhsn" and signal and signal not in NHSN_DELPHI:
            self.audit.append({"dataset_key": key, "source_path": artifact.relative_path,
                               "status": "excluded", "message": "Not an NHSN patient-count measure"})
            return
        spec = CATALOG.get(key)
        resolutions = (spec.geographic_resolutions if spec else
                       tuple(artifact.dataset.get("geographic_resolutions", ())))
        match = re.search(r"(?:^|/)geo_type=([^/]+)", artifact.relative_path)
        path_geo_type = match.group(1).lower() if match else None
        if ((path_geo_type and path_geo_type not in {"state", "nation", "national"}) or
                (resolutions and not set(resolutions).intersection({"state", "nation", "national"}))):
            self.audit.append({"dataset_key": key, "source_path": artifact.relative_path,
                               "status": "excluded", "message": "Post-intake filter: state and national observations only"})
            return
        for table in super()._table_sources(artifact):
            def rows(original=table.iter_rows, path=table.source_path):
                for row in original():
                    if observation_geography(row, resolutions, path_geo_type) and selected_row(key, path, row):
                        yield row
            table = replace(table, iter_rows=rows, geographic_resolutions=resolutions)
            if key in HUB_FILES or key == "hub_rsvnet":
                table = replace(table, iter_rows=self._validated_hub_rows(table))
            yield table

    def selected_tables(self, *, group: str | None = None, dataset_key: str | None = None) -> Iterator[TableSource]:
        """Consume each yielded table before advancing; tar files are temporary."""
        for artifact in self.artifacts():
            key = str(artifact.dataset["key"])
            if (group and group not in source_groups(key)) or (dataset_key and key != dataset_key):
                continue
            for table in self._table_sources(artifact):
                if group and key.startswith("hub_") and key.endswith("_current"):
                    from dataclasses import replace
                    def rows(original=table.iter_rows, key=key, path=table.source_path):
                        for row in original():
                            if describe(key, "observation", path, row)["source_group"] == group:
                                yield row
                    table = replace(table, iter_rows=rows)
                yield table

    def iter_records(self, *, group: str | None = None, dataset_key: str | None = None,
                     available_by: str | None = None) -> Iterator[SelectedRecord]:
        """Stream native selected records, optionally excluding releases after a cutoff.

        All eligible revisions are emitted. Consumers must resolve full snapshots
        and revisions themselves, rather than treating them as independent events.
        Unknown historical availability is excluded when a cutoff is supplied.
        """
        cutoff = _timestamp(available_by) if available_by else None
        catalog = self._catalog()
        for table in self.selected_tables(group=group, dataset_key=dataset_key):
            manifest = json.loads((self.data_root / "raw" / table.dataset_key / "snapshots" /
                                   table.snapshot_id / "manifest.json").read_text())
            fallback = manifest.get("source_state", {}).get("commit_time") or manifest.get("retrieved_at")
            for row in table.iter_rows():
                columns = measure_columns(table.dataset_key, tuple(row), table.source_path)
                if family(table.dataset_key) == table.dataset_key and not table.dataset_key.startswith("hub_"):
                    spec = CATALOG.get(table.dataset_key)
                    context = set(spec.natural_key if spec else ()) | {
                        "state", "location", "geo_value", "geo_type", "region", "level", "geography",
                        "season", "respseason", "season_week", "population", "population_served",
                        "date", "week", "week_end", "weekendingdate", "mmwrweek_end", "epiweek",
                    }
                    columns = tuple(k for k in columns if k not in context
                                    and not k.lower().endswith(("_id", "_code", "_fips"))
                                    and _numeric_or_missing(row[k]))
                available = row.get(table.vintage_column) if table.vintage_column else None
                available = str(available or fallback or "") or None
                if cutoff and (not available or _timestamp(available) > cutoff):
                    continue
                record_group = family(table.dataset_key)
                if table.dataset_key.startswith("hub_") and columns:
                    record_group = describe(table.dataset_key, columns[0], table.source_path, row)["source_group"]
                yield SelectedRecord(record_group, table.dataset_key, table.snapshot_id,
                                     table.source_path, {k: row[k] for k in columns},
                                     {k: v for k, v in row.items() if k not in columns}, available, catalog)

    def summary(self) -> dict[str, Any]:
        groups: dict[str, list[str]] = {}
        for key in CATALOG:
            for group in sorted(source_groups(key)):
                groups.setdefault(group, []).append(key)
        downloaded = {str(a.dataset["key"]) for a in self.artifacts()}
        return {"policy_version": POLICY_VERSION, "raw_catalog_datasets": len(CATALOG),
                "logical_source_groups": len(groups), "nhsn_count_measures": len(NHSN_MEASURES),
                "groups": groups, "downloaded_selected_datasets": sorted(downloaded),
                "missing_datasets": sorted(set(CATALOG) - downloaded)}


def _numeric_or_missing(value: Any) -> bool:
    if value is None or str(value).strip().lower() in {"", "*", ".", "na", "n/a", "nan", "null", "none"}:
        return True
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect the shared post-intake selection policy")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)
    print(json.dumps(SelectedData(args.data_root).summary(), indent=2))


if __name__ == "__main__":
    main()
