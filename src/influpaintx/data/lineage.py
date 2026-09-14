"""Publisher lineage, shared by the acquisition catalog and explorer.

Rules match a signal, target, or file, since a Hub can contain multiple parents.
Unmapped derived columns remain explicitly unknown rather than guessed.
"""
from __future__ import annotations

from dataclasses import replace
from fnmatch import fnmatch
from typing import Any, Mapping

from .models import DatasetSpec

FLUSIGHT_README = "https://github.com/cdcepi/FluSight-forecast-hub/blob/main/target-data/README.md"
DELPHI_DOCS = "https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/"


NHSN_DELPHI = {
    "confirmed_admissions_covid_ew": "totalconfc19newadm",
    "confirmed_admissions_flu_ew": "totalconfflunewadm",
    "confirmed_admissions_rsv_ew": "totalconfrsvnewadm",
    "hosprep_confirmed_admissions_covid_ew": "totalconfc19newadmhosprep",
    "hosprep_confirmed_admissions_flu_ew": "totalconfflunewadmhosprep",
    "hosprep_confirmed_admissions_rsv_ew": "totalconfrsvnewadmhosprep",
    "inpatient_beds_ew": "numinptbeds",
    # Despite the signal name, Delphi documents an occupied-bed count.
    "inpatient_beds_occupied_pct_ew": "numinptbedsocc",
}

def with_lineage(spec: DatasetSpec) -> DatasetSpec:
    key = spec.key
    rules = []
    if key.startswith("cdc_"):
        parent = key.split("_")[1].upper()
        rules = [{"column": "*", "parent_dataset": parent, "parent_column": "{column}",
                  "transform": "identity", "source_url": spec.source_url}]
    elif key.startswith("delphi_"):
        parent = key.removeprefix("delphi_").upper()
        if parent == "NHSN":
            rules = [{"signal": signal, "column": "value", "parent_dataset": parent,
                      "parent_column": column, "transform": "identity at native support; parent aggregates retain geographic support",
                      "source_url": DELPHI_DOCS + "nhsn.html"} for signal, column in NHSN_DELPHI.items()]
        elif parent == "NSSP":
            cdc_columns = {
                "combined": "percent_visits_combined",
                "covid": "percent_visits_covid",
                "influenza": "percent_visits_influenza",
                "rsv": "percent_visits_rsv",
            }
            for pathogen in ("ari", "combined", "covid", "influenza", "rsv"):
                for prefix in ("", "smoothed_"):
                    if pathogen == "ari" and prefix:
                        continue
                    rules.append({"signal": f"{prefix}pct_ed_visits_{pathogen}", "column": "value",
                                  "parent_dataset": parent,
                                  "parent_column": (({
                                      "combined": "percent_visits_smoothed",
                                      "influenza": "percent_visits_smoothed_1",
                                      "covid": "percent_visits_smoothed_covid",
                                      "rsv": "percent_visits_smoothed_rsv",
                                  }[pathogen] if prefix else cdc_columns.get(pathogen))),
                                  "transform": "publisher percentage; geographic aggregation may differ",
                                  "source_url": DELPHI_DOCS + "nssp.html"})
        elif parent == "NWSS":
            mapping = {
                f"{pathogen}_{suffix}": f"pcr_target_{cdc_suffix}"
                for pathogen in ("covid", "flu", "rsv")
                for suffix, cdc_suffix in (
                    ("avg_conc", "avg_conc"),
                    ("avg_conc_lin", "avg_conc_lin"),
                    ("flowpop_lin", "flowpop_lin"),
                    ("mic_lin", "mic_lin"),
                )
            }
            rules = [{"signal": signal, "column": "value", "parent_dataset": parent,
                      "parent_column": column, "transform": "identity; Delphi signal grouped with the corresponding CDC measure",
                      "source_url": DELPHI_DOCS + "nwss.html"} for signal, column in mapping.items()]
    elif key.endswith("_current"):
        parent = "NHSN / NSSP"
        pathogen, disease = {"hub_flusight_current": ("influenza", "flu"),
                             "hub_covid_current": ("covid", "c19"),
                             "hub_rsv_current": ("rsv", "rsv")}[key]
        for selector in ({"path": "*hospital-admissions*"}, {"target": "*hosp*"}):
            rules.append({**selector, "parent_dataset": "NHSN",
                          "parent_column": f"totalconf{disease}newadm", "transform": "weekly admissions count",
                          "source_url": FLUSIGHT_README if disease == "flu" else spec.source_url.removesuffix(".git") + "/tree/main/target-data"})
        for selector in ({"path": "*ed-visits*"}, {"target": "*ed visits*"}):
            rules.append({**selector, "parent_dataset": "NSSP",
                          "parent_column": f"percent_visits_{pathogen}",
                          "transform": "percent / 100; state rows use county=All",
                          "source_url": FLUSIGHT_README if disease == "flu" else spec.source_url.removesuffix(".git") + "/tree/main/target-data"})
    else:
        parent = {"hub_flusight_legacy": "HHS Protect", "hub_covid_legacy": "JHU CSSE / HHS Protect",
                  "hub_rsvnet": "RSV-NET"}[key]
    return replace(spec, parent_dataset=parent, column_lineage=tuple(rules))


def series_lineage(metadata: Mapping[str, Any], column: str, path: str,
                   dimensions: Mapping[str, Any]) -> dict[str, Any]:
    signal = str(dimensions.get("signal", ""))
    if not signal:
        signal = next((part.split("=", 1)[1] for part in path.split("/") if part.startswith("signal=")), "")
    result = {"parent_dataset": metadata.get("parent_dataset") or metadata.get("key", "Unknown"),
              "parent_column": None, "parent_transform": None, "lineage_source_url": None,
              "lineage_status": "unmapped"}
    values = {"column": column, "signal": signal, "path": path,
              "target": str(dimensions.get("target", dimensions.get("target_variable", "")))}
    for rule in metadata.get("column_lineage", ()):
        # Canonical current Hub outcomes use observation; older exports use value.
        if str(metadata.get("key", "")).endswith("_current") and column not in {"value", "observation"}:
            continue
        if all(fnmatch(values[key].lower(), str(rule[key]).lower()) for key in values if key in rule):
            result.update(parent_dataset=rule["parent_dataset"],
                          parent_column=column if rule["parent_column"] == "{column}" else rule["parent_column"],
                          parent_transform=rule.get("transform"), lineage_source_url=rule.get("source_url"),
                          lineage_status="mapped")
            break
    return result
