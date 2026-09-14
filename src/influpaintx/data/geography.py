"""Normalize state identifiers and select native state/national observations."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}

STATE_ALIASES = {
    **{key.lower(): key for key in STATE_NAMES},
    **{value.lower(): key for key, value in STATE_NAMES.items()},
    "district of columbia": "DC",
    "washington dc": "DC",
    "washington, d.c.": "DC",
}


def normalize_state(value: Any) -> str | None:
    """Return a postal abbreviation for a state-like publisher value."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    if not text:
        return None
    lowered = re.sub(r"\s+", " ", text).lower()
    if lowered in STATE_ALIASES:
        return STATE_ALIASES[lowered]
    compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if compact in STATE_NAMES:
        return compact
    if compact.startswith("US") and compact[2:] in STATE_FIPS:
        return STATE_FIPS[compact[2:]]
    if compact.isdigit():
        return STATE_FIPS.get(compact.zfill(2))
    return None


def observation_geography(
    row: Mapping[str, Any], resolutions: Sequence[str], path_geo_type: str | None = None,
) -> str | None:
    """Return a native state code or US; reject finer and regional observations.

    A state column on a county/site row is context, not its observation support.
    Only a national-only source may omit geographic columns altogether.
    """
    national = {"us", "usa", "united states", "united states of america", "national", "nation"}
    aggregate = {"", "all", "total", "na", "n/a", "none", "null"}
    for key in ("county", "county_fips", "county_name", "hsa", "hsa_nci_id",
                "site", "site_id", "sewershed", "sewershed_id", "wwtp_id"):
        if str(row.get(key) or "").strip().lower() not in aggregate:
            return None
    levels = [str(row.get(k) or "").strip().lower()
              for k in ("geo_type", "geography_type", "geographic_level", "level")]
    if path_geo_type:
        levels.append(path_geo_type.lower())
    for level in filter(None, levels):
        if level not in {"state", "states", "state-level", "state/territory", *national}:
            return None
    values = [row.get(k) for k in (
        "jurisdiction", "state", "state_code", "state_abbr", "state_abbreviation",
        "state_territory", "geography", "geo_value", "location", "location_code",
        "location_name", "fips", "area", "region",
    ) if row.get(k) is not None and str(row[k]).strip()]
    if any(str(v).strip().lower() in national for v in values) or any(l in national for l in levels):
        return "US"
    if not values and set(resolutions).issubset({"nation", "national"}) and resolutions:
        return "US"
    if resolutions and "state" not in resolutions:
        return None
    for value in values:
        state = normalize_state(value)
        if state:
            return state
    return None
