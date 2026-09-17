"""Six native weekly channels; no imputation, scaling, or provider fallback."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from tapestry.data.geography import STATE_NAMES, observation_geography
from tapestry.data.selection import SelectedData
from tapestry.models.provenance import CALENDAR_START

CHANNELS = (
    "nhsn_flu_admissions", "nhsn_covid_admissions", "nhsn_rsv_admissions",
    "nssp_flu_proportion", "nssp_covid_proportion", "nssp_rsv_proportion",
)
SOURCES = {
    "cdc_nhsn_final": (0, ("totalconfflunewadm", "totalconfc19newadm", "totalconfrsvnewadm")),
    "cdc_nssp_trajectories": (3, ("percent_visits_influenza", "percent_visits_covid", "percent_visits_rsv")),
}


def saturday(value):
    day = date.fromisoformat(str(value)[:10])
    if day.weekday() != 5:
        raise ValueError(f"Expected a Saturday week end: {value}")
    return day


def season(day):
    """CDC epiweek 31–30, as proposed in the B design, not challenge dates."""
    # Week 1 contains January 4; CDC weeks start Sunday.
    def boundary(year):
        jan4 = date(year, 1, 4)
        return jan4 - timedelta(days=(jan4.weekday() + 1) % 7) + timedelta(weeks=30)
    year = day.year if day >= boundary(day.year) else day.year - 1
    return f"{year}-{year + 1}"


@dataclass
class FinalizedDataset:
    panel: np.ndarray  # [week, channel, (value, mask), location]
    dates: tuple[str, ...]
    locations: tuple[str, ...]
    metadata: dict

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, panel=self.panel, dates=self.dates,
                                locations=self.locations, metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["panel"], tuple(data["dates"].tolist()),
                       tuple(data["locations"].tolist()), json.loads(str(data["metadata"])))

    def _slice(self, dates, locations):
        result = np.zeros((len(dates), 6, 2, len(locations)), dtype=self.panel.dtype)
        indices = [self.locations.index(loc) for loc in locations]
        lookup = {d: i for i, d in enumerate(self.dates)}
        for i, day in enumerate(dates):
            if day.isoformat() in lookup:
                result[i] = np.take(self.panel[lookup[day.isoformat()]], indices, axis=-1)
        return result

    def query(self, context_end, *, lookback=8, locations=None, horizons=(1, 2, 3, 4),
              target_start=None, target_end=None):
        """History ends inclusively at context_end; horizons are weeks AFTER it.

        Default targets correspond to FluSight h=0..3 when reference Saturday
        is context_end + 7 days. No finalized revision-nowcast training here.
        Target bounds mask labels, independently of the context mask.
        """
        end = saturday(context_end)
        if not isinstance(lookback, int) or lookback < 1:
            raise ValueError("lookback must be a positive integer")
        horizons = tuple(horizons)
        if not horizons or len(set(horizons)) != len(horizons):
            raise ValueError("Provide nonempty, unique horizons")
        if target_start and target_end and date.fromisoformat(target_start) > date.fromisoformat(target_end):
            raise ValueError("target_start must not be after target_end")
        if any(not isinstance(h, int) or h < 1 for h in horizons):
            raise ValueError("Finalized pilot requires positive integer future horizons")
        locations = tuple(self.locations if locations is None else locations)
        if not locations or len(set(locations)) != len(locations):
            raise ValueError("Provide nonempty, unique locations")
        days = [end - timedelta(weeks=i) for i in reversed(range(lookback))]
        targets = [end + timedelta(weeks=h) for h in horizons]
        y = self._slice(targets, locations)
        for i, day in enumerate(targets):
            if ((target_start and day < date.fromisoformat(target_start)) or
                    (target_end and day > date.fromisoformat(target_end))):
                y[i] = 0
        return {"X": self._slice(days, locations), "Y": y,
                "context_dates": tuple(d.isoformat() for d in days),
                "target_dates": tuple(d.isoformat() for d in targets),
                "channels": CHANNELS, "locations": locations, "season": season(end)}

    def windows(self, *, season_id=None, start=None, end=None, **kwargs):
        """Yield weekly episodes; season filters context end, label bounds are explicit.

        Incomplete leading histories stay padded/masked; skip episodes with no labels.
        Pass target_start/target_end to enforce a training/validation partition.
        """
        for day in self.dates:
            if (start and day < start) or (end and day > end):
                continue
            if season_id and season(date.fromisoformat(day)) != season_id:
                continue
            episode = self.query(day, **kwargs)
            if episode["Y"][:, :, 1, :].any():
                yield episode


def build_dataset(data_root="data", *, start=CALENDAR_START, end=None):
    """Read only the pinned latest direct CDC products via the shared selector."""
    start_day = date.fromisoformat(start)
    end_day = date.fromisoformat(end) if end else None
    selected = SelectedData(data_root)
    locations = tuple(sorted(STATE_NAMES)) + ("US",)
    observations, provenance = {}, []
    reasons = {"missing": 0, "invalid": 0}
    for key, (offset, columns) in SOURCES.items():
        found = False
        for table in selected.selected_tables(dataset_key=key):
            found = True
            snapshot = Path(data_root) / "raw" / key / "snapshots" / table.snapshot_id
            manifest = snapshot / "manifest.json"
            provenance.append({"dataset": key, "snapshot_id": table.snapshot_id,
                               "source_path": table.source_path,
                               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                               "manifest": json.loads(manifest.read_text())})
            for row in table.iter_rows():
                day = saturday(row[table.event_date_column])
                if day < start_day or (end_day and day > end_day):
                    continue
                loc = observation_geography(row, table.geographic_resolutions)
                if loc not in locations:
                    continue
                for j, column in enumerate(columns):
                    raw = row.get(column)
                    reason = None
                    if raw is None or str(raw).strip().lower() in {"", "*", "--", ".", "na", "n/a", "nan", "null", "none"}:
                        value, reason = None, "missing"
                    else:
                        try:
                            value = float(raw)
                            if not np.isfinite(value) or value < 0 or (offset == 3 and value > 100):
                                value, reason = None, "invalid"
                        except (TypeError, ValueError):
                            value, reason = None, "invalid"
                    if reason:
                        reasons[reason] += 1
                    if value is not None and offset == 3:
                        value /= 100
                    identity = (day, offset + j, loc)
                    if identity in observations and observations[identity] != value:
                        raise ValueError(f"Conflicting native observations: {identity}")
                    observations[identity] = value
        if not found:
            raise ValueError(f"Required source is unavailable: {key}")
    if not observations:
        raise ValueError("No observations in requested date range")
    first = start_day + timedelta(days=(5 - start_day.weekday()) % 7)
    last = end_day or max(key[0] for key in observations)
    days = tuple(first + timedelta(weeks=i) for i in range((last - first).days // 7 + 1))
    panel = np.zeros((len(days), 6, 2, len(locations)), dtype=np.float32)
    dates_index = {day: i for i, day in enumerate(days)}
    for (day, channel, loc), value in observations.items():
        if value is not None:
            panel[dates_index[day], channel, :, locations.index(loc)] = (value, 1)
    coverage = {}
    for label in sorted({season(day) for day in days}):
        subset = panel[[season(day) == label for day in days], :, 1, :]
        coverage[label] = {ch: {loc: int(subset[:, c, l].sum()) for l, loc in enumerate(locations)}
                           for c, ch in enumerate(CHANNELS)}
    metadata = {"version": 1, "mode": "finalized_nhsn_frozen_latest_nssp",
                "channels": CHANNELS, "units": ["admissions"] * 3 + ["proportion"] * 3,
                "axes": ["week", "channel", "value_mask", "location"],
                "start": start, "season_policy": "CDC epiweek 31 through 30",
                "finality_note": "NSSP has no finality flag; latest reported unsmoothed snapshot is assumed evaluation truth. Not an operational as-of backtest.",
                "geography_note": "50 states + DC + native US; no broadcasting or aggregation. Territories excluded by existing selection policy.",
                "provenance": provenance, "raw_missing_or_invalid": reasons,
                "coverage_by_season_channel_location": coverage}
    return FinalizedDataset(panel, tuple(d.isoformat() for d in days), locations, metadata)
