# Vintages and geography

## Revision semantics

`versioned` means the source can recover a historical information state; it does
not merely mean that local downloads are timestamped.

| Revision mode | Interpretation | Versioned |
|---|---|---:|
| `snapshot_only` | Current mutable publisher view; local pulls remain immutable | no |
| `initial_release` | Frozen first-publication product | yes |
| `as_of_column` | Rows carry publisher release dates | yes |
| `report_time` | Delphi archive carries retained report-time revisions | yes |
| `git_history` | Files are recoverable at an exact Git commit | yes |

The raw repository preserves all available vintage fields. The explorer supports an explicit as-of cutoff and overlays of multiple versions.
Report-time archives select the latest eligible revision per event date; full
`as_of` tables select a complete release. Unversioned sources apply only an
event-date cutoff and cannot reconstruct past revisions. The explorer reads only releases present in the selected snapshot; it does not
traverse Git history. A future training materializer must also enforce release availability.

## Geographic support

Native support is never silently relabeled:

- State observations are valid state-level inputs.
- National observations are stored once and shown with explicit national-context labels.
- HHS, HSA, and county observations are excluded at the shared post-intake filter.
- Catchment/network data are not treated as statewide values.
- Wastewater site and sewershed rows remain in raw storage but are excluded from
  the selected stream and explorer. No site-to-state average is computed.

This distinction is essential for the later mask: native state availability,
parent context, and unavailable state data need not share the same mask policy.
