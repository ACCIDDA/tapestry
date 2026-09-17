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
event-date cutoff and cannot reconstruct past revisions.

**No change does not mean no version.** Delphi is a change history: for each
observation, the value advertised at `report_time` applies until another advertised
change replaces it. No new row on a Wednesday is required. A Hub commit gives the
actual repository state, including unchanged observations; that state remains in
force between target-file changes. Skipping duplicate blobs or unchanged rows is
storage compression, not a break in coverage. Explicit nulls, removals, and dates
before the first advertised version remain distinct from unchanged valid values.
This persistence is along publication time for the same observation; it never
fills a different event week or carries information backward before publication.

Hub acquisitions now include `git-history.ndjson.gz` and `git-history.json` for
configured canonical target CSVs without native release dates. Intake walks the
pinned main branch's first-parent history and saves complete target-file states,
including empty releases after deletion. The selected stream and explorer consume
that saved history; they do not run Git on each query. In the explorer these are
**Git commit history** variants beside the native `as_of` variants of the same
measure. The date cutoff selects a complete eligible Git snapshot, preserving
omissions and retractions rather than carrying removed values forward.

**Repository ground truth:** a commit fixes the complete target-file contents.
**Timing assumption:** its main-branch committer timestamp places that state on
the availability timeline; it does not establish the provider release time. Author time, observation dates,
filename dates, and retrieval time are not substituted. Backdated target changes
cannot predate the preceding target state; equal-time changes resolve to the last
first-parent state. Only history reachable from the pinned commit is exported.
Original commit hashes, paths and blob hashes are retained. CSV `date` and
`target_end_date` are explicit event-date aliases, never release dates. Current
Hub ED CSV values are already proportions; no percentage conversion is applied.
Once a preferred filename has appeared, an obsolete alias cannot replace it
after deletion, even if that older file remains in the Git tree.

`hub-history` augments the latest acquisition at its existing pinned commit;
ordinary Hub pulls also create these histories. No user's worktree is checked out.

```bash
.venv/bin/python -m tapestry.data --data-root data hub-history \
  hub_flusight_current hub_covid_current hub_rsv_current hub_flusight_legacy
.venv/bin/python -m tapestry.explorer --data-root data index --force
.venv/bin/python -m tapestry.explorer --data-root data export
```

The path policy covers historical current-FluSight admissions and ED files,
current-COVID admissions, and legacy FluSight/COVID primary truth files. Current
RSV has native `as_of` history and no configured unversioned predecessor. Legacy
COVID LFS pointers still require their actual payloads; a pointer is not a usable
historical observation. RSV-NET catchment files are a separate source outside the
six scored NHSN/NSSP targets and are not folded into statewide Hub outcomes.

The September 17 backfill, pinned to the existing September 16 acquisitions:

| Source | Complete Git releases | Native state/DC/US rows | Commit-time range |
|---|---:|---:|---|
| Current FluSight | 132 | 1,577,922 | 2023-10-03–2026-07-09 |
| Current COVID | 93 | 241,072 | 2024-11-18–2026-09-09 |
| Current RSV | 0 | 0 | Native `as_of` history already present |
| Legacy FluSight | 527 | 3,736,470 | 2021-12-07–2023-11-22 |

Repeated event weeks appear in multiple complete releases; these counts are not
independent observations. Local acquisition IDs and exact pinned commit hashes
are in `data/processed/hub-git-history-summary.json`.

B1 uses native Hub `as_of` coverage first, Git snapshots outside that coverage,
then Delphi outside both. Holes within established native/Git coverage remain
missing in vintage resolution; the separately documented supplied-final policy
then fills missing recent inputs. Adding Git history does not revert that policy. The rebuilt B1 retains 154
calendar episodes and uses 560 recent Git input cells. Relative to the previous
native-Hub/Delphi-only build, 61 recent reports become available and 249 formerly
used Delphi cells fall inside Git Hub coverage with no current Hub value. Those
are filled by flagged finals, not passed off as Wednesday reports. All changes
are influenza ED. Git history therefore does not simply reduce missingness;
source precedence and full-snapshot omissions matter. See
`data/processed/b1-git-history-impact.json`.

## Log

- September 17, 2026: materialized Git target-file history into the canonical
  acquisition and explorer; previously only chosen-commit exports existed and
  historical no-`as_of` files were excluded by current-Hub selection. Emily's
  configs and Influpaint's cutoff checkouts motivated this integration. Preserve
  source dates and the pinned commit; expose Git publication semantics explicitly.
  The local explorer rebuild reused the 62 unchanged payloads after verifying
  the previous complete source fingerprint and matching every payload hash,
  then indexed the four new history artifacts. Unchanged series retain their
  original immutable acquisition IDs. Ordinary `index --force` remains the
  full-rebuild reproduction path.

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
