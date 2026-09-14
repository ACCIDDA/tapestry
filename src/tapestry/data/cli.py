"""Command-line interface for reproducible raw covariate pulls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .catalog import CATALOG, get_spec, specs_in_group
from .repository import RawDataRepository
from .sources import DelphiV5Fetcher, HubverseFetcher, SocrataFetcher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tapestry-data",
        description="Build and inspect the Tapestry raw covariate repository.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Repository root (default: data)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("catalog", help="Print the machine-readable source catalog")
    subparsers.add_parser("init", help="Initialize an empty data repository")

    pull = subparsers.add_parser("pull", help="Pull named datasets or a catalog group")
    pull.add_argument("datasets", nargs="*", help="Catalog dataset keys")
    pull.add_argument("--group", choices=("all", "core", "cdc", "delphi", "hubverse"))
    pull.add_argument("--dry-run", action="store_true", help="Print the pull plan only")
    pull.add_argument("--page-size", type=int, default=50_000, help="Socrata rows per page")
    pull.add_argument("--where", help="Optional Socrata $where expression")
    pull.add_argument("--mode", choices=("archive", "snapshot"), help="Delphi version view")
    pull.add_argument("--snapshot-date", help="Delphi snapshot date, YYYY-MM-DD")
    pull.add_argument("--report-time", help="Delphi archive report-time filter, e.g. =2026-08-28")
    pull.add_argument("--fill-method", help="Optional Delphi fill variant; default keeps all published variants")
    pull.add_argument("--signal", action="append", dest="signals", help="Limit Delphi V5 signals; repeatable")
    pull.add_argument("--geo-type", action="append", dest="geo_types", help="Limit Delphi V5 geography types; repeatable")
    pull.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent Delphi requests (default: 4)",
    )
    pull.add_argument(
        "--resume-from",
        type=Path,
        help="Reuse gzip CRC-valid Delphi V5 CSV gzip files from an interrupted staging directory",
    )
    pull.add_argument("--hub-ref", help="Hub Git branch, tag, or commit (default: catalog branch)")
    pull.add_argument(
        "--hub-as-of",
        help="Latest first-parent Hub commit at this UTC date/time (date means end of day)",
    )

    show = subparsers.add_parser("show", help="Show saved snapshots for a dataset")
    show.add_argument("dataset", choices=tuple(sorted(CATALOG)))
    verify = subparsers.add_parser(
        "verify", help="Verify checksums for one dataset's latest snapshot"
    )
    verify.add_argument("dataset", choices=tuple(sorted(CATALOG)))
    return parser


def _selected_specs(args: argparse.Namespace):
    if args.group and args.datasets:
        raise SystemExit("Choose dataset keys or --group, not both")
    if args.group:
        return specs_in_group(args.group)
    if not args.datasets:
        raise SystemExit("Provide at least one dataset key or --group")
    return tuple(get_spec(key) for key in args.datasets)


def _pull_one(repository: RawDataRepository, spec, args: argparse.Namespace):
    if spec.fetcher == "socrata":
        return SocrataFetcher().fetch(
            repository,
            spec,
            page_size=args.page_size,
            where=args.where,
        )
    if spec.fetcher == "delphi_v5":
        return DelphiV5Fetcher().fetch(
            repository,
            spec,
            mode=args.mode or "archive",
            snapshot_date=args.snapshot_date,
            report_time=args.report_time,
            fill_method=args.fill_method,
            signals=tuple(args.signals) if args.signals else None,
            geo_types=tuple(args.geo_types) if args.geo_types else None,
            workers=args.workers,
            resume_from=args.resume_from,
        )
    if spec.fetcher == "hubverse":
        return HubverseFetcher().fetch(
            repository,
            spec,
            ref=args.hub_ref,
            as_of=args.hub_as_of,
        )
    raise ValueError(f"Unsupported fetcher {spec.fetcher!r} for {spec.key}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = RawDataRepository(args.data_root)

    if args.command == "catalog":
        print(json.dumps([CATALOG[key].to_dict() for key in sorted(CATALOG)], indent=2))
        return 0
    if args.command == "init":
        repository.initialize(CATALOG)
        print(repository.root)
        return 0
    if args.command == "show":
        manifests = repository.list_snapshots(args.dataset)
        print(json.dumps([manifest.to_dict() for manifest in manifests], indent=2))
        return 0
    if args.command == "verify":
        manifest = repository.latest(args.dataset)
        repository.verify_snapshot(manifest)
        print(
            json.dumps(
                {
                    "dataset_key": manifest.dataset_key,
                    "snapshot_id": manifest.snapshot_id,
                    "files_verified": len(manifest.files),
                    "status": "ok",
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "pull":
        specs = _selected_specs(args)
        if args.dry_run:
            print(json.dumps([spec.to_dict() for spec in specs], indent=2))
            return 0
        repository.initialize(CATALOG)
        for spec in specs:
            print(f"Pulling {spec.key} ({spec.title})...", flush=True)
            manifest = _pull_one(repository, spec, args)
            print(
                json.dumps(
                    {
                        "dataset_key": manifest.dataset_key,
                        "snapshot_id": manifest.snapshot_id,
                        "files": len(manifest.files),
                        "rows": sum(item.rows or 0 for item in manifest.files),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
