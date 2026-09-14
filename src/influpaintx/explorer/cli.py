"""Build, validate, and serve the disposable explorer index."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from typing import Sequence

from .index import ExplorerIndex
from .server import DEFAULT_PORT, make_server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index and explore downloaded InfluPaintX data")
    parser.add_argument("--data-root", default="data", help="Raw data repository root (default: data)")
    parser.add_argument("--index-path", help="Override the disposable SQLite index location")
    parser.add_argument("--ledger-path", help="Override the Parquet ledger location")
    subparsers = parser.add_subparsers(dest="command")
    index_parser = subparsers.add_parser("index", help="Build or refresh the explorer index")
    index_parser.add_argument("--force", action="store_true", help="Rebuild even when the index is current")
    for name, help_text in (("status", "Inspect saved build metadata and source errors"),
                            ("validate", "Check SQLite integrity and the Parquet footer; no raw scan")):
        subparsers.add_parser(name, help=help_text)
    serve_parser = subparsers.add_parser("serve", help="Build if needed and start the local explorer")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    serve_parser.add_argument("--no-index", action="store_true", help="Serve the existing index without checking raw data")
    serve_parser.add_argument("--rebuild", action="store_true", help="Force a full index rebuild before serving")
    for build_parser in (index_parser, serve_parser):
        build_parser.add_argument("--batch-rows", type=int, default=100_000,
                                  help="Maximum buffered revision keys per flush (default: 100000)")
        build_parser.add_argument("--cache-mb", type=int, default=256,
                                  help="SQLite page-cache budget in MiB (default: 256)")
        build_parser.add_argument("--strict", action="store_true",
                                  help="Refuse builds or reuse with source errors")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    # Preserve the documented default command while applying serve defaults.
    if arguments.command is None:
        arguments = parser.parse_args([*(argv if argv is not None else sys.argv[1:]), "serve"])
    command = arguments.command or "serve"
    if getattr(arguments, "batch_rows", 1) < 1 or getattr(arguments, "cache_mb", 1) < 1:
        parser.error("--batch-rows and --cache-mb must be positive")
    index = ExplorerIndex(
        arguments.data_root, arguments.index_path, arguments.ledger_path,
        batch_rows=getattr(arguments, "batch_rows", 100_000),
        cache_mb=getattr(arguments, "cache_mb", 256),
        strict=getattr(arguments, "strict", False),
    )
    if command in {"status", "validate"}:
        result = index.status(validate=command == "validate")
        print(json.dumps(result, indent=2))
        return int(bool(result["errors"]))
    if command == "index":
        index.ensure(rebuild=arguments.force)
        return 0
    if not arguments.no_index:
        index.ensure(rebuild=arguments.rebuild)
    elif not index.index_path.is_file():
        raise FileNotFoundError(f"No explorer index exists at {index.index_path}")
    server = make_server(index, arguments.host, arguments.port)
    url = f"http://{arguments.host}:{server.server_address[1]}/"
    print(f"InfluPaintX explorer: {url}")
    if not arguments.no_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping explorer")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
