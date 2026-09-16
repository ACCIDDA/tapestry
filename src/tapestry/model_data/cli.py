"""Build B0 windows or use build-wednesday for B1 vintage arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .finalized import CHANNELS, FinalizedDataset, build_dataset

DEFAULT_DATASET = "data/processed/build_b_finalized.npz"


def summary(dataset):
    return {
        "shape": list(dataset.panel.shape),
        "axes": ["week", "channel", "value_mask", "location"],
        "date_range": [dataset.dates[0], dataset.dates[-1]],
        "locations": dataset.locations,
        "channels": CHANNELS,
        "seasons": sorted(dataset.metadata.get("coverage_by_season_channel_location", {})),
        "observed_by_channel": {
            channel: int(dataset.panel[:, c, 1, :].sum()) for c, channel in enumerate(CHANNELS)
        },
    }


def save_episodes(path, episodes, dataset_path, *, batched):
    """NPZ contains named arrays and JSON provenance, never pickled Python objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.stack([episode[key] for episode in episodes]) if batched else episodes[0][key]
        for key in ("X", "Y", "context_dates", "target_dates", "season")
    }
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays, channels=CHANNELS,
                            locations=episodes[0]["locations"],
                            metadata=json.dumps({"dataset_path": str(Path(dataset_path).resolve()),
                                                 "fields": ["value", "mask"]}))
    return {"output": str(path), "episodes": len(episodes),
            "X_shape": list(arrays["X"].shape), "Y_shape": list(arrays["Y"].shape)}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'build-wednesday':
        from .wednesday import main as build_wednesday_main
        return build_wednesday_main(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Materialize the two local CDC snapshots")
    build.add_argument("--data-root", default="data")
    build.add_argument("--start", default="2023-09-01")
    build.add_argument("--end")
    build.add_argument("--output", default=DEFAULT_DATASET)
    inspect = commands.add_parser("inspect", help="Show dimensions and observed coverage")
    inspect.add_argument("--dataset", default=DEFAULT_DATASET)
    for name in ("query", "windows"):
        command = commands.add_parser(name, help="Export one window" if name == "query" else "Export a batch of windows")
        command.add_argument("--dataset", default=DEFAULT_DATASET)
        command.add_argument("--lookback", type=int, default=8)
        command.add_argument("--locations", nargs="+", help="Postal codes or US; preserves requested order")
        command.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4],
                             help="Positive week offsets after context end")
        command.add_argument("--target-start")
        command.add_argument("--target-end")
        command.add_argument("--output", required=True)
        if name == "query":
            command.add_argument("--context-end", required=True, help="Inclusive Saturday YYYY-MM-DD")
        else:
            command.add_argument("--season", dest="season_id", help="For example 2023-2024")
            command.add_argument("--start", help="First context-end date")
            command.add_argument("--end", help="Last context-end date")
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Keep the original builder invocation working.
    if not arguments or arguments[0].startswith("--") and arguments[0] not in {"--help", "-h"}:
        arguments.insert(0, "build")
    args = parser.parse_args(arguments)
    try:
        if args.command == "build":
            dataset = build_dataset(args.data_root, start=args.start, end=args.end)
            dataset.save(args.output)
            Path(args.output).with_suffix(".json").write_text(json.dumps(dataset.metadata, indent=2) + "\n")
            result = {"output": args.output, **summary(dataset)}
        else:
            dataset = FinalizedDataset.load(args.dataset)
            if args.command == "inspect":
                result = summary(dataset)
            else:
                options = {key: getattr(args, key) for key in
                           ("lookback", "locations", "horizons", "target_start", "target_end")}
                if args.command == "query":
                    episodes = [dataset.query(args.context_end, **options)]
                else:
                    episodes = list(dataset.windows(season_id=args.season_id, start=args.start, end=args.end, **options))
                if not episodes:
                    raise ValueError("No windows with observed targets match these filters")
                if Path(args.output).resolve() == Path(args.dataset).resolve():
                    raise ValueError("Query output must differ from the source dataset")
                result = save_episodes(args.output, episodes, args.dataset, batched=args.command == "windows")
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
