"""PopHIVE Ingest standard tables with their Git commit history as report times.

PopHIVE republishes each source as ``data/<source>/standard/data.csv.gz`` and
commits it on every update. A row's availability is therefore the time of the
first commit that contains it, and a later commit that changes or removes it is
a revision. The full Ingest repository is over 10 GB, so this fetcher lists the
commits that touched one file through the GitHub API and downloads that file at
each commit from raw.githubusercontent.com instead of mirroring the repository.

Commits made before a source's ingest was scheduled can be backfills: a
report time is the moment the value entered PopHIVE, not when its publisher
first released it.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import re
from contextlib import closing
from datetime import UTC, datetime
from urllib.parse import quote

from ..geography import STATE_FIPS
from ..http import HttpClient, with_query
from ..models import DatasetSpec
from ..repository import RawDataRepository

PAYLOAD = "archive.csv.gz"
COMMITS = "commits.json"
ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _location(geography: str) -> tuple[str, str]:
    """PopHIVE geography codes are FIPS: 00 is national, two digits a state."""
    if geography == "00":
        return "nation", "US"
    if geography in STATE_FIPS:
        return "state", STATE_FIPS[geography]
    raise ValueError(f"Unsupported PopHIVE geography {geography!r}; only national and state FIPS are indexed")


def _parse_version(payload: bytes, measures: tuple[str, ...] | None):
    """Rows of one committed file as {(geography, day): measure tuple}, plus skipped rows."""
    reader = csv.reader(io.StringIO(gzip.decompress(payload).decode("utf-8-sig")))
    header = next(reader)
    if header[:2] != ["geography", "time"] or len(header) < 3:
        raise ValueError(f"Unexpected PopHIVE standard header: {header}")
    if measures is not None and tuple(header[2:]) != measures:
        raise ValueError(f"PopHIVE measure columns changed: {tuple(header[2:])} != {measures}")
    rows, skipped = {}, 0
    for row in reader:
        # The first Kinsa commit contains one row whose time is the string NA.
        if len(row) != len(header) or not ISO_DAY.fullmatch(row[1]):
            skipped += 1
            continue
        rows[row[0], row[1]] = tuple(value if value not in ("NA", "") else "" for value in row[2:])
    return tuple(header[2:]), rows, skipped


class PopHiveGitFetcher:
    """Write every value change between commits as one report-time archive."""

    def __init__(self, client: HttpClient | None = None):
        self.client = client or HttpClient(timeout=120.0)

    def _commits(self, repository: str, path: str, branch: str) -> list[dict]:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        headers = {"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {token}"} if token else {})}
        commits, page = [], 1
        while True:
            url = with_query(f"https://api.github.com/repos/{repository}/commits",
                             {"path": path, "sha": branch, "per_page": 100, "page": page})
            batch = self.client.get_json(url, headers=headers)
            commits.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        if not commits:
            raise RuntimeError(f"No commits touch {path} on {repository}@{branch}")
        return [dict(sha=c["sha"], committed_at=c["commit"]["committer"]["date"],
                     message=c["commit"]["message"].splitlines()[0]) for c in reversed(commits)]

    def fetch(self, repository: RawDataRepository, spec: DatasetSpec):
        remote, path, branch = (str(spec.config[key]) for key in ("repository", "path", "branch"))
        commits = self._commits(remote, path, branch)
        measures, state, events, skipped_rows = None, {}, [], 0
        for commit in commits:
            url = f"https://raw.githubusercontent.com/{remote}/{commit['sha']}/{quote(path)}"
            with closing(self.client.open(url)) as response:
                payload = response.read()
            measures, rows, skipped = _parse_version(payload, measures)
            report_time = datetime.fromisoformat(commit["committed_at"].replace("Z", "+00:00")).astimezone(UTC)
            commit.update(sha256=hashlib.sha256(payload).hexdigest(), rows=len(rows), skipped_rows=skipped)
            skipped_rows += skipped
            stamp = report_time.strftime("%Y-%m-%dT%H:%M:%S")
            for key in sorted(rows, key=lambda k: (k[1], k[0])):
                if state.get(key) != rows[key]:
                    events.append((stamp, key, rows[key]))
            # A row that vanishes is a retraction: an explicit empty value, not silence.
            events.extend((stamp, key, ("",) * len(measures)) for key in sorted(state) if key not in rows)
            state = rows
        if not events:
            raise RuntimeError(f"PopHIVE {path} produced no rows")

        with repository.begin_snapshot(spec) as snapshot:
            destination = snapshot.path(PAYLOAD, rows=len(events), media_type="text/csv+gzip")
            with (destination.open("wb") as raw,
                  gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
                  io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text):
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(["report_time", "geo_type", "geo_value", "reference_time", *measures])
                for stamp, (geography, day), values in events:
                    geo_type, geo_value = _location(geography)
                    writer.writerow([stamp, geo_type, geo_value, day, *values])
            snapshot.write_json(COMMITS, dict(repository=remote, path=path, branch=branch, commits=commits))
            return snapshot.commit(
                selector=dict(repository=remote, path=path, branch=branch),
                source_state=dict(head_commit=commits[-1]["sha"], head_committed_at=commits[-1]["committed_at"],
                                  first_commit=commits[0]["sha"], first_committed_at=commits[0]["committed_at"],
                                  commits=len(commits), revision_rows=len(events), skipped_rows=skipped_rows,
                                  final_rows=len(state)),
            )
