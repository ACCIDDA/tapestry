"""Read-only Hubverse Git mirrors and immutable target-data exports."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import BinaryIO

from ..models import DatasetSpec
from ..repository import RawDataRepository


class _BoundedReader:
    """Expose exactly one object from a persistent ``git cat-file`` stream."""

    def __init__(self, stream: BinaryIO, size: int):
        self.stream = stream
        self.remaining = size

    def read(self, size: int = -1) -> bytes:
        if self.remaining == 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        chunks = []
        received = 0
        while received < size:
            chunk = self.stream.read(size - received)
            if not chunk:
                raise EOFError("git cat-file ended before the object was complete")
            chunks.append(chunk)
            received += len(chunk)
        self.remaining -= received
        return b"".join(chunks)


def _git_error(error: subprocess.CalledProcessError) -> RuntimeError:
    detail = error.stderr or error.stdout or ""
    if isinstance(detail, bytes):
        detail = detail.decode("utf-8", errors="replace")
    detail = detail.strip()
    suffix = f": {detail}" if detail else ""
    return RuntimeError(f"Git command failed with exit status {error.returncode}{suffix}")


class HubMirror:
    """A bare Git mirror that never checks out or mutates a user's worktree."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()

    def sync(self, remote_url: str) -> None:
        if self.path.exists():
            actual = self.run("remote", "get-url", "origin").strip()
            if actual != remote_url:
                raise RuntimeError(
                    f"Mirror {self.path} points to {actual!r}, expected {remote_url!r}"
                )
            self.run("remote", "update", "--prune")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.path.name}.clone-", dir=self.path.parent)
        )
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--mirror",
                    "--filter=blob:none",
                    remote_url,
                    str(temporary),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            temporary.replace(self.path)
        except subprocess.CalledProcessError as error:
            raise _git_error(error) from error
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def run(self, *arguments: str, text: bool = True):
        try:
            result = subprocess.run(
                ["git", "--git-dir", str(self.path), *arguments],
                check=True,
                text=text,
                capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            raise _git_error(error) from error
        return result.stdout

    def resolve(self, ref: str) -> str:
        return self.run("rev-parse", "--verify", f"{ref}^{{commit}}").strip()

    def commit_at(self, ref: str, cutoff: str | date | datetime) -> str:
        if isinstance(cutoff, str):
            if len(cutoff) == 10:
                parsed = datetime.combine(
                    date.fromisoformat(cutoff), time(23, 59, 59), tzinfo=UTC
                )
            else:
                parsed = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
        elif isinstance(cutoff, date) and not isinstance(cutoff, datetime):
            parsed = datetime.combine(cutoff, time(23, 59, 59), tzinfo=UTC)
        else:
            parsed = cutoff
        if parsed.tzinfo is None:
            raise ValueError("A datetime cutoff must include an explicit timezone")
        before = parsed.astimezone(UTC).isoformat()
        commit = self.run(
            "rev-list", "-1", "--first-parent", f"--before={before}", ref
        ).strip()
        if not commit:
            raise ValueError(f"No commit in {ref!r} at or before {before}")
        return commit

    def commit_time(self, commit: str) -> str:
        return self.run("show", "-s", "--format=%cI", commit).strip()

    def list_files(self, commit: str, prefix: str | None = None) -> tuple[str, ...]:
        arguments = ["ls-tree", "-r", "--name-only", commit]
        if prefix:
            arguments.extend(["--", prefix])
        output = self.run(*arguments)
        return tuple(line for line in output.splitlines() if line)

    def read_file(self, commit: str, path: str) -> bytes:
        return self.run("show", f"{commit}:{path}", text=False)

    def _is_partial(self) -> bool:
        result = subprocess.run(
            [
                "git",
                "--git-dir",
                str(self.path),
                "config",
                "--bool",
                "--get",
                "remote.origin.promisor",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _tree_entries(
        self, commit: str, prefixes: tuple[str, ...]
    ) -> tuple[tuple[str, str, str, str], ...]:
        raw = self.run(
            "ls-tree", "-r", "-t", "-z", commit, "--", *prefixes, text=False
        )
        entries = []
        for item in raw.split(b"\0"):
            if not item:
                continue
            metadata, raw_path = item.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            entries.append(
                (mode, object_type, object_id, raw_path.decode("utf-8"))
            )
        return tuple(entries)

    def _prefetch_blobs(self, object_ids: tuple[str, ...]) -> None:
        if not object_ids or not self._is_partial():
            return
        print(
            f"Fetching {len(object_ids):,} selected Hub data objects into {self.path}...",
            flush=True,
        )
        try:
            subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(self.path),
                    "-c",
                    "fetch.negotiationAlgorithm=noop",
                    "fetch",
                    "origin",
                    "--no-tags",
                    "--no-write-fetch-head",
                    "--recurse-submodules=no",
                    "--stdin",
                ],
                input="\n".join(object_ids) + "\n",
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            raise _git_error(error) from error

    def _write_archive(
        self,
        commit: str,
        entries: tuple[tuple[str, str, str, str], ...],
        destination: Path,
    ) -> None:
        commit_time = int(self.run("show", "-s", "--format=%ct", commit).strip())
        environment = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
        process = subprocess.Popen(
            ["git", "--git-dir", str(self.path), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            with destination.open("wb") as raw_stream, gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_stream, mtime=0
            ) as compressed, tarfile.open(
                fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
            ) as archive:
                for mode, object_type, object_id, path in entries:
                    info = tarfile.TarInfo(path + ("/" if object_type == "tree" else ""))
                    info.mtime = commit_time
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    if object_type == "tree":
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        archive.addfile(info)
                        continue
                    if object_type != "blob":
                        raise RuntimeError(
                            f"Cannot archive Git object type {object_type!r} at {path!r}"
                        )

                    process.stdin.write(f"{object_id}\n".encode("ascii"))
                    process.stdin.flush()
                    header = process.stdout.readline().decode("ascii").strip().split()
                    if len(header) != 3 or header[1] != "blob":
                        raise RuntimeError(
                            f"Could not read Git blob {object_id} for {path!r}: {' '.join(header)}"
                        )
                    size = int(header[2])
                    reader = _BoundedReader(process.stdout, size)
                    info.mode = int(mode, 8) & 0o7777
                    if mode == "120000":
                        info.type = tarfile.SYMTYPE
                        info.linkname = reader.read().decode("utf-8")
                        archive.addfile(info)
                    else:
                        info.size = size
                        archive.addfile(info, reader)
                    if reader.remaining or process.stdout.read(1) != b"\n":
                        raise RuntimeError(f"Malformed git cat-file output for {path!r}")
        except BaseException:
            process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            raise
        else:
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            if return_code:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"git cat-file failed with exit status {return_code}: {detail}"
                )

    def archive(self, commit: str, prefixes: tuple[str, ...], destination: Path) -> tuple[str, ...]:
        selected = tuple(
            prefix for prefix in prefixes if self.list_files(commit, prefix=prefix)
        )
        if not selected:
            raise FileNotFoundError(
                f"None of {prefixes!r} exist at commit {commit} in {self.path}"
            )
        entries = self._tree_entries(commit, selected)
        blobs = tuple(
            dict.fromkeys(
                object_id
                for _, object_type, object_id, _ in entries
                if object_type == "blob"
            )
        )
        self._prefetch_blobs(blobs)
        self._write_archive(commit, entries, destination)
        return selected


class HubverseFetcher:
    def fetch(
        self,
        repository: RawDataRepository,
        spec: DatasetSpec,
        *,
        ref: str | None = None,
        as_of: str | None = None,
    ):
        remote_url = str(spec.config["remote_url"])
        default_branch = str(spec.config["default_branch"])
        selected_ref = ref or default_branch
        archive_paths = tuple(str(item) for item in spec.config["archive_paths"])
        mirror = HubMirror(repository.mirrors_dir / f"{spec.key}.git")
        mirror.sync(remote_url)
        commit = (
            mirror.commit_at(selected_ref, as_of)
            if as_of is not None
            else mirror.resolve(selected_ref)
        )
        commit_time = mirror.commit_time(commit)
        files = mirror.list_files(commit)

        with repository.begin_snapshot(spec) as snapshot:
            selected_paths = mirror.archive(
                commit,
                archive_paths,
                snapshot.path("hub-data.tar.gz", media_type="application/gzip"),
            )
            snapshot.write_json(
                "mirror.json",
                {
                    "remote_url": remote_url,
                    "mirror_path": str(mirror.path),
                    "default_branch": default_branch,
                    "selected_ref": selected_ref,
                    "as_of": as_of,
                    "commit": commit,
                    "commit_time": commit_time,
                    "archived_prefixes": selected_paths,
                },
            )
            file_list = snapshot.path("repository-files.txt", media_type="text/plain")
            file_list.write_text("\n".join(files) + "\n", encoding="utf-8")
            return snapshot.commit(
                selector={"ref": selected_ref, "as_of": as_of, "commit": commit},
                source_state={
                    "remote_url": remote_url,
                    "commit": commit,
                    "commit_time": commit_time,
                    "file_count": len(files),
                    "mirror_path": str(mirror.path),
                },
            )
