"""Read-only HTTP endpoints and static files for the local explorer."""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from .index import ExplorerIndex

DEFAULT_PORT = 8765
STATIC_DIR = Path(__file__).with_name("static")


class ExplorerHandler(BaseHTTPRequestHandler):
    index: ExplorerIndex

    def do_GET(self) -> None:  # noqa: N802 - inherited HTTP method name
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._json({"ok": True, "index": str(self.index.index_path)})
            elif parsed.path == "/api/catalog":
                self._json(self.index.overview())
            elif parsed.path == "/api/series":
                query = parse_qs(parsed.query)
                self._json(self.index.list_series(
                    self._one(query, "state", required=True),
                    query=self._one(query, "q", default=""),
                    cadence=self._one(query, "cadence", default=""),
                    vintage=self._one(query, "vintage", default=""),
                    support=self._one(query, "support", default=""),
                    freshness=self._one(query, "freshness", default=""),
                    limit=int(self._one(query, "limit", default="250")),
                    offset=int(self._one(query, "offset", default="0")),
                ))
            elif parsed.path == "/api/versions":
                query = parse_qs(parsed.query)
                ids = [int(value) for value in self._one(query, "series", default="").split(",") if value]
                self._json(self.index.versions(self._one(query, "state", required=True), ids))
            elif parsed.path == "/api/data":
                query = parse_qs(parsed.query)
                raw_ids = self._one(query, "series", required=True)
                ids = [int(value) for value in raw_ids.split(",") if value]
                scale = self._one(query, "scale", default="false").lower() in {"1", "true", "yes"}
                self._json(self.index.data(self._one(query, "state", required=True), ids, scale=scale, as_of=self._one(query, "as_of", default="latest")))
            elif parsed.path in {"/", "/index.html"}:
                self._file("index.html", "text/html; charset=utf-8")
            elif parsed.path == "/app.js":
                self._file("app.js", "text/javascript; charset=utf-8")
            elif parsed.path == "/style.css":
                self._file("style.css", "text/css; charset=utf-8")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as error:
            self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as error:
            self._json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._json({"error": f"{type(error).__name__}: {error}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    @staticmethod
    def _one(query: Mapping[str, list[str]], key: str, *, default: str | None = None, required: bool = False) -> str:
        values = query.get(key)
        if values:
            return values[0]
        if required:
            raise ValueError(f"Missing query parameter: {key}")
        assert default is not None
        return default

    def _json(self, value: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name: str, content_type: str) -> None:
        body = (STATIC_DIR / name).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def make_server(index: ExplorerIndex, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundExplorerHandler", (ExplorerHandler,), {"index": index})
    return ThreadingHTTPServer((host, port), handler)


