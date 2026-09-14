"""Small standard-library HTTP client with bounded retries."""

from __future__ import annotations

import email.utils
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from typing import Any, Callable, Iterator


RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class HttpClient:
    def __init__(
        self,
        *,
        timeout: float = 120.0,
        max_attempts: int = 5,
        backoff_seconds: float = 1.0,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.sleeper = sleeper

    def open(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        method: str = "GET",
        data: bytes | None = None,
    ):
        request_headers = {
            "Accept": "application/json, text/csv;q=0.9, */*;q=0.1",
            "User-Agent": "tapestry-data/0.1",
            **(headers or {}),
        }
        request = urllib.request.Request(
            url, headers=request_headers, method=method, data=data
        )
        for attempt in range(1, self.max_attempts + 1):
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                if error.code not in RETRYABLE_STATUSES or attempt == self.max_attempts:
                    raise
                delay = self._retry_delay(error.headers.get("Retry-After"), attempt)
                self.sleeper(delay)
            except urllib.error.URLError:
                if attempt == self.max_attempts:
                    raise
                self.sleeper(self._retry_delay(None, attempt))
        raise AssertionError("retry loop exhausted unexpectedly")

    def get_json(
        self, url: str, *, headers: dict[str, str] | None = None
    ) -> Any:
        with closing(self.open(url, headers=headers)) as response:
            return json.load(response)

    def iter_bytes(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        chunk_size: int = 1 << 20,
    ) -> Iterator[bytes]:
        with closing(self.open(url, headers=headers)) as response:
            for chunk in iter(lambda: response.read(chunk_size), b""):
                yield chunk

    def _retry_delay(self, retry_after: str | None, attempt: int) -> float:
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                parsed = email.utils.parsedate_to_datetime(retry_after)
                return min(60.0, max(0.0, parsed.timestamp() - time.time()))
        jitter = random.uniform(0.8, 1.2)
        return self.backoff_seconds * (2 ** (attempt - 1)) * jitter


def with_query(url: str, params: dict[str, Any]) -> str:
    """Append non-None query parameters without changing existing ones."""

    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.extend((key, str(value)) for key, value in params.items() if value is not None)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), parts.fragment)
    )
