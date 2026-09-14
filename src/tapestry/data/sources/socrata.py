"""Complete, deterministically ordered downloads from CDC's Socrata API."""

from __future__ import annotations

import gzip
import json
import os
from typing import Any

from ..http import HttpClient, with_query
from ..columns import cdc_columns
from ..models import DatasetSpec
from ..repository import RawDataRepository


class SocrataFetcher:
    def __init__(self, client: HttpClient | None = None):
        self.client = client or HttpClient()

    def fetch(
        self,
        repository: RawDataRepository,
        spec: DatasetSpec,
        *,
        page_size: int = 50_000,
        where: str | None = None,
        order_by: str | None = None,
    ):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        dataset_id = str(spec.config["dataset_id"])
        resource_url = str(
            spec.config.get(
                "resource_url", f"https://data.cdc.gov/resource/{dataset_id}.json"
            )
        )
        metadata_url = str(
            spec.config.get(
                "metadata_url", f"https://data.cdc.gov/api/views/{dataset_id}.json"
            )
        )
        stable_order = order_by or str(spec.config.get("order_by", ":id"))
        order_columns = {item.strip().split()[0] for item in stable_order.split(",")}
        if ":id" not in order_columns:
            stable_order = f"{stable_order},:id"
        token = os.environ.get("CDC_APP_TOKEN") or os.environ.get("SOCRATA_APP_TOKEN")
        headers = {"X-App-Token": token} if token else {}

        metadata = self.client.get_json(metadata_url, headers=headers)
        count_url = with_query(
            resource_url,
            {"$select": "count(*)", "$where": where},
        )
        count_result = self.client.get_json(count_url, headers=headers)
        if not isinstance(count_result, list) or len(count_result) != 1 or "count" not in count_result[0]:
            raise TypeError(f"Socrata {dataset_id} returned an invalid count response")
        expected_rows = int(count_result[0]["count"])
        page_count = 0
        row_count = 0

        with repository.begin_snapshot(spec) as snapshot:
            snapshot.write_json("metadata.json", metadata)
            snapshot.write_json("columns.json", {
                "source_url": metadata_url, "columns": cdc_columns(metadata),
            })
            output = snapshot.path(
                "data.ndjson.gz", media_type="application/x-ndjson+gzip"
            )
            with output.open("wb") as raw_stream, gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_stream, mtime=0
            ) as compressed:
                offset = 0
                while True:
                    url = with_query(
                        resource_url,
                        {
                            "$limit": page_size,
                            "$offset": offset,
                            "$order": stable_order,
                            "$where": where,
                        },
                    )
                    page = self.client.get_json(url, headers=headers)
                    if not isinstance(page, list):
                        raise TypeError(
                            f"Socrata {dataset_id} returned {type(page).__name__}, expected list"
                        )
                    page_count += 1
                    for row in page:
                        compressed.write(
                            json.dumps(row, separators=(",", ":"), sort_keys=True).encode(
                                "utf-8"
                            )
                        )
                        compressed.write(b"\n")
                    row_count += len(page)
                    if len(page) < page_size:
                        break
                    offset += len(page)

            if row_count == 0:
                raise RuntimeError(f"Socrata dataset {dataset_id} returned no rows")
            if row_count != expected_rows:
                raise RuntimeError(
                    f"Socrata dataset {dataset_id} changed or paginated incompletely: "
                    f"expected {expected_rows} rows, fetched {row_count}"
                )
            metadata_after = self.client.get_json(metadata_url, headers=headers)
            if metadata_after.get("rowsUpdatedAt") != metadata.get("rowsUpdatedAt"):
                raise RuntimeError(
                    f"Socrata dataset {dataset_id} changed during pagination; retry the pull"
                )
            snapshot.set_file_details(
                "data.ndjson.gz",
                rows=row_count,
                media_type="application/x-ndjson+gzip",
            )
            return snapshot.commit(
                selector={
                    "where": where,
                    "order_by": stable_order,
                    "page_size": page_size,
                },
                source_state={
                    "dataset_id": dataset_id,
                    "metadata_name": metadata.get("name"),
                    "metadata_rows_updated_at": metadata.get("rowsUpdatedAt"),
                    "metadata_rows_updated_by": metadata.get("rowsUpdatedBy"),
                    "pages": page_count,
                    "rows": row_count,
                    "expected_rows": expected_rows,
                },
            )
