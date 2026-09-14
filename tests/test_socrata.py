from __future__ import annotations

import gzip
import json
import tempfile
import unittest
import urllib.parse

from tapestry.data.catalog import CATALOG
from tapestry.data.repository import RawDataRepository
from tapestry.data.sources.socrata import SocrataFetcher


class FakeSocrataClient:
    def __init__(self):
        self.rows = [{"row": str(index)} for index in range(5)]
        self.page_queries = []
        self.metadata_calls = 0

    def get_json(self, url, *, headers=None):
        parsed = urllib.parse.urlsplit(url)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        if "/api/views/" in parsed.path:
            self.metadata_calls += 1
            return {"name": "test", "rowsUpdatedAt": 123, "id": "ua7e-t2fy", "columns": [
                {"fieldName": "row", "name": "Publisher measure name", "description": "CDC definition", "dataTypeName": "number"},
                {"fieldName": ":id", "name": "System identifier"},
            ]}
        if query.get("$select") == "count(*)":
            return [{"count": str(len(self.rows))}]
        self.page_queries.append(query)
        offset = int(query["$offset"])
        limit = int(query["$limit"])
        return self.rows[offset : offset + limit]


class SocrataTests(unittest.TestCase):
    def test_complete_deterministic_pagination(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = RawDataRepository(temporary)
            repository.initialize(CATALOG)
            client = FakeSocrataClient()
            manifest = SocrataFetcher(client).fetch(
                repository, CATALOG["cdc_nhsn_final"], page_size=2
            )

            self.assertEqual(client.metadata_calls, 2)
            self.assertEqual([item["$offset"] for item in client.page_queries], ["0", "2", "4"])
            self.assertTrue(all(item["$order"] == "weekendingdate,jurisdiction,:id" for item in client.page_queries))
            data_file = repository.snapshot_path(manifest) / "data.ndjson.gz"
            with gzip.open(data_file, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(rows, client.rows)
            columns = json.loads((data_file.parent / "columns.json").read_text())["columns"]
            self.assertEqual(set(columns), {"row"})
            self.assertEqual(columns["row"]["name"], "Publisher measure name")
            self.assertEqual(columns["row"]["description"], "CDC definition")
            self.assertTrue(any(item.path == "columns.json" for item in manifest.files))
            file_record = next(item for item in manifest.files if item.path == "data.ndjson.gz")
            self.assertEqual(file_record.rows, 5)


if __name__ == "__main__":
    unittest.main()
