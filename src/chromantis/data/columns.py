"""CDC API field identifiers and publisher-authored labels, without renaming raw rows."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


def cdc_columns(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        col["fieldName"]: {
            "field_name": col["fieldName"],
            "name": col.get("name") or col["fieldName"],
            "description": col.get("description", ""),
            "data_type": col.get("dataTypeName"),
        }
        for col in metadata.get("columns", [])
        if col.get("fieldName") and not col["fieldName"].startswith(":")
    }


@lru_cache(maxsize=64)
def saved_columns(path: Path, modified_ns: int) -> dict[str, dict[str, Any]]:
    # Timestamp participates in the cache key so replacing a local file is visible.
    payload = json.loads(path.read_text())
    return payload["columns"] if path.name == "columns.json" else cdc_columns(payload)
