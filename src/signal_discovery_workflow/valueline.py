"""Pinned Chapman Value Line research tables."""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq

from . import pins, r2_store

BUCKET = "r2://financial-data-wrds-private"
RELEASE_ID = "20261124T074240Z"
CATALOG = {
    "valueline__screener": "260df2ba264b572b3704f2f9af3b3e88a42d20cf100aaed948654e17fb1bbafc",
    "valueline__reports": "3e40be23a42f9b834017e6aa2bf4d3835bab2f8e06dc1706e1f16b20960c15d5",
    "valueline__history": "c779b8fde1bf19dfb1a8b251846b26aa5859f16aa2069091c78df4d67907dad0",
}


def load_table(dataset_id: str) -> pa.Table:
    """Read and verify an immutable Value Line table from private R2."""
    expected = CATALOG[dataset_id]
    prefix = f"raw/chapman/{dataset_id}/releases/{RELEASE_ID}/"
    manifest_uri = r2_store.join(BUCKET, prefix + "manifest.json")
    body = r2_store.read_bytes(manifest_uri)
    if pins.sha256_bytes(body) != expected:
        raise pins.PinMismatch(f"Value Line manifest changed: {manifest_uri}")
    manifest = json.loads(body)
    if manifest["dataset_id"] != dataset_id or manifest["run_id"] != RELEASE_ID:
        raise ValueError(f"Unexpected Value Line manifest identity: {manifest_uri}")

    parts = []
    for part in manifest["parts"]:
        key = part["key"]
        uri = r2_store.join(BUCKET, key)
        if not key.startswith(prefix):
            raise ValueError(f"Value Line part is outside the pinned release: {uri}")
        data = r2_store.read_bytes(uri)
        if len(data) != part["bytes"] or pins.sha256_bytes(data) != part["sha256"]:
            raise pins.PinMismatch(f"Value Line part changed: {uri}")
        table = pq.read_table(io.BytesIO(data))
        if table.num_rows != part["rows"]:
            raise ValueError(f"Unexpected Value Line part row count: {uri}")
        parts.append(table)

    combined = pa.concat_tables(parts)
    if combined.num_rows != manifest["row_count"]:
        raise ValueError(f"Unexpected Value Line table row count: {manifest_uri}")
    return combined
