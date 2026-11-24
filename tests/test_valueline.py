"""Verification checks for the Value Line catalog."""

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from signal_discovery_workflow import pins, valueline


def test_load_table_checks_manifest_and_parquet(monkeypatch):
    dataset_id = "valueline__reports"
    prefix = f"raw/chapman/{dataset_id}/releases/{valueline.RELEASE_ID}"
    table = pa.table({"ticker": ["AAPL"]})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    data = buffer.getvalue()
    part_key = f"{prefix}/part-000000.parquet"
    manifest = json.dumps({
        "dataset_id": dataset_id,
        "run_id": valueline.RELEASE_ID,
        "row_count": 1,
        "parts": [{"key": part_key, "sha256": pins.sha256_bytes(data), "bytes": len(data), "rows": 1}],
    }).encode()
    manifest_uri = f"{valueline.BUCKET}/{prefix}/manifest.json"
    part_uri = f"{valueline.BUCKET}/{part_key}"
    objects = {manifest_uri: manifest, part_uri: data}
    monkeypatch.setitem(valueline.CATALOG, dataset_id, pins.sha256_bytes(manifest))
    monkeypatch.setattr(valueline.r2_store, "read_bytes", objects.__getitem__)

    assert valueline.load_table(dataset_id).equals(table)
    objects[part_uri] = bytes(len(data))
    with pytest.raises(pins.PinMismatch, match="part changed"):
        valueline.load_table(dataset_id)
    objects[manifest_uri] = b"{}"
    with pytest.raises(pins.PinMismatch, match="manifest changed"):
        valueline.load_table(dataset_id)
