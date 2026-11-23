# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Precomputed R2 Feature-Store Snapshot (Speed Upgrade).

Rebuilding 19 Lakehouse v2 panels (104 Parquet parts), 4 SEC quarterly
acceleration panels (678 Parquet parts), FRED macro regimes, and 755+
cross-sectional Barra-style OLS regressions on every Modal invocation costs
~20-40 minutes of startup overhead before the first LLM call.

Format ``numpy_zlib_v1`` (default on write):
* Stores shared ``index.json`` (dates + column labels) once.
* Stores each aligned wide panel as a C-contiguous ``float32`` (or ``bool``)
  buffer compressed with ``zlib`` (level 1) under ``<uri>/panels/<name>.bin.zlib``.
* Eliminates PyArrow's ~5s-per-panel overhead of converting 25,822 wide Parquet
  columns into pandas blocks, cutting 38-42 panel startup from ~3.5 minutes to
  ~3-5 seconds while preserving exact SHA-256 manifest pins.
* ``load_feature_store`` stays fully backward-compatible with legacy
  ``.parquet`` feature-store manifests.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import r2_store

logger = logging.getLogger(__name__)

DEFAULT_MONTHLY_FEATURE_STORE_URI = (
    "r2://financial-data-wrds-lakehouse/research/feature_store/monthly_v1"
)
DEFAULT_WEEKLY_FEATURE_STORE_URI = (
    "r2://financial-data-wrds-lakehouse/research/feature_store/weekly_v1"
)

_NUMPY_ZLIB_FORMAT = "numpy_zlib_v1"


def _df_to_parquet_bytes(frame: pd.DataFrame, *, as_bool: bool = False) -> bytes:
    """Serialize a Date x Security wide DataFrame to zstd-compressed Parquet bytes."""
    df = frame.copy()
    if not as_bool:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(np.float32)
    else:
        df = df.fillna(False).astype(bool)
    df.index = pd.DatetimeIndex(df.index, name="Date")
    table = pa.Table.from_pandas(df, preserve_index=True)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def _parquet_bytes_to_df(raw: bytes) -> pd.DataFrame:
    df = pq.read_table(io.BytesIO(raw)).to_pandas()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="Date")
    return df


def _df_to_zlib_bytes(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    columns: pd.Index,
    *,
    as_bool: bool = False,
) -> bytes:
    """Serialize an aligned wide DataFrame as zlib-compressed contiguous float32/bool bytes."""
    aligned = frame.reindex(index=dates, columns=columns)
    if as_bool:
        arr = np.ascontiguousarray(aligned.fillna(False).to_numpy(dtype=np.bool_))
    else:
        arr = np.ascontiguousarray(aligned.to_numpy(dtype=np.float32, na_value=np.nan))
    return zlib.compress(arr.tobytes(), level=1)


def _zlib_bytes_to_df(
    raw: bytes,
    dates: pd.DatetimeIndex,
    columns: pd.Index,
    *,
    as_bool: bool = False,
) -> pd.DataFrame:
    dtype = np.bool_ if as_bool else np.float32
    buf = zlib.decompress(raw)
    arr = np.frombuffer(buf, dtype=dtype).reshape((len(dates), len(columns))).copy()
    return pd.DataFrame(arr, index=dates, columns=columns, copy=False)


def save_feature_store(
    store_uri: str,
    panels: dict[str, pd.DataFrame],
    *,
    sector_map: dict[str, str] | None = None,
    universe_mask: pd.DataFrame | None = None,
    residual_returns: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Write pre-aligned wide panels + metadata to an R2 feature-store prefix."""
    close = panels.get("Close")
    if close is None and panels:
        close = next(iter(panels.values()))
    if close is None:
        raise ValueError("Cannot save an empty panels dictionary to feature store")

    dates = pd.DatetimeIndex(pd.to_datetime(close.index), name="Date")
    columns = pd.Index([str(c) for c in close.columns])
    index_bytes = json.dumps(
        {
            "dates": [d.isoformat() for d in dates],
            "columns": list(columns),
        }
    ).encode("utf-8")
    r2_store.write_bytes(r2_store.join(store_uri, "index.json"), index_bytes)
    index_sha = hashlib.sha256(index_bytes).hexdigest()

    panel_hashes: dict[str, str] = {}

    def _write_panel(item: tuple[str, pd.DataFrame]) -> tuple[str, str, int]:
        name, frame = item
        raw = _df_to_zlib_bytes(frame, dates, columns, as_bool=False)
        uri = r2_store.join(store_uri, f"panels/{name}.bin.zlib")
        r2_store.write_bytes(uri, raw)
        return name, hashlib.sha256(raw).hexdigest(), len(raw)

    items = list(panels.items())
    total_bytes = len(index_bytes)
    workers = max(1, min(int(max_workers), len(items) or 1))
    if workers == 1:
        results = [_write_panel(it) for it in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_write_panel, items))
    for name, digest, nbytes in results:
        panel_hashes[name] = digest
        total_bytes += nbytes

    extras: dict[str, str] = {"index": index_sha}
    if sector_map:
        sec_bytes = json.dumps(sector_map, sort_keys=True).encode("utf-8")
        r2_store.write_bytes(r2_store.join(store_uri, "sector_map.json"), sec_bytes)
        extras["sector_map"] = hashlib.sha256(sec_bytes).hexdigest()

    if universe_mask is not None and not universe_mask.empty:
        mask_bytes = _df_to_zlib_bytes(universe_mask, dates, columns, as_bool=True)
        r2_store.write_bytes(r2_store.join(store_uri, "universe_mask.bin.zlib"), mask_bytes)
        extras["universe_mask"] = hashlib.sha256(mask_bytes).hexdigest()
        total_bytes += len(mask_bytes)

    if residual_returns is not None and not residual_returns.empty:
        res_bytes = _df_to_zlib_bytes(residual_returns, dates, columns, as_bool=False)
        r2_store.write_bytes(r2_store.join(store_uri, "residual_returns.bin.zlib"), res_bytes)
        extras["residual_returns"] = hashlib.sha256(res_bytes).hexdigest()
        total_bytes += len(res_bytes)

    manifest = {
        "schema": "qsd-feature-store/1.1.0",
        "panel_format": _NUMPY_ZLIB_FORMAT,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_panels": len(panel_hashes),
        "panels": sorted(panel_hashes),
        "panel_sha256": panel_hashes,
        "extras_sha256": extras,
        "shape": [len(dates), len(columns)],
        "first_date": str(dates.min().date()) if len(dates) else None,
        "last_date": str(dates.max().date()) if len(dates) else None,
        "total_bytes": total_bytes,
        "metadata": metadata or {},
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    manifest_uri = r2_store.join(store_uri, "manifest.json")
    r2_store.write_bytes(manifest_uri, manifest_bytes)
    manifest["manifest_uri"] = manifest_uri
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    logger.info(
        f"Saved feature store to {store_uri}: {len(panel_hashes)} panels ({_NUMPY_ZLIB_FORMAT}), "
        f"{total_bytes / 1e6:.1f} MB, sha={manifest['manifest_sha256'][:12]}"
    )
    return manifest


def load_feature_store(
    store_uri: str,
    max_workers: int = 12,
) -> dict[str, Any]:
    """Load a precomputed feature-store bundle from R2 concurrently in memory."""
    manifest_uri = r2_store.join(store_uri, "manifest.json")
    manifest_bytes = r2_store.read_bytes(manifest_uri)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    panel_names: list[str] = list(manifest.get("panels") or [])
    if not panel_names:
        raise ValueError(f"Feature store manifest at {manifest_uri} lists no panels")

    use_zlib = manifest.get("panel_format") == _NUMPY_ZLIB_FORMAT
    extras_hashes = manifest.get("extras_sha256") or {}

    if use_zlib:
        idx_obj = json.loads(r2_store.read_text(r2_store.join(store_uri, "index.json")))
        dates = pd.DatetimeIndex(pd.to_datetime(idx_obj["dates"]), name="Date")
        columns = pd.Index(idx_obj["columns"])

        def _read_panel(name: str) -> tuple[str, pd.DataFrame]:
            uri = r2_store.join(store_uri, f"panels/{name}.bin.zlib")
            return name, _zlib_bytes_to_df(r2_store.read_bytes(uri), dates, columns, as_bool=False)
    else:
        dates = pd.DatetimeIndex([])
        columns = pd.Index([])

        def _read_panel(name: str) -> tuple[str, pd.DataFrame]:
            uri = r2_store.join(store_uri, f"panels/{name}.parquet")
            return name, _parquet_bytes_to_df(r2_store.read_bytes(uri))

    workers = max(1, min(int(max_workers), len(panel_names)))
    if workers == 1:
        loaded_pairs = [_read_panel(n) for n in panel_names]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded_pairs = list(pool.map(_read_panel, panel_names))
    panels = dict(loaded_pairs)

    sector_map: dict[str, str] = {}
    if "sector_map" in extras_hashes:
        sector_map = json.loads(r2_store.read_text(r2_store.join(store_uri, "sector_map.json")))

    universe_mask: pd.DataFrame | None = None
    if "universe_mask" in extras_hashes:
        if use_zlib:
            universe_mask = _zlib_bytes_to_df(
                r2_store.read_bytes(r2_store.join(store_uri, "universe_mask.bin.zlib")),
                dates,
                columns,
                as_bool=True,
            )
        else:
            universe_mask = _parquet_bytes_to_df(
                r2_store.read_bytes(r2_store.join(store_uri, "universe_mask.parquet"))
            ).astype(bool)

    residual_returns: pd.DataFrame | None = None
    if "residual_returns" in extras_hashes:
        if use_zlib:
            residual_returns = _zlib_bytes_to_df(
                r2_store.read_bytes(r2_store.join(store_uri, "residual_returns.bin.zlib")),
                dates,
                columns,
                as_bool=False,
            )
        else:
            residual_returns = _parquet_bytes_to_df(
                r2_store.read_bytes(r2_store.join(store_uri, "residual_returns.parquet"))
            )

    logger.info(
        f"Loaded feature store from {store_uri}: {len(panels)} panels "
        f"(format={manifest.get('panel_format', 'parquet_v1')}, "
        f"sha={manifest_sha[:12]}, built={manifest.get('built_at_utc')})"
    )
    return {
        "panels": panels,
        "sector_map": sector_map,
        "universe_mask": universe_mask,
        "residual_returns": residual_returns,
        "manifest": manifest,
        "manifest_uri": manifest_uri,
        "manifest_sha256": manifest_sha,
    }
