# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
One place that knows how to read and write bytes and parquet.

Everything in this workflow that touches data goes through here, so there is a
single definition of "remote". ``r2://bucket/key`` and ``s3://bucket/key`` are
read and written **in place** through pyarrow's S3 filesystem: no staging to
local disk, no temporary CSV, no full download. A plain path is accepted only so
the test suite and an explicit local smoke test can work; production
configuration points at R2 and a missing credential raises instead of silently
falling back to the filesystem.

Credential names match the fund's ``phf_research/r2.py`` so a Modal secret works
as-is.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CRED_ALIASES = {
    "R2_ACCESS_KEY_ID": ("R2_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
    "R2_SECRET_ACCESS_KEY": ("R2_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
    "R2_S3_ENDPOINT": ("R2_S3_ENDPOINT", "AWS_ENDPOINT_URL"),
}

_filesystem: Any = None


def is_remote(uri: str | Path | None) -> bool:
    return str(uri or "").startswith(("r2://", "s3://"))


def _cred(name: str) -> str:
    for candidate in _CRED_ALIASES[name]:
        value = os.environ.get(candidate)
        if value:
            return value
    raise RuntimeError(
        f"missing {name}; set R2_* (or the AWS_* aliases used by the Modal secret). "
        "No local fallback is used for a remote URI."
    )


def filesystem() -> Any:
    """Cached S3 filesystem for R2."""
    global _filesystem
    if _filesystem is None:
        import pyarrow.fs as pafs

        _filesystem = pafs.S3FileSystem(
            endpoint_override=_cred("R2_S3_ENDPOINT"),
            access_key=_cred("R2_ACCESS_KEY_ID"),
            secret_key=_cred("R2_SECRET_ACCESS_KEY"),
            region="auto",
        )
    return _filesystem


def _split(uri: str | Path) -> tuple[Any, str]:
    """Return ``(filesystem_or_None, path)``."""
    text = str(uri)
    if is_remote(text):
        return filesystem(), text.split("://", 1)[1]
    return None, text


def read_bytes(uri: str | Path) -> bytes:
    fs, path = _split(uri)
    if fs is None:
        return Path(path).read_bytes()
    with fs.open_input_stream(path) as stream:
        return stream.read()


def read_bytes_prefix(uri: str | Path, max_bytes: int = 250_000) -> bytes:
    """Read at most ``max_bytes`` from the start of a local or R2 object."""
    limit = max(1, int(max_bytes))
    fs, path = _split(uri)
    if fs is None:
        with Path(path).open("rb") as fh:
            return fh.read(limit)
    with fs.open_input_stream(path) as stream:
        return stream.read(limit)


def write_bytes(uri: str | Path, data: bytes) -> None:
    fs, path = _split(uri)
    if fs is None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return
    with fs.open_output_stream(path) as stream:
        stream.write(data)


def exists(uri: str | Path) -> bool:
    fs, path = _split(uri)
    if fs is None:
        return Path(path).exists()
    # Do not swallow errors here: a credential failure must not look like "the
    # ledger is empty", which would silently start a fresh search history.
    return fs.get_file_info(path).type.name != "NotFound"


def stat(uri: str | Path) -> dict[str, Any] | None:
    """Object size/type/mtime, or ``None`` when it does not exist."""
    fs, path = _split(uri)
    if fs is None:
        target = Path(path)
        if not target.exists():
            return None
        info = target.stat()
        return {
            "path": str(target),
            "size": info.st_size,
            "mtime": info.st_mtime,
            "is_dir": target.is_dir(),
        }
    info = fs.get_file_info(path)
    if info.type.name == "NotFound":
        return None
    return {
        "path": info.path,
        "size": info.size,
        "mtime": info.mtime_ns or 0,
        "is_dir": info.type.name == "Directory",
    }


def read_text(uri: str | Path) -> str:
    return read_bytes(uri).decode("utf-8")


def write_text(uri: str | Path, text: str) -> None:
    write_bytes(uri, text.encode("utf-8"))


def join(base: str, name: str) -> str:
    return f"{str(base).rstrip('/')}/{name.lstrip('/')}"


def list_entries(prefix: str | Path) -> list[dict[str, Any]]:
    """Immediate children of ``prefix`` with modification time.

    Used to resolve the newest release without editing configuration every
    time a dataset is republished. ``mtime`` is nanoseconds on a remote
    filesystem and seconds on a local path; callers only compare it.

    ``path`` is a full URI (``r2://bucket/key``) on a remote filesystem. It
    must be: callers join the manifest name onto it and pass the result back to
    ``stat``, which would otherwise treat a bare ``bucket/key`` as a local path.
    """
    fs, path = _split(prefix)
    path = str(path).rstrip("/")
    entries: list[dict[str, Any]] = []
    if fs is None:
        root = Path(path)
        if not root.exists():
            return entries
        for child in root.iterdir():
            stat = child.stat()
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "mtime": stat.st_mtime,
                    "is_dir": child.is_dir(),
                }
            )
        return sorted(entries, key=lambda entry: entry["name"])

    import pyarrow.fs as pafs

    scheme = str(prefix).split("://", 1)[0]

    # S3/R2 has no real directory objects, and pyarrow's handling of a prefix
    # depends on whether it is written with a trailing slash. Try the bare
    # form first (it synthesises the child directory) and fall back to the
    # slashed form; whichever returns children wins.
    for candidate in (path, path + "/"):
        selector = pafs.FileSelector(candidate, recursive=False, allow_not_found=True)
        for info in fs.get_file_info(selector):
            if info.type.name == "NotFound":
                continue
            name = info.path.rstrip("/").rsplit("/", 1)[-1]
            if not name:
                continue
            entries.append(
                {
                    "name": name,
                    "path": f"{scheme}://{info.path.rstrip('/')}",
                    "mtime": info.mtime_ns or 0,
                    "is_dir": info.type.name == "Directory",
                }
            )
        if entries:
            break
    return sorted(entries, key=lambda entry: entry["name"])


def parquet_schema(uri: str | Path) -> set[str]:
    import pyarrow.parquet as pq

    fs, path = _split(uri)
    if fs is None:
        return set(pq.ParquetFile(path).schema_arrow.names)
    return set(pq.ParquetFile(path, filesystem=fs).schema_arrow.names)


def read_parquet(uri: str | Path, columns: list[str]) -> Any:
    """Read selected columns into a pandas frame without staging the file."""
    import pyarrow.parquet as pq

    fs, path = _split(uri)
    if fs is None:
        import pandas as pd

        return pd.read_parquet(path, columns=columns)
    return pq.read_table(path, columns=columns, filesystem=fs).to_pandas()


def _parquet_parts(uri: str | Path) -> list[Any]:
    """Every parquet part under a release directory, or the single file itself."""
    fs, path = _split(uri)
    if fs is None:
        root = Path(path)
        return sorted(root.rglob("*.parquet")) if root.is_dir() else [root]

    import pyarrow.fs as pafs

    target = path.rstrip("/")
    if target.lower().endswith(".parquet"):
        info = fs.get_file_info(target)
        return [target] if info.type.name == "File" else []

    selector = pafs.FileSelector(target + "/", recursive=True, allow_not_found=True)
    return [info.path for info in fs.get_file_info(selector) if info.path.endswith(".parquet")]


def parquet_rows(uri: str | Path) -> int:
    """Row count from parquet metadata, without materializing the table."""
    import pyarrow.parquet as pq

    fs, path = _split(uri)
    handle = pq.ParquetFile(path, filesystem=fs) if fs is not None else pq.ParquetFile(path)
    return int(handle.metadata.num_rows)


def dataset_schema(uri: str | Path) -> set[str]:
    """Schema of a release: a single parquet file or a directory of parts."""
    import pyarrow.parquet as pq

    parts = _parquet_parts(uri)
    if not parts:
        raise FileNotFoundError(f"no parquet parts under {uri}")
    fs, _ = _split(uri)
    if fs is None:
        return set(pq.ParquetFile(str(parts[0])).schema_arrow.names)
    return set(pq.ParquetFile(parts[0], filesystem=fs).schema_arrow.names)


def read_parquet_parts(
    uri: str | Path,
    columns: list[str],
    filters: list[tuple] | None = None,
    max_workers: int = 12,
) -> Any:
    """Read every parquet part of one release in place and concatenate them.

    ``filters`` is pushed down to pyarrow, so a long-format fact table can be
    reduced to the metrics of interest before it is materialized.

    Parts are fetched concurrently. A release with hundreds of small parts is
    network-latency bound, not CPU bound: reading them one at a time costs two
    S3 round-trips per part and turned a 20-minute load into a preemption.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = _parquet_parts(uri)
    if not parts:
        raise FileNotFoundError(f"no parquet parts under {uri}")
    fs, _ = _split(uri)

    # A release can be a union of exports with different columns (the factor
    # returns release has one AQR schema with ``sheet`` and one JKP schema
    # without it). Reading the first part's column list against every part
    # raises "No match for FieldRef.Name"; intersect per part instead so a
    # heterogeneous release is still readable.
    filter_columns = {
        str(flt[0]) for flt in (filters or []) if isinstance(flt, (list, tuple)) and flt
    }

    def _read_part(part: Any) -> Any | None:
        if fs is None:
            handle = pq.ParquetFile(str(part))
        else:
            handle = pq.ParquetFile(part, filesystem=fs)
        names = set(handle.schema_arrow.names)
        if filter_columns - names:
            logger.info(f"Skipping {part}: missing filter columns {sorted(filter_columns - names)}")
            return None
        use = [c for c in columns if c in names]
        if fs is None:
            return pq.read_table(str(part), columns=use, filters=filters)
        return pq.read_table(part, columns=use, filters=filters, filesystem=fs)

    workers = max(1, min(int(max_workers), len(parts)))
    if workers == 1:
        tables = [t for t in (_read_part(p) for p in parts) if t is not None]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            tables = [t for t in pool.map(_read_part, parts) if t is not None]
    if not tables:
        raise FileNotFoundError(f"no readable parquet parts under {uri}")
    return pa.concat_tables(tables, promote_options="permissive").to_pandas()
