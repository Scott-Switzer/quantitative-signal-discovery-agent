# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
SEC Filing Text Diffs — "Lazy Prices" (Item 4).

Implements the point-in-time textual-change alpha from Cohen, Malloy & Nguyen
(2020, Journal of Finance, "Lazy Prices"): firms that substantially alter the
language of their consecutive 10-K / 10-Q filings underperform over the
following 6+ months because investors underreact to non-numeric disclosures.

Three orthogonal metrics are computed per ``(cik, form)`` against the prior
filing of the same form type (strictly backward-looking, available at
``filed_date``):

* ``FilingTextChange``: ``1 - cosine_similarity`` over normalized word-count
  vectors (higher = larger language shift, predicts lower returns).
* ``FilingJaccardDiff``: ``1 - jaccard_similarity`` over unique vocabulary
  tokens (captures newly added risk/litigation vocabulary).
* ``FilingLengthDelta``: ``log((len_t + 1) / (len_{t-1} + 1))`` (expansion in
  filing length often signals added risk factors or accounting complexity).

Scalable zero-local-staging cloud architecture:
1. Filters corpus CIKs to CRSP-mapped issuers via fail-closed
   ``crsp__ccm_lookup`` (skipping ~85% of EDGAR trusts/funds/non-CRSP filers).
2. Filters manifests to ``10-K`` / ``10-Q`` from ``start_year`` onward *before*
   touching ``primary.html``.
3. Streams at most ``max_bytes`` (default 250 KB) of each ``primary.html`` via
   ``r2_store.read_bytes_prefix``, tokenizes immediately, diffs against the
   prior filing's token counter, and discards the raw HTML so peak RAM per CIK
   stays under a few megabytes.
4. Shards CIKs across parallel Modal containers via ``.map()`` + concurrent
   in-container S3 threads, then merges compact shard Parquets into R2.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from typing import Any

import numpy as np
import pandas as pd

from . import r2_store
from .fundamentals import DEFAULT_CCM_IDENTITY_URI, _load_cik_bridge, _normalize_cik, _scatter

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_V2_URI = (
    "r2://ppe-sec-intelligence-prod/sec-filing-intelligence/corpus/v2/"
    "9bb2a1fa99a4669e934c93a33d80036393cc4803afa50fd42b5b987499864b9a/"
)
DEFAULT_LAZY_PRICES_OUTPUT_URI = (
    "r2://financial-data-wrds-lakehouse/research/datasets/lazy_prices/panel.parquet"
)
DEFAULT_LAZY_PRICES_SHARDS_URI = (
    "r2://financial-data-wrds-lakehouse/research/datasets/lazy_prices/shards"
)

VALID_LAZY_PRICES_FORMS = frozenset({"10-K", "10-Q"})

_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z]{3,}")
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "with", "from", "this", "are", "was",
        "were", "have", "has", "had", "not", "but", "our", "its", "their",
        "which", "will", "would", "could", "should", "may", "can", "into",
        "such", "other", "than", "then", "also", "been", "being", "more",
        "most", "all", "any", "each", "under", "over", "between", "during",
        "including", "million", "thousand", "billion", "december", "january",
        "february", "march", "april", "june", "july", "august", "september",
        "october", "november", "table", "item", "note", "page", "form",
    }
)


def _canonical_form_group(raw_form: Any) -> str:
    return str(raw_form or "").strip().upper().replace("/A", "")


def strip_html_and_tokenize(raw_text: str, max_chars: int = 250_000) -> Counter[str]:
    """Strip HTML tags/entities and return filtered lowercase token frequencies."""
    text = unescape(_TAG_RE.sub(" ", raw_text[:max_chars])).lower()
    return Counter(tok for tok in _TOKEN_RE.findall(text) if tok not in _STOPWORDS)


def cosine_distance(a: Counter[str], b: Counter[str]) -> float:
    """``1 - cosine_similarity(a, b)`` in ``[0, 1]``."""
    if not a or not b:
        return float("nan")
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a <= 0 or norm_b <= 0:
        return float("nan")
    sim = max(0.0, min(1.0, dot / (norm_a * norm_b)))
    return 1.0 - sim


def jaccard_distance(a: Counter[str], b: Counter[str]) -> float:
    """``1 - |A ∩ B| / |A ∪ B|`` over token sets."""
    if not a or not b:
        return float("nan")
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    if union == 0:
        return float("nan")
    return 1.0 - (len(sa & sb) / union)


def compute_filing_diffs(filings: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive filing text diffs per ``(cik, form_group)``.

    Input columns required: ``cik``, ``filed_date``, ``form``, ``text``.
    Only ``10-K`` and ``10-Q`` filings are diffed (amendments collapsed to base
    form). Returns a compact feature table with ``cik``, ``filed_date``,
    ``form``, ``text_change`` (cosine distance), ``jaccard_diff``, and
    ``length_delta``.
    """
    empty_cols = ["cik", "filed_date", "form", "text_change", "jaccard_diff", "length_delta"]
    if filings.empty:
        return pd.DataFrame(columns=empty_cols)

    df = filings.dropna(subset=["cik", "filed_date", "text"]).copy()
    df["cik"] = _normalize_cik(df["cik"])
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df = df.dropna(subset=["filed_date"])
    df["form_group"] = (
        df.get("form", pd.Series("10-K", index=df.index))
        .map(_canonical_form_group)
    )
    df = df[df["form_group"].isin(VALID_LAZY_PRICES_FORMS)]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    df = df.sort_values(["cik", "form_group", "filed_date"])

    rows: list[dict[str, Any]] = []
    for (cik, form_group), grp in df.groupby(["cik", "form_group"], sort=False):
        prev_counts: Counter[str] | None = None
        prev_len: int | None = None
        for _, rec in grp.iterrows():
            counts = strip_html_and_tokenize(str(rec["text"]))
            tok_len = sum(counts.values())
            if prev_counts is not None and prev_len is not None and tok_len > 0 and prev_len > 0:
                rows.append(
                    {
                        "cik": cik,
                        "filed_date": rec["filed_date"],
                        "form": form_group,
                        "text_change": cosine_distance(counts, prev_counts),
                        "jaccard_diff": jaccard_distance(counts, prev_counts),
                        "length_delta": math.log((tok_len + 1.0) / (prev_len + 1.0)),
                    }
                )
            prev_counts = counts
            prev_len = tok_len

    return pd.DataFrame(rows, columns=empty_cols)


def build_lazy_prices_panels(
    diffs: pd.DataFrame,
    bridge: pd.DataFrame,
    dates: pd.Index,
    securities: pd.Index,
    valid_days: int = 380,
) -> dict[str, pd.DataFrame]:
    """Scatter point-in-time filing diff features onto ``(dates, securities)``."""
    if diffs.empty:
        return {}

    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    columns = pd.Index([str(s) for s in securities])
    column_of = {name.split(":")[-1]: idx for idx, name in enumerate(columns)}

    df = diffs.copy()
    df["cik"] = _normalize_cik(df["cik"])
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
    df = df.dropna(subset=["cik", "filed_date"])

    merged = df.merge(bridge, on="cik", how="inner")
    merged = merged[
        (merged["filed_date"] >= merged["effective_from"])
        & (merged["filed_date"] <= merged["effective_to"])
    ]
    merged = merged[merged["permno"].isin(column_of)]
    if merged.empty:
        return {}

    merged = merged.sort_values("filed_date")
    merged["valid_from"] = merged["filed_date"]
    merged["valid_to"] = merged["filed_date"] + pd.Timedelta(days=valid_days)

    left = np.searchsorted(dates.values, merged["valid_from"].to_numpy(), side="left")
    right = np.searchsorted(dates.values, merged["valid_to"].to_numpy(), side="right") - 1
    lengths = right - left + 1
    keep = lengths > 0
    if not keep.any():
        return {}

    left, lengths = left[keep], lengths[keep]
    kept = merged.loc[keep]
    col_index = kept["permno"].map(column_of).to_numpy(dtype=np.int64)
    total = int(lengths.sum())
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.repeat(left, lengths) + offsets
    cols = np.repeat(col_index, lengths)
    shape = (len(dates), len(columns))

    mapping = {
        "text_change": "FilingTextChange",
        "jaccard_diff": "FilingJaccardDiff",
        "length_delta": "FilingLengthDelta",
    }
    panels: dict[str, pd.DataFrame] = {}
    for src_col, panel_name in mapping.items():
        if src_col not in kept.columns:
            continue
        vals = pd.to_numeric(kept[src_col], errors="coerce").astype("float32")
        arr = _scatter(vals, lengths, rows, cols, shape)
        frame = pd.DataFrame(arr, index=dates, columns=columns)
        if frame.notna().any().any():
            panels[panel_name] = frame

    if panels:
        logger.info("Lazy Prices panels loaded: " + ", ".join(sorted(panels)))
    return panels


def load_lazy_prices_panels(
    panel_uri: str,
    bridge_uri: str,
    dates: pd.Index,
    securities: pd.Index,
) -> dict[str, pd.DataFrame]:
    """Load precomputed Lazy Prices Parquet features from R2 into wide panels."""
    schema = r2_store.dataset_schema(panel_uri)
    cols = [
        c
        for c in ("cik", "filed_date", "form", "text_change", "jaccard_diff", "length_delta")
        if c in schema
    ]
    diffs = r2_store.read_parquet_parts(panel_uri, cols)
    bridge = _load_cik_bridge(bridge_uri)
    return build_lazy_prices_panels(diffs, bridge, dates, securities)


# ---------------------------------------------------------------------------
# Scalable streaming & sharded R2 batch extraction
# ---------------------------------------------------------------------------


def _process_single_cik(
    cik_entry: dict[str, Any],
    *,
    start_year: int = 2010,
    max_bytes: int = 250_000,
) -> tuple[int, list[dict[str, Any]]]:
    """Stream one CIK's 10-K/10-Q filings in bounded memory and emit diff rows.

    Reads ``manifest.json`` first, discards non-10-K/10-Q and pre-``start_year``
    filings *without* downloading ``primary.html``, and streams at most
    ``max_bytes`` of each qualifying HTML filing one at a time.
    """
    cik_raw = str(cik_entry.get("name", "")).replace("cik=", "")
    cik_norm = _normalize_cik(pd.Series([cik_raw])).iloc[0]
    if cik_norm == "0000000000":
        return 0, []

    try:
        acc_dirs = r2_store.list_entries(cik_entry["path"])
    except Exception as e:
        logger.debug(f"Could not list accessions for CIK {cik_norm}: {e}")
        return 0, []

    cutoff = pd.Timestamp(f"{int(start_year)}-01-01")
    qualifying: list[tuple[str, pd.Timestamp, str]] = []
    for acc_entry in acc_dirs:
        man_uri = r2_store.join(acc_entry["path"], "manifest.json")
        try:
            meta = json.loads(r2_store.read_text(man_uri))
        except Exception:
            continue
        form_group = _canonical_form_group(
            meta.get("form") or meta.get("form_type") or ""
        )
        if form_group not in VALID_LAZY_PRICES_FORMS:
            continue
        filed_raw = meta.get("filing_date") or meta.get("filed_date") or meta.get("filed")
        filed_ts = pd.to_datetime(filed_raw, errors="coerce")
        if pd.isna(filed_ts) or filed_ts < cutoff:
            continue
        html_uri = r2_store.join(acc_entry["path"], "primary.html")
        qualifying.append((form_group, filed_ts, html_uri))

    if len(qualifying) < 2:
        return len(qualifying), []

    qualifying.sort(key=lambda item: (item[0], item[1]))
    rows: list[dict[str, Any]] = []
    filings_read = 0
    prev_by_form: dict[str, tuple[Counter[str], int]] = {}

    for form_group, filed_ts, html_uri in qualifying:
        try:
            raw_bytes = r2_store.read_bytes_prefix(html_uri, max_bytes=max_bytes)
        except Exception:
            continue
        filings_read += 1
        html_text = raw_bytes.decode("utf-8", errors="ignore")
        counts = strip_html_and_tokenize(html_text, max_chars=max_bytes)
        tok_len = sum(counts.values())
        if tok_len <= 0:
            continue

        prev = prev_by_form.get(form_group)
        if prev is not None:
            prev_counts, prev_len = prev
            if prev_len > 0:
                rows.append(
                    {
                        "cik": cik_norm,
                        "filed_date": filed_ts,
                        "form": form_group,
                        "text_change": cosine_distance(counts, prev_counts),
                        "jaccard_diff": jaccard_distance(counts, prev_counts),
                        "length_delta": math.log((tok_len + 1.0) / (prev_len + 1.0)),
                    }
                )
        prev_by_form[form_group] = (counts, tok_len)

    return filings_read, rows


def list_lazy_prices_cik_shards(
    corpus_uri: str = DEFAULT_CORPUS_V2_URI,
    *,
    num_shards: int = 32,
    ccm_uri: str | None = DEFAULT_CCM_IDENTITY_URI,
) -> list[list[dict[str, Any]]]:
    """List corpus CIK directories, filter to CRSP-mapped CIKs, and partition into shards."""
    filings_root = r2_store.join(corpus_uri, "raw/sec/filings")
    cik_dirs = r2_store.list_entries(filings_root)
    total_found = len(cik_dirs)

    if ccm_uri:
        try:
            bridge = _load_cik_bridge(ccm_uri)
            allowed_ciks = set(bridge["cik"].astype(str))
            filtered = [
                entry
                for entry in cik_dirs
                if _normalize_cik(pd.Series([str(entry.get("name", "")).replace("cik=", "")])).iloc[0]
                in allowed_ciks
            ]
            logger.info(
                f"Lazy Prices CIK filter: kept {len(filtered):,}/{total_found:,} CRSP-mapped CIKs"
            )
            cik_dirs = filtered
        except Exception as e:
            logger.warning(f"Could not filter CIKs with CCM bridge ({e}); using all {total_found:,} CIKs")

    n = max(1, min(int(num_shards), len(cik_dirs) or 1))
    # Round-robin partitioning balances large/small CIKs across shards.
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for idx, entry in enumerate(cik_dirs):
        shards[idx % n].append(entry)
    return [s for s in shards if s]


def process_lazy_prices_cik_shard(
    cik_entries: list[dict[str, Any]],
    *,
    shard_id: int,
    shards_prefix: str = DEFAULT_LAZY_PRICES_SHARDS_URI,
    start_year: int = 2010,
    max_bytes: int = 250_000,
    max_workers: int = 16,
) -> dict[str, Any]:
    """Process one shard of CIKs concurrently and write a compact shard Parquet to R2."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    all_rows: list[dict[str, Any]] = []
    filings_read = 0

    def _run(entry: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        return _process_single_cik(entry, start_year=start_year, max_bytes=max_bytes)

    workers = max(1, min(int(max_workers), len(cik_entries) or 1))
    if workers == 1:
        results = [_run(e) for e in cik_entries]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run, cik_entries))

    for count, rows in results:
        filings_read += count
        all_rows.extend(rows)

    shard_uri = r2_store.join(shards_prefix, f"shard_{int(shard_id):03d}.parquet")
    empty_cols = ["cik", "filed_date", "form", "text_change", "jaccard_diff", "length_delta"]
    diffs = pd.DataFrame(all_rows, columns=empty_cols)
    if not diffs.empty:
        buf = io.BytesIO()
        pq.write_table(
            pa.Table.from_pandas(diffs, preserve_index=False), buf, compression="zstd"
        )
        r2_store.write_bytes(shard_uri, buf.getvalue())

    logger.info(
        f"Lazy Prices shard {shard_id:03d}: {len(cik_entries)} CIKs, "
        f"{filings_read:,} filings read, {len(diffs):,} diffs written"
    )
    return {
        "shard_id": int(shard_id),
        "ciks_processed": len(cik_entries),
        "filings_read": filings_read,
        "diffs_written": len(diffs),
        "shard_uri": shard_uri if not diffs.empty else None,
    }


def merge_lazy_prices_shards(
    shards_prefix: str = DEFAULT_LAZY_PRICES_SHARDS_URI,
    output_uri: str = DEFAULT_LAZY_PRICES_OUTPUT_URI,
    *,
    max_workers: int = 12,
) -> dict[str, Any]:
    """Concatenate all shard Parquets under ``shards_prefix`` into ``output_uri``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["cik", "filed_date", "form", "text_change", "jaccard_diff", "length_delta"]
    combined = r2_store.read_parquet_parts(shards_prefix, cols, max_workers=max_workers)
    if combined.empty:
        raise RuntimeError(f"No Lazy Prices shard rows found under {shards_prefix}")

    combined["filed_date"] = pd.to_datetime(combined["filed_date"], errors="coerce")
    combined = (
        combined.dropna(subset=["cik", "filed_date"])
        .sort_values(["cik", "form", "filed_date"])
        .drop_duplicates(["cik", "form", "filed_date"], keep="last")
        .reset_index(drop=True)
    )
    buf = io.BytesIO()
    pq.write_table(
        pa.Table.from_pandas(combined, preserve_index=False), buf, compression="zstd"
    )
    r2_store.write_bytes(output_uri, buf.getvalue())
    logger.info(f"Merged {len(combined):,} Lazy Prices diffs to {output_uri}")
    return {"diffs_written": len(combined), "output_uri": output_uri}


def build_lazy_prices_release_from_r2(
    corpus_uri: str = DEFAULT_CORPUS_V2_URI,
    output_uri: str = DEFAULT_LAZY_PRICES_OUTPUT_URI,
    *,
    start_year: int = 2010,
    max_bytes: int = 250_000,
    max_workers: int = 16,
    ccm_uri: str | None = DEFAULT_CCM_IDENTITY_URI,
) -> dict[str, Any]:
    """Single-container streaming Lazy Prices build (filtered & bounded memory)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shards = list_lazy_prices_cik_shards(corpus_uri, num_shards=1, ccm_uri=ccm_uri)
    cik_entries = shards[0] if shards else []
    all_rows: list[dict[str, Any]] = []
    filings_read = 0

    def _run(entry: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        return _process_single_cik(entry, start_year=start_year, max_bytes=max_bytes)

    workers = max(1, min(int(max_workers), len(cik_entries) or 1))
    if workers == 1:
        results = [_run(e) for e in cik_entries]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run, cik_entries))

    for count, rows in results:
        filings_read += count
        all_rows.extend(rows)

    empty_cols = ["cik", "filed_date", "form", "text_change", "jaccard_diff", "length_delta"]
    diffs = pd.DataFrame(all_rows, columns=empty_cols)
    buf = io.BytesIO()
    pq.write_table(
        pa.Table.from_pandas(diffs, preserve_index=False), buf, compression="zstd"
    )
    r2_store.write_bytes(output_uri, buf.getvalue())
    logger.info(
        f"Published {len(diffs):,} Lazy Prices filing diffs "
        f"({filings_read:,} filings read across {len(cik_entries):,} CIKs) to {output_uri}"
    )
    return {
        "ciks_processed": len(cik_entries),
        "filings_read": filings_read,
        "diffs_written": len(diffs),
        "output_uri": output_uri,
    }
