# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Persistent trial registry and multiple-testing control.

The discovery loop is a search over a very large formula space. Two things go
wrong if every run starts from an empty slate:

1. The model regenerates the same handful of momentum ideas, so the search never
   explores. Recording every formula lets the prompt say "already tried".
2. Nobody counts the trials. Accepting the best of N tests at p<0.05 guarantees
   false discoveries as N grows. Every trial carries its p-value, and
   Benjamini-Hochberg q-values are computed over the whole history.

The ledger is one JSON object per line, stored wherever ``path`` points:
``r2://bucket/key`` is read and written in place through ``r2_store`` so the
history lives with the results, not on a laptop. A plain path is accepted for
tests. The whole ledger is cached in memory for the life of the process because
every read is a network round trip otherwise.

Concurrency: writes are read-modify-write. A single research process per ledger
is assumed; two simultaneous runs will lose one run's lines. That is acceptable
for an exploratory search and is called out rather than hidden behind a lock
that only works inside one process.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import r2_store

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).parent / "output" / "trials.jsonl"


def normalize_formula(formula: str | None) -> str:
    """Whitespace- and case-insensitive key for de-duplicating formulas."""
    return re.sub(r"\s+", "", formula or "").lower()


def benjamini_hochberg(pvalues: list[float], q: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg step-up FDR control.

    Returns ``(qvalues, rejected)`` aligned with the input order.
    """
    n = len(pvalues)
    if n == 0:
        return [], []

    order = sorted(range(n), key=lambda i: pvalues[i])
    qvalues = [1.0] * n
    running_min = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        value = pvalues[idx] * n / rank
        running_min = min(running_min, value)
        qvalues[idx] = min(1.0, running_min)

    return qvalues, [qv <= q for qv in qvalues]


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


class TrialRegistry:
    """Append-only record of every signal the loop has evaluated.

    When ``sharded=True``, each process writes only to its own immutable shard
    file under ``<parent>/trials/run_<utc>_<uuid>.jsonl`` while ``load()``
    merges the legacy ``trials.jsonl`` (if present) plus all shards. That
    eliminates concurrent read-modify-write overwrites across parallel Modal
    runs and guarantees a failed read can never truncate prior history.
    """

    def __init__(self, path: str | Path | None = None, sharded: bool = False):
        self.path = path if path else DEFAULT_REGISTRY_PATH
        self.sharded = bool(sharded)
        self._cache: list[dict[str, Any]] | None = None
        self._run_trials: list[dict[str, Any]] = []
        self._shard_uri: str | Path | None = None
        if self.sharded:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            shard_name = f"run_{ts}_{uuid.uuid4().hex[:8]}.jsonl"
            self._shard_uri = self._shard_dir_join(shard_name)

    def _shard_dir(self) -> str:
        text = str(self.path)
        if text.endswith(".jsonl"):
            parent = text.rsplit("/", 1)[0] if "/" in text else "."
            return r2_store.join(parent, "trials")
        return r2_store.join(text, "trials")

    def _shard_dir_join(self, name: str) -> str:
        return r2_store.join(self._shard_dir(), name)

    # -- persistence --------------------------------------------------------

    def load(self) -> list[dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        trials: list[dict[str, Any]] = []

        if r2_store.exists(self.path):
            try:
                trials.extend(_parse_jsonl(r2_store.read_text(self.path)))
            except Exception as e:
                logger.warning(f"Could not read trial ledger {self.path}: {e}")

        if self.sharded:
            try:
                entries = [
                    e for e in r2_store.list_entries(self._shard_dir())
                    if str(e.get("name", "")).endswith(".jsonl")
                ]
                if entries:
                    def _read_shard(entry: dict[str, Any]) -> list[dict[str, Any]]:
                        try:
                            return _parse_jsonl(r2_store.read_text(entry["path"]))
                        except Exception as e:
                            logger.warning(f"Could not read shard {entry['path']}: {e}")
                            return []

                    workers = max(1, min(12, len(entries)))
                    if workers == 1:
                        for entry in entries:
                            trials.extend(_read_shard(entry))
                    else:
                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            for rows in pool.map(_read_shard, entries):
                                trials.extend(rows)
            except Exception as e:
                logger.warning(f"Could not list trial shards under {self._shard_dir()}: {e}")

        self._cache = trials
        return self._cache

    def _write(self) -> None:
        if self.sharded and self._shard_uri is not None:
            body = "\n".join(json.dumps(t, default=str) for t in self._run_trials)
            r2_store.write_text(self._shard_uri, body + ("\n" if body else ""))
            return
        body = "\n".join(json.dumps(t, default=str) for t in self._cache or [])
        r2_store.write_text(self.path, body + ("\n" if body else ""))

    def record(self, trial: dict[str, Any]) -> dict[str, Any]:
        """Append one trial and return it with ``trial_id`` and ``recorded_at`` set."""
        trials = self.load()
        record = dict(trial)
        if self.sharded:
            record.setdefault("trial_id", f"t_{uuid.uuid4().hex[:10]}")
        else:
            record.setdefault("trial_id", f"t{len(trials) + 1:05d}")
        record.setdefault("recorded_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        trials.append(record)
        self._run_trials.append(record)
        self._write()
        return record

    # -- query --------------------------------------------------------------

    def tried_formulas(self, limit: int = 80) -> list[str]:
        """Distinct formulas already evaluated, most recent first."""
        seen: dict[str, str] = {}
        for trial in reversed(self.load()):
            formula = trial.get("formula")
            if not formula:
                continue
            key = trial.get("trial_key") or normalize_formula(formula)
            if key and key not in seen:
                seen[key] = formula
            if len(seen) >= limit:
                break
        return list(seen.values())

    def tried_keys(self) -> set[str]:
        return {
            t.get("trial_key") or normalize_formula(t.get("formula"))
            for t in self.load()
            if t.get("formula")
        }

    def _scored(self) -> list[dict[str, Any]]:
        return [
            t for t in self.load()
            if t.get("mean_ic") is not None and t.get("p_value") is not None
        ]

    def multiple_testing(self, q: float = 0.05) -> dict[str, Any]:
        """BH-FDR over every scored trial in the registry history."""
        scored = self._scored()
        pvalues = [float(t["p_value"]) for t in scored]
        qvalues, rejected = benjamini_hochberg(pvalues, q=q)
        by_id = {
            str(trial.get("trial_id")): {"q_value": qv, "significant_fdr": bool(rej)}
            for trial, qv, rej in zip(scored, qvalues, rejected)
        }
        return {
            "num_trials": len(scored),
            "q": q,
            "num_significant": int(sum(rejected)),
            "expected_false_positives_at_naive_0_05": round(0.05 * len(scored), 2),
            "by_trial": by_id,
        }

    def leaderboard(self, top: int = 10) -> list[dict[str, Any]]:
        """Best trials by out-of-sample |IC|, with FDR annotation."""
        annotation_by_trial = self.multiple_testing()["by_trial"]
        enriched = [
            {**trial, **annotation_by_trial.get(str(trial.get("trial_id")), {})}
            for trial in self._scored()
        ]
        enriched.sort(key=lambda t: abs(float(t.get("mean_ic") or 0.0)), reverse=True)
        return enriched[:top]

    def summary(self) -> dict[str, Any]:
        trials = self.load()
        families: dict[str, int] = {}
        for trial in trials:
            family = str(trial.get("category") or "unknown")
            families[family] = families.get(family, 0) + 1
        return {
            "num_trials": len(trials),
            "num_accepted": sum(1 for t in trials if t.get("accepted")),
            "num_unique_formulas": len(self.tried_keys()),
            "families": dict(sorted(families.items(), key=lambda kv: -kv[1])),
            "multiple_testing": {
                k: v for k, v in self.multiple_testing().items() if k != "by_trial"
            },
        }
