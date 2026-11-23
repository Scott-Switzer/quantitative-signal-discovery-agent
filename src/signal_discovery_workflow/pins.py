# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""Byte-level pins for every dataset a run reads.

A dataset is named by a release, and a release is identified by the SHA-256 of
its manifest object's exact bytes. Reading without a pin means a result cannot
say what it was computed from: the object could be replaced, the release could
be re-cut under the same name, and the numbers would silently change.

Two policies exist because data is republished on its own schedule:

* ``enforce`` -- a configured hash that does not match raises. Use this to
  reproduce a published result or a paper.
* ``record`` -- a mismatch is logged and the bytes actually read are recorded
  with ``verified: false``. Use this for daily research, where the point is to
  know exactly what was used rather than to freeze the dataset forever.

Every result carries the pin record either way. ``resolve_release`` finds the
newest sub-release under a prefix using object modification time, so a rolling
dataset can be consumed without editing a URI and without losing the audit
trail.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from . import r2_store

logger = logging.getLogger(__name__)

POLICIES = ("enforce", "record")


class PinMismatch(RuntimeError):
    """The manifest bytes on storage are not the pinned bytes."""


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def pin(uri: str, role: str | None = None) -> dict[str, Any]:
    """Hash one manifest object and describe the release it pins."""
    body = r2_store.read_bytes(uri)
    record: dict[str, Any] = {
        "uri": str(uri),
        "sha256": sha256_bytes(body),
        "bytes": len(body),
    }
    if role:
        record["role"] = role
    return record


def verify(
    uri: str,
    expected_sha256: str,
    role: str | None = None,
    policy: str = "enforce",
) -> dict[str, Any]:
    """Pin ``uri`` and compare it with ``expected_sha256``.

    ``policy="record"`` never raises on a mismatch; it returns
    ``verified: False`` plus the hash that was actually read. The caller is
    expected to surface that in the result payload.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown pin policy {policy!r}; expected one of {POLICIES}")
    record = pin(uri, role=role)
    if not expected_sha256:
        return {**record, "verified": False, "policy": policy, "reason": "no expected hash configured"}
    if record["sha256"] != expected_sha256:
        message = (
            f"{role or uri}: pinned {expected_sha256}, read {record['sha256']} "
            f"({record['bytes']} bytes). The release changed under the run."
        )
        if policy == "enforce":
            raise PinMismatch(message)
        logger.warning(f"{message} pin_policy=record, continuing with the bytes just read")
        return {
            **record,
            "verified": False,
            "mismatch": True,
            "expected_sha256": expected_sha256,
            "policy": policy,
        }
    return {**record, "verified": True, "policy": policy}


def manifest_for(release_uri: str, name: str = "manifest.json") -> str:
    """Manifest URI inside a release directory."""
    return r2_store.join(release_uri, name)


def resolve_release(
    prefix: str,
    artifact: str | None = "panel.parquet",
    manifest: str = "manifest.json",
) -> dict[str, Any]:
    """Resolve the newest release under ``prefix``.

    ``prefix`` may already point at a release (it contains the artifact and the
    manifest) or at a family directory with dated/hashed children. The newest
    child is chosen by object modification time, not by name: release hashes
    and dates do not sort chronologically.

    Returns ``{release_uri, artifact_uri, manifest_uri, release_mtime,
    candidates}``. Nothing is downloaded here except directory metadata.
    """
    prefix = str(prefix).rstrip("/")
    entries = r2_store.list_entries(prefix)
    names = {entry["name"] for entry in entries}
    if manifest in names and (artifact is None or artifact in names):
        release = prefix
        anchor = artifact or manifest
        mtime = next(entry["mtime"] for entry in entries if entry["name"] == anchor)
        candidates = 1
    else:
        releases = []
        for entry in entries:
            # A child is a usable release only if both objects exist. Do not
            # trust the listing's directory flag: S3 has no directory objects
            # and pyarrow reports them inconsistently across prefix forms.
            if r2_store.stat(r2_store.join(entry["path"], manifest)) is None:
                continue
            anchor = r2_store.join(entry["path"], artifact or manifest)
            info = r2_store.stat(anchor)
            if info is None:
                continue
            releases.append({**entry, "mtime": info["mtime"], "size": info["size"]})
        if not releases:
            raise FileNotFoundError(
                f"{prefix} is neither a release nor a directory of usable releases "
                f"(looked for {artifact} + {manifest}); "
                f"entries={len(entries)} names={sorted(names)[:8]} "
                f"module={r2_store.__file__}"
            )
        releases.sort(key=lambda entry: (entry.get("mtime") or 0, entry["name"]))
        newest = releases[-1]
        release = newest["path"]
        mtime = newest.get("mtime")
        candidates = len(releases)
        logger.info(
            f"Resolved newest release under {prefix}: {release} "
            f"({candidates} candidates)"
        )
    return {
        "release_uri": release,
        "artifact_uri": r2_store.join(release, artifact) if artifact else None,
        "manifest_uri": r2_store.join(release, manifest),
        "release_mtime": mtime,
        "candidates": candidates,
    }
