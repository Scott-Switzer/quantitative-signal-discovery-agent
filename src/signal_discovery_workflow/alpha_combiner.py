# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0

"""
Two-stage weak-alpha pool, cross-run R2 library & heterogeneous meta-combiner.

Institutional alpha factories (Kakushadze 2016 "101 Formulaic Alphas",
WorldQuant BRAIN, Shi et al. "AlphaForge", Microsoft RD-Agent(Q) 2025,
Heterogeneous Ensemble Learning 2026) do not rely on a single formula clearing
standalone Sharpe >= 0.50:

1. **Stage 1 (Pool Admission):** Admit signals with consistent train/validation
   sign, modest |IC| (e.g. >= 0.010), and **low pairwise correlation** against
   existing pool members (e.g. < 0.50).
2. **Cross-Run R2 Persistence (``AlphaPoolLibrary``):** Persist admitted weak
   signals as immutable per-run JSONL shards in R2 so parallel Modal fan-out
   runs accumulate a shared library of orthogonal parents across themes/weeks.
3. **Stage 2 (Walk-Forward Heterogeneous Stacking):** Standardize each admitted
   signal cross-sectionally and combine them via:
   * Train-only **L2 Ridge** (``combine_ridge``), and
   * **Heterogeneous Ridge + shallow GBDT** (``combine_heterogeneous``):
     blends the linear Ridge forecast with a shallow, heavily regularized
     gradient-boosted tree (LightGBM if available, else scikit-learn
     ``HistGradientBoostingRegressor``, else a quadratic cross-interaction
     Ridge fallback) fitted strictly on ``<= train_end_date``.
"""

from __future__ import annotations

import json
import logging
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import r2_store
from .trial_registry import normalize_formula

logger = logging.getLogger(__name__)


def _cs_rank_zscore(signal: pd.DataFrame) -> pd.DataFrame:
    """Map each row cross-sectionally to a uniform [-1, 1] centered rank score."""
    ranks = signal.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True)
    return (ranks - 0.5) * 2.0


def cross_sectional_signal_correlation(
    sig_a: pd.DataFrame,
    sig_b: pd.DataFrame,
    sample_dates: int = 48,
) -> float:
    """Mean cross-sectional rank correlation between two signal panels."""
    dates = sig_a.index.intersection(sig_b.index)
    cols = sig_a.columns.intersection(sig_b.columns)
    if len(dates) == 0 or len(cols) < 10:
        return 0.0
    if len(dates) > sample_dates:
        step = max(1, len(dates) // sample_dates)
        dates = dates[::step]

    a = _cs_rank_zscore(sig_a.loc[dates, cols])
    b = _cs_rank_zscore(sig_b.loc[dates, cols])
    valid = a.notna() & b.notna()
    a_c = a.where(valid)
    b_c = b.where(valid)
    a_c = a_c.sub(a_c.mean(axis=1), axis=0)
    b_c = b_c.sub(b_c.mean(axis=1), axis=0)
    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt(a_c.pow(2).sum(axis=1) * b_c.pow(2).sum(axis=1))
    corr = (num / den).where(den > 1e-12).dropna()
    return float(corr.mean()) if not corr.empty else 0.0


def _fit_nonlinear_stacker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_all: np.ndarray,
    *,
    max_depth: int = 3,
    n_estimators: int = 60,
    reg_lambda: float = 10.0,
) -> tuple[np.ndarray, str]:
    """Fit a shallow, heavily regularized non-linear stacker on train cells only.

    Prefers LightGBM, falls back to scikit-learn HistGradientBoostingRegressor,
    and finally to a pure-NumPy pairwise interaction Ridge model so the
    heterogeneous stacker works in any environment without crashing.
    """
    try:
        import lightgbm as lgb  # type: ignore[import-not-found]

        model = lgb.LGBMRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            num_leaves=min(15, 2 ** int(max_depth)),
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=float(reg_lambda),
            min_child_samples=50,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        return np.asarray(model.predict(X_all), dtype=float), "lightgbm"
    except Exception:
        pass

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-not-found]

        model = HistGradientBoostingRegressor(
            max_iter=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=0.05,
            l2_regularization=float(reg_lambda),
            min_samples_leaf=50,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return np.asarray(model.predict(X_all), dtype=float), "hist_gbdt"
    except Exception:
        pass

    # Pure-NumPy non-linear fallback: linear + pairwise product + squared terms with L2 Ridge.
    k = X_train.shape[1]

    def _expand(X: np.ndarray) -> np.ndarray:
        cols = [X, X ** 2]
        for i in range(k):
            for j in range(i + 1, k):
                cols.append((X[:, i] * X[:, j])[:, None])
        return np.hstack(cols)

    Z_tr = _expand(X_train)
    Z_all = _expand(X_all)
    p = Z_tr.shape[1]
    gram = Z_tr.T @ Z_tr + float(reg_lambda) * max(1.0, len(y_train) / 100.0) * np.eye(p)
    coef = np.linalg.solve(gram, Z_tr.T @ y_train)
    return Z_all @ coef, "poly_interaction_ridge"


class AlphaPool:
    """Orthogonal weak-alpha pool with pairwise correlation gating and stacking."""

    def __init__(self, max_pairwise_corr: float = 0.50, max_size: int = 25):
        self.max_pairwise_corr = float(max_pairwise_corr)
        self.max_size = int(max_size)
        self.members: list[dict[str, Any]] = []

    def try_admit(
        self,
        *,
        name: str,
        formula: str | None,
        values: pd.DataFrame,
        direction: float,
        train_ic: float | None,
        val_ic: float | None,
        val_p: float | None,
        val_sharpe: float | None = None,
        val_monotonicity: float | None = None,
    ) -> tuple[bool, float, str]:
        """Admit a candidate if it is orthogonal to existing pool members.

        Returns ``(admitted, max_corr_seen, reason)``.
        """
        if len(self.members) >= self.max_size:
            return False, 0.0, f"pool full ({self.max_size})"

        oriented = values * float(direction)
        max_corr = 0.0
        for member in self.members:
            c = abs(cross_sectional_signal_correlation(oriented, member["oriented"]))
            if c > max_corr:
                max_corr = c
            if c >= self.max_pairwise_corr:
                return (
                    False,
                    max_corr,
                    f"corr={c:.2f} >= {self.max_pairwise_corr:.2f} with {member['name']}",
                )

        self.members.append(
            {
                "name": name,
                "formula": formula,
                "oriented": oriented,
                "direction": float(direction),
                "train_ic": train_ic,
                "val_ic": val_ic,
                "val_p": val_p,
                "val_sharpe": val_sharpe,
                "val_monotonicity": val_monotonicity,
            }
        )
        return True, max_corr, "admitted"

    def combine_ridge(
        self,
        forward_returns: pd.DataFrame,
        train_end_date: str,
        ridge_alpha: float = 10.0,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Fit L2 Ridge weights strictly on ``<= train_end_date`` and project across all dates."""
        if not self.members:
            raise ValueError("Cannot combine an empty AlphaPool")
        if len(self.members) == 1:
            m = self.members[0]
            return _cs_rank_zscore(m["oriented"]), {m["name"]: 1.0}

        train_end = pd.Timestamp(train_end_date)
        ret = forward_returns.replace([np.inf, -np.inf], np.nan)
        standardized = [_cs_rank_zscore(m["oriented"]) for m in self.members]

        train_dates = ret.index[ret.index <= train_end]
        k = len(self.members)
        factor_rets = np.full((len(train_dates), k), np.nan, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            r_train = ret.loc[train_dates]
            for j, sig in enumerate(standardized):
                s_train = sig.reindex_like(r_train)
                valid = s_train.notna() & r_train.notna()
                num = (s_train * r_train).where(valid).sum(axis=1)
                den = s_train.abs().where(valid).sum(axis=1).replace(0.0, np.nan)
                factor_rets[:, j] = (num / den).to_numpy(dtype=float)

        valid_rows = np.all(np.isfinite(factor_rets), axis=1)
        F = factor_rets[valid_rows]
        if len(F) < max(12, k + 2):
            w = np.full(k, 1.0 / k, dtype=float)
        else:
            mu = F.mean(axis=0)
            cov = np.cov(F, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            diag_scale = np.diag(np.maximum(np.diag(cov), 1e-6))
            reg = cov + float(ridge_alpha) * diag_scale
            raw_w = np.linalg.solve(reg, mu)
            raw_w = np.maximum(raw_w, 0.0)
            w = raw_w / raw_w.sum() if raw_w.sum() > 1e-12 else np.full(k, 1.0 / k)

        combined = pd.DataFrame(0.0, index=ret.index, columns=ret.columns, dtype=float)
        weight_sum = pd.DataFrame(0.0, index=ret.index, columns=ret.columns, dtype=float)
        for weight, sig in zip(w, standardized):
            if weight <= 0:
                continue
            aligned = sig.reindex(index=ret.index, columns=ret.columns)
            present = aligned.notna()
            combined = combined + aligned.fillna(0.0) * float(weight)
            weight_sum = weight_sum + present.astype(float) * float(weight)

        combined = combined.div(weight_sum.replace(0.0, np.nan))
        weights_map = {m["name"]: float(wt) for m, wt in zip(self.members, w)}
        logger.info(
            f"Combined {k} orthogonal signals via train-only Ridge: "
            + ", ".join(f"{n}={wt:.2f}" for n, wt in weights_map.items())
        )
        return combined, weights_map

    def combine_heterogeneous(
        self,
        forward_returns: pd.DataFrame,
        train_end_date: str,
        *,
        ridge_alpha: float = 10.0,
        gbdt_weight: float = 0.45,
        max_depth: int = 3,
        n_estimators: int = 60,
        max_train_samples: int = 120_000,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Blend train-only L2 Ridge with a shallow regularized GBDT on rank features.

        Following Heterogeneous Ensemble Learning (2026), combining a low-variance
        linear predictor with a low-depth non-linear tree captures conditional
        interactions (e.g. value x quality or regime x leverage) while reducing
        prediction error covariance. Zero leakage: both models fit strictly on
        ``<= train_end_date``.
        """
        ridge_df, ridge_weights = self.combine_ridge(
            forward_returns, train_end_date=train_end_date, ridge_alpha=ridge_alpha
        )
        if len(self.members) < 2 or gbdt_weight <= 0.0:
            return ridge_df, {"backend": "ridge_only", "ridge_weights": ridge_weights}

        train_end = pd.Timestamp(train_end_date)
        ret = forward_returns.replace([np.inf, -np.inf], np.nan)
        ret_rank = _cs_rank_zscore(ret)
        standardized = [
            _cs_rank_zscore(m["oriented"]).reindex(index=ret.index, columns=ret.columns)
            for m in self.members
        ]

        # Build aligned 3D tensor (T, N, K) -> flatten valid rows for train & all dates.
        k = len(standardized)
        T, N = ret.shape
        stack = np.stack([df.to_numpy(dtype=np.float32) for df in standardized], axis=-1)
        y_arr = ret_rank.to_numpy(dtype=np.float32)
        has_any_x = np.any(np.isfinite(stack), axis=-1)

        train_date_mask = np.asarray(ret.index <= train_end, dtype=bool)[:, None]
        train_cell_mask = train_date_mask & has_any_x & np.isfinite(y_arr)
        n_train = int(train_cell_mask.sum())
        if n_train < max(200, k * 25):
            return ridge_df, {
                "backend": "ridge_fallback_insufficient_train",
                "ridge_weights": ridge_weights,
            }

        X_tr = np.nan_to_num(stack[train_cell_mask], nan=0.0)
        y_tr = y_arr[train_cell_mask]
        if len(y_tr) > max_train_samples:
            step = max(1, len(y_tr) // max_train_samples)
            X_tr = X_tr[::step]
            y_tr = y_tr[::step]

        X_all = np.nan_to_num(stack[has_any_x], nan=0.0)
        preds_flat, backend = _fit_nonlinear_stacker(
            X_tr,
            y_tr,
            X_all,
            max_depth=max_depth,
            n_estimators=n_estimators,
            reg_lambda=ridge_alpha,
        )

        nl_grid = np.full((T, N), np.nan, dtype=np.float64)
        nl_grid[has_any_x] = preds_flat
        nl_df = _cs_rank_zscore(pd.DataFrame(nl_grid, index=ret.index, columns=ret.columns))
        ridge_norm = _cs_rank_zscore(ridge_df)

        w_nl = float(max(0.0, min(1.0, gbdt_weight)))
        blended = (1.0 - w_nl) * ridge_norm + w_nl * nl_df
        logger.info(
            f"Heterogeneous stacker ({backend}, w_nl={w_nl:.2f}) combined {k} signals "
            f"using {len(y_tr):,} train cells"
        )
        return blended, {
            "backend": backend,
            "gbdt_weight": w_nl,
            "ridge_weights": ridge_weights,
            "train_cells": int(len(y_tr)),
        }


class AlphaPoolLibrary:
    """Sharded, append-only R2 library of orthogonal weak-alpha formulas across runs.

    Writes immutable per-run shards under ``<library_uri>/shards/<ts>_<id>.jsonl``
    so concurrent Modal fan-out containers never collide. On read, merges all
    shards and deduplicates by normalized formula, keeping the highest
    validation-tail-spread record per formula.
    """

    def __init__(self, library_uri: str | Path | None):
        self.library_uri = str(library_uri) if library_uri else None
        self._shard_name = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.jsonl"
        )

    def _shards_prefix(self) -> str:
        assert self.library_uri is not None
        return r2_store.join(self.library_uri, "shards")

    def save_members(
        self,
        members: list[dict[str, Any]],
        *,
        run_id: str,
        horizon: int,
        periods_per_year: int | None = None,
    ) -> str | None:
        """Persist admitted pool member metadata as an immutable shard."""
        if not self.library_uri or not members:
            return None
        lines: list[str] = []
        now = datetime.now(timezone.utc).isoformat()
        for m in members:
            rec = {
                "timestamp": now,
                "run_id": run_id,
                "horizon": int(horizon),
                "periods_per_year": int(periods_per_year) if periods_per_year is not None else None,
                "name": m.get("name"),
                "formula": m.get("formula"),
                "formula_key": normalize_formula(m.get("formula")),
                "direction": float(m.get("direction", 1.0)),
                "train_ic": m.get("train_ic"),
                "val_ic": m.get("val_ic"),
                "val_p": m.get("val_p"),
                "val_sharpe": m.get("val_sharpe"),
                "val_monotonicity": m.get("val_monotonicity"),
            }
            lines.append(json.dumps(rec, default=str))
        shard_uri = r2_store.join(self._shards_prefix(), self._shard_name)
        r2_store.write_text(shard_uri, "\n".join(lines) + "\n")
        logger.info(f"Persisted {len(lines)} AlphaPool members to {shard_uri}")
        return shard_uri

    def load_all(
        self,
        *,
        horizon: int | None = None,
        periods_per_year: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load and deduplicate all persisted pool members across shards concurrently."""
        if not self.library_uri:
            return []
        try:
            entries = [
                e for e in r2_store.list_entries(self._shards_prefix())
                if str(e.get("name", "")).endswith(".jsonl")
            ]
        except Exception:
            return []
        if not entries:
            return []

        def _read_text_safe(entry: dict[str, Any]) -> str:
            try:
                return r2_store.read_text(entry["path"])
            except Exception:
                return ""

        workers = max(1, min(12, len(entries)))
        if workers == 1:
            texts = [_read_text_safe(e) for e in entries]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                texts = list(pool.map(_read_text_safe, entries))

        by_key: dict[str, dict[str, Any]] = {}
        for text in texts:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if horizon is not None and rec.get("horizon") != int(horizon):
                    continue
                if periods_per_year is not None:
                    rec_freq = rec.get("periods_per_year")
                    # Legacy shards without periods_per_year belong to the monthly (12) sleeve.
                    effective_freq = int(rec_freq) if rec_freq is not None else 12
                    if effective_freq != int(periods_per_year):
                        continue
                key = rec.get("formula_key") or normalize_formula(rec.get("formula"))
                if not key:
                    continue
                prev = by_key.get(key)
                if prev is None or abs(rec.get("val_ic") or 0.0) > abs(prev.get("val_ic") or 0.0):
                    by_key[key] = rec

        def _score(r: dict[str, Any]) -> tuple[float, float]:
            sh = r.get("val_sharpe")
            return (float(sh) if sh is not None else -99.0, abs(float(r.get("val_ic") or 0.0)))

        return sorted(by_key.values(), key=_score, reverse=True)

    def top_parents(
        self,
        n: int = 6,
        *,
        horizon: int | None = None,
        periods_per_year: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top N archived parent signals for evolutionary mutation/crossover."""
        return self.load_all(horizon=horizon, periods_per_year=periods_per_year)[: int(n)]
