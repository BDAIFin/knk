# drift_report_train_vs_check.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


IDCOL = "id"
LABEL = "fraud"


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_quantile(a: np.ndarray, q: float) -> float:
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    return float(np.quantile(a, q))


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    if not _HAS_SCIPY:
        # fallback: KS via empirical CDF on pooled sorted values (O(n log n))
        x = np.sort(np.unique(np.r_[a, b]))
        if x.size == 0:
            return np.nan
        fa = np.searchsorted(np.sort(a), x, side="right") / a.size
        fb = np.searchsorted(np.sort(b), x, side="right") / b.size
        return float(np.max(np.abs(fa - fb)))
    return float(ks_2samp(a, b).statistic)


def _psi_from_edges(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    if edges is None or len(edges) < 3:
        return 0.0

    a_cnt, _ = np.histogram(a, bins=edges)
    b_cnt, _ = np.histogram(b, bins=edges)

    a_p = a_cnt / (a_cnt.sum() + 1e-12)
    b_p = b_cnt / (b_cnt.sum() + 1e-12)

    a_p = np.clip(a_p, 1e-12, None)
    b_p = np.clip(b_p, 1e-12, None)

    return float(np.sum((b_p - a_p) * np.log(b_p / a_p)))


def _psi_numeric(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(a, qs))
    if edges.size < 3:
        return 0.0
    return _psi_from_edges(a, b, edges)


def _psi_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan

    pa1 = float(np.mean(a != 0))
    pb1 = float(np.mean(b != 0))
    pa0 = 1.0 - pa1
    pb0 = 1.0 - pb1

    pa0 = max(pa0, 1e-12)
    pa1 = max(pa1, 1e-12)
    pb0 = max(pb0, 1e-12)
    pb1 = max(pb1, 1e-12)

    return float((pb0 - pa0) * np.log(pb0 / pa0) + (pb1 - pa1) * np.log(pb1 / pa1))


def _infer_kind(train_s: pd.Series, check_s: pd.Series) -> str:
    tr = _to_num(train_s)
    ck = _to_num(check_s)

    tr_u = tr.dropna().unique()
    ck_u = ck.dropna().unique()
    u = np.unique(np.r_[tr_u, ck_u])

    if u.size <= 2 and set(u.tolist()).issubset({0, 1}):
        return "binary"
    if u.size <= 10 and np.all(np.isclose(u, np.round(u))):
        return "discrete"
    return "numeric"


def _basic_stats(x: np.ndarray) -> dict:
    x_nn = x[~np.isnan(x)]
    if x_nn.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p50": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(x_nn.size),
        "mean": float(np.mean(x_nn)),
        "std": float(np.std(x_nn, ddof=0)),
        "p01": _safe_quantile(x_nn, 0.01),
        "p05": _safe_quantile(x_nn, 0.05),
        "p10": _safe_quantile(x_nn, 0.10),
        "p50": _safe_quantile(x_nn, 0.50),
        "p90": _safe_quantile(x_nn, 0.90),
        "p95": _safe_quantile(x_nn, 0.95),
        "p99": _safe_quantile(x_nn, 0.99),
        "min": float(np.min(x_nn)),
        "max": float(np.max(x_nn)),
    }


def make_drift_report(
    train_df: pd.DataFrame,
    check_df: pd.DataFrame,
    idcol: str = IDCOL,
    label: str = LABEL,
    bins_numeric: int = 10,
) -> pd.DataFrame:
    common = [c for c in train_df.columns if c in check_df.columns and c not in [idcol, label]]
    rows = []

    for c in common:
        tr_raw = train_df[c]
        ck_raw = check_df[c]

        kind = _infer_kind(tr_raw, ck_raw)
        tr = _to_num(tr_raw).to_numpy(dtype="float64")
        ck = _to_num(ck_raw).to_numpy(dtype="float64")

        tr_miss = float(np.mean(np.isnan(tr)))
        ck_miss = float(np.mean(np.isnan(ck)))

        tr_stats = _basic_stats(tr)
        ck_stats = _basic_stats(ck)

        if kind == "binary":
            psi = _psi_binary(tr, ck)
            ks = _ks(tr, ck)
            tr_rate = float(np.mean((tr[~np.isnan(tr)] != 0))) if tr_stats["count"] else np.nan
            ck_rate = float(np.mean((ck[~np.isnan(ck)] != 0))) if ck_stats["count"] else np.nan
            rows.append({
                "feature": c,
                "kind": kind,
                "psi": psi,
                "ks": ks,
                "train_missing": tr_miss,
                "check_missing": ck_miss,
                "train_rate_1": tr_rate,
                "check_rate_1": ck_rate,
                "abs_rate_diff": (abs(ck_rate - tr_rate) if (np.isfinite(tr_rate) and np.isfinite(ck_rate)) else np.nan),
                "train_mean": tr_stats["mean"],
                "check_mean": ck_stats["mean"],
                "train_p50": tr_stats["p50"],
                "check_p50": ck_stats["p50"],
                "train_p99": tr_stats["p99"],
                "check_p99": ck_stats["p99"],
            })
            continue

        if kind == "discrete":
            # PSI with train-quantile edges is OK; discrete tends to have repeated edges -> PSI may be 0
            psi = _psi_numeric(tr, ck, bins=bins_numeric)
            ks = _ks(tr, ck)
            rows.append({
                "feature": c,
                "kind": kind,
                "psi": psi,
                "ks": ks,
                "train_missing": tr_miss,
                "check_missing": ck_miss,
                "train_mean": tr_stats["mean"],
                "check_mean": ck_stats["mean"],
                "mean_diff": (ck_stats["mean"] - tr_stats["mean"]) if (np.isfinite(tr_stats["mean"]) and np.isfinite(ck_stats["mean"])) else np.nan,
                "train_std": tr_stats["std"],
                "check_std": ck_stats["std"],
                "train_p01": tr_stats["p01"],
                "check_p01": ck_stats["p01"],
                "train_p50": tr_stats["p50"],
                "check_p50": ck_stats["p50"],
                "train_p99": tr_stats["p99"],
                "check_p99": ck_stats["p99"],
                "train_min": tr_stats["min"],
                "check_min": ck_stats["min"],
                "train_max": tr_stats["max"],
                "check_max": ck_stats["max"],
            })
            continue

        # numeric
        psi = _psi_numeric(tr, ck, bins=bins_numeric)
        ks = _ks(tr, ck)

        rows.append({
            "feature": c,
            "kind": kind,
            "psi": psi,
            "ks": ks,
            "train_missing": tr_miss,
            "check_missing": ck_miss,
            "train_mean": tr_stats["mean"],
            "check_mean": ck_stats["mean"],
            "mean_diff": (ck_stats["mean"] - tr_stats["mean"]) if (np.isfinite(tr_stats["mean"]) and np.isfinite(ck_stats["mean"])) else np.nan,
            "train_std": tr_stats["std"],
            "check_std": ck_stats["std"],
            "std_ratio": (ck_stats["std"] / (tr_stats["std"] + 1e-12)) if (np.isfinite(tr_stats["std"]) and np.isfinite(ck_stats["std"])) else np.nan,
            "train_p01": tr_stats["p01"],
            "check_p01": ck_stats["p01"],
            "train_p05": tr_stats["p05"],
            "check_p05": ck_stats["p05"],
            "train_p10": tr_stats["p10"],
            "check_p10": ck_stats["p10"],
            "train_p50": tr_stats["p50"],
            "check_p50": ck_stats["p50"],
            "train_p90": tr_stats["p90"],
            "check_p90": ck_stats["p90"],
            "train_p95": tr_stats["p95"],
            "check_p95": ck_stats["p95"],
            "train_p99": tr_stats["p99"],
            "check_p99": ck_stats["p99"],
            "train_min": tr_stats["min"],
            "check_min": ck_stats["min"],
            "train_max": tr_stats["max"],
            "check_max": ck_stats["max"],
        })

    out = pd.DataFrame(rows)

    # sorting: PSI desc, then KS desc
    if "psi" in out.columns:
        out = out.sort_values(["psi", "ks"], ascending=[False, False], na_position="last").reset_index(drop=True)

    # quick flags
    out["flag_psi_ge_0_25"] = (out["psi"] >= 0.25).fillna(False)
    out["flag_psi_ge_0_10"] = (out["psi"] >= 0.10).fillna(False)
    out["flag_ks_ge_0_15"] = (out["ks"] >= 0.15).fillna(False)

    if "abs_rate_diff" in out.columns:
        out["flag_rate_diff_ge_0_03"] = (out["abs_rate_diff"] >= 0.03).fillna(False)

    out.insert(0, "n_train", len(train_df))
    out.insert(1, "n_check", len(check_df))
    return out


def main():
    TRAIN_PATH = Path("../../../DATA/dataset/TRAIN_stage2")
    CHECK_PATH = Path("../../../DATA/dataset/CHECK_stage2")

    OUT_DIR = Path("../../../DATA/drift_reports")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(TRAIN_PATH)
    check_df = pd.read_parquet(CHECK_PATH)

    report = make_drift_report(train_df, check_df, idcol=IDCOL, label=LABEL, bins_numeric=10)

    out_csv = OUT_DIR / "drift_train_vs_check.csv"
    out_parquet = OUT_DIR / "drift_train_vs_check.parquet"
    report.to_csv(out_csv, index=False)
    report.to_parquet(out_parquet, index=False)

    print("saved:", str(out_csv))
    print("saved:", str(out_parquet))
    print(report.head(30))


if __name__ == "__main__":
    main()