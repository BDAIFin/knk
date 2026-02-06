from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import os

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
@dataclass
class Stage2Config:
    label_col: str = "fraud"
    id_cols: Tuple[str, ...] = ("client_id", "card_id", "merchant_id")
    feature_cols: Optional[List[str]] = None

    # preprocessing
    use_scaler: bool = True
    fillna_value: float = 0.0

    # feature에서 자동 제외할 컬럼
    drop_feature_cols: Tuple[str, ...] = ("_row_id",)

    # categorical index를 one-hot으로
    onehot_client_top_mccg_prev: bool = False
    client_top_mccg_prev_col: str = "client_top_mccg_prev"
    client_top_mccg_prev_n_classes: int = 11  # 0..10, -1 unknown

    # HDBSCAN base (if not tuning)
    hdb_min_cluster_size: int = 50
    hdb_min_samples: Optional[int] = None
    hdb_metric: str = "euclidean"
    hdb_cluster_selection_method: str = "eom"  # "eom" | "leaf"

    # Two-stage split
    two_stage_enabled: bool = True
    dominant_share_threshold: float = 0.90  # 90% 이상이면 dominant 재클러스터링
    dominant_label_policy: str = "replace"   # "replace" | "keep_major"
    # replace: dominant cluster 내부 라벨을 subcluster로 교체해 최종 타입을 더 쪼갬
    # keep_major: major는 유지하고 subcluster는 별도 컬럼으로만 저장

    # Tuning (choose "best" HDBSCAN)
    tune_enabled: bool = True
    tune_min_cluster_sizes: Tuple[int, ...] = (20, 30, 50, 80, 120)
    tune_min_samples: Tuple[Optional[int], ...] = (None,)  # 필요 시 (None, 5, 10) 등으로 늘려도 됨
    tune_sample_size_for_silhouette: int = 30000  # silhouette 계산용 샘플 (너무 크면 느림)
    tune_require_min_clusters: int = 2            # noise 제외 최소 클러스터 수
    tune_score_noise_penalty: float = 0.75        # score = sil * (1 - noise_ratio)^penalty

    # Reporting
    topn: int = 10


# -----------------------------
# Feature matrix
# -----------------------------
def _infer_feature_cols(df: pd.DataFrame, cfg: Stage2Config) -> List[str]:
    if cfg.feature_cols is not None:
        cols = list(cfg.feature_cols)
    else:
        cols = [c for c in df.columns if c != cfg.label_col and c not in cfg.id_cols]

    drop_set = set(cfg.drop_feature_cols or ())
    cols = [c for c in cols if c not in drop_set]
    return cols


def _make_onehot_client_top_mccg_prev(df: pd.DataFrame, col: str, n_classes: int) -> pd.DataFrame:
    s = df[col].fillna(-1).astype(int)
    out = pd.DataFrame(index=df.index)
    out[f"{col}_unknown"] = (s == -1).astype(np.int8)
    for k in range(n_classes):
        out[f"{col}_{k}"] = (s == k).astype(np.int8)
    return out


def make_stage2_matrix(
    df: pd.DataFrame,
    cfg: Stage2Config,
) -> Tuple[np.ndarray, List[str], Optional[StandardScaler]]:
    feat_cols = _infer_feature_cols(df, cfg)
    Xdf = df[feat_cols].copy()

    # categorical index one-hot
    if cfg.onehot_client_top_mccg_prev and cfg.client_top_mccg_prev_col in Xdf.columns:
        oh = _make_onehot_client_top_mccg_prev(
            Xdf,
            col=cfg.client_top_mccg_prev_col,
            n_classes=cfg.client_top_mccg_prev_n_classes,
        )
        Xdf = Xdf.drop(columns=[cfg.client_top_mccg_prev_col])
        Xdf = pd.concat([Xdf, oh], axis=1)
        feat_cols = list(Xdf.columns)

    # bool -> int
    for c in Xdf.columns:
        if Xdf[c].dtype == bool:
            Xdf[c] = Xdf[c].astype(np.int8)

    # non-numeric guard
    non_num = [c for c in Xdf.columns if not np.issubdtype(Xdf[c].dtype, np.number)]
    if non_num:
        raise TypeError(f"Non-numeric feature cols found: {non_num[:20]} (showing up to 20)")

    # inf/nan
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(cfg.fillna_value)

    scaler = None
    if cfg.use_scaler:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(Xdf.to_numpy(dtype=np.float64, copy=False))
        return Xs, feat_cols, scaler

    return Xdf.to_numpy(dtype=np.float64, copy=False), feat_cols, None


# HDBSCAN core

def _fit_hdbscan(X: np.ndarray, cfg: Stage2Config, mcs: int, ms: Optional[int]):
    if not _HDBSCAN_AVAILABLE:
        raise RuntimeError("hdbscan not available. pip install hdbscan")

    cl = hdbscan.HDBSCAN(
        min_cluster_size=int(mcs),
        min_samples=ms,
        metric=cfg.hdb_metric,
        cluster_selection_method=cfg.hdb_cluster_selection_method,
        prediction_data=True,
    )
    labels = cl.fit_predict(X)
    return cl, labels


def _hdbscan_quality(
    X: np.ndarray,
    labels: np.ndarray,
    cfg: Stage2Config,
) -> Dict[str, Any]:

    n = int(X.shape[0])
    counts = pd.Series(labels).value_counts(dropna=False)
    noise_ratio = float((labels == -1).mean())

    # clusters excluding noise
    clus_ids = [c for c in counts.index.tolist() if c != -1]
    n_clusters = len(clus_ids)

    sil = np.nan
    score = -np.inf

    if n_clusters >= cfg.tune_require_min_clusters:
        # silhouette sample
        idx_non_noise = np.flatnonzero(labels != -1)
        if idx_non_noise.size >= 2000:
            rng = np.random.RandomState(42)
            take = int(min(cfg.tune_sample_size_for_silhouette, idx_non_noise.size))
            samp = rng.choice(idx_non_noise, size=take, replace=False)

            y = labels[samp]
            # ensure at least 2 labels in sample
            if len(np.unique(y)) >= 2:
                sil = float(silhouette_score(X[samp], y))
                score = sil * ((1.0 - noise_ratio) ** cfg.tune_score_noise_penalty)

    largest_share = float(counts.max() / n) if n > 0 else 0.0
    return {
        "n": n,
        "n_clusters_ex_noise": int(n_clusters),
        "noise_ratio": float(noise_ratio),
        "largest_share": float(largest_share),
        "silhouette_non_noise": float(sil) if np.isfinite(sil) else np.nan,
        "score": float(score),
    }


def fit_hdbscan_best(
    X: np.ndarray,
    cfg: Stage2Config,
) -> Tuple[Any, np.ndarray, Dict[str, Any]]:

    if not cfg.tune_enabled:
        model, labels = _fit_hdbscan(X, cfg, cfg.hdb_min_cluster_size, cfg.hdb_min_samples)
        rep = _hdbscan_quality(X, labels, cfg)
        rep["picked"] = {"min_cluster_size": cfg.hdb_min_cluster_size, "min_samples": cfg.hdb_min_samples}
        return model, labels, rep

    best = {"score": -np.inf}
    best_tuple = None

    for mcs in cfg.tune_min_cluster_sizes:
        for ms in cfg.tune_min_samples:
            model, labels = _fit_hdbscan(X, cfg, mcs, ms)
            rep = _hdbscan_quality(X, labels, cfg)
            rep["picked"] = {"min_cluster_size": int(mcs), "min_samples": ms}

            # choose by score, tie-breaker: lower noise, more clusters
            better = False
            if rep["score"] > best["score"]:
                better = True
            elif np.isfinite(rep["score"]) and rep["score"] == best["score"]:
                if rep["noise_ratio"] < best.get("noise_ratio", 1.0):
                    better = True
                elif rep["noise_ratio"] == best.get("noise_ratio", 1.0) and rep["n_clusters_ex_noise"] > best.get("n_clusters_ex_noise", 0):
                    better = True

            if better:
                best = rep
                best_tuple = (model, labels)

    if best_tuple is None:
        # fallback: run with default
        model, labels = _fit_hdbscan(X, cfg, cfg.hdb_min_cluster_size, cfg.hdb_min_samples)
        best = _hdbscan_quality(X, labels, cfg)
        best["picked"] = {"min_cluster_size": cfg.hdb_min_cluster_size, "min_samples": cfg.hdb_min_samples}
        return model, labels, best

    model, labels = best_tuple
    return model, labels, best



# Profiling / table

def profile_clusters(
    df_src: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    topn: int = 10,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    df = df_src.copy()
    df["_cluster"] = labels

    counts = df["_cluster"].value_counts(dropna=False).sort_index()
    out["cluster_counts"] = counts.to_dict()
    out["n_clusters_ex_noise"] = int(len([c for c in counts.index if c != -1]))
    out["noise_ratio"] = float((df["_cluster"] == -1).mean()) if (-1 in counts.index) else 0.0

    base_mean = df[feature_cols].mean(numeric_only=True)

    cluster_means: Dict[int, Dict[str, float]] = {}
    top_features: Dict[int, List[str]] = {}

    for c, sub in df.groupby("_cluster", sort=True):
        mu = sub[feature_cols].mean(numeric_only=True)
        diff = (mu - base_mean).abs().sort_values(ascending=False)
        cluster_means[int(c)] = mu.to_dict()
        top_features[int(c)] = diff.head(topn).index.tolist()

    out["cluster_means"] = cluster_means
    out["cluster_top_features_absdiff"] = top_features
    return out


def build_cluster_table(
    labels: np.ndarray,
    prof: Dict[str, Any],
    stage: str,
) -> pd.DataFrame:
    counts = prof["cluster_counts"]
    total = int(sum(counts.values())) if counts else int(len(labels))

    rows = []
    for cid, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        rows.append({
            "stage": stage,
            "cluster_id": int(cid),
            "n": int(n),
            "share": float(n / total) if total > 0 else 0.0,
            "top_features_absdiff": " | ".join(prof["cluster_top_features_absdiff"].get(int(cid), [])),
        })
    return pd.DataFrame(rows)


# Stage2 main (HDBSCAN + optional 2-stage)

def fit_stage2_hdbscan(
    df_feat: pd.DataFrame,
    X: np.ndarray,
    feat_cols: List[str],
    cfg: Stage2Config,
) -> Dict[str, Any]:
    """
    Returns:
    - labels_major, labels_minor, labels_final
    - reports (tuning report major/minor)
    - tables: df_table (summary)
    - profiles: prof_major, prof_minor(optional)
    """
    # 1) major hdbscan (best)
    hdb_major, labels_major, rep_major = fit_hdbscan_best(X, cfg)
    prof_major = profile_clusters(df_feat, labels_major, feat_cols, topn=cfg.topn)

    # dominant cluster 판단 (noise 제외하고 최대 cluster)
    s = pd.Series(labels_major)
    counts_ex = s[s != -1].value_counts()
    if counts_ex.empty:
        dominant_id = None
        dominant_share = 0.0
    else:
        dominant_id = int(counts_ex.index[0])
        dominant_share = float(counts_ex.iloc[0] / len(labels_major))

    # 2) optional second pass on dominant cluster
    labels_minor = np.full_like(labels_major, fill_value=-99)
    prof_minor = None
    rep_minor = None

    do_second = (
        cfg.two_stage_enabled
        and (dominant_id is not None)
        and (dominant_share >= cfg.dominant_share_threshold)
    )

    if do_second:
        mask_dom = (labels_major == dominant_id)
        X2 = X[mask_dom]
        df2 = df_feat.loc[mask_dom].copy()

        # re-tune on subset as well (same cfg; tuning will pick best for subset)
        hdb_minor, lab_minor, rep_minor = fit_hdbscan_best(X2, cfg)
        labels_minor[mask_dom] = lab_minor
        prof_minor = profile_clusters(df2, lab_minor, feat_cols, topn=cfg.topn)

        # 3) build final labels
        labels_final = labels_major.copy()
        if cfg.dominant_label_policy == "replace":

            for i in np.flatnonzero(mask_dom):
                mn = int(labels_minor[i])
                if mn == -1:
                    labels_final[i] = -1
                else:
                    labels_final[i] = int(dominant_id * 1000 + mn)

        elif cfg.dominant_label_policy == "keep_major":
            # final은 major 그대로, minor는 별도 컬럼에서만 활용
            labels_final = labels_major.copy()
        else:
            raise ValueError(f"Unknown dominant_label_policy={cfg.dominant_label_policy}")

    else:
        labels_final = labels_major.copy()

    # 4) profiling for final labels
    prof_final = profile_clusters(df_feat, labels_final, feat_cols, topn=cfg.topn)

    # 5) tables
    df_tbl_major = build_cluster_table(labels_major, prof_major, stage="major")
    df_tbl_final = build_cluster_table(labels_final, prof_final, stage="final")
    df_tbl = pd.concat([df_tbl_major, df_tbl_final], axis=0, ignore_index=True)

    if prof_minor is not None:
        df_tbl_minor = build_cluster_table(labels_minor[labels_minor != -99], prof_minor, stage="minor_dominant")
        df_tbl = pd.concat([df_tbl, df_tbl_minor], axis=0, ignore_index=True)

    return {
        "labels_major": labels_major,
        "labels_minor": labels_minor,
        "labels_final": labels_final,
        "rep_major": rep_major,
        "rep_minor": rep_minor,
        "dominant_cluster_id": dominant_id,
        "dominant_cluster_share": dominant_share,
        "prof_major": prof_major,
        "prof_minor": prof_minor,
        "prof_final": prof_final,
        "table": df_tbl,
    }



# Save package (model/scaler + reports/tables)

def save_stage2_package(
    out_dir: str,
    scaler: Optional[StandardScaler],
    feature_cols: List[str],
    cfg: Stage2Config,
    result: Dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # scaler
    if scaler is not None:
        dump(scaler, os.path.join(out_dir, "stage2_scaler.joblib"))

    # schema
    schema = {
        "feature_cols": feature_cols,
        "cfg": asdict(cfg),
        "dominant_cluster_id": result.get("dominant_cluster_id"),
        "dominant_cluster_share": result.get("dominant_cluster_share"),
        "dominant_label_policy": cfg.dominant_label_policy,
    }
    with open(os.path.join(out_dir, "stage2_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    # reports
    report = {
        "rep_major": result.get("rep_major"),
        "rep_minor": result.get("rep_minor"),
        "prof_major": result.get("prof_major"),
        "prof_minor": result.get("prof_minor"),
        "prof_final": result.get("prof_final"),
    }
    with open(os.path.join(out_dir, "stage2_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # table csv
    df_tbl: pd.DataFrame = result["table"]
    df_tbl.to_csv(os.path.join(out_dir, "stage2_cluster_table.csv"), index=False, encoding="utf-8")

    # labels parquet 
    labels_df = pd.DataFrame({
        "labels_major": result["labels_major"].astype(np.int32),
        "labels_minor": result["labels_minor"].astype(np.int32),
        "labels_final": result["labels_final"].astype(np.int32),
    })
    labels_df.to_parquet(os.path.join(out_dir, "stage2_labels.parquet"), index=False)
