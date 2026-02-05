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


@dataclass
class TypeDiscoveryConfig:
    label_col: str = "fraud"
    id_cols: Tuple[str, ...] = ("client_id", "card_id", "merchant_id")
    feature_cols: Optional[List[str]] = None

    # preprocessing
    use_scaler: bool = True
    fillna_value: float = 0.0

    # feature에서 자동 제외할 컬럼
    drop_feature_cols: Tuple[str, ...] = ("_row_id",)  # client_state에서 생기는 임시 정렬키

    # categorical index를 one-hot으로 
    onehot_client_top_mccg_prev: bool = False
    client_top_mccg_prev_col: str = "client_top_mccg_prev"
    client_top_mccg_prev_n_classes: int = 11  # MCCG_COLS 개수 (0~10), -1은 unknown

    # gmm
    gmm_k_min: int = 2
    gmm_k_max: int = 12
    gmm_covariance_type: str = "full"
    gmm_n_init: int = 3
    gmm_max_iter: int = 300
    gmm_random_state: int = 42

    # silhouette
    compute_silhouette: bool = True
    silhouette_sample_size: int = 50000

    # hdbscan
    hdb_min_cluster_size: int = 50
    hdb_min_samples: Optional[int] = None
    hdb_metric: str = "euclidean"
    hdb_cluster_selection_method: str = "eom"


def _infer_feature_cols(df: pd.DataFrame, cfg: TypeDiscoveryConfig) -> List[str]:
    if cfg.feature_cols is not None:
        cols = list(cfg.feature_cols)
    else:
        cols = [c for c in df.columns if c != cfg.label_col and c not in cfg.id_cols]

    # drop_feature_cols 제거 
    drop_set = set(cfg.drop_feature_cols or ())
    cols = [c for c in cols if c not in drop_set]

    return cols


def _make_onehot_client_top_mccg_prev(
    df: pd.DataFrame,
    col: str,
    n_classes: int,
) -> pd.DataFrame:
    """
    client_top_mccg_prev: -1(unknown), 0..n_classes-1
    -> one-hot: topmccg_prev_0..topmccg_prev_{n_classes-1}, topmccg_prev_unknown
    """
    s = df[col].fillna(-1).astype(int)
    out = pd.DataFrame(index=df.index)

    # unknown
    out[f"{col}_unknown"] = (s == -1).astype(np.int8)

    # known classes
    for k in range(n_classes):
        out[f"{col}_{k}"] = (s == k).astype(np.int8)

    return out


def make_type_discovery_matrix(
    df: pd.DataFrame,
    cfg: TypeDiscoveryConfig,
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
        # 원본 index col 제거 후 one-hot 붙이기
        Xdf = Xdf.drop(columns=[cfg.client_top_mccg_prev_col])
        Xdf = pd.concat([Xdf, oh], axis=1)
        # feat_cols도 업데이트
        feat_cols = list(Xdf.columns)

    # bool -> int
    for c in Xdf.columns:
        if Xdf[c].dtype == bool:
            Xdf[c] = Xdf[c].astype(np.int8)

    # object/datetime 방지
    non_num = [c for c in Xdf.columns if not np.issubdtype(Xdf[c].dtype, np.number)]
    if non_num:
        raise TypeError(f"Non-numeric feature cols found: {non_num[:20]} (showing up to 20)")

    # inf/nan 처리
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(cfg.fillna_value)

    scaler = None
    if cfg.use_scaler:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(Xdf.to_numpy(dtype=np.float64, copy=False))
        return Xs, feat_cols, scaler

    return Xdf.to_numpy(dtype=np.float64, copy=False), feat_cols, None


def _fit_gmm_for_k(X: np.ndarray, k: int, cfg: TypeDiscoveryConfig) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=int(k),
        covariance_type=cfg.gmm_covariance_type,
        n_init=cfg.gmm_n_init,
        max_iter=cfg.gmm_max_iter,
        random_state=cfg.gmm_random_state,
    )
    gmm.fit(X)
    return gmm


def select_gmm_k(X: np.ndarray, cfg: TypeDiscoveryConfig) -> Tuple[int, pd.DataFrame]:
    rows = []
    n = X.shape[0]

    sample_n = int(min(cfg.silhouette_sample_size, n))
    sample_idx = None
    if cfg.compute_silhouette and sample_n >= 2000:
        rng = np.random.RandomState(cfg.gmm_random_state)
        sample_idx = rng.choice(n, size=sample_n, replace=False)

    for k in range(cfg.gmm_k_min, cfg.gmm_k_max + 1):
        gmm = _fit_gmm_for_k(X, k, cfg)
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))

        sil = np.nan
        if sample_idx is not None:
            labels = gmm.predict(X[sample_idx])
            if len(np.unique(labels)) >= 2:
                sil = float(silhouette_score(X[sample_idx], labels))

        rows.append({"k": k, "bic": bic, "aic": aic, "silhouette": sil})

    df_scores = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_k = int(df_scores.loc[df_scores["bic"].idxmin(), "k"])
    return best_k, df_scores


def fit_gmm_best(X: np.ndarray, cfg: TypeDiscoveryConfig) -> Tuple[GaussianMixture, int, pd.DataFrame]:
    best_k, scores = select_gmm_k(X, cfg)
    gmm = _fit_gmm_for_k(X, best_k, cfg)
    return gmm, best_k, scores


def fit_hdbscan(X: np.ndarray, cfg: TypeDiscoveryConfig):
    if not _HDBSCAN_AVAILABLE:
        raise RuntimeError("hdbscan not available. pip install hdbscan")

    cl = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdb_min_cluster_size,
        min_samples=cfg.hdb_min_samples,
        metric=cfg.hdb_metric,
        cluster_selection_method=cfg.hdb_cluster_selection_method,
        prediction_data=True,
    )
    labels = cl.fit_predict(X)  # noise = -1
    return cl, labels


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


def save_type_package(
    out_dir: str,
    algo: str,
    model: Any,
    scaler: Optional[StandardScaler],
    feature_cols: List[str],
    cfg: TypeDiscoveryConfig,
    extra: Dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    dump(model, os.path.join(out_dir, f"{algo}_model.joblib"))
    if scaler is not None:
        dump(scaler, os.path.join(out_dir, f"{algo}_scaler.joblib"))

    schema = {"algo": algo, "feature_cols": feature_cols, "cfg": asdict(cfg)}
    with open(os.path.join(out_dir, "type_feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "type_discovery_report.json"), "w", encoding="utf-8") as f:
        json.dump(extra, f, ensure_ascii=False, indent=2)
