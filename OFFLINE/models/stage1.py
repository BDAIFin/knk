from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import pandas as pd
from joblib import dump

from tqdm.auto import tqdm

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

# ---- optional deps
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HGB_AVAILABLE = True
except Exception:
    _HGB_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except Exception:
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except Exception:
    _LGBM_AVAILABLE = False



# Config

@dataclass
class Stage1Config:
    label_col: str = "fraud"
    drop_cols: Tuple[str, ...] = ("client_id", "card_id", "merchant_id")
    valid_ratio: float = 0.21

    target_recall: float = 0.70
    min_threshold: float = 0.0
    max_threshold: float = 1.0

    # model_name: "logit" | "hgb" | "xgb" | "lgbm"
    model_name: str = "logit"

    # --- logit
    logit_C: float = 1.0
    logit_max_iter: int = 200
    logit_n_jobs: int = -1

    # --- hgb
    hgb_max_depth: int = 6
    hgb_max_iter: int = 200
    hgb_learning_rate: float = 0.06

    # --- preprocessing 
    use_scaler_for_xgb: bool = False
    use_scaler_for_lgbm: bool = False

    # --- XGB
    xgb_n_estimators: int = 600
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 1.0
    xgb_min_child_weight: float = 1.0
    xgb_gamma: float = 0.0
    xgb_tree_method: str = "hist"
    xgb_n_jobs: int = -1
    xgb_random_state: int = 42

    # --- LGBM
    lgbm_n_estimators: int = 800
    lgbm_num_leaves: int = 63
    lgbm_learning_rate: float = 0.05
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    lgbm_min_child_samples: int = 40
    lgbm_reg_lambda: float = 0.0
    lgbm_n_jobs: int = -1
    lgbm_random_state: int = 42


# Split / Schema

def _make_time_index(df: pd.DataFrame) -> pd.Series:
    # ✅ FIX: tx_day 누락으로 time_split이 실제 시간순이 아니게 섞이는 문제 방지
    # year-month-day-hour 단위로 증가하도록 안전한 자리수로 구성
    return (
        df["tx_year"].astype(np.int64) * 1000000
        + df["tx_month"].astype(np.int64) * 10000
        + df["tx_day"].astype(np.int64) * 100
        + df["tx_hour"].astype(np.int64)
    )

def time_split(df: pd.DataFrame, cfg: Stage1Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.copy()
    df2["_t"] = _make_time_index(df2)
    df2 = df2.sort_values(["_t"]).reset_index(drop=True)

    n = len(df2)
    cut = int(np.floor(n * (1.0 - cfg.valid_ratio)))
    cut = max(1, min(cut, n - 1))

    train_df = df2.iloc[:cut].drop(columns=["_t"])
    valid_df = df2.iloc[cut:].drop(columns=["_t"])
    return train_df, valid_df

def build_feature_columns(df: pd.DataFrame, cfg: Stage1Config) -> List[str]:
    return [c for c in df.columns if c != cfg.label_col and c not in cfg.drop_cols]

def make_xy(df: pd.DataFrame, feature_cols: List[str], label_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_cols].copy()
    y = df[label_col].astype(np.int8).to_numpy()
    return X, y



# Threshold selection

def choose_threshold_by_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
) -> Dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision/recall 길이가 thresholds보다 1 더 김
    prec_t = precision[1:]
    rec_t = recall[1:]
    thr_t = thresholds

    # threshold 범위 제한
    in_range = (thr_t >= float(min_threshold)) & (thr_t <= float(max_threshold))
    if np.any(in_range):
        prec_t = prec_t[in_range]
        rec_t = rec_t[in_range]
        thr_t = thr_t[in_range]

    if len(thr_t) == 0:
        # 범위가 너무 좁아 후보가 사라진 경우 fallback
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        prec_t = precision[1:]
        rec_t = recall[1:]
        thr_t = thresholds

    ok = rec_t >= target_recall
    if not np.any(ok):
        best = int(np.argmax(rec_t))
        return {
            "threshold": float(thr_t[best]),
            "precision": float(prec_t[best]),
            "recall": float(rec_t[best]),
            "note": "target_recall_not_met; picked max_recall_point"
        }

    cand_idx = np.where(ok)[0]
    best_prec = np.max(prec_t[cand_idx])
    bests = cand_idx[prec_t[cand_idx] == best_prec]
    best = int(bests[np.argmax(thr_t[bests])])
    return {
        "threshold": float(thr_t[best]),
        "precision": float(prec_t[best]),
        "recall": float(rec_t[best]),
        "note": "picked_best_precision_under_recall_constraint"
    }


# Model builders

def _scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / max(pos, 1.0)

def build_model_pipeline(
    feature_cols: List[str],
    cfg: Stage1Config,
    y_train: Optional[np.ndarray] = None,
) -> Pipeline:
    numeric_features = feature_cols
    name = cfg.model_name.lower()

    # --- LOGIT
    if name == "logit":
        pre = ColumnTransformer(
            transformers=[("num", StandardScaler(with_mean=True, with_std=True), numeric_features)],
            remainder="drop",
        )
        clf = LogisticRegression(
            C=cfg.logit_C,
            max_iter=cfg.logit_max_iter,
            n_jobs=cfg.logit_n_jobs,
            class_weight="balanced",
            solver="lbfgs",
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    # --- HGB
    if name == "hgb":
        if not _HGB_AVAILABLE:
            raise RuntimeError("HistGradientBoostingClassifier not available")
        pre = ColumnTransformer(
            transformers=[("num", "passthrough", numeric_features)],
            remainder="drop",
        )
        clf = HistGradientBoostingClassifier(
            max_depth=cfg.hgb_max_depth,
            max_iter=cfg.hgb_max_iter,
            learning_rate=cfg.hgb_learning_rate,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    # --- XGB
    if name == "xgb":
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost not available. pip install xgboost")

        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=True, with_std=True), numeric_features)
            ] if cfg.use_scaler_for_xgb else [
                ("num", "passthrough", numeric_features)
            ],
            remainder="drop",
        )

        spw = _scale_pos_weight(y_train) if y_train is not None else 1.0

        clf = XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            reg_lambda=cfg.xgb_reg_lambda,
            min_child_weight=cfg.xgb_min_child_weight,
            gamma=cfg.xgb_gamma,
            tree_method=cfg.xgb_tree_method,
            n_jobs=cfg.xgb_n_jobs,
            random_state=cfg.xgb_random_state,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=spw,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    # --- LGBM
    if name == "lgbm":
        if not _LGBM_AVAILABLE:
            raise RuntimeError("lightgbm not available. pip install lightgbm")

        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=True, with_std=True), numeric_features)
            ] if cfg.use_scaler_for_lgbm else [
                ("num", "passthrough", numeric_features)
            ],
            remainder="drop",
        )

        clf = LGBMClassifier(
            n_estimators=cfg.lgbm_n_estimators,
            num_leaves=cfg.lgbm_num_leaves,
            learning_rate=cfg.lgbm_learning_rate,
            subsample=cfg.lgbm_subsample,
            colsample_bytree=cfg.lgbm_colsample_bytree,
            min_child_samples=cfg.lgbm_min_child_samples,
            reg_lambda=cfg.lgbm_reg_lambda,
            n_jobs=cfg.lgbm_n_jobs,
            random_state=cfg.lgbm_random_state,
            objective="binary",
            class_weight="balanced",
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    raise ValueError(f"Unknown model_name: {cfg.model_name}")



# Train

def train_stage1(df: pd.DataFrame, cfg: Stage1Config) -> Dict[str, Any]:
    pbar = tqdm(total=6, desc=f"Stage1({cfg.model_name})", leave=True)

    # 1) Split
    train_df, valid_df = time_split(df, cfg)
    pbar.update(1)

    # 2) Feature schema
    feature_cols = build_feature_columns(train_df, cfg)
    X_tr, y_tr = make_xy(train_df, feature_cols, cfg.label_col)
    X_va, y_va = make_xy(valid_df, feature_cols, cfg.label_col)
    pbar.update(1)

    # 3) Build + fit
    pipe = build_model_pipeline(feature_cols, cfg, y_train=y_tr)
    pipe.fit(X_tr, y_tr)
    pbar.update(1)

    # 4) Predict
    proba_va = pipe.predict_proba(X_va)[:, 1]
    pbar.update(1)

    # 5) Metrics
    pr_auc = float(average_precision_score(y_va, proba_va))
    roc_auc = float(roc_auc_score(y_va, proba_va))
    pbar.update(1)

    # 6) Threshold
    thr_info = choose_threshold_by_recall(
        y_true=y_va,
        y_proba=proba_va,
        target_recall=cfg.target_recall,
        min_threshold=cfg.min_threshold,
        max_threshold=cfg.max_threshold,
    )
    thr = float(thr_info["threshold"])

    yhat_va = (proba_va >= thr).astype(np.int8)

    cm = confusion_matrix(y_va, yhat_va).tolist()
    # 기존 포맷 유지: dict로 저장
    cls = classification_report(y_va, yhat_va, digits=4, output_dict=True)

    feature_schema = {
        "label_col": cfg.label_col,
        "drop_cols": list(cfg.drop_cols),
        "feature_cols": feature_cols,
        "dtypes": {c: str(df[c].dtype) for c in feature_cols},
        "model_name": cfg.model_name,
        "cfg": asdict(cfg),
    }

    report = {
        "model_name": cfg.model_name,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "pos_rate_train": float(train_df[cfg.label_col].mean()),
        "pos_rate_valid": float(valid_df[cfg.label_col].mean()),
        "pr_auc_valid": pr_auc,
        "roc_auc_valid": roc_auc,
        "threshold": thr_info,
        "confusion_matrix_valid": cm,
        "classification_report_valid": cls,
    }

    pbar.update(1)
    pbar.close()

    return {
        "model": pipe,
        "feature_cols": feature_cols,
        "feature_schema": feature_schema,
        "threshold": thr_info,
        "report": report,
    }


# Save artifacts
def save_stage1_artifacts(
    model: Any,
    feature_schema: Dict[str, Any],
    threshold: Dict[str, Any],
    out_dir: str,
    model_name: str = "stage1_model.joblib",
    schema_name: str = "feature_schema.json",
    threshold_name: str = "threshold.json",
) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)

    dump(model, os.path.join(out_dir, model_name))

    with open(os.path.join(out_dir, schema_name), "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, threshold_name), "w", encoding="utf-8") as f:
        json.dump(threshold, f, ensure_ascii=False, indent=2)
