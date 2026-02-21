import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "fraud"
AMT_COL = "log_abs_amount"

# ==========================================
# Stage1 Features (ONLY buildable from 24cols)
# ==========================================
STAGE1_FEATURES_AVAILABLE = [
    # 1) Amount
    "log_abs_amount",
    # "high_amount",                      # DROP
    # "amount_vs_client_avg_diff",        # 중복 위험 (선택적 DROP 가능)
    "amount_deviation",

    # 2) Error
    # "has_error",                        # DROP
    # "err_bad_cvv",                      # DROP
    # "err_bad_card_number",              # DROP
    # "err_bad_expiration",               # DROP
    # "card_error_last1",                 # DROP
    # "client_error_last1",               # DROP

    # 3) Fraud history
    # "card_fraud_last1",                 # card_fraud_last3로 대체
    "client_fraud_last1",
    "card_fraud_last3",

    # 4) Time
    "tx_hour",
    # "tx_month",                         # DROP
    "hour_cos",
    "is_highrisk_weekday",

    # 5) Velocity
    "seconds_since_prev_tx",
    "card_velocity_spike_ratio",

    # 6) MCC / Merchant novelty
    # "card_mcc_is_new",                  # DROP
    "client_mcc_is_new",
    # "card_merchant_is_new",             # DROP
    "client_merchant_is_new",
    # "merchant_is_new_x_has_error",      # DROP
]

# =========================
# Utils
# =========================
def ensure_sorted(df: pd.DataFrame, keys) -> pd.DataFrame:
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)

def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _tx_1h_count_by_entity(df_sorted: pd.DataFrame, entity_col: str) -> np.ndarray:
    """
    df_sorted must be sorted by [entity_col, date].
    counts tx in the last 3600s INCLUDING current tx.
    O(N) with searchsorted per-entity.
    """
    ent = df_sorted[entity_col].to_numpy()
    t = df_sorted["date"].to_numpy(dtype="datetime64[s]").astype(np.int64)

    out = np.empty(len(df_sorted), dtype=np.int32)
    n = len(df_sorted)

    start = 0
    while start < n:
        end = start + 1
        while end < n and ent[end] == ent[start]:
            end += 1

        tt = t[start:end]
        left = np.searchsorted(tt, tt - 3600, side="left")
        out[start:end] = (np.arange(end - start) - left + 1).astype(np.int32)

        start = end

    return out

# =========================
# Artifacts (ONLY buildable)
# =========================
def fit_stage1_artifacts_minimal(
    df_train: pd.DataFrame,
    high_amount_q: float = 0.90,
    high_risk_days=(0, 4, 6),  # 월(0) 금(4) 일(6) 같은 식으로
) -> dict:
    high_amount_thr = float(df_train[AMT_COL].quantile(high_amount_q))
    base_rate = float(df_train[LABEL].mean())
    return {
        "base_rate": base_rate,
        "high_amount_q": float(high_amount_q),
        "high_amount_thr": float(high_amount_thr),
        "high_risk_days": list(high_risk_days),
    }

# =========================
# Feature Builder (Stage1)
# =========================
def build_stage1_dataset_from_min_df(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy()

    # ensure datetime
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("date has NaT after to_datetime")

    # ensure required
    need = ["id","date","client_id","card_id","merchant_id","mcc",LABEL,"has_error",
            "err_bad_cvv","err_bad_card_number","err_bad_expiration","tx_hour","tx_month","weekday",AMT_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required cols: {missing}")

    # type
    df["mcc"] = df["mcc"].astype(str)

    # -------------------------
    # (A) Amount features
    # -------------------------
    thr = float(artifacts["high_amount_thr"])
    df["high_amount"] = (df[AMT_COL] >= thr).astype("int8")

    # amount_vs_client_avg_diff (past mean, excluding current)
    df = ensure_sorted(df, ["client_id", "date"])
    grp_c = df.groupby("client_id", sort=False)[AMT_COL]

    amt_shift = grp_c.shift(1)
    amt_cumsum = amt_shift.fillna(0).groupby(df["client_id"], sort=False).cumsum()
    cnt_past = df.groupby("client_id", sort=False).cumcount()

    client_avg_amt_prev = np.where(
        cnt_past.to_numpy() > 0,
        (amt_cumsum.to_numpy() / cnt_past.to_numpy()),
        df[AMT_COL].to_numpy(),
    ).astype("float32")

    df["amount_vs_client_avg_diff"] = (df[AMT_COL].to_numpy() - client_avg_amt_prev).astype("float32")

    # amount_deviation (per-client z-score; offline용)
    client_mean = grp_c.transform("mean").astype("float32")
    client_std = grp_c.transform("std").fillna(0).astype("float32")
    df["amount_deviation"] = ((df[AMT_COL] - client_mean) / (client_std + 1e-6)).astype("float32")

    # -------------------------
    # (B) Error lag features
    # -------------------------
    df = ensure_sorted(df, ["card_id", "date"])
    g_card_err = df.groupby("card_id", sort=False)["has_error"]
    df["card_error_last1"] = g_card_err.shift(1).fillna(0).astype("int8")

    df = ensure_sorted(df, ["client_id", "date"])
    g_client_err = df.groupby("client_id", sort=False)["has_error"]
    df["client_error_last1"] = g_client_err.shift(1).fillna(0).astype("int8")

    # -------------------------
    # (C) Fraud history (shift only)
    # -------------------------
    df = ensure_sorted(df, ["card_id", "date"])
    g_card_f = df.groupby("card_id", sort=False)[LABEL]
    f1 = g_card_f.shift(1).fillna(0).astype("int8")
    f2 = g_card_f.shift(2).fillna(0).astype("int8")
    f3 = g_card_f.shift(3).fillna(0).astype("int8")
    df["card_fraud_last1"] = f1
    df["card_fraud_last3"] = (f1 + f2 + f3).astype("int8")

    df = ensure_sorted(df, ["client_id", "date"])
    g_client_f = df.groupby("client_id", sort=False)[LABEL]
    df["client_fraud_last1"] = g_client_f.shift(1).fillna(0).astype("int8")

    # -------------------------
    # (D) Time features
    # -------------------------
    df["hour_cos"] = np.cos(2 * np.pi * df["tx_hour"].to_numpy() / 24.0).astype("float32")

    high_risk_days = set(artifacts["high_risk_days"])
    df["is_highrisk_weekday"] = df["weekday"].isin(high_risk_days).astype("int8")

    # -------------------------
    # (E) Velocity features
    # -------------------------
    # seconds_since_prev_tx (per client)
    df = ensure_sorted(df, ["client_id", "date"])
    prev_t = df.groupby("client_id", sort=False)["date"].shift(1)
    df["seconds_since_prev_tx"] = (df["date"] - prev_t).dt.total_seconds().fillna(0).astype("float32")

    # card_velocity_spike_ratio (1h tx count / past avg)
    df = ensure_sorted(df, ["card_id", "date"])
    card_tx_1h = _tx_1h_count_by_entity(df, "card_id").astype("float32")
    df["_card_tx_1h"] = card_tx_1h

    card_tx_1h_shift = df.groupby("card_id", sort=False)["_card_tx_1h"].shift(1).fillna(0)
    card_tx_1h_cumsum = card_tx_1h_shift.groupby(df["card_id"], sort=False).cumsum()
    card_tx_cnt_past = df.groupby("card_id", sort=False).cumcount()

    card_tx_1h_avg_prev = np.where(
        card_tx_cnt_past.to_numpy() > 0,
        (card_tx_1h_cumsum.to_numpy() / card_tx_cnt_past.to_numpy()),
        df["_card_tx_1h"].to_numpy(),
    ).astype("float32")

    df["card_velocity_spike_ratio"] = (df["_card_tx_1h"].to_numpy() / (card_tx_1h_avg_prev + 1e-6)).astype("float32")
    df.drop(columns=["_card_tx_1h"], inplace=True)

    # -------------------------
    # (F) MCC / Merchant novelty
    # -------------------------
    df = ensure_sorted(df, ["client_id", "date"])
    df["_client_mcc_prior"] = df.groupby(["client_id", "mcc"], sort=False).cumcount()
    df["client_mcc_is_new"] = (df["_client_mcc_prior"] == 0).astype("int8")
    df.drop(columns=["_client_mcc_prior"], inplace=True)

    df = ensure_sorted(df, ["card_id", "date"])
    df["_card_mcc_prior"] = df.groupby(["card_id", "mcc"], sort=False).cumcount()
    df["card_mcc_is_new"] = (df["_card_mcc_prior"] == 0).astype("int8")
    df.drop(columns=["_card_mcc_prior"], inplace=True)

    df = ensure_sorted(df, ["client_id", "date"])
    df["client_merchant_is_new"] = (
        df.groupby(["client_id", "merchant_id"], sort=False).cumcount().eq(0).astype("int8")
    )

    df = ensure_sorted(df, ["card_id", "date"])
    df["card_merchant_is_new"] = (
        df.groupby(["card_id", "merchant_id"], sort=False).cumcount().eq(0).astype("int8")
    )

    df["merchant_is_new_x_has_error"] = (df["card_merchant_is_new"] * df["has_error"]).astype("int8")

    # -------------------------
    # Final select
    # -------------------------
    out_cols = ["id", LABEL] + STAGE1_FEATURES_AVAILABLE
    missing = [c for c in out_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns after build: {missing}")

    df_out = df[out_cols].copy()

    # dtype downcast (optional)
    # keep float32
    float32_cols = [
        "log_abs_amount",
        "amount_vs_client_avg_diff",
        "amount_deviation",
        "hour_cos",
        "seconds_since_prev_tx",
        "card_velocity_spike_ratio",
    ]
    for c in float32_cols:
        if c in df_out.columns:
            df_out[c] = df_out[c].astype("float32")

    return df_out


def main():
    in_path = "DATA/dataset/transactions_train"   # <- 너가 실제 저장한 경로로
    out_parquet = "DATA/dataset/train_stage1_min24cols.parquet"
    artifacts_path = "DATA/artifacts/stage1_artifacts_min.json"

    df = pd.read_parquet(in_path)

    # artifacts (minimal)
    artifacts = fit_stage1_artifacts_minimal(df)
    save_json(artifacts, artifacts_path)

    # build
    df_stage1 = build_stage1_dataset_from_min_df(df, artifacts)

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_stage1.to_parquet(out_parquet, index=False)

    mem_mb = df_stage1.memory_usage(deep=True).sum() / 1024**2
    print("saved:", out_parquet)
    print("shape:", df_stage1.shape)
    print("n_features:", len([c for c in df_stage1.columns if c not in ["id", LABEL]]))
    print("mem(MB):", round(mem_mb, 2))
    print("artifacts:", artifacts_path)


if __name__ == "__main__":
    main()