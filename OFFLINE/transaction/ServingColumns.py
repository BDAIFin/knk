import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "fraud"
AMT_COL = "log_abs_amount"

# =========================
# Config: Stage1 Features
# =========================
STAGE1_FEATURES = [
    # 1) Amount
    "log_abs_amount",
    "high_amount",
    "amount_vs_client_avg_diff",
    "amount_deviation",

    # 2) Error
    "has_error",
    "err_bad_cvv",
    "err_bad_card_number",
    "err_bad_expiration",
    "card_error_last1",
    "client_error_last1",

    # 3) Fraud history
    "card_fraud_last1",
    "client_fraud_last1",
    "card_fraud_last3",

    # 4) Time
    "tx_hour",
    "tx_month",
    "hour_cos",
    "is_highrisk_weekday",

    # 5) Velocity
    "seconds_since_prev_tx",
    "card_velocity_spike_ratio",

    # 6) MCC / Merchant
    "mcc_highrisk_90",
    "card_mcc_is_new",
    "client_mcc_is_new",
    "card_merchant_is_new",
    "client_merchant_is_new",
    "merchant_is_new_x_has_error",
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


def fit_stage1_artifacts(
    df_train: pd.DataFrame,
    high_amount_q: float = 0.90,
    mcc_min_cnt: int = 1000,
    mcc_rate_mult: float = 3.0,
    high_risk_days=(0, 4, 6),
) -> dict:
    # high_amount threshold (log_abs_amount quantile)
    high_amount_thr = float(df_train[AMT_COL].quantile(high_amount_q))
    base_rate = float(df_train[LABEL].mean())

    mcc_stats = (
        df_train.groupby("mcc", sort=False)[LABEL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "tx_count"})
    )

    highrisk_mcc = mcc_stats[
        (mcc_stats["tx_count"] >= mcc_min_cnt)
        & (mcc_stats["fraud_rate"] >= base_rate * mcc_rate_mult)
    ].index.tolist()

    return {
        "base_rate": base_rate,
        "high_amount_q": high_amount_q,
        "high_amount_thr": high_amount_thr,
        "mcc_min_cnt": mcc_min_cnt,
        "mcc_rate_mult": mcc_rate_mult,
        "highrisk_mcc": highrisk_mcc,
        "high_risk_days": list(high_risk_days),
    }


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


# Feature Builder

def build_stage1_dataset(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy()

    # ensure datetime
    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # -------------------------
    # (A) Amount features
    # -------------------------
    # high_amount
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

    # amount_deviation (z-score using per-client mean/std)
    # (offline에서는 transform OK. online이면 state 필요)
    client_mean = grp_c.transform("mean").astype("float32")
    client_std = grp_c.transform("std").fillna(0).astype("float32")
    df["amount_deviation"] = ((df[AMT_COL] - client_mean) / (client_std + 1e-6)).astype("float32")

    # -------------------------
    # (B) Error lag features
    # -------------------------
    # card_error_last1
    df = ensure_sorted(df, ["card_id", "date"])
    g_card_err = df.groupby("card_id", sort=False)["has_error"]
    df["card_error_last1"] = g_card_err.shift(1).fillna(0).astype("int8")

    # client_error_last1
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
    # hour_cos
    df["hour_cos"] = np.cos(2 * np.pi * df["tx_hour"].to_numpy() / 24.0).astype("float32")

    # is_highrisk_weekday
    high_risk_days = set(artifacts["high_risk_days"])
    df["is_highrisk_weekday"] = df["weekday"].isin(high_risk_days).astype("int8")

    # -------------------------
    # (E) Velocity features
    # -------------------------
    # seconds_since_prev_tx (per client)
    df = ensure_sorted(df, ["client_id", "date"])
    prev_t = df.groupby("client_id", sort=False)["date"].shift(1)
    df["seconds_since_prev_tx"] = (df["date"] - prev_t).dt.total_seconds().fillna(0).astype("float32")

    # card_velocity_spike_ratio:
    #   - card_tx_1h: last 1 hour count per card (including current)
    #   - card_tx_1h_avg_prev: mean of card_tx_1h over past txs (excluding current)
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
    # (F) MCC / Merchant features
    # -------------------------
    highrisk_mcc = set(artifacts["highrisk_mcc"])
    df["mcc_highrisk_90"] = df["mcc"].isin(highrisk_mcc).astype("int8")

    # client_mcc_is_new / card_mcc_is_new
    df = ensure_sorted(df, ["client_id", "date"])
    df["_client_mcc_prior"] = df.groupby(["client_id", "mcc"], sort=False).cumcount()
    df["client_mcc_is_new"] = (df["_client_mcc_prior"] == 0).astype("int8")
    df.drop(columns=["_client_mcc_prior"], inplace=True)

    df = ensure_sorted(df, ["card_id", "date"])
    df["_card_mcc_prior"] = df.groupby(["card_id", "mcc"], sort=False).cumcount()
    df["card_mcc_is_new"] = (df["_card_mcc_prior"] == 0).astype("int8")
    df.drop(columns=["_card_mcc_prior"], inplace=True)

    # merchant new flags (client/card)
    df = ensure_sorted(df, ["client_id", "date"])
    df["client_merchant_is_new"] = (
        df.groupby(["client_id", "merchant_id"], sort=False).cumcount().eq(0).astype("int8")
    )

    df = ensure_sorted(df, ["card_id", "date"])
    df["card_merchant_is_new"] = (
        df.groupby(["card_id", "merchant_id"], sort=False).cumcount().eq(0).astype("int8")
    )

    # interaction: merchant_is_new_x_has_error
    # (your definition: merchant_is_new == card_merchant_is_new)
    df["merchant_is_new_x_has_error"] = (df["card_merchant_is_new"] * df["has_error"]).astype("int8")

    # -------------------------
    # Final: keep exactly (fraud + STAGE1_FEATURES)
    # -------------------------
    META_COLS = ["id"]

    out_cols = META_COLS + [LABEL] + STAGE1_FEATURES

    missing = [c for c in out_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns after feature build: {missing}")

    df_stage1 = df[out_cols].copy()

    # enforce dtypes (optional but recommended)
    # keep float32 to save memory
    float32_cols = [
        "log_abs_amount",
        "amount_vs_client_avg_diff",
        "amount_deviation",
        "hour_cos",
        "seconds_since_prev_tx",
        "card_velocity_spike_ratio",
    ]
    for c in float32_cols:
        if c in df_stage1.columns:
            df_stage1[c] = df_stage1[c].astype("float32")

    return df_stage1


def main():
    in_path = "DATA/dataset/transactions_train"
    out_parquet = "DATA/dataset/train_stage1"
    artifacts_path = "DATA/artifacts/stage1_artifacts.json"

    df = pd.read_parquet(in_path)

    artifacts = fit_stage1_artifacts(df)
    save_json(artifacts, artifacts_path)

    df_stage1 = build_stage1_dataset(df, artifacts)

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_stage1.to_parquet(out_parquet)

    mem_mb = df_stage1.memory_usage(deep=True).sum() / 1024**2
    print("saved:", out_parquet)
    print("shape:", df_stage1.shape)
    print("cols:", df_stage1.columns.tolist())
    print("mem(MB):", round(mem_mb, 2))
    print("artifacts:", artifacts_path)


if __name__ == "__main__":
    main()
