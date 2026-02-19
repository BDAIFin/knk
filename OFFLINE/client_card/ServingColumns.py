import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "fraud"
AMT_COL = "log_abs_amount"



# Stage2 CORE KEEP FEATURES 

STAGE2_KEEP_COLS = [
    # Fraud history
    "card_fraud_last3",
    "client_fraud_last3",

    # Amount anomaly
    "amount_deviation",

    # Merchant novelty + interaction
    "client_merchant_is_new",
    "card_merchant_is_new",
    "mccnew_x_velocity",
    "dev_x_mccnew",

    # Behavior pattern
    "client_mcc_repeat_cnt_last5",
    "card_mcc_change_cnt_last5",
    "merchant_change_cnt_last5",

    # Velocity context
    "client_tx_1h_avg_prev",
    "card_velocity_spike_ratio",

    # Time context
    "tx_hour",
    "hour_sin",
    "hour_cos",

    # Demographic context
    "current_age",
    "log_yearly_income",
]


# -----------------------------
# Utils
# -----------------------------
def ensure_sorted(df: pd.DataFrame, keys) -> pd.DataFrame:
    # mergesort: stable sort (중요)
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Artifacts
# -----------------------------
def fit_artifacts(
    df_train: pd.DataFrame,
    amt_q: float = 0.9,
    mcc_min_cnt: int = 1000,
    mcc_rate_mult: float = 3.0,
    high_risk_days=(0, 4, 6),
) -> dict:
    # CORE_STAGE2에서는 실제로 사용하지 않아도,
    # artifacts 저장/재현성 유지 위해 남겨둠.
    thr_amt = float(df_train[AMT_COL].quantile(amt_q))
    base_rate = float(df_train[LABEL].mean())

    mcc_stats = (
        df_train.groupby("mcc")[LABEL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "tx_count"})
    )

    highrisk_mcc = mcc_stats[
        (mcc_stats["tx_count"] >= mcc_min_cnt)
        & (mcc_stats["fraud_rate"] >= base_rate * mcc_rate_mult)
    ].index.tolist()

    return {
        "amt_q": amt_q,
        "thr_amt": thr_amt,
        "base_rate": base_rate,
        "mcc_min_cnt": mcc_min_cnt,
        "mcc_rate_mult": mcc_rate_mult,
        "highrisk_mcc": highrisk_mcc,
        "high_risk_days": list(high_risk_days),
    }



# Fast 1h-window counts

def feature_client_tx_1h(df: pd.DataFrame) -> pd.Series:
    df = ensure_sorted(df, ["client_id", "date"])
    cid = df["client_id"].to_numpy()
    t = df["date"].to_numpy(dtype="datetime64[s]").astype(np.int64)

    out = np.empty(len(df), dtype=np.int32)

    n = len(df)
    start = 0
    while start < n:
        end = start + 1
        while end < n and cid[end] == cid[start]:
            end += 1

        tt = t[start:end]
        left = np.searchsorted(tt, tt - 3600, side="left")
        out[start:end] = (np.arange(end - start) - left + 1).astype(np.int32)

        start = end

    return pd.Series(out, index=df.index, name="client_tx_1h")


def feature_card_tx_1h(df: pd.DataFrame) -> pd.Series:
    df = ensure_sorted(df, ["card_id", "date"])
    cid = df["card_id"].to_numpy()
    t = df["date"].to_numpy(dtype="datetime64[s]").astype(np.int64)

    out = np.empty(len(df), dtype=np.int32)

    n = len(df)
    start = 0
    while start < n:
        end = start + 1
        while end < n and cid[end] == cid[start]:
            end += 1

        tt = t[start:end]
        left = np.searchsorted(tt, tt - 3600, side="left")
        out[start:end] = (np.arange(end - start) - left + 1).astype(np.int32)

        start = end

    return pd.Series(out, index=df.index, name="card_tx_1h")


# Feature builder (CORE only)

def build_features(df: pd.DataFrame, artifacts: dict | None = None) -> pd.DataFrame:
    df = df.copy()

    # row 정합성 보장용 id (정렬을 여러 번 해도 최종에 복원)
    df["_row_id"] = np.arange(len(df), dtype=np.int32)


    need_cols = ["id", "client_id", "card_id", "merchant_id", "mcc", "date", "tx_hour", AMT_COL, "current_age", "log_yearly_income"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required raw cols: {missing}")

    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("date column has NaT after to_datetime; check raw data.")


    df["current_age"] = pd.to_numeric(df["current_age"], downcast="integer")
    df["log_yearly_income"] = pd.to_numeric(df["log_yearly_income"], downcast="float")


    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    df["hour_sin"] = np.sin(2 * np.pi * df["tx_hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["tx_hour"] / 24).astype("float32")


    # 2) MCC repeat cnt (client) + is_new (internal)

    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    df["client_mcc_prior_count"] = df.groupby(["client_id", "mcc"], sort=False).cumcount().astype("int32")
    df["client_mcc_is_new"] = (df["client_mcc_prior_count"] == 0).astype("int8")

    g_mcc = df.groupby("client_id", sort=False)["mcc"]
    m1 = df["mcc"].eq(g_mcc.shift(1))
    m2 = df["mcc"].eq(g_mcc.shift(2))
    m3 = df["mcc"].eq(g_mcc.shift(3))
    m4 = df["mcc"].eq(g_mcc.shift(4))
    m5 = df["mcc"].eq(g_mcc.shift(5))

    df["client_mcc_repeat_cnt_last5"] = (
        m1.fillna(False).astype("int8")
        + m2.fillna(False).astype("int8")
        + m3.fillna(False).astype("int8")
        + m4.fillna(False).astype("int8")
        + m5.fillna(False).astype("int8")
    ).astype("int8")

    # 3) card MCC change cnt (KEEP)

    df = ensure_sorted(df, ["card_id", "date", "_row_id"])
    g_card_mcc = df.groupby("card_id", sort=False)["mcc"]
    prev1c = g_card_mcc.shift(1)
    prev2c = g_card_mcc.shift(2)
    prev3c = g_card_mcc.shift(3)
    prev4c = g_card_mcc.shift(4)
    prev5c = g_card_mcc.shift(5)

    df["card_mcc_change_cnt_last5"] = (
        prev1c.ne(prev2c).fillna(False).astype("int8")
        + prev2c.ne(prev3c).fillna(False).astype("int8")
        + prev3c.ne(prev4c).fillna(False).astype("int8")
        + prev4c.ne(prev5c).fillna(False).astype("int8")
    ).astype("int8")


    # 4) Merchant novelty + change cnt (KEEP)

    # novelty
    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    df["client_merchant_is_new"] = (
        df.groupby(["client_id", "merchant_id"], sort=False).cumcount().eq(0).astype("int8")
    )

    df = ensure_sorted(df, ["card_id", "date", "_row_id"])
    df["card_merchant_is_new"] = (
        df.groupby(["card_id", "merchant_id"], sort=False).cumcount().eq(0).astype("int8")
    )

    # change cnt on card (rolling)
    prev_merchant = df.groupby("card_id", sort=False)["merchant_id"].shift(1)
    df["_merchant_changed"] = df["merchant_id"].ne(prev_merchant).fillna(True).astype("int8")

    df["merchant_change_cnt_last5"] = (
        df.groupby("card_id", sort=False)["_merchant_changed"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .astype("int8")
    )
    df.drop(columns=["_merchant_changed"], inplace=True)

    # 5) Velocity (1h) (KEEP)

    df_sorted_client = ensure_sorted(df[["_row_id", "client_id", "date"]], ["client_id", "date", "_row_id"])
    client_tx_1h = feature_client_tx_1h(df_sorted_client.rename(columns={"_row_id": "_row_id"}))
    df_sorted_client["client_tx_1h"] = client_tx_1h.to_numpy()

    df_sorted_client["client_tx_1h_shift"] = df_sorted_client.groupby("client_id", sort=False)["client_tx_1h"].shift(1)
    df_sorted_client["client_tx_1h_cumsum"] = (
        df_sorted_client["client_tx_1h_shift"].fillna(0).groupby(df_sorted_client["client_id"], sort=False).cumsum()
    ).astype("float32")
    df_sorted_client["client_tx_cnt_past"] = df_sorted_client.groupby("client_id", sort=False).cumcount().astype("int32")

    df_sorted_client["client_tx_1h_avg_prev"] = np.where(
        df_sorted_client["client_tx_cnt_past"] > 0,
        df_sorted_client["client_tx_1h_cumsum"] / df_sorted_client["client_tx_cnt_past"],
        df_sorted_client["client_tx_1h"],
    ).astype("float32")

    df = df.merge(
        df_sorted_client[["_row_id", "client_tx_1h_avg_prev", "client_tx_1h"]],
        on="_row_id",
        how="left",
        validate="one_to_one",
    )

    df_sorted_card = ensure_sorted(df[["_row_id", "card_id", "date"]], ["card_id", "date", "_row_id"])
    card_tx_1h = feature_card_tx_1h(df_sorted_card.rename(columns={"_row_id": "_row_id"}))
    df_sorted_card["card_tx_1h"] = card_tx_1h.to_numpy()

    df_sorted_card["card_tx_1h_shift"] = df_sorted_card.groupby("card_id", sort=False)["card_tx_1h"].shift(1)
    df_sorted_card["card_tx_1h_cumsum"] = (
        df_sorted_card["card_tx_1h_shift"].fillna(0).groupby(df_sorted_card["card_id"], sort=False).cumsum()
    ).astype("float32")
    df_sorted_card["card_tx_cnt_past"] = df_sorted_card.groupby("card_id", sort=False).cumcount().astype("int32")

    df_sorted_card["card_tx_1h_avg_prev"] = np.where(
        df_sorted_card["card_tx_cnt_past"] > 0,
        df_sorted_card["card_tx_1h_cumsum"] / df_sorted_card["card_tx_cnt_past"],
        df_sorted_card["card_tx_1h"],
    ).astype("float32")

    df = df.merge(
        df_sorted_card[["_row_id", "card_tx_1h", "card_tx_1h_avg_prev"]],
        on="_row_id",
        how="left",
        validate="one_to_one",
    )

    df["card_velocity_spike_ratio"] = (df["card_tx_1h"] / (df["card_tx_1h_avg_prev"] + 1e-6)).astype("float32")

    # -------------------------
    # 6) Amount deviation (KEEP) + dev_x_mccnew (KEEP)
    # -------------------------
    # client avg amount prev (dev_x_mccnew용)
    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    df["_amt_shift"] = df.groupby("client_id", sort=False)[AMT_COL].shift(1)
    df["_amt_cumsum"] = df["_amt_shift"].fillna(0).groupby(df["client_id"], sort=False).cumsum().astype("float32")
    df["_amt_cnt_past"] = df.groupby("client_id", sort=False).cumcount().astype("int32")
    df["_client_avg_amt_prev"] = np.where(
        df["_amt_cnt_past"] > 0, df["_amt_cumsum"] / df["_amt_cnt_past"], df[AMT_COL]
    ).astype("float32")

    # amount_deviation
    client_amt_mean = df.groupby("client_id", sort=False)[AMT_COL].transform("mean")
    client_amt_std = df.groupby("client_id", sort=False)[AMT_COL].transform("std")
    df["amount_deviation"] = ((df[AMT_COL] - client_amt_mean) / (client_amt_std + 1e-6)).astype("float32")

    # -------------------------
    # 7) Fraud history last3 (KEEP)
    # -------------------------
    if LABEL not in df.columns:
        raise KeyError(
            "LABEL column (fraud) not found. "
            "This builder is TRAIN-time feature builder. "
            "For SERVING, fraud-history features must come from confirmed fraud event logs."
        )

    f1 = df.groupby("card_id", sort=False)[LABEL].shift(1)
    f2 = df.groupby("card_id", sort=False)[LABEL].shift(2)
    f3 = df.groupby("card_id", sort=False)[LABEL].shift(3)
    df["card_fraud_last3"] = (f1.fillna(0).astype("int8") + f2.fillna(0).astype("int8") + f3.fillna(0).astype("int8")).astype("int8")

    f1 = df.groupby("client_id", sort=False)[LABEL].shift(1)
    f2 = df.groupby("client_id", sort=False)[LABEL].shift(2)
    f3 = df.groupby("client_id", sort=False)[LABEL].shift(3)
    df["client_fraud_last3"] = (f1.fillna(0).astype("int8") + f2.fillna(0).astype("int8") + f3.fillna(0).astype("int8")).astype("int8")

    # -------------------------
    # 8) Interactions (KEEP)
    # -------------------------
    df["_velocity_spike_ratio"] = (df["client_tx_1h"] / (df["client_tx_1h_avg_prev"] + 1e-6)).astype("float32")
    df["mccnew_x_velocity"] = (df["client_mcc_is_new"].astype("float32") * df["_velocity_spike_ratio"]).astype("float32")

    # dev_x_mccnew
    eps = 1e-6
    K = 10
    df["_client_recent_avg_amt"] = (
        df.groupby("client_id", sort=False)[AMT_COL]
        .shift(1)
        .rolling(K, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    df["_client_recent_avg_amt"] = df["_client_recent_avg_amt"].fillna(df["_client_avg_amt_prev"]).astype("float32")
    df["_amount_vs_recent_window_avg"] = (df[AMT_COL] / (df["_client_recent_avg_amt"] + eps)).astype("float32")
    df["_log_amount_vs_recent_window_avg"] = np.log1p(df["_amount_vs_recent_window_avg"]).astype("float32")

    df["dev_x_mccnew"] = (df["_log_amount_vs_recent_window_avg"] * df["client_mcc_is_new"]).astype("float32")


    # 9) row 순서 복원 + 컬럼 슬림

    df = df.sort_values("_row_id", kind="mergesort").reset_index(drop=True)

    # 중간 컬럼 제거 (필요한 것만 남김)
    keep = set(["id"] + STAGE2_KEEP_COLS + [LABEL, "_row_id"])
    df_out = df[[c for c in df.columns if c in keep]].copy()

    # _row_id는 최종엔 필요 없으면 제거
    df_out.drop(columns=["_row_id"], inplace=True)

    # dtype 정리
    df_out[LABEL] = df_out[LABEL].astype("int8")
    for c in df_out.columns:
        if c != LABEL and pd.api.types.is_float_dtype(df_out[c]):
            df_out[c] = pd.to_numeric(df_out[c], downcast="float")
        if c in ("current_age",) and pd.api.types.is_integer_dtype(df_out[c]):
            df_out[c] = pd.to_numeric(df_out[c], downcast="integer")

    return df_out


# -----------------------------
# Make model df: KEEP-only 방식
# -----------------------------
def make_df_model(df_feat: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in STAGE2_KEEP_COLS if c not in df_feat.columns]
    if missing:
        raise KeyError(f"Missing required Stage2 cols: {missing}")

    df_model = df_feat[["id"] + STAGE2_KEEP_COLS + [LABEL]].copy()
    df_model[LABEL] = df_model[LABEL].astype("int8")

    for c in df_model.columns:
        if c != LABEL and pd.api.types.is_float_dtype(df_model[c]):
            df_model[c] = pd.to_numeric(df_model[c], downcast="float")
        if c == "current_age":
            df_model[c] = pd.to_numeric(df_model[c], downcast="integer")

    return df_model


def sanity_report(df_model: pd.DataFrame) -> None:
    cols = [c for c in df_model.columns if c != LABEL]
    extra = [c for c in cols if c not in STAGE2_KEEP_COLS]
    missing = [c for c in STAGE2_KEEP_COLS if c not in cols]
    print("=== Sanity Check ===")
    print("n_cols(except label):", len(cols))
    print("missing:", missing)
    print("extra:", extra)
    print("col_order_ok:", cols == STAGE2_KEEP_COLS)


def main():
    in_path = "DATA/dataset/train_stage2"          # input parquet
    out_parquet = "DATA/dataset/stage2_model_df"   # output parquet (KEEP-only)
    artifacts_path = "DATA/artifacts/stage2_artifacts.json"

    df = pd.read_parquet(in_path)

    artifacts = fit_artifacts(df)
    save_json(artifacts, artifacts_path)

    # CORE features only (already slim)
    df_feat = build_features(df, artifacts)
    df_model = make_df_model(df_feat)

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_model.to_parquet(out_parquet)

    print("saved:", out_parquet, "| shape:", df_model.shape)
    sanity_report(df_model)


if __name__ == "__main__":
    main()
