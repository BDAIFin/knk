import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL = "fraud"
AMT_COL = "log_abs_amount"


def ensure_sorted(df: pd.DataFrame, keys) -> pd.DataFrame:
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fit_artifacts(
    df_train: pd.DataFrame,
    amt_q: float = 0.9,
    mcc_min_cnt: int = 1000,
    mcc_rate_mult: float = 3.0,
    high_risk_days=(0, 4, 6),
) -> dict:
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


def build_features(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy()

    df = ensure_sorted(df, ["client_id", "date"])

    thr = artifacts["thr_amt"]
    df["refund_high_amount"] = (
        (df["is_refund"].astype("int8") == 1) & (df[AMT_COL] > thr)
    ).astype("int8")

    if "amount" in df.columns:
        df.drop(columns=["amount"], inplace=True)

    df = ensure_sorted(df, ["card_id", "date"])
    g_err = df.groupby("card_id", sort=False)["has_error"]

    e1 = g_err.shift(1).fillna(0).astype("int8")
    e2 = g_err.shift(2).fillna(0).astype("int8")
    e3 = g_err.shift(3).fillna(0).astype("int8")
    df["card_error_last1"] = e1
    df["card_error_last3"] = (e1 + e2 + e3).astype("int8")

    df["card_fraud_cum_prev"] = (
        df.groupby("card_id", sort=False)[LABEL]
        .cumsum()
        .shift(1)
        .fillna(0)
        .astype("int32")
    )
    df["card_has_fraud_history"] = (df["card_fraud_cum_prev"] > 0).astype("int8")
    df["card_hist_x_error"] = (df["card_has_fraud_history"] * df["card_error_last1"]).astype(
        "int8"
    )
    df.drop(columns=["card_fraud_cum_prev", "card_has_fraud_history"], inplace=True)

    drop_err = [
        "err_insufficient_balance",
        "err_technical_glitch",
        "err_bad_zipcode",
        "err_bad_pin",
    ]
    df.drop(columns=[c for c in drop_err if c in df.columns], inplace=True)

    df = ensure_sorted(df, ["client_id", "date"])
    df["hour_sin"] = np.sin(2 * np.pi * df["tx_hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["tx_hour"] / 24).astype("float32")

    df["sin_shift"] = (
        df.groupby("client_id", sort=False)["hour_sin"]
        .shift(1)
        .fillna(df["hour_sin"])
        .astype("float32")
    )
    df["cos_shift"] = (
        df.groupby("client_id", sort=False)["hour_cos"]
        .shift(1)
        .fillna(df["hour_cos"])
        .astype("float32")
    )

    df["sin_cumsum"] = df.groupby("client_id", sort=False)["sin_shift"].cumsum()
    df["cos_cumsum"] = df.groupby("client_id", sort=False)["cos_shift"].cumsum()
    df["cnt_past"] = df.groupby("client_id", sort=False).cumcount()

    df["client_sin_mean_past"] = np.where(
        df["cnt_past"] > 0, df["sin_cumsum"] / df["cnt_past"], df["hour_sin"]
    ).astype("float32")
    df["client_cos_mean_past"] = np.where(
        df["cnt_past"] > 0, df["cos_cumsum"] / df["cnt_past"], df["hour_cos"]
    ).astype("float32")

    df["hour_circular_distance"] = np.sqrt(
        (df["hour_sin"] - df["client_sin_mean_past"]) ** 2
        + (df["hour_cos"] - df["client_cos_mean_past"]) ** 2
    ).astype("float32")

    df["client_weekday_prev"] = (
        df.groupby("client_id", sort=False)["weekday"]
        .shift(1)
        .fillna(df["weekday"])
        .astype(df["weekday"].dtype)
    )
    df["client_weekday_match_last1"] = (df["weekday"] == df["client_weekday_prev"]).astype(
        "int8"
    )

    if "cb_Discover" in df.columns and "err_bad_cvv" in df.columns:
        df["discover_x_cvv"] = (
            df["cb_Discover"].astype("int8") * df["err_bad_cvv"].astype("int8")
        ).astype("int8")

    if "is_credit" in df.columns and "is_prepaid" in df.columns:
        card_type = np.select(
            [df["is_credit"] == 1, df["is_prepaid"] == 1],
            ["credit", "debit(prepaid)"],
            default="debit(non_prepaid)",
        )
        df["prepaid_logamount_interaction"] = (
            (card_type == "debit(prepaid)").astype("int8") * df[AMT_COL]
        ).astype("float32")

    highrisk_mcc = set(artifacts["highrisk_mcc"])
    df["mcc_highrisk_90"] = df["mcc"].isin(highrisk_mcc).astype("int8")

    high_risk_days = set(artifacts["high_risk_days"])
    df["is_highrisk_weekday"] = df["weekday"].isin(high_risk_days).astype("int8")

    df = ensure_sorted(df, ["client_id", "date"])
    df["client_mcc_prior_count"] = df.groupby(["client_id", "mcc"], sort=False).cumcount()
    df["client_mcc_is_new"] = (df["client_mcc_prior_count"] == 0).astype("int8")

    g_mcc = df.groupby("client_id", sort=False)["mcc"]
    m1 = df["mcc"].eq(g_mcc.shift(1))
    m2 = df["mcc"].eq(g_mcc.shift(2))
    m3 = df["mcc"].eq(g_mcc.shift(3))
    m4 = df["mcc"].eq(g_mcc.shift(4))
    m5 = df["mcc"].eq(g_mcc.shift(5))

    prev1 = g_mcc.shift(1)
    prev2 = g_mcc.shift(2)
    prev3 = g_mcc.shift(3)
    prev4 = g_mcc.shift(4)
    prev5 = g_mcc.shift(5)

    df["client_mcc_change_cnt_last5"] = (
        prev1.ne(prev2).fillna(False).astype("int8")
        + prev2.ne(prev3).fillna(False).astype("int8")
        + prev3.ne(prev4).fillna(False).astype("int8")
        + prev4.ne(prev5).fillna(False).astype("int8")
    ).astype("int8")

    df["client_mcc_seen_last5"] = (m1 | m2 | m3 | m4 | m5).fillna(False).astype("int8")
    df["client_mcc_repeat_cnt_last5"] = (
        m1.fillna(False).astype("int8")
        + m2.fillna(False).astype("int8")
        + m3.fillna(False).astype("int8")
        + m4.fillna(False).astype("int8")
        + m5.fillna(False).astype("int8")
    ).astype("int8")
    df["client_mcc_repeat_ratio_last5"] = (df["client_mcc_repeat_cnt_last5"] / 5.0).astype(
        "float32"
    )

    df = ensure_sorted(df, ["card_id", "date"])
    df["card_mcc_prior_count"] = df.groupby(["card_id", "mcc"], sort=False).cumcount()
    df["card_mcc_is_new"] = (df["card_mcc_prior_count"] == 0).astype("int8")

    g_card_mcc = df.groupby("card_id", sort=False)["mcc"]
    prev1 = g_card_mcc.shift(1)
    prev2 = g_card_mcc.shift(2)
    prev3 = g_card_mcc.shift(3)
    prev4 = g_card_mcc.shift(4)
    prev5 = g_card_mcc.shift(5)

    df["card_mcc_change_cnt_last5"] = (
        prev1.ne(prev2).fillna(False).astype("int8")
        + prev2.ne(prev3).fillna(False).astype("int8")
        + prev3.ne(prev4).fillna(False).astype("int8")
        + prev4.ne(prev5).fillna(False).astype("int8")
    ).astype("int8")

    df = ensure_sorted(df, ["client_id", "date"])
    df["client_merchant_is_new"] = (
        df.groupby(["client_id", "merchant_id"], sort=False)
        .cumcount()
        .eq(0)
        .astype("int8")
    )
    df["card_merchant_is_new"] = (
        df.groupby(["card_id", "merchant_id"], sort=False)
        .cumcount()
        .eq(0)
        .astype("int8")
    )

    prev_merchant = df.groupby("card_id", sort=False)["merchant_id"].shift(1)
    df["merchant_changed"] = df["merchant_id"].ne(prev_merchant).fillna(True).astype("int8")

    df["merchant_change_cnt_last5"] = (
        df.groupby("card_id", sort=False)["merchant_changed"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .astype("int8")
    )
    df.drop(columns=["merchant_changed"], inplace=True)

    df["merchant_is_new"] = df["card_merchant_is_new"].astype("int8")
    df["merchant_is_new_x_mcc_is_new"] = (
        df["merchant_is_new"].astype("int8") * df["card_mcc_is_new"].astype("int8")
    ).astype("int8")
    df["merchant_is_new_x_has_error"] = (
        df["merchant_is_new"].astype("int8") * df["has_error"].astype("int8")
    ).astype("int8")

    df["prev_tx_time"] = df.groupby("client_id", sort=False)["date"].shift(1)
    df["seconds_since_prev_tx"] = (df["date"] - df["prev_tx_time"]).dt.total_seconds().fillna(0)

    df["log_interval"] = np.log1p(df["seconds_since_prev_tx"]).astype("float32")
    df["log_interval_shift"] = df.groupby("client_id", sort=False)["log_interval"].shift(1)

    df["interval_cumsum"] = (
        df["log_interval_shift"].fillna(0).groupby(df["client_id"]).cumsum()
    ).astype("float32")
    df["interval_cnt_past"] = df.groupby("client_id", sort=False).cumcount()

    df["client_avg_interval_prev"] = np.where(
        df["interval_cnt_past"] > 0,
        df["interval_cumsum"] / df["interval_cnt_past"],
        df["log_interval"],
    ).astype("float32")

    df["client_tx_1h"] = feature_client_tx_1h(df)
    df["client_tx_1h_shift"] = df.groupby("client_id", sort=False)["client_tx_1h"].shift(1)

    df["client_tx_1h_cumsum"] = (
        df["client_tx_1h_shift"].fillna(0).groupby(df["client_id"]).cumsum()
    ).astype("float32")
    df["client_tx_cnt_past"] = df.groupby("client_id", sort=False).cumcount()

    df["client_tx_1h_avg_prev"] = np.where(
        df["client_tx_cnt_past"] > 0,
        df["client_tx_1h_cumsum"] / df["client_tx_cnt_past"],
        df["client_tx_1h"],
    ).astype("float32")

    df["card_tx_1h"] = feature_card_tx_1h(df)
    df["card_tx_1h_shift"] = df.groupby("card_id", sort=False)["card_tx_1h"].shift(1)

    df["card_tx_1h_cumsum"] = (
        df["card_tx_1h_shift"].fillna(0).groupby(df["card_id"]).cumsum()
    ).astype("float32")
    df["card_tx_cnt_past"] = df.groupby("card_id", sort=False).cumcount()

    df["card_tx_1h_avg_prev"] = np.where(
        df["card_tx_cnt_past"] > 0,
        df["card_tx_1h_cumsum"] / df["card_tx_cnt_past"],
        df["card_tx_1h"],
    ).astype("float32")

    df["card_velocity_spike_ratio"] = (
        df["card_tx_1h"] / (df["card_tx_1h_avg_prev"] + 1e-6)
    ).astype("float32")

    df = ensure_sorted(df, ["client_id", "date"])
    df["amt_shift"] = df.groupby("client_id", sort=False)[AMT_COL].shift(1)

    df["amt_cumsum"] = df["amt_shift"].fillna(0).groupby(df["client_id"]).cumsum().astype(
        "float32"
    )
    df["amt_cnt_past"] = df.groupby("client_id", sort=False).cumcount()

    df["client_avg_amt_prev"] = np.where(
        df["amt_cnt_past"] > 0, df["amt_cumsum"] / df["amt_cnt_past"], df[AMT_COL]
    ).astype("float32")

    df["amount_vs_client_avg_diff"] = (df[AMT_COL] - df["client_avg_amt_prev"]).astype(
        "float32"
    )

    f1 = df.groupby("card_id", sort=False)[LABEL].shift(1)
    f2 = df.groupby("card_id", sort=False)[LABEL].shift(2)
    f3 = df.groupby("card_id", sort=False)[LABEL].shift(3)
    df["card_fraud_last3"] = (
        f1.fillna(0).astype("int8") + f2.fillna(0).astype("int8") + f3.fillna(0).astype("int8")
    ).astype("int8")

    f1 = df.groupby("client_id", sort=False)[LABEL].shift(1)
    f2 = df.groupby("client_id", sort=False)[LABEL].shift(2)
    f3 = df.groupby("client_id", sort=False)[LABEL].shift(3)
    df["client_fraud_last3"] = (
        f1.fillna(0).astype("int8") + f2.fillna(0).astype("int8") + f3.fillna(0).astype("int8")
    ).astype("int8")

    client_amt_mean = df.groupby("client_id", sort=False)[AMT_COL].transform("mean")
    client_amt_std = df.groupby("client_id", sort=False)[AMT_COL].transform("std")

    df["amount_deviation"] = (
        (df[AMT_COL] - client_amt_mean) / (client_amt_std + 1e-6)
    ).astype("float32")

    df["velocity_spike_ratio"] = (
        df["client_tx_1h"] / (df["client_tx_1h_avg_prev"] + 1e-6)
    ).astype("float32")

    df["mccnew_x_error"] = (df["client_mcc_is_new"] * df["has_error"]).astype("int8")
    df["mccnew_x_velocity"] = (df["client_mcc_is_new"] * df["velocity_spike_ratio"]).astype(
        "float32"
    )

    eps = 1e-6
    K = 10

    df["client_recent_avg_amt"] = (
        df.groupby("client_id", sort=False)[AMT_COL]
        .shift(1)
        .rolling(K, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    df["client_recent_avg_amt"] = df["client_recent_avg_amt"].fillna(df["client_avg_amt_prev"]).astype(
        "float32"
    )

    df["amount_vs_recent_window_avg"] = (df[AMT_COL] / (df["client_recent_avg_amt"] + eps)).astype(
        "float32"
    )
    df["log_amount_vs_recent_window_avg"] = np.log1p(df["amount_vs_recent_window_avg"]).astype(
        "float32"
    )

    DEV = "log_amount_vs_recent_window_avg"
    df["dev_x_mccnew"] = (df[DEV] * df["client_mcc_is_new"]).astype("float32")
    df["dev_x_velocity"] = (df[DEV] * df["card_velocity_spike_ratio"]).astype("float32")

    return df


def make_df_model(df_feat: pd.DataFrame) -> pd.DataFrame:
    df = df_feat.copy()

    if "date" in df.columns:
        df.drop(columns=["date"], inplace=True)

    drop_cols = ["client_id", "card_id", "merchant_id", "mcc"]
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

    drop_more = [
        "amount_income_ratio",
        "amount_limit_ratio",
        "abs_amount",
        "expires_year",
        "years_since_pin_change",
        "tx_day",
        "weekday",
        "tx_year",
        "client_sin_mean_past",
        "sin_cumsum",
        "client_cos_mean_past",
        "client_weekday_prev",
        "cos_cumsum",
        "cnt_past",
        "credit_limit",
        "total_debt",
        "yearly_income",
        "per_capita_income",
        "income_ratio_region",
        "months_to_expire",
        "months_from_account",
        "male",
        "expires_month",
        "num_cards_issued",
        "credit_score",
        "cb_Visa",
        "cb_Mastercard",
        "is_credit",
        "client_mcc_prior_count",
        "card_mcc_prior_count",
        "prev_tx_time",
        "log_interval",
        "log_interval_shift",
        "interval_cumsum",
        "interval_cnt_past",
        "client_tx_1h",
        "client_tx_1h_shift",
        "client_tx_1h_cumsum",
        "client_tx_cnt_past",
        "card_tx_1h",
        "card_tx_1h_shift",
        "card_tx_1h_cumsum",
        "card_tx_cnt_past",
        "card_tx_1h_avg_prev",
        "amt_shift",
        "amt_cumsum",
        "amt_cnt_past",
        "client_avg_amt_prev",
        "log_amount_vs_recent_window_avg",
        "amount_vs_recent_window_avg",
        "client_recent_avg_amt",
        "velocity_spike_ratio",
    ]
    df_model = df_model.drop(columns=[c for c in drop_more if c in df_model.columns])

    return df_model


def main():
    in_path = "DATA/dataset/train_stage2"
    out_parquet = "DATA/dataset/test_add"
    artifacts_path = "DATA/artifacts/stage2_artifacts.json"

    df = pd.read_parquet(in_path)

    artifacts = fit_artifacts(df)
    save_json(artifacts, artifacts_path)

    df_feat = build_features(df, artifacts)
    df_model = make_df_model(df_feat)

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_model.to_parquet(out_parquet)
    print("saved:", out_parquet, "| shape:", df_model.shape)


if __name__ == "__main__":
    main()
