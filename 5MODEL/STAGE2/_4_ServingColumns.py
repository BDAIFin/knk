import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "fraud"
AMT_COL = "log_abs_amount"   # 이미 만들어진 로그 금액(양수 가정)
RAW_AMT_COL = "amount"       # raw amount가 있으면 더 좋음 (없으면 log로 근사)

# ===========================
# Stage2 Feature Set (<= 101)
# ===========================
STAGE2_FEATURES_101 = [
    # 1) Refund / Amount
    "refund_high_amount",
    "log_abs_amount",

    # 2) Error flags
    "err_bad_cvv",
    "card_hist_x_error",
    "client_hist_x_error",
    "err_bad_card_number",
    "err_insufficient_balance",

    # 3) Error history
    "client_error_last5",
    "card_error_last5",
    "client_error_last3",
    "card_error_last3",
    "client_error_last1",
    "card_error_last1",

    # 4) Time basic + cyclic
    "hour_cos",
    "hour_sin",
    "tx_hour",
    "weekday",
    "tx_month",
    "tx_year",
    "tx_day",

    # 5) Behavioral time-pattern
    "client_weekday_match_last1",
    "cos_shift",
    "sin_shift",
    "client_weekday_prev",
    "cnt_past",
    "client_cos_mean_past",
    "client_sin_mean_past",
    "client_weekday_prior_count",

    # 6) Demographic / credit
    "log_yearly_income",
    "credit_limit",
    "num_credit_cards",
    "log_income_ratio_region",
    "credit_score",
    "current_age",
    "years_to_retirement",
    "months_from_account",
    "total_debt",
    "per_capita_income",
    "yearly_income",
    "income_ratio_region",

    # 7) Card brand / type
    "cb_Visa",
    "cb_Discover",
    "cb_Mastercard",
    "is_prepaid",
    "discover_x_cvv",
    "prepaid_logamount_interaction",
    "cb_Amex",

    # 8) MCC risk
    "mcc_smoothed_risk",
    "mcc_risk_level",
    "is_highrisk_weekday",
    "high_mcc_high_weekday",

    # 9) Merchant novelty / change
    "merchant_is_new_x_has_error",
    "merchant_is_new_x_mcc_is_new",
    "merchant_change_cnt_last5",
    "card_merchant_is_new",
    "client_merchant_is_new",

    # 10) Interval / velocity
    "seconds_since_prev_tx",
    "client_avg_interval_prev",
    "log_interval_dev",
    "client_tx_1h",
    "client_tx_1h_avg_prev",
    "velocity_spike_ratio",
    "card_tx_1h",
    "card_velocity_spike_ratio",

    # 11) Velocity interactions / flags
    "vel_x_error",
    "vel_x_mcc_risk",
    "vel_x_high_mcc",
    "vel_x_merchant_new",
    "vel_x_client_mcc_new",
    "vel_x_card_mcc_new",
    "high_vel_flag",
    "high_vel_error",
    "high_vel_new_mcc",

    # 12) Amount context
    "amount_vs_client_avg_ratio",
    "amount_vs_client_avg_diff",
    "amount_vs_recent_window_avg",
    "amount_vs_client_quantile_q95",
    "amount_vs_client_quantile_q99",
    "amt_over_q95",
    "amt_over_q99",

    # 13) Fraud history
    "client_fraud_last1",
    "client_fraud_last3",
    "card_fraud_last1",
    "card_fraud_last3",

    # 14) Ratios
    "amount_income_ratio",
    "amount_limit_ratio",

    # 15) Basic error
    "has_error",

    # 16) Minimal customer/card context duplicates (keep as-is if present)
    
    # - credit_limit
    # - current_age
    # - is_credit
    # - has_chip
    "is_credit",
    "has_chip",

    # 17) Interaction Tier (93~101)
    "amount_ratio_x_mcc_smoothed_risk",
    "client_merchant_is_new_x_mcc_smoothed_risk",
    "amount_ratio_x_client_merchant_new",
    "card_fraud_last3_x_client_merchant_new",
    "client_fraud_last3_x_client_merchant_new",
    "client_mcc_is_new_x_mcc_smoothed_risk",
    "client_merchant_is_new_x_log_interval_dev",
    "amount_ratio_x_client_mcc_new",
    "client_mcc_is_new_x_log_interval_dev",
]

# -----------------------------
# Utils
# -----------------------------
def ensure_sorted(df: pd.DataFrame, keys) -> pd.DataFrame:
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)

def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# Artifacts (SMALL json)
# + mcc_stats parquet (LARGE)
# -----------------------------
def fit_artifacts(
    df_train: pd.DataFrame,
    amt_q: float = 0.9,
    mcc_min_cnt: int = 1000,
    mcc_rate_mult: float = 3.0,
    high_risk_days=(0, 4, 6),
    smooth_alpha: float = 50.0,  # smoothing strength
) -> tuple[dict, pd.DataFrame]:
    """
    returns:
      - artifacts_small: json-friendly (thresholds, lists)
      - mcc_stats_df: parquet로 저장 권장 (mcc -> smoothed_rate 등)
    """
    thr_amt = float(df_train[AMT_COL].quantile(amt_q))
    base_rate = float(df_train[LABEL].mean())

    mcc_agg = df_train.groupby("mcc")[LABEL].agg(["sum", "count"])
    mcc_agg.rename(columns={"sum": "fraud_sum", "count": "tx_count"}, inplace=True)
    mcc_agg["fraud_rate"] = mcc_agg["fraud_sum"] / mcc_agg["tx_count"]

    # smoothed_rate: (fraud_sum + alpha*base_rate) / (count + alpha)
    mcc_agg["smoothed_rate"] = (mcc_agg["fraud_sum"] + smooth_alpha * base_rate) / (mcc_agg["tx_count"] + smooth_alpha)

    highrisk_mcc = mcc_agg[
        (mcc_agg["tx_count"] >= mcc_min_cnt)
        & (mcc_agg["fraud_rate"] >= base_rate * mcc_rate_mult)
    ].index.astype(str).tolist()

    artifacts_small = {
        "amt_q": amt_q,
        "thr_amt": thr_amt,
        "base_rate": base_rate,
        "mcc_min_cnt": mcc_min_cnt,
        "mcc_rate_mult": mcc_rate_mult,
        "highrisk_mcc": highrisk_mcc,
        "high_risk_days": list(high_risk_days),
        "smooth_alpha": float(smooth_alpha),
    }

    mcc_stats_df = mcc_agg.reset_index().copy()
    mcc_stats_df["mcc"] = mcc_stats_df["mcc"].astype(str)

    return artifacts_small, mcc_stats_df

# -----------------------------
# Fast 1h-window counts (vectorized by segments)
# -----------------------------
def _count_1h_within_groups(df_sorted: pd.DataFrame, key: str) -> np.ndarray:
    """
    df_sorted must be sorted by [key, date].
    returns int32 array length=len(df_sorted)
    """
    gid = df_sorted[key].to_numpy()
    t = df_sorted["date"].to_numpy(dtype="datetime64[s]").astype(np.int64)

    out = np.empty(len(df_sorted), dtype=np.int32)

    n = len(df_sorted)
    start = 0
    while start < n:
        end = start + 1
        while end < n and gid[end] == gid[start]:
            end += 1

        tt = t[start:end]
        left = np.searchsorted(tt, tt - 3600, side="left")
        out[start:end] = (np.arange(end - start) - left + 1).astype(np.int32)
        start = end

    return out

# -----------------------------
# Interaction columns (93~101)
# -----------------------------
def add_interaction_columns(df: pd.DataFrame) -> None:
    """
    df에 interaction 컬럼을 in-place로 추가
    """
    binary_cols = [
        "client_merchant_is_new",
        "client_mcc_is_new",
        "card_merchant_is_new",
        "card_mcc_is_new",
        "refund_high_amount",
        "err_bad_cvv",
        "client_fraud_last3",
        "card_fraud_last3",
    ]

    for c in binary_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("int8")

    cont_cols = [
        "mcc_smoothed_risk",
        "log_interval_dev",
        "amount_vs_client_avg_ratio",
    ]
    for c in cont_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    interactions = {
        # Tier 1
        "client_merchant_is_new_x_mcc_smoothed_risk": ("client_merchant_is_new", "mcc_smoothed_risk"),
        "client_merchant_is_new_x_log_interval_dev":  ("client_merchant_is_new", "log_interval_dev"),
        "client_mcc_is_new_x_log_interval_dev":       ("client_mcc_is_new",      "log_interval_dev"),
        "client_mcc_is_new_x_mcc_smoothed_risk":      ("client_mcc_is_new",      "mcc_smoothed_risk"),

        # Tier 2
        "amount_ratio_x_client_merchant_new": ("amount_vs_client_avg_ratio", "client_merchant_is_new"),
        "amount_ratio_x_client_mcc_new":      ("amount_vs_client_avg_ratio", "client_mcc_is_new"),
        "amount_ratio_x_mcc_smoothed_risk":   ("amount_vs_client_avg_ratio", "mcc_smoothed_risk"),

        # Tier 3
        "client_fraud_last3_x_client_merchant_new": ("client_fraud_last3", "client_merchant_is_new"),
        "card_fraud_last3_x_client_merchant_new":   ("card_fraud_last3",   "client_merchant_is_new"),
    }

    for new_col, (a, b) in interactions.items():
        if new_col in df.columns:
            continue
        if a not in df.columns or b not in df.columns:
            continue

        df[new_col] = df[a] * df[b]
        # binary*binary => int8, else float32
        if (a in binary_cols) and (b in binary_cols):
            df[new_col] = df[new_col].fillna(0).astype("int8")
        else:
            df[new_col] = df[new_col].astype("float32")

# ===========================
# Feature builder (<=101)
# ===========================
def build_features_101(
    df: pd.DataFrame,
    artifacts: dict,
    mcc_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()
    df["_row_id"] = np.arange(len(df), dtype=np.int32)

    # ---- required raw cols (최소)
    base_need = ["id", "client_id", "card_id", "merchant_id", "mcc", "date", AMT_COL]
    missing = [c for c in base_need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required raw cols: {missing}")

    # datetime
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("date column has NaT after to_datetime; check raw data.")

    # ensure types
    df["mcc"] = df["mcc"].astype(str)

    # ---- time parts
    if "tx_hour" not in df.columns:
        df["tx_hour"] = df["date"].dt.hour.astype("int16")
    df["weekday"] = df["date"].dt.weekday.astype("int8")
    df["tx_month"] = df["date"].dt.month.astype("int8")
    df["tx_year"] = df["date"].dt.year.astype("int16")
    df["tx_day"] = df["date"].dt.day.astype("int8")

    df["hour_sin"] = np.sin(2 * np.pi * df["tx_hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["tx_hour"] / 24).astype("float32")

    # ---- refund_high_amount
    if "is_refund" in df.columns:
        thr = float(artifacts["thr_amt"])
        df["refund_high_amount"] = ((df["is_refund"].fillna(0).astype("int8") == 1) & (df[AMT_COL] > thr)).astype("int8")
    else:
        df["refund_high_amount"] = 0

    # ---- has_error + error flags (없으면 0으로)
    err_cols = [
        "has_error",
        "err_bad_cvv",
        "err_bad_card_number",
        "err_insufficient_balance",
    ]
    for c in err_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("int8")
        else:
            df[c] = 0

    # ---- error history last1/3/5 (client/card)
    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    g = df.groupby("client_id", sort=False)["has_error"]
    df["client_error_last1"] = g.shift(1).fillna(0).astype("int8")
    df["client_error_last3"] = (g.shift(1).fillna(0) + g.shift(2).fillna(0) + g.shift(3).fillna(0)).astype("int8")
    df["client_error_last5"] = (g.shift(1).fillna(0) + g.shift(2).fillna(0) + g.shift(3).fillna(0) + g.shift(4).fillna(0) + g.shift(5).fillna(0)).astype("int8")

    df = ensure_sorted(df, ["card_id", "date", "_row_id"])
    g = df.groupby("card_id", sort=False)["has_error"]
    df["card_error_last1"] = g.shift(1).fillna(0).astype("int8")
    df["card_error_last3"] = (g.shift(1).fillna(0) + g.shift(2).fillna(0) + g.shift(3).fillna(0)).astype("int8")
    df["card_error_last5"] = (g.shift(1).fillna(0) + g.shift(2).fillna(0) + g.shift(3).fillna(0) + g.shift(4).fillna(0) + g.shift(5).fillna(0)).astype("int8")

    # ---- fraud history features (train-time only)
    if LABEL in df.columns:
        df[LABEL] = df[LABEL].fillna(0).astype("int8")

        df = ensure_sorted(df, ["client_id", "date", "_row_id"])
        df["client_fraud_last1"] = df.groupby("client_id", sort=False)[LABEL].shift(1).fillna(0).astype("int8")
        df["client_fraud_last3"] = (
            df.groupby("client_id", sort=False)[LABEL].shift(1).fillna(0)
            + df.groupby("client_id", sort=False)[LABEL].shift(2).fillna(0)
            + df.groupby("client_id", sort=False)[LABEL].shift(3).fillna(0)
        ).astype("int8")

        df = ensure_sorted(df, ["card_id", "date", "_row_id"])
        df["card_fraud_last1"] = df.groupby("card_id", sort=False)[LABEL].shift(1).fillna(0).astype("int8")
        df["card_fraud_last3"] = (
            df.groupby("card_id", sort=False)[LABEL].shift(1).fillna(0)
            + df.groupby("card_id", sort=False)[LABEL].shift(2).fillna(0)
            + df.groupby("card_id", sort=False)[LABEL].shift(3).fillna(0)
        ).astype("int8")

        # hist_x_error
        df["card_fraud_cum_prev"] = df.groupby("card_id", sort=False)[LABEL].cumsum().shift(1).fillna(0).astype("int32")
        df["card_has_fraud_history"] = (df["card_fraud_cum_prev"] > 0).astype("int8")
        df["card_hist_x_error"] = (df["card_has_fraud_history"] * df["card_error_last1"]).astype("int8")

        df["client_fraud_cum_prev"] = df.groupby("client_id", sort=False)[LABEL].cumsum().shift(1).fillna(0).astype("int32")
        df["client_has_fraud_history"] = (df["client_fraud_cum_prev"] > 0).astype("int8")
        df["client_hist_x_error"] = (df["client_has_fraud_history"] * df["has_error"]).astype("int8")
    else:
        # serving-time placeholder (원하면 외부 confirmed-log에서 join해서 넣어야 함)
        for c in ["client_fraud_last1","client_fraud_last3","card_fraud_last1","card_fraud_last3","card_hist_x_error","client_hist_x_error"]:
            df[c] = 0

    # ---- client weekday pattern
    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    df["client_weekday_prev"] = df.groupby("client_id", sort=False)["weekday"].shift(1)
    df["client_weekday_match_last1"] = (df["weekday"] == df["client_weekday_prev"]).fillna(False).astype("int8")

    df["cos_shift"] = df.groupby("client_id", sort=False)["hour_cos"].shift(1).fillna(df["hour_cos"]).astype("float32")
    df["sin_shift"] = df.groupby("client_id", sort=False)["hour_sin"].shift(1).fillna(df["hour_sin"]).astype("float32")

    df["cnt_past"] = df.groupby("client_id", sort=False).cumcount().astype("int32")

    # mean past of cos/sin (shifted)
    cos_shift = df.groupby("client_id", sort=False)["hour_cos"].shift(1).fillna(df["hour_cos"]).astype("float32")
    sin_shift = df.groupby("client_id", sort=False)["hour_sin"].shift(1).fillna(df["hour_sin"]).astype("float32")
    df["_cos_cumsum"] = cos_shift.groupby(df["client_id"], sort=False).cumsum().astype("float32")
    df["_sin_cumsum"] = sin_shift.groupby(df["client_id"], sort=False).cumsum().astype("float32")

    df["client_cos_mean_past"] = np.where(df["cnt_past"] > 0, df["_cos_cumsum"] / df["cnt_past"], np.nan).astype("float32")
    df["client_sin_mean_past"] = np.where(df["cnt_past"] > 0, df["_sin_cumsum"] / df["cnt_past"], np.nan).astype("float32")
    df.drop(columns=["_cos_cumsum", "_sin_cumsum"], inplace=True)

    df["client_weekday_prior_count"] = df.groupby(["client_id", "weekday"], sort=False).cumcount().astype("int32")

    # ---- mcc_smoothed_risk
    m = mcc_stats_df[["mcc", "smoothed_rate"]].copy()
    m["mcc"] = m["mcc"].astype(str)
    m = m.drop_duplicates("mcc")
    m_map = dict(zip(m["mcc"].values, m["smoothed_rate"].astype("float32").values))
    df["mcc_smoothed_risk"] = df["mcc"].map(m_map).astype("float32")

    # ---- mcc_risk_level (네가 준 hard-coded list 그대로)
    high = {"5732", "5712", "5045", "5816", "5651", "4411"}
    mid  = {"5193", "5311", "7996"}
    df["mcc_risk_level"] = 0
    df.loc[df["mcc"].isin(mid), "mcc_risk_level"] = 1
    df.loc[df["mcc"].isin(high), "mcc_risk_level"] = 2
    df["mcc_risk_level"] = df["mcc_risk_level"].astype("int8")

    # ---- high risk weekday + joint
    high_risk_days = set(artifacts["high_risk_days"])
    df["is_highrisk_weekday"] = df["weekday"].isin(high_risk_days).astype("int8")
    df["high_mcc_high_weekday"] = ((df["mcc_risk_level"] == 2) & (df["is_highrisk_weekday"] == 1)).astype("int8")

    # ---- merchant novelty (global time-aware) + client/card merchant is_new
    df = ensure_sorted(df, ["date", "_row_id"])
    df["merchant_is_new"] = df.groupby("merchant_id", sort=False).cumcount().eq(0).astype("int8")

    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    df["client_merchant_is_new"] = df.groupby(["client_id", "merchant_id"], sort=False).cumcount().eq(0).astype("int8")

    df = ensure_sorted(df, ["card_id", "date", "_row_id"])
    df["card_merchant_is_new"] = df.groupby(["card_id", "merchant_id"], sort=False).cumcount().eq(0).astype("int8")

    # ---- mcc is_new (client/card) for interactions
    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    df["client_mcc_prior_count"] = df.groupby(["client_id", "mcc"], sort=False).cumcount().astype("int32")
    df["client_mcc_is_new"] = (df["client_mcc_prior_count"] == 0).astype("int8")

    df = ensure_sorted(df, ["card_id", "date", "_row_id"])
    df["card_mcc_prior_count"] = df.groupby(["card_id", "mcc"], sort=False).cumcount().astype("int32")
    df["card_mcc_is_new"] = (df["card_mcc_prior_count"] == 0).astype("int8")

    # ---- merchant_is_new interactions
    df["merchant_is_new_x_has_error"] = (df["merchant_is_new"] * df["has_error"]).astype("int8")
    df["merchant_is_new_x_mcc_is_new"] = (df["merchant_is_new"] * df["card_mcc_is_new"]).astype("int8")

    # ---- merchant_change_cnt_last5 (card rolling)
    df = ensure_sorted(df, ["card_id", "date", "_row_id"])
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

    # ---- seconds_since_prev_tx (client 기준; 원하면 card 기준으로 바꿔도 됨)
    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    prev_time = df.groupby("client_id", sort=False)["date"].shift(1)
    df["seconds_since_prev_tx"] = (df["date"] - prev_time).dt.total_seconds().fillna(0).astype("float32")

    df["log_interval"] = np.log1p(df["seconds_since_prev_tx"]).astype("float32")
    df["interval_cnt_past"] = df.groupby("client_id", sort=False).cumcount().astype("int32")
    df["interval_shift"] = df.groupby("client_id", sort=False)["log_interval"].shift(1).fillna(df["log_interval"]).astype("float32")
    df["interval_cumsum"] = df["interval_shift"].groupby(df["client_id"], sort=False).cumsum().astype("float32")
    df["client_avg_interval_prev"] = np.where(
        df["interval_cnt_past"] > 0,
        df["interval_cumsum"] / df["interval_cnt_past"],
        df["log_interval"],
    ).astype("float32")
    df["log_interval_dev"] = (df["log_interval"] - df["client_avg_interval_prev"]).astype("float32")
    df.drop(columns=["interval_shift", "interval_cumsum", "interval_cnt_past", "log_interval"], inplace=True)

    # ---- 1h tx counts + avg prev + velocity_spike_ratio
    df_client = ensure_sorted(df[["_row_id", "client_id", "date"]], ["client_id", "date", "_row_id"])
    df_client["client_tx_1h"] = _count_1h_within_groups(df_client, "client_id")
    df_client["client_tx_1h_shift"] = df_client.groupby("client_id", sort=False)["client_tx_1h"].shift(1)
    df_client["client_tx_1h_cumsum"] = df_client["client_tx_1h_shift"].fillna(0).groupby(df_client["client_id"], sort=False).cumsum().astype("float32")
    df_client["client_tx_cnt_past"] = df_client.groupby("client_id", sort=False).cumcount().astype("int32")
    df_client["client_tx_1h_avg_prev"] = np.where(
        df_client["client_tx_cnt_past"] > 0,
        df_client["client_tx_1h_cumsum"] / df_client["client_tx_cnt_past"],
        df_client["client_tx_1h"],
    ).astype("float32")
    df = df.merge(df_client[["_row_id", "client_tx_1h", "client_tx_1h_avg_prev"]], on="_row_id", how="left", validate="one_to_one")
    df["velocity_spike_ratio"] = (df["client_tx_1h"] / (df["client_tx_1h_avg_prev"] + 1e-6)).astype("float32")

    df_card = ensure_sorted(df[["_row_id", "card_id", "date"]], ["card_id", "date", "_row_id"])
    df_card["card_tx_1h"] = _count_1h_within_groups(df_card, "card_id")
    df_card["card_tx_1h_shift"] = df_card.groupby("card_id", sort=False)["card_tx_1h"].shift(1)
    df_card["card_tx_1h_cumsum"] = df_card["card_tx_1h_shift"].fillna(0).groupby(df_card["card_id"], sort=False).cumsum().astype("float32")
    df_card["card_tx_cnt_past"] = df_card.groupby("card_id", sort=False).cumcount().astype("int32")
    df_card["card_tx_1h_avg_prev"] = np.where(
        df_card["card_tx_cnt_past"] > 0,
        df_card["card_tx_1h_cumsum"] / df_card["card_tx_cnt_past"],
        df_card["card_tx_1h"],
    ).astype("float32")
    df = df.merge(df_card[["_row_id", "card_tx_1h", "card_tx_1h_avg_prev"]], on="_row_id", how="left", validate="one_to_one")
    df["card_velocity_spike_ratio"] = (df["card_tx_1h"] / (df["card_tx_1h_avg_prev"] + 1e-6)).astype("float32")

    # ---- vel interactions / flags
    df["vel_x_error"] = (df["velocity_spike_ratio"] * df["has_error"]).astype("float32")
    df["vel_x_mcc_risk"] = (df["velocity_spike_ratio"] * df["mcc_risk_level"]).astype("float32")
    df["vel_x_high_mcc"] = (df["velocity_spike_ratio"] * (df["mcc_risk_level"] == 2).astype("int8")).astype("float32")
    df["vel_x_merchant_new"] = (df["velocity_spike_ratio"] * df["merchant_is_new"]).astype("float32")
    df["vel_x_client_mcc_new"] = (df["velocity_spike_ratio"] * df["client_mcc_is_new"]).astype("float32")
    df["vel_x_card_mcc_new"] = (df["velocity_spike_ratio"] * df["card_mcc_is_new"]).astype("float32")

    high_vel_thr = float(df["velocity_spike_ratio"].quantile(0.95))
    df["high_vel_flag"] = (df["velocity_spike_ratio"] > high_vel_thr).astype("int8")
    df["high_vel_error"] = (df["high_vel_flag"] * df["has_error"]).astype("int8")
    df["high_vel_new_mcc"] = (df["high_vel_flag"] * df["client_mcc_is_new"]).astype("int8")

    # ---- amount context 
    if RAW_AMT_COL in df.columns:
        amt = pd.to_numeric(df[RAW_AMT_COL], errors="coerce").fillna(0).astype("float32").abs()
    else:
        # fallback: log_abs_amount -> expm1 근사
        amt = np.expm1(pd.to_numeric(df[AMT_COL], errors="coerce").fillna(0).astype("float32")).astype("float32")

    eps = 1e-6
    df = ensure_sorted(df, ["client_id", "date", "_row_id"])
    a_shift = df.groupby("client_id", sort=False)[amt.name if isinstance(amt, pd.Series) else AMT_COL].shift(1) if isinstance(amt, pd.Series) else df.groupby("client_id", sort=False)[AMT_COL].shift(1)

    df["_amt"] = amt.to_numpy()
    df["_amt_shift"] = df.groupby("client_id", sort=False)["_amt"].shift(1)
    df["_amt_cumsum"] = df["_amt_shift"].fillna(0).groupby(df["client_id"], sort=False).cumsum().astype("float32")
    df["_amt_cnt_past"] = df.groupby("client_id", sort=False).cumcount().astype("int32")
    df["_client_avg_amt_prev"] = np.where(df["_amt_cnt_past"] > 0, df["_amt_cumsum"] / df["_amt_cnt_past"], df["_amt"]).astype("float32")

    df["amount_vs_client_avg_ratio"] = (df["_amt"] / (df["_client_avg_amt_prev"] + eps)).astype("float32")
    df["amount_vs_client_avg_diff"] = (df["_amt"] - df["_client_avg_amt_prev"]).astype("float32")

    # recent window avg (K=10)
    K = 10
    df["_recent_avg"] = (
        df.groupby("client_id", sort=False)["_amt"]
          .shift(1)
          .rolling(K, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    ).astype("float32")
    df["_recent_avg"] = df["_recent_avg"].fillna(df["_client_avg_amt_prev"]).astype("float32")
    df["amount_vs_recent_window_avg"] = (df["_amt"] / (df["_recent_avg"] + eps)).astype("float32")

    # rolling quantiles (min_periods=10)
    KQ = 50
    q95 = (
        df.groupby("client_id", sort=False)["_amt"]
          .shift(1)
          .rolling(KQ, min_periods=10)
          .quantile(0.95)
          .reset_index(level=0, drop=True)
    ).astype("float32")
    q99 = (
        df.groupby("client_id", sort=False)["_amt"]
          .shift(1)
          .rolling(KQ, min_periods=10)
          .quantile(0.99)
          .reset_index(level=0, drop=True)
    ).astype("float32")

    q95 = q95.fillna(df["_client_avg_amt_prev"]).astype("float32")
    q99 = q99.fillna(df["_client_avg_amt_prev"]).astype("float32")

    df["amount_vs_client_quantile_q95"] = (df["_amt"] / (q95 + eps)).astype("float32")
    df["amount_vs_client_quantile_q99"] = (df["_amt"] / (q99 + eps)).astype("float32")
    df["amt_over_q95"] = (df["_amt"] > q95).astype("int8")
    df["amt_over_q99"] = (df["_amt"] > q99).astype("int8")

    df.drop(columns=["_amt","_amt_shift","_amt_cumsum","_amt_cnt_past","_client_avg_amt_prev","_recent_avg"], inplace=True)

    # ---- ratios: amount_income_ratio, amount_limit_ratio
    if "yearly_income" in df.columns:
        df["yearly_income"] = pd.to_numeric(df["yearly_income"], errors="coerce").astype("float32")
        df["amount_income_ratio"] = (amt / (df["yearly_income"].fillna(0) + 1.0)).astype("float32")
    else:
        df["amount_income_ratio"] = np.nan

    if "credit_limit" in df.columns:
        df["credit_limit"] = pd.to_numeric(df["credit_limit"], errors="coerce").astype("float32")
        df["amount_limit_ratio"] = (amt / (df["credit_limit"].fillna(0) + 1.0)).astype("float32")
    else:
        df["amount_limit_ratio"] = np.nan

    # ---- prepaid_logamount_interaction
    if "card_type" in df.columns:
        df["prepaid_logamount_interaction"] = ((df["card_type"] == "debit(prepaid)").astype("int8") * df[AMT_COL]).astype("float32")
    else:
        df["prepaid_logamount_interaction"] = 0.0

    # ---- discover_x_cvv
    if "cb_Discover" in df.columns:
        df["cb_Discover"] = df["cb_Discover"].fillna(0).astype("int8")
        df["discover_x_cvv"] = (df["cb_Discover"] * df["err_bad_cvv"]).astype("int8")
    else:
        df["discover_x_cvv"] = 0

    # ---- add interaction tier 93~101
    add_interaction_columns(df)

    # ---- final: restore row order + slim to <=101 (+label if exists)
    df = df.sort_values("_row_id", kind="mergesort").reset_index(drop=True)

    keep = ["id"] + STAGE2_FEATURES_101 + ([LABEL] if LABEL in df.columns else [])
    keep = [c for c in keep if c in df.columns]  # 존재하는 것만

    df_out = df[keep].copy()

        # -----------------------------
    # Final sanitize: NaN/Inf 제거
    # -----------------------------
    # 1) time pattern 3개는 의미 보존하면서 채움
    if "client_weekday_prev" in df_out.columns:
        # "이전 요일 없음" 센티널
        df_out["client_weekday_prev"] = df_out["client_weekday_prev"].fillna(-1)

    # cos/sin mean past는 "과거 평균 없음"이면 현재 hour_cos/sin으로 대체
    if "client_cos_mean_past" in df_out.columns:
        base = df_out["hour_cos"] if "hour_cos" in df_out.columns else 0.0
        df_out["client_cos_mean_past"] = df_out["client_cos_mean_past"].fillna(base)

    if "client_sin_mean_past" in df_out.columns:
        base = df_out["hour_sin"] if "hour_sin" in df_out.columns else 0.0
        df_out["client_sin_mean_past"] = df_out["client_sin_mean_past"].fillna(base)

    # 2) 전체 numeric에 대해 Inf -> NaN -> 0 일괄 처리 (학습/서빙 안전)
    feat_cols = [c for c in df_out.columns if c not in ["id", LABEL]]
    df_out[feat_cols] = df_out[feat_cols].replace([np.inf, -np.inf], np.nan)

    # 숫자 컬럼만 0으로 채움 (object는 건드리지 않음)
    num_cols = df_out[feat_cols].select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].fillna(0)

    # 3) 라벨도 혹시 모를 NaN 방지
    if LABEL in df_out.columns:
        df_out[LABEL] = df_out[LABEL].fillna(0).astype("int8")


    for c in df_out.columns:
        if c == LABEL:
            continue
        if pd.api.types.is_bool_dtype(df_out[c]):
            df_out[c] = df_out[c].astype("int8")
        if pd.api.types.is_integer_dtype(df_out[c]):
            df_out[c] = pd.to_numeric(df_out[c], downcast="integer")
        if pd.api.types.is_float_dtype(df_out[c]):
            df_out[c] = pd.to_numeric(df_out[c], downcast="float")

    return df_out

# ===========================
# Sanity
# ===========================
def sanity_report(df_feat: pd.DataFrame) -> None:
    cols = [c for c in df_feat.columns if c != LABEL and c != "id"]
    missing = [c for c in STAGE2_FEATURES_101 if c not in df_feat.columns]
    extra = [c for c in cols if c not in STAGE2_FEATURES_101]
    print("=== Sanity Check ===")
    print("n_features:", len(cols))
    print("missing:", missing)
    print("extra:", extra)

# ===========================
# Main
# ===========================
def main():
    in_path = "DATA/dataset/train_stage2"                  # input parquet
    out_parquet = "DATA/dataset/stage2_feat_101.parquet"   # output
    artifacts_json = "DATA/artifacts/stage2_artifacts.json"
    mcc_stats_parquet = "DATA/artifacts/mcc_stats.parquet"

    df = pd.read_parquet(in_path)

    # train-time artifacts
    artifacts, mcc_stats_df = fit_artifacts(df)
    save_json(artifacts, artifacts_json)
    Path(mcc_stats_parquet).parent.mkdir(parents=True, exist_ok=True)
    mcc_stats_df.to_parquet(mcc_stats_parquet, index=False)

    # build
    df_feat = build_features_101(df, artifacts, mcc_stats_df)

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out_parquet, index=False)

    print("saved:", out_parquet, "| shape:", df_feat.shape)
    sanity_report(df_feat)

if __name__ == "__main__":
    main()