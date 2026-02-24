from pathlib import Path
import numpy as np
import pandas as pd


LABEL = "fraud"
IDCOL = "id"

IN_STAGE2_PATH_TRAIN  = "../../DATA/dataset/train_stage2"
STAGE1_PATH_TRAIN     = "../../DATA/dataset/TRAIN_stage1"
OUT_STAGE2_PATH_TRAIN = "../../DATA/dataset/TRAIN_stage2"


STAGE2_COLS = [
    "client_error_last3",
    "client_mcc_is_new",
    "vel_x_merchant_new",
    "vel_x_mcc_risk",
    "amount_limit_ratio",
    "limit_ratio_extreme",
    "merchant_change_last5_x_amount_limit_ratio",
    "log_interval_dev",
    "seconds_since_prev_tx",
    "credit_limit",
    "is_refund",
]


def _require_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")


def _ensure_datetime(df, col="date"):
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _build_1h_counts_sorted(df, group_col, time_col="date"):
    n = len(df)
    out = np.zeros(n, dtype=np.int32)

    groups = df.groupby(group_col, sort=False).groups
    for _, idx in groups.items():
        g = df.loc[idx]
        times = g[time_col].values.astype("datetime64[s]").astype("int64")

        cum = np.arange(len(g), dtype=np.int32)
        t_minus_1h = times - 3600
        left = np.searchsorted(times, t_minus_1h, side="left")

        out[idx] = cum - left + 1

    return out


def build_train_stage2(
    in_stage2_path=IN_STAGE2_PATH_TRAIN,
    stage1_path=STAGE1_PATH_TRAIN,
    out_stage2_path=OUT_STAGE2_PATH_TRAIN,
):
    df = pd.read_parquet(in_stage2_path)
    df = _ensure_datetime(df, "date")

    _require_cols(df, [IDCOL, "client_id", "card_id", "merchant_id", "mcc", "date", LABEL, "is_refund"])
    _require_cols(df, ["amount_limit_ratio", "credit_limit"])

    df[IDCOL] = pd.to_numeric(df[IDCOL], errors="coerce").astype("int64")
    df = df.sort_values(["client_id", "date", IDCOL]).reset_index(drop=True)

    # ---------- helper bases (create if missing) ----------
    if "mcc_risk_level" not in df.columns:
        df["mcc_risk_level"] = np.int8(0)

    if "merchant_is_new" not in df.columns:
        df["client_merchant_prior_count"] = df.groupby(["client_id", "merchant_id"], sort=False).cumcount()
        df["merchant_is_new"] = (df["client_merchant_prior_count"] == 0).astype("int8")

    if "merchant_changed" not in df.columns:
        prev_merchant = df.groupby("card_id", sort=False)["merchant_id"].shift(1)
        df["merchant_changed"] = (df["merchant_id"] != prev_merchant).fillna(False).astype("int8")

    # ---------- card_fraud_last3 ----------
    f1 = df.groupby("card_id", sort=False)[LABEL].shift(1)
    f2 = df.groupby("card_id", sort=False)[LABEL].shift(2)
    f3 = df.groupby("card_id", sort=False)[LABEL].shift(3)
    df["card_fraud_last3"] = (
        f1.fillna(0).astype("int8") +
        f2.fillna(0).astype("int8") +
        f3.fillna(0).astype("int8")
    )

    # ---------- client_error_last3 (요청대로: fraud shift 합) ----------
    g1 = df.groupby("client_id", sort=False)[LABEL].shift(1)
    g2 = df.groupby("client_id", sort=False)[LABEL].shift(2)
    g3 = df.groupby("client_id", sort=False)[LABEL].shift(3)
    df["client_error_last3"] = (
        g1.fillna(0).astype("int8") +
        g2.fillna(0).astype("int8") +
        g3.fillna(0).astype("int8")
    )

    # ---------- client_mcc_is_new ----------
    df["client_mcc_prior_count"] = df.groupby(["client_id", "mcc"], sort=False).cumcount()
    df["client_mcc_is_new"] = (df["client_mcc_prior_count"] == 0).astype("int8")

    # ---------- velocity_spike_ratio ----------
    df = df.sort_values(["client_id", "date", IDCOL]).reset_index(drop=True)

    df["client_tx_1h"] = _build_1h_counts_sorted(df, "client_id", "date").astype("int32")
    df["client_tx_1h_shift"] = df.groupby("client_id", sort=False)["client_tx_1h"].shift(1)

    df["client_tx_1h_cumsum"] = (
        df["client_tx_1h_shift"].fillna(0)
          .groupby(df["client_id"])
          .cumsum()
    )
    df["client_tx_cnt_past"] = df.groupby("client_id", sort=False).cumcount()

    df["client_tx_1h_avg_prev"] = np.where(
        df["client_tx_cnt_past"] > 0,
        df["client_tx_1h_cumsum"] / df["client_tx_cnt_past"],
        df["client_tx_1h"].astype(float),
    )

    df["velocity_spike_ratio"] = (
        df["client_tx_1h"] / (df["client_tx_1h_avg_prev"] + 1e-6)
    ).astype("float32")

    df["vel_x_merchant_new"] = (df["velocity_spike_ratio"] * df["merchant_is_new"]).astype("float32")
    df["vel_x_mcc_risk"] = (df["velocity_spike_ratio"] * df["mcc_risk_level"]).astype("float32")

    # ---------- limit_ratio_extreme ----------
    thr = float(df["amount_limit_ratio"].quantile(0.999))
    df["limit_ratio_extreme"] = (df["amount_limit_ratio"] >= thr).astype("int8")

    # ---------- merchant_change_last5_x_amount_limit_ratio ----------
    df["merchant_change_cnt_last5"] = (
        df.groupby("card_id", sort=False)["merchant_changed"]
          .rolling(window=5, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
          .astype("int8")
    )
    df["merchant_change_last5_x_amount_limit_ratio"] = (
        df["merchant_change_cnt_last5"].astype("float32") * df["amount_limit_ratio"].astype("float32")
    ).astype("float32")

    # ---------- seconds_since_prev_tx / log_interval_dev ----------
    df["prev_tx_time"] = df.groupby("client_id", sort=False)["date"].shift(1)
    df["seconds_since_prev_tx"] = (
        (df["date"] - df["prev_tx_time"]).dt.total_seconds()
    ).fillna(0.0).astype("float32")

    df["log_interval"] = np.log1p(df["seconds_since_prev_tx"]).astype("float32")

    df["log_interval_shift"] = df.groupby("client_id", sort=False)["log_interval"].shift(1).fillna(0.0)
    df["interval_cumsum_prev"] = df["log_interval_shift"].groupby(df["client_id"]).cumsum().astype("float32")
    df["interval_cnt_past"] = df.groupby("client_id", sort=False).cumcount().astype("int32")

    df["client_avg_interval_prev"] = np.where(
        df["interval_cnt_past"] > 0,
        df["interval_cumsum_prev"] / df["interval_cnt_past"],
        df["log_interval"].astype(float),
    ).astype("float32")

    df["log_interval_dev"] = (df["log_interval"] - df["client_avg_interval_prev"]).astype("float32")

    # ---------- keep only stage2 cols (+id,fraud) ----------
    _require_cols(df, [IDCOL, LABEL] + STAGE2_COLS)
    df_out = df[[IDCOL, LABEL] + STAGE2_COLS].copy()

    # ---------- merge stage1 features ----------
    st1 = pd.read_parquet(stage1_path)
    _require_cols(st1, [IDCOL, LABEL])
    st1[IDCOL] = pd.to_numeric(st1[IDCOL], errors="coerce").astype("int64")

    stage1_feat_cols = [c for c in st1.columns if c not in [IDCOL, LABEL]]
    st1_small = st1[[IDCOL] + stage1_feat_cols].copy()

    df_out = df_out.merge(st1_small, on=IDCOL, how="left", validate="one_to_one")

    out_stage2_path = Path(out_stage2_path)
    out_stage2_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_stage2_path, index=False)

    mem_mb = df_out.memory_usage(deep=True).sum() / 1024**2
    print("saved:", str(out_stage2_path))
    print("shape:", df_out.shape)
    print("n_features:", len([c for c in df_out.columns if c not in [IDCOL, LABEL]]))
    print("mem(MB):", round(mem_mb, 2))
    print("stage1_added_cols:", len(stage1_feat_cols))
    print("stage2_cols:", len(STAGE2_COLS))


if __name__ == "__main__":
    build_train_stage2()