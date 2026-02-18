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


def fit_stage1_artifacts(
    df_train: pd.DataFrame,
    mcc_min_cnt: int = 1000,
    mcc_rate_mult: float = 3.0,
    high_risk_days=(0, 4, 6),
) -> dict:
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
        "base_rate": base_rate,
        "mcc_min_cnt": mcc_min_cnt,
        "mcc_rate_mult": mcc_rate_mult,
        "highrisk_mcc": highrisk_mcc,
        "high_risk_days": list(high_risk_days),
    }


def build_stage1_features(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy()

    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

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

    highrisk_mcc = set(artifacts["highrisk_mcc"])
    df["mcc_highrisk_90"] = df["mcc"].isin(highrisk_mcc).astype("int8")

    high_risk_days = set(artifacts["high_risk_days"])
    df["is_highrisk_weekday"] = df["weekday"].isin(high_risk_days).astype("int8")

    stage1_cols = [
        LABEL,
        "has_error",
        "err_bad_card_number",
        "err_bad_expiration",
        "err_bad_cvv",
        "tx_month",
        "tx_hour",
        "is_refund",
        AMT_COL,
        "hour_sin",
        "hour_cos",
        "sin_shift",
        "cos_shift",
        "hour_circular_distance",
        "mcc_highrisk_90",
        "is_highrisk_weekday",
    ]

    df_stage1 = df[stage1_cols].copy()
    df_stage1["hour_circular_distance"] = df_stage1["hour_circular_distance"].astype("float32")

    return df_stage1


def main():
    in_path = "DATA/dataset/transactions_train"  
    out_parquet = "DATA/dataset/train_stage1"
    artifacts_path = "DATA/artifacts/stage1_artifacts.json"

    df = pd.read_parquet(in_path)

    artifacts = fit_stage1_artifacts(df)
    save_json(artifacts, artifacts_path)

    df_stage1 = build_stage1_features(df, artifacts)

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_stage1.to_parquet(out_parquet)
    print("saved:", out_parquet, "| shape:", df_stage1.shape, "| mem(MB):", df_stage1.memory_usage(deep=True).sum() / 1024**2)


if __name__ == "__main__":
    main()
