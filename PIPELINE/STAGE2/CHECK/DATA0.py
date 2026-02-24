# build_check_stage1.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "fraud"
IDCOL = "id"
MCC_COL = "mcc"

STAGE1_FEATURES_AVAILABLE = [
    "id",
    "fraud",
    "mcc_smoothed_risk",
    "mcc_risk_level",
    "hour_sin",
    "hour_cos",
    "weekday",
    "month_sin",
    "is_risky_wd_hour",
    "is_risky_month",
    "log_abs_amount",
    "err_bad_cvv",
]


def _require_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_stage1_dataset_from_min_df(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    need = [
        IDCOL,
        MCC_COL,
        "tx_hour",
        "tx_month",
        "weekday",
        "log_abs_amount",
        "err_bad_cvv",
    ]
    _require_cols(df, need)

    out = df.copy()

    out[IDCOL] = pd.to_numeric(out[IDCOL], errors="coerce").astype("int64")
    out[MCC_COL] = out[MCC_COL].astype(str)

    if LABEL in out.columns:
        out[LABEL] = pd.to_numeric(out[LABEL], errors="coerce").fillna(0).astype("int8")
    else:
        out[LABEL] = np.int8(-1)

    mcc_map = artifacts.get("mcc_smoothed_risk_map", {})
    global_rate = float(artifacts.get("global_rate", 0.0))
    out["mcc_smoothed_risk"] = (
        out[MCC_COL].map(mcc_map).fillna(global_rate).astype("float32")
    )

    high = set(artifacts.get("mcc_risk_level", {}).get("high", []))
    mid = set(artifacts.get("mcc_risk_level", {}).get("mid", []))
    out["mcc_risk_level"] = np.int8(0)
    out.loc[out[MCC_COL].isin(mid), "mcc_risk_level"] = np.int8(1)
    out.loc[out[MCC_COL].isin(high), "mcc_risk_level"] = np.int8(2)

    hr = pd.to_numeric(out["tx_hour"], errors="coerce").fillna(0).astype("int16")
    out["hour_sin"] = np.sin(2 * np.pi * hr / 24.0).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * hr / 24.0).astype("float32")

    mo = pd.to_numeric(out["tx_month"], errors="coerce").fillna(1).astype("int16")
    out["month_sin"] = np.sin(2 * np.pi * mo / 12.0).astype("float32")

    high_months = set(int(x) for x in artifacts.get("high_risk_months", []))
    out["is_risky_month"] = out["tx_month"].isin(high_months).astype("int8")

    pairs = artifacts.get("risky_wd_hour", {}).get("pairs", [])
    pair_set = set((int(wd), int(hr_)) for wd, hr_ in pairs)
    wd = pd.to_numeric(out["weekday"], errors="coerce").fillna(0).astype("int16")
    out["is_risky_wd_hour"] = [
        1 if (int(w), int(h)) in pair_set else 0
        for w, h in zip(wd.tolist(), hr.tolist())
    ]
    out["is_risky_wd_hour"] = out["is_risky_wd_hour"].astype("int8")

    out["log_abs_amount"] = (
        pd.to_numeric(out["log_abs_amount"], errors="coerce")
        .fillna(0.0)
        .astype("float32")
    )

    out["err_bad_cvv"] = (
        pd.to_numeric(out["err_bad_cvv"], errors="coerce")
        .fillna(0)
        .astype("int8")
    )

    out = out[STAGE1_FEATURES_AVAILABLE].copy()
    return out


def run_build_check_stage1(
    in_path: str,
    out_parquet: str,
    artifacts_path: str,
):
    df = pd.read_parquet(in_path)
    artifacts = _load_json(artifacts_path)

    df_stage1 = build_stage1_dataset_from_min_df(df, artifacts)

    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_stage1.to_parquet(out_parquet, index=False)

    mem_mb = df_stage1.memory_usage(deep=True).sum() / 1024**2
    print("saved:", str(out_parquet))
    print("shape:", df_stage1.shape)
    print("n_features:", len([c for c in df_stage1.columns if c not in [IDCOL, LABEL]]))
    print("mem(MB):", round(mem_mb, 2))


if __name__ == "__main__":
    IN_PATH = "../../DATA/dataset/check_stage1"
    OUT_PATH = "../../DATA/dataset/CHECK_stage1"
    ART_PATH = "../../DATA/artifacts/stage1_artifacts_min.json"

    run_build_check_stage1(
        in_path=IN_PATH,
        out_parquet=OUT_PATH,
        artifacts_path=ART_PATH,
    )