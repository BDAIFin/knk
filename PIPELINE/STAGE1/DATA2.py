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


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _require_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")


def fit_stage1_artifacts_minimal(
    df: pd.DataFrame,
    alpha: float = 1000.0,
    high_risk_mcc=None,
    mid_risk_mcc=None,
    high_risk_months=None,
    risky_wd_hour_quantile: float = 0.90,
    min_bin_count: int = 200,
):
    _require_cols(df, [LABEL, MCC_COL, "weekday", "tx_hour", "tx_month"])

    if high_risk_mcc is None:
        high_risk_mcc = ["5732", "5712", "5045", "5816", "5651", "4411"]
    if mid_risk_mcc is None:
        mid_risk_mcc = ["5193", "5311", "7996"]
    if high_risk_months is None:
        high_risk_months = [7, 8, 10, 12]

    tmp = df[[LABEL, MCC_COL, "weekday", "tx_hour", "tx_month"]].copy()
    tmp[MCC_COL] = tmp[MCC_COL].astype(str)

    global_rate = float(tmp[LABEL].mean())

    mcc_stats = (
        tmp.groupby(MCC_COL)[LABEL]
        .agg(tx_count="count", raw_rate="mean")
        .reset_index()
    )
    mcc_stats["mcc_smoothed_risk"] = (
        (mcc_stats["raw_rate"] * mcc_stats["tx_count"] + global_rate * float(alpha))
        / (mcc_stats["tx_count"] + float(alpha))
    )

    mcc_map = dict(
        zip(
            mcc_stats[MCC_COL].astype(str).tolist(),
            mcc_stats["mcc_smoothed_risk"].astype(float).tolist(),
        )
    )

    wh = (
        tmp.groupby(["weekday", "tx_hour"])[LABEL]
        .agg(bin_count="count", bin_rate="mean")
        .reset_index()
    )
    wh_ok = wh[wh["bin_count"] >= int(min_bin_count)].copy()
    if len(wh_ok) > 0:
        thr = float(wh_ok["bin_rate"].quantile(float(risky_wd_hour_quantile)))
        risky_pairs = wh_ok.loc[wh_ok["bin_rate"] >= thr, ["weekday", "tx_hour"]].values.tolist()
    else:
        thr = None
        risky_pairs = []

    artifacts = {
        "label": LABEL,
        "id_col": IDCOL,
        "mcc_col": MCC_COL,
        "alpha": float(alpha),
        "global_rate": float(global_rate),
        "mcc_smoothed_risk_map": mcc_map,
        "mcc_risk_level": {
            "mid": [str(x) for x in mid_risk_mcc],
            "high": [str(x) for x in high_risk_mcc],
        },
        "high_risk_months": [int(x) for x in high_risk_months],
        "risky_wd_hour": {
            "quantile": float(risky_wd_hour_quantile),
            "min_bin_count": int(min_bin_count),
            "threshold_rate": thr,
            "pairs": [[int(wd), int(hr)] for wd, hr in risky_pairs],
        },
    }
    return artifacts


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
    pair_set = set((int(wd), int(hr)) for wd, hr in pairs)
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

def run_pipeline(in_path, out_parquet, artifacts_path=None, fit_artifacts=False):

    df = pd.read_parquet(in_path)

    if fit_artifacts:
        artifacts = fit_stage1_artifacts_minimal(df)
        if artifacts_path is not None:
            save_json(artifacts, artifacts_path)
    else:
        with open(artifacts_path, "r") as f:
            artifacts = json.load(f)

    df_stage1 = build_stage1_dataset_from_min_df(df, artifacts)

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_stage1.to_parquet(out_parquet, index=False)

    mem_mb = df_stage1.memory_usage(deep=True).sum() / 1024**2
    print("saved:", out_parquet)
    print("shape:", df_stage1.shape)
    print("n_features:", len([c for c in df_stage1.columns if c not in [IDCOL, LABEL]]))
    print("mem(MB):", round(mem_mb, 2))


def main():

    train_in = "../../DATA/dataset/train_stage1"
    train_out = "../../DATA/dataset/TRAIN_stage1"
    artifacts_path = "../../DATA/artifacts/stage1_artifacts_min.json"

    test_in = "../../DATA/dataset/test_stage1"
    test_out = "../../DATA/dataset/TEST_stage1"

    print("=== TRAIN STAGE1 BUILD ===")
    run_pipeline(
        in_path=train_in,
        out_parquet=train_out,
        artifacts_path=artifacts_path,
        fit_artifacts=True,
    )

    print("=== TEST STAGE1 BUILD ===")
    run_pipeline(
        in_path=test_in,
        out_parquet=test_out,
        artifacts_path=artifacts_path,
        fit_artifacts=False,
    )


if __name__ == "__main__":
    main()