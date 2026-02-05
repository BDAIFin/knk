import pandas as pd
import numpy as np

MCCG_COLS = [
    "mccg_Food_Daily",
    "mccg_Transport_Travel",
    "mccg_Digital_Online",
    "mccg_Financial",
    "mccg_Retail",
    "mccg_Medical",
    "mccg_Entertainment",
    "mccg_Automotive_Home",
    "mccg_Utilities_Government",
    "mccg_Professional_Services",
    "mccg_Industrial_/_Manufacturing",
]

# time sort
def sort_like_stream(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["_row_id"] = out.index.to_numpy()

    out = out.sort_values(
        ["tx_year", "tx_month", "tx_day", "tx_hour", "_row_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    return out

# add feature
def add_client_state_feature(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = sort_like_stream(df)
    g = out.groupby(cfg.client_col, sort=False)

    # [hour mean prev]
    txn_cnt_prev = g.cumcount()
    out["is_new_client"] = (txn_cnt_prev == 0).astype("int8")

    hour_oh = pd.get_dummies(out["tx_hour"], prefix="h", dtype=np.int16)

    for h in range(24):
        col = f"h_{h}"
        if col not in hour_oh.columns:
            hour_oh[col] = 0
    hour_oh = hour_oh[[f"h_{h}" for h in range(24)]]

    hour_cnt_prev = (
        hour_oh.groupby(out[cfg.client_col], sort=False)
        .cumsum()
        .shift(1)
        .fillna(0)
        .astype(np.int32)
    )

    out["client_hour_mode_prev"] = hour_cnt_prev.to_numpy().argmax(axis=1).astype(np.int8)
    out.loc[out["is_new_client"] == 1, "client_hour_mode_prev"] = -1

    # [weekend rate prev]
    wk_cum_prev = g["is_weekend"].cumsum().shift(1)
    out["client_weekend_rate_prev"] = (wk_cum_prev / txn_cnt_prev.replace(0, np.nan)).fillna(0.0)
    out["weekend_diff"] = out["client_weekend_rate_prev"] - out["is_weekend"]

    # [amount]
    amount_cum = g["amount"].cumsum().shift(1)
    out["amount_mean"] = (amount_cum / txn_cnt_prev.replace(0, np.nan)).fillna(0.0)
    out["amount_diff"] = out["amount_mean"] - out["amount"]

    # [MCC]
    for c in MCCG_COLS:
        out[f"client_{c}_cnt_prev"] = g[c].cumsum().shift(1).fillna(0).astype("int32")

    prev_cnt_cols = [f"client_{c}_cnt_prev" for c in MCCG_COLS]
    prev_cnt_mat = out[prev_cnt_cols].to_numpy()
    top_idx = prev_cnt_mat.argmax(axis=1)
    all_zero = (prev_cnt_mat.sum(axis=1) == 0)
    top_idx = top_idx.astype("int16")
    top_idx[all_zero] = -1
    out["client_top_mccg_prev"] = top_idx

    return out
