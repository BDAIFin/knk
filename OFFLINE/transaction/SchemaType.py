from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import ijson  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("ijson is required for streaming JSON label loading") from e


@dataclass(frozen=True)
class TransactionSchemaConfig:
    date_format: str = "%Y-%m-%d %H:%M:%S"
    mcc_nullable_int_dtype: str = "Int64"
    amount_dtype: str = "float32"
    log_amount_dtype: str = "float32"
    min_year_dtype: str = "int16"
    month_day_hour_dtype: str = "int8"
    binary_dtype: str = "int8"


ERROR_PATTERNS: Dict[str, str] = {
    "err_bad_card_number": "Bad Card Number",
    "err_bad_expiration": "Bad Expiration",
    "err_bad_cvv": "Bad CVV",
    "err_bad_pin": "Bad PIN",
    "err_bad_zipcode": "Bad Zipcode",
    "err_insufficient_balance": "Insufficient Balance",
    "err_technical_glitch": "Technical Glitch",
}


MCC_GROUP: Dict[str, List[str]] = {
    "Food & Daily": [
        "5812", "5814", "5813", "5411", "5499", "5912", "5921",
        "5300", "5310", "5311",
    ],
    "Transport & Travel": [
        "4111", "4121", "4131", "4112",
        "3722", "3771", "3775",
        "4511", "4411",
        "4722", "7011", "4784",
        "4214",
    ],
    "Digital & Online": ["5815", "5816", "4814", "4899", "3780"],
    "Financial": ["4829", "6300", "7276", "8931"],
    "Retail": [
        "5045", "5732", "5733",
        "5941", "5942", "5947",
        "5661", "5651", "5655", "5621",
        "5977", "5970", "5932",
        "5192", "5193",
        "5712", "5719", "5722",
        "5094",
    ],
    "Medical": ["8011", "8021", "8041", "8043", "8049", "8062", "8099"],
    "Entertainment": ["7832", "7922", "7996", "7801", "7802", "7995"],
    "Automotive & Home": [
        "5541",
        "7531", "7538", "7542", "7549", "5533",
        "1711", "5251", "5261", "5211", "3504",
        "7210", "7230", "7349",
        "3640",
    ],
    "Utilities & Government": ["4900", "9402"],
    "Professional Services": ["8111", "7393"],
    "Industrial / Manufacturing": [
        "3000", "3001", "3005", "3006", "3007", "3008", "3009",
        "3058", "3066", "3075",
        "3132", "3144", "3174",
        "3256", "3260",
        "3359", "3387", "3389", "3390", "3393", "3395", "3405",
        "3509", "3596", "3684",
        "3730",
    ],
}


def load_fraud_labels_json_stream(labels_json_path: str | Path) -> pd.DataFrame:
    ids: List[int] = []
    labels: List[int] = []

    with open(labels_json_path, "rb") as f:
        for k, v in ijson.kvitems(f, "target"):
            kid = int(str(k).strip())
            ids.append(kid)
            labels.append(1 if v == "Yes" else 0)

    return pd.DataFrame({"id": ids, "fraud": labels}).astype({"fraud": "int8"})


def _sanitize_group_col_name(group_name: str) -> str:
    s = group_name.replace(" & ", "_").replace(" ", "_").replace("/", "_")
    s = s.replace("__", "_")
    return s


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def build_transaction_schema(
    trans_parquet_path: str | Path,
    labels_json_path: str | Path,
    cfg: TransactionSchemaConfig = TransactionSchemaConfig(),
    mcc_group: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    trans = pd.read_parquet(trans_parquet_path)
    labels_df = load_fraud_labels_json_stream(labels_json_path)

    _ensure_columns(trans, ["id", "date", "amount", "use_chip", "errors", "mcc"])
    trans = trans.merge(labels_df, on="id", how="inner")

    trans["amount"] = trans["amount"].astype(str).str.replace("$", "", regex=False).astype(cfg.amount_dtype)

    err = trans["errors"]
    trans["has_error"] = err.notna().astype(cfg.binary_dtype)

    for out_col, pat in ERROR_PATTERNS.items():
        trans[out_col] = err.astype(str).str.contains(pat, na=False).astype(cfg.binary_dtype)

    trans = trans.drop(columns=["errors"])

    drop_cols = [c for c in ["merchant_state", "zip", "merchant_city"] if c in trans.columns]
    if drop_cols:
        trans = trans.drop(columns=drop_cols)

    trans["date"] = pd.to_datetime(trans["date"], format=cfg.date_format, errors="coerce")
    trans = trans.sort_values(["date"]).reset_index(drop=True)

    trans["tx_year"] = trans["date"].dt.year.astype(cfg.min_year_dtype)
    trans["tx_month"] = trans["date"].dt.month.astype(cfg.month_day_hour_dtype)
    trans["tx_day"] = trans["date"].dt.day.astype(cfg.month_day_hour_dtype)
    trans["tx_hour"] = trans["date"].dt.hour.astype(cfg.month_day_hour_dtype)

    trans["is_weekend"] = (trans["date"].dt.weekday >= 5).astype(cfg.binary_dtype)

    trans["is_refund"] = (trans["amount"] < 0).astype(cfg.binary_dtype)
    trans["log_abs_amount"] = np.log1p(np.abs(trans["amount"].astype("float64"))).astype(cfg.log_amount_dtype)

    if "use_chip" in trans.columns:
        trans = trans.drop(columns=["use_chip"])

    trans["mcc"] = pd.to_numeric(trans["mcc"], errors="coerce")
    trans["mcc"] = trans["mcc"].astype(cfg.mcc_nullable_int_dtype)

    group_def = mcc_group if mcc_group is not None else MCC_GROUP
    trans["mcc"] = trans["mcc"].astype(str)

    for group_name, mcc_list in group_def.items():
        col_name = f"mccg_{_sanitize_group_col_name(group_name)}"
        trans[col_name] = trans["mcc"].isin(list(map(str, mcc_list))).astype(cfg.binary_dtype)

    return trans
