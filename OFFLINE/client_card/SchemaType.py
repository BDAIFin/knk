import numpy as np
import pandas as pd
from pathlib import Path


class ClientCardSchemaConfig:
    def __init__(self, trans_path: str, users_path: str, cards_path: str):
        self.trans_path = Path(trans_path)
        self.users_path = Path(users_path)
        self.cards_path = Path(cards_path)


def _load_base_trans(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in: {path}")
        return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return pd.read_parquet(path)


def _clean_currency(col: pd.Series) -> pd.Series:
    return col.astype(str).str.replace("$", "", regex=False).astype("float32")


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def _ensure_key_columns(trans: pd.DataFrame) -> pd.DataFrame:
    df = trans.copy()

    if "client_id" not in df.columns:
        for cand in ("user_id", "customer_id", "client", "user"):
            if cand in df.columns:
                df = df.rename(columns={cand: "client_id"})
                break

    if "card_id" not in df.columns:
        for cand in ("card", "cardid"):
            if cand in df.columns:
                df = df.rename(columns={cand: "card_id"})
                break

    missing = [c for c in ("client_id", "card_id") if c not in df.columns]
    if missing:
        raise KeyError(
            f"trans is missing key columns: {missing}. "
            f"trans columns={list(df.columns)[:40]}..."
        )

    return df


def build_stage2_dataset(cfg: ClientCardSchemaConfig) -> pd.DataFrame:
    trans = _ensure_key_columns(_load_base_trans(cfg.trans_path))

    users = pd.read_csv(cfg.users_path)
    cards = pd.read_csv(cfg.cards_path)

    users = users.rename(columns={"id": "client_id"})
    cards = cards.rename(columns={"id": "card_id"})

    if "client_id" not in users.columns:
        raise KeyError(f"users missing client_id after rename. cols={users.columns.tolist()}")

    if "card_id" not in cards.columns:
        raise KeyError(f"cards missing card_id after rename. cols={cards.columns.tolist()}")

    if "client_id" not in cards.columns:
        raise KeyError(f"cards missing client_id. cols={cards.columns.tolist()}")

    df = trans.merge(users, on="client_id", how="left", validate="m:1")
    df = df.merge(cards, on=["card_id", "client_id"], how="left", validate="m:1")

    df.drop(columns=["id", "cvv", "card_on_dark_web"], errors="ignore", inplace=True)

    if "year_pin_last_changed" in df.columns:
        df["year_pin_last_changed"] = df["year_pin_last_changed"].astype("Int16")

    df["acct_open_date"] = pd.to_datetime(df["acct_open_date"], format="%m/%Y", errors="coerce")
    df["acct_open_year"] = df["acct_open_date"].dt.year.astype("int16")
    df["acct_open_month"] = df["acct_open_date"].dt.month.astype("int8")

    df["expires"] = pd.to_datetime(df["expires"], format="%m/%Y", errors="coerce")
    df["expires_year"] = df["expires"].dt.year.astype("int16")
    df["expires_month"] = df["expires"].dt.month.astype("int8")

    ref_date = df["date"]

    df["months_to_expire"] = (
        (df["expires_year"] - ref_date.dt.year) * 12
        + (df["expires_month"] - ref_date.dt.month)
    ).astype("int16")

    df = df[df["months_to_expire"] >= 0].copy()
    df.drop(columns=["expires", "acct_open_date"], inplace=True)

    df["credit_limit"] = _clean_currency(df["credit_limit"])
    df["total_debt"] = _clean_currency(df["total_debt"])
    df["yearly_income"] = _clean_currency(df["yearly_income"])
    df["per_capita_income"] = _clean_currency(df["per_capita_income"])

    df = df[df["credit_limit"] > 0].copy()

    if "has_chip" in df.columns:
        df["has_chip"] = df["has_chip"].replace({"YES": 1, "NO": 0}).astype("int8")

    df["is_credit"] = df["card_type"].astype(str).str.upper().str.contains("CREDIT").astype("int8")
    df["is_prepaid"] = df["card_type"].astype(str).str.upper().str.contains("PREPAID").astype("int8")
    df.drop(columns=["card_type"], inplace=True)

    df["num_credit_cards"] = df["num_credit_cards"].astype("int8")

    df["male"] = (df["gender"] == "Male").astype("int8")
    df.drop(columns=["gender"], inplace=True)

    df.drop(columns=["birth_month", "birth_year"], inplace=True)
    df.drop(columns=["card_number"], errors="ignore", inplace=True)

    df["card_brand"] = df["card_brand"].astype(str).str.strip().str.title()
    for b in ["Visa", "Mastercard", "Amex", "Discover"]:
        df[f"cb_{b}"] = (df["card_brand"] == b).astype("int8")
    df.drop(columns=["card_brand"], inplace=True)

    df["months_from_account"] = (
        (
            df["date"].dt.to_period("M")
            - pd.to_datetime(
                dict(year=df["acct_open_year"], month=df["acct_open_month"], day=1)
            ).dt.to_period("M")
        )
        .apply(lambda x: x.n)
        .astype("int16")
    )
    df.drop(columns=["acct_open_year", "acct_open_month"], inplace=True)

    if "year_pin_last_changed" in df.columns:
        df["years_since_pin_change"] = (
            df["date"].dt.year - df["year_pin_last_changed"]
        ).clip(lower=0).astype("int8")
    else:
        df["years_since_pin_change"] = pd.Series(0, index=df.index, dtype="int8")

    if "retirement_age" in df.columns and "current_age" in df.columns:
        df["years_to_retirement"] = (
            pd.to_numeric(df["retirement_age"], errors="coerce")
            - pd.to_numeric(df["current_age"], errors="coerce")
        ).clip(lower=0).astype("int16")
    else:
        df["years_to_retirement"] = pd.Series(0, index=df.index, dtype="int16")

    df.drop(columns=["retirement_age"], errors="ignore", inplace=True)

    home_loc = (
        df.groupby("client_id")[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "home_lat", "longitude": "home_lon"})
    )
    df = df.join(home_loc, on="client_id")

    df["income_ratio_region"] = (df["yearly_income"] / (df["per_capita_income"] + 1e-6)).astype("float32")
    df["log_yearly_income"] = np.log1p(df["yearly_income"])
    df["log_income_ratio_region"] = np.log1p(df["income_ratio_region"])

    df.drop(columns=["latitude", "longitude", "address"], errors="ignore", inplace=True)
    df.drop(columns=["home_lat", "home_lon"], inplace=True)

    df["amount_income_ratio"] = (df["amount"] / (df["yearly_income"] + 1e-6)).astype("float32")
    df["amount_limit_ratio"] = (df["amount"] / (df["credit_limit"] + 1e-6)).astype("float32")

    df = df[df["months_from_account"] >= 0].copy()

    return df
