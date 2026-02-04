# OFFLINE/config.py
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class CFG:
    # Paths
    project_root: Path = Path(".")
    data_root: Path = project_root / "data"
    online_dir: Path = data_root / "online"
    artifacts_dir: Path = project_root / "artifacts"

    # input files
    train_path: Path = online_dir / "train_oss.parquet"
    test_path: Path = online_dir / "test_raw.parquet"

    # Core column names
    label_col: str = "fraud"
    client_col: str = "client_id"
    card_col: str = "card_id"

    time_col: str | None = None

    amount_col: str = "amount"

    # time features
    time_feature_cols: list[str] = field(default_factory=lambda: [
        "tx_year",
        "tx_month",
        "tx_hour",
        "is_weekend",
    ])

    # Numeric raw features
    num_cols_raw: list[str] = field(default_factory=lambda: [
        "amount",
        "current_age",
        "retirement_age",
        "birth_year",
        "latitude",
        "longitude",
        "per_capita_income",
        "yearly_income",
        "total_debt",
        "credit_score",
        "credit_limit",
        "months_to_expire",
        "num_cards_issued",
    ])

    # Binary / categorical 
    binary_cols: list[str] = field(default_factory=lambda: [
        # card / channel
        "is_online",
        "is_credit",
        "is_prepaid",
        "has_chip",

        # errors
        "has_error",
        "err_bad_card_number",
        "err_bad_expiration",
        "err_bad_cvv",
        "err_bad_pin",
        "err_bad_zipcode",
        "err_insufficient_balance",
        "err_technical_glitch",

        # demographics
        "male",

        # distance
        "distance_imputed",
    ])

    # MCC group one-hot
    mcc_group_cols: list[str] = field(default_factory=lambda: [
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
    ])

    # Card brand one-hot
    card_brand_cols: list[str] = field(default_factory=lambda: [
        "cb_Visa",
        "cb_Mastercard",
        "cb_Amex",
        "cb_Discover",
    ])

    # Client aggregation
    use_client_agg: bool = True
    use_amount_z: bool = True

    # Train/validation split
    valid_ratio: float = 0.2
    random_state: int = 42

    # Output
    feature_parquet_dir: Path = artifacts_dir / "feature_sets"

    # Helper: final model columns
    def model_input_cols(self) -> list[str]:
        cols = (
            self.num_cols_raw
            + self.time_feature_cols
            + self.binary_cols
            + self.mcc_group_cols
            + self.card_brand_cols
        )
        return list(dict.fromkeys(cols))  # deduplicate

CFG = CFG()