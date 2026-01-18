## How to Deal with Parquet

```python
import pandas as pd

df = pd.read_parquet("transactions_clean.parquet")

df["date"] = pd.to_datetime(df["date"])

df["client_id"]   = df["client_id"].astype("int32")
df["card_id"]     = df["card_id"].astype("int32")
df["merchant_id"] = df["merchant_id"].astype("int32")
df["mcc"]         = df["mcc"].astype("int16")

df["amount"] = df["amount"].astype("float32")

for c in ["use_chip", "merchant_city", "merchant_state", "zip"]:
    df[c] = df[c].astype("category")

for c in [
    "has_error",
    "err_card_credential",
    "err_authentication",
    "err_financial",
    "err_system"
]:
    df[c] = df[c].astype("int8")

df["fraud"] = df["fraud"].astype("int8")

df.info(memory_usage="deep")
