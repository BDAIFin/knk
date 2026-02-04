project/
├─ OFFLINE/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ io.py
│  ├─ validate.py
│  ├─ features/
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  ├─ time_features.py
│  │  ├─ client_agg.py
│  │  ├─ client_state.py
│  │  └─ assemble.py
│  ├─ split.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ stage1.py
│  │  └─ type_package.py
│  ├─ evaluate.py
│  └─ artifacts.py
│
├─ notebooks/
│  ├─ 01_offline_train.ipynb
│  └─ 02_offline_type_package.ipynb
│
├─ artifacts/
│  ├─ hgb/
│  │   ├─ feature_schema.json
│  │   ├─ stage1_hgb.joblib
│  │   └─ threshold.json
│  ├─ lgbm/
│  │   ├─ feature_schema.json
│  │   ├─ stage1_lgbm.joblib
│  │   └─ threshold.json
│  └─ logit/
│  │   ├─ feature_schema.json
│  │   ├─ stage1_logit.joblib
│  │   └─ threshold.json
│  └─ xgb/
│      ├─ feature_schema.json
│      ├─ stage1_xgb.joblib
│      └─ threshold.json
└─ data/
│     ├─ offline/
│     │  ├─ test_raw.parquet
│     │  └─ test_raw.parquet 
│     ├─ online/
│     │  ├─ test_raw.parquet
│     │  └─ test_raw.parquet 
│     ├─ original/
│     │  ├─ cards_data.csv
│     │  ├─ mcc_codes.json
│     │  ├─ train_fraud_labels.json
│     │  ├─ transactions_data.csv
│     │  ├─ users_data.csv
│     │  └─ uszips.csv
│     ├─ split/
│     │  ├─ offline
│     │  └─ online
│     ├─ data.ipynb
│     └─ trans.parquet