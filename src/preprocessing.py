"""
FraudSentinel: Preprocessing & Time-Bucket Spike Detection Engine.

Handles data loading, memory optimization, structural imputation,
rare-category pooling for interpretable SHAP attributions, rolling
time-bucket anomaly z-scores, and stratified train/test partitioning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("FraudSentinel-Preprocess")


@dataclass(frozen=True)
class PreprocessingConfig:
    target_col: str = "isFraud"
    join_key: str = "TransactionID"
    time_col: str = "TransactionDT"
    bucket_seconds: int = 3600          # 1-hour temporal buckets
    rolling_window_buckets: int = 24    # 24-hour moving window for baseline
    rare_category_threshold: int = 500  # Minimum frequency to retain categorical value
    test_split_size: float = 0.20
    random_seed: int = 42


def reduce_memory_footprint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts numeric types to optimize memory for IEEE-CIS scale (~590k records).
    """
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        # Strictly check for numeric types to prevent string/float comparison crashes
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            col_type = df[col].dtype
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == "float":
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory reduced: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


def load_and_merge_data(
    transaction_path: str,
    identity_path: str,
    config: PreprocessingConfig
) -> pd.DataFrame:
    """
    Left-joins transaction and identity tables on TransactionID.
    """
    logger.info(f"Loading transactions from {transaction_path}...")
    df_tx = pd.read_csv(transaction_path)
    logger.info(f"Loading identities from {identity_path}...")
    df_id = pd.read_csv(identity_path)

    logger.info(f"Merging datasets on {config.join_key} (Left Join)...")
    merged_df = pd.merge(df_tx, df_id, on=config.join_key, how="left")
    logger.info(f"Merged shape: {merged_df.shape}")

    return reduce_memory_footprint(merged_df)


def engineer_spike_features(
    df: pd.DataFrame,
    config: PreprocessingConfig
) -> pd.DataFrame:
    """
    Computes rolling-window z-scores for velocity and financial volume.
    """
    logger.info("Computing rolling-window anomaly spike indicators...")
    
    # 1. Isolate temporal features to prevent DataFrame fragmentation
    time_bucket = (df[config.time_col] // config.bucket_seconds).astype(np.int32)
    
    temp_df = pd.DataFrame({
        "time_bucket": time_bucket,
        "TransactionAmt": df["TransactionAmt"]
    })
    
    # 2. Group by the standalone time bucket
    bucket_stats = (
        temp_df.groupby("time_bucket")
        .agg(
            bucket_tx_count=("TransactionAmt", "count"),
            bucket_total_amt=("TransactionAmt", "sum")
        )
        .reset_index()
        .sort_values("time_bucket")
    )
    
    # 3. Calculate rolling baseline (excluding future lookahead)
    bucket_stats["rolling_count_mean"] = (
        bucket_stats["bucket_tx_count"]
        .rolling(window=config.rolling_window_buckets, min_periods=1)
        .mean()
    )
    bucket_stats["rolling_count_std"] = (
        bucket_stats["bucket_tx_count"]
        .rolling(window=config.rolling_window_buckets, min_periods=1)
        .std()
        .fillna(1.0)
    )
    bucket_stats["rolling_amt_mean"] = (
        bucket_stats["bucket_total_amt"]
        .rolling(window=config.rolling_window_buckets, min_periods=1)
        .mean()
    )
    bucket_stats["rolling_amt_std"] = (
        bucket_stats["bucket_total_amt"]
        .rolling(window=config.rolling_window_buckets, min_periods=1)
        .std()
        .fillna(1.0)
    )
    
    # 4. Z-scores: (Observed - Baseline Mean) / Baseline Std
    eps = 1e-5
    bucket_stats["tx_volume_zscore"] = (
        (bucket_stats["bucket_tx_count"] - bucket_stats["rolling_count_mean"]) 
        / (bucket_stats["rolling_count_std"] + eps)
    ).astype(np.float32)
    
    bucket_stats["tx_amount_zscore"] = (
        (bucket_stats["bucket_total_amt"] - bucket_stats["rolling_amt_mean"]) 
        / (bucket_stats["rolling_amt_std"] + eps)
    ).astype(np.float32)
    
    # 5. Map back to original row alignment via left merge on the temporary DataFrame
    merged_features = pd.merge(
        temp_df[["time_bucket"]], 
        bucket_stats[["time_bucket", "tx_volume_zscore", "tx_amount_zscore"]], 
        on="time_bucket", 
        how="left"
    )
    
    # 6. Drop the mapping key and concat all new features in one shot
    merged_features.drop(columns=["time_bucket"], inplace=True)
    merged_features.index = df.index  # Guarantee strict index alignment
    
    df = pd.concat([df, merged_features], axis=1)
    
    return df


class SentinelPreprocessor:
    """
    Transforms raw tabular data into interpretable, clean ML arrays.
    
    Missing value strategy:
    - Categorical: Explicit 'MISSING' token. In fraud detection, missing metadata
      (e.g., no email domain, missing proxy info) is a strong positive fraud signal.
      Imputing the mode would erase this operational risk signal.
    - Numerical: Median imputation with explicit missing indicator tracking where useful.
    
    Categorical Encoding strategy:
    - High-cardinality features (e.g., P_emaildomain) pool rare strings (< 500 count)
      into 'OTHER_RARE' before Ordinal Encoding.
    - Ordinal encoding preserves feature cardinality for TreeSHAP without exploding
      dimensionality like One-Hot Encoding, keeping explanations human-readable.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.encoder: Optional[OrdinalEncoder] = None
        self.categorical_cols: List[str] = [
            "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"
        ]
        self.numerical_cols: List[str] = [
            "TransactionAmt", "card1", "card2", "card3", "card5",
            "addr1", "addr2", "dist1", "tx_volume_zscore", "tx_amount_zscore"
        ]
        self.rare_category_maps: Dict[str, set] = {}
        self.medians: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "SentinelPreprocessor":
        logger.info("Fitting Preprocessor (calculating medians, pooling rare categories)...")
        
        # 1. Learn medians on numerical training columns
        for col in self.numerical_cols:
            if col in df.columns:
                self.medians[col] = float(df[col].median(skipna=True))

        # 2. Identify frequent categories for high-cardinality columns
        for col in self.categorical_cols:
            if col in df.columns:
                freq = df[col].fillna("MISSING").value_counts()
                frequent = set(freq[freq >= self.config.rare_category_threshold].index)
                self.rare_category_maps[col] = frequent

        # 3. Fit Ordinal Encoder on normalized categorical columns
        clean_cats = self._clean_categoricals(df)
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2
        )
        self.encoder.fit(clean_cats)
        return self

    def _clean_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_df = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            if col in df.columns:
                series = df[col].fillna("MISSING").astype(str)
                frequent = self.rare_category_maps.get(col, set())
                cat_df[col] = series.apply(lambda x: x if x in frequent else "OTHER_RARE")
            else:
                cat_df[col] = "MISSING"
        return cat_df

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        logger.info("Transforming dataset...")
        output_df = pd.DataFrame(index=df.index)

        # Impute numerical features using training medians
        for col in self.numerical_cols:
            if col in df.columns:
                output_df[col] = df[col].fillna(self.medians.get(col, 0.0)).astype(np.float32)
            else:
                output_df[col] = self.medians.get(col, 0.0)

        # Encode categorical features
        clean_cats = self._clean_categoricals(df)
        encoded_cats = self.encoder.transform(clean_cats)
        for idx, col in enumerate(self.categorical_cols):
            output_df[col] = encoded_cats[:, idx].astype(np.float32)

        y = df[self.config.target_col].astype(np.int8) if self.config.target_col in df.columns else None
        return output_df, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: PreprocessingConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test split.
    """
    logger.info("Executing Stratified Holdout Split...")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.test_split_size,
        random_state=config.random_seed
    )
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    fraud_rate_train = y_train.mean() * 100
    fraud_rate_test = y_test.mean() * 100
    logger.info(f"Train size: {len(X_train)} (Fraud rate: {fraud_rate_train:.2f}%)")
    logger.info(f"Test size:  {len(X_test)}  (Fraud rate: {fraud_rate_test:.2f}%)")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Smoke test execution pipeline
    import os
    config = PreprocessingConfig()

    tx_path = "data/raw/train_transaction.csv"
    id_path = "data/raw/train_identity.csv"

    if os.path.exists(tx_path) and os.path.exists(id_path):
        raw_data = load_and_merge_data(tx_path, id_path, config)
        featured_data = engineer_spike_features(raw_data, config)
        
        preprocessor = SentinelPreprocessor(config)
        preprocessor.fit(featured_data)
        X, y = preprocessor.transform(featured_data)
        
        X_train, X_test, y_train, y_test = split_data(X, y, config)
        
        os.makedirs("data/processed", exist_ok=True)
        X_train.to_parquet("data/processed/X_train.parquet")
        X_test.to_parquet("data/processed/X_test.parquet")
        y_train.to_frame().to_parquet("data/processed/y_train.parquet")
        y_test.to_frame().to_parquet("data/processed/y_test.parquet")
        logger.info("Data processing complete. Artifacts saved to data/processed/")
    else:
        logger.warning(
            f"Files not found at '{tx_path}' and '{id_path}'. Place raw CSVs in data/raw/ to execute the full run."
        )