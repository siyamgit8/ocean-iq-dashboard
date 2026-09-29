"""
train_forecast_model.py - Machine Learning Forecasting Pipeline for Coking Coal Freight Rates
Trains an XGBoost Regressor with 7-day lag features and evaluates against a Linear Regression baseline.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Reconfigure stdout/stderr for clean UTF-8 output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, inspect, text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_forecast_model")

# File & Path Configuration
BASE_DIR = Path(__file__).resolve().parent
DB_FILE_CANDIDATES = [
    BASE_DIR / "logistics_platform.db",
    Path.cwd() / "logistics_platform.db",
    BASE_DIR.parent / "logistics_platform.db",
    Path.cwd().parent / "logistics_platform.db",
    Path.home() / "Downloads" / "SIH_NEW" / "Antigravity_sih" / "logistics_platform.db",
    Path.home() / "Downloads" / "SIH_NEW" / "logistics_platform.db",
]

MODEL_ARTIFACT_PATH = BASE_DIR / "freight_xgboost_model.pkl"
FEATURES_ARTIFACT_PATH = BASE_DIR / "model_features.pkl"
METRICS_ARTIFACT_PATH = BASE_DIR / "metrics.json"


def resolve_database_path() -> Path:
    """Locate the logistics_platform.db database file across candidate paths."""
    for candidate in DB_FILE_CANDIDATES:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            logger.info("Connected to database at: %s", candidate.resolve())
            return candidate.resolve()
            
    searched = "\n  - ".join(str(p) for p in DB_FILE_CANDIDATES[:4])
    error_msg = f"Database 'logistics_platform.db' not found. Searched:\n  - {searched}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def load_data_from_db() -> pd.DataFrame:
    """Read historical_freight table from SQLite database using SQLAlchemy and pandas."""
    db_path = resolve_database_path()
    engine = create_engine(f"sqlite:///{db_path.as_posix()}", echo=False)
    
    inspector = inspect(engine)
    if "historical_freight" not in inspector.get_table_names():
        raise ValueError("Table 'historical_freight' does not exist in logistics_platform.db")

    query = "SELECT * FROM historical_freight"
    logger.info("Executing query: %s", query)
    df = pd.read_sql(query, engine)
    logger.info("Loaded %d rows with columns: %s", len(df), df.columns.tolist())
    return df


def prepare_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data according to requirements:
    1. Sort by date ascending
    2. Filter out rows where is_feasible = 0
    3. Extract month and day_of_year from date column
    4. Create 7-day lag features for vlsfo_price and bdi_index per (origin, destination, vessel_class) group
    5. Drop NaN rows created by lagging
    6. One-hot encode categorical columns: origin, destination, vessel_class, charter_type
    7. Add artificial noise to freight_rate_usd: df['freight_rate_usd'] += np.random.normal(0, 2.5, len(df))
    """
    df = df_raw.copy()

    # Ensure required columns exist with sensible defaults if table schema is legacy
    if "is_feasible" not in df.columns:
        df["is_feasible"] = 1
    if "vessel_class" not in df.columns:
        # Default representative dry bulk vessel class if not explicitly partitioned
        df["vessel_class"] = "Capesize"
    if "charter_type" not in df.columns:
        df["charter_type"] = "Spot"
    if "coal_price_usd" not in df.columns:
        df["coal_price_usd"] = 250.0 + (df["bdi_index"].fillna(1500) * 0.05)
    if "port_congestion_origin" not in df.columns:
        df["port_congestion_origin"] = 2.5
    if "port_congestion_dest" not in df.columns:
        df["port_congestion_dest"] = 3.0

    # 1. Sort by date ascending
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    logger.info("Data sorted chronologically from %s to %s", df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d"))

    # 2. Filter out rows where is_feasible = 0
    initial_len = len(df)
    df = df[df["is_feasible"] != 0].copy()
    filtered_out = initial_len - len(df)
    if filtered_out > 0:
        logger.info("Filtered out %d unfeasible rows (is_feasible == 0). Remaining: %d", filtered_out, len(df))

    # 3. Extract month and day_of_year from date column
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # 4. Create 7-day lag features for vlsfo_price and bdi_index per (origin, destination, vessel_class) group using .shift(7)
    group_cols = [c for c in ["origin", "destination", "vessel_class"] if c in df.columns]
    logger.info("Generating 7-day lag features grouped by: %s", group_cols)
    df["vlsfo_price_lag7"] = df.groupby(group_cols)["vlsfo_price"].shift(7)
    df["bdi_index_lag7"] = df.groupby(group_cols)["bdi_index"].shift(7)

    # 5. Drop NaN rows created by lagging
    before_drop = len(df)
    df = df.dropna(subset=["vlsfo_price_lag7", "bdi_index_lag7"]).reset_index(drop=True)
    logger.info("Dropped %d NaN rows from lag creation. Working dataset: %d rows", before_drop - len(df), len(df))

    # 7. Add a small amount of artificial noise to freight_rate_usd before training
    np.random.seed(42)
    noise = np.random.normal(0, 2.5, len(df))
    df["freight_rate_usd"] += noise
    logger.info("Added artificial Gaussian noise (std=2.5) to 'freight_rate_usd' for realistic simulation.")

    # 6. One-hot encode categorical columns: origin, destination, vessel_class, charter_type
    cat_cols = [c for c in ["origin", "destination", "vessel_class", "charter_type"] if c in df.columns]
    logger.info("One-hot encoding categorical features: %s", cat_cols)
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    # Define feature column list (exclude metadata/target columns)
    exclude_cols = {"id", "date", "is_feasible", "freight_rate_usd"}
    feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]
    
    logger.info("Feature engineering complete. Total feature columns: %d", len(feature_cols))
    return df_encoded, feature_cols


def train_and_evaluate(
    df_encoded: pd.DataFrame, feature_cols: List[str]
) -> Tuple[xgb.XGBRegressor, LinearRegression, Dict[str, Any]]:
    """
    Split data chronologically (80% train, 20% test),
    Train XGBoost Regressor and Linear Regression baseline,
    Compute all evaluation metrics.
    """
    # Chronological 80/20 split
    split_idx = int(len(df_encoded) * 0.8)
    train_df = df_encoded.iloc[:split_idx]
    test_df = df_encoded.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df["freight_rate_usd"]
    X_test = test_df[feature_cols]
    y_test = test_df["freight_rate_usd"]

    logger.info("Train set: %d samples (%s to %s)", len(X_train), train_df["date"].min().strftime("%Y-%m-%d"), train_df["date"].max().strftime("%Y-%m-%d"))
    logger.info("Test set: %d samples (%s to %s)", len(X_test), test_df["date"].min().strftime("%Y-%m-%d"), test_df["date"].max().strftime("%Y-%m-%d"))

    # 1. Train XGBoost Regressor
    logger.info("Training XGBoost Regressor (n_estimators=500, max_depth=6, lr=0.05)...")
    xgb_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics computation
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_r2 = float(r2_score(y_train, y_train_pred))

    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))

    # 2. Train Linear Regression baseline
    logger.info("Training Linear Regression baseline...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_test_pred = lr_model.predict(X_test)
    lr_test_rmse = float(np.sqrt(mean_squared_error(y_test, lr_test_pred)))
    lr_test_mae = float(mean_absolute_error(y_test, lr_test_pred))
    lr_test_r2 = float(r2_score(y_test, lr_test_pred))

    # 3. Top 10 Feature Importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    top_10_features = importance_df.head(10).to_dict(orient="records")

    # Package metrics dictionary
    metrics = {
        "model_architecture": "xgboost.XGBRegressor",
        "hyperparameters": xgb_params,
        "dataset_split": {
            "total_records": len(df_encoded),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "train_percentage": 80.0,
            "test_percentage": 20.0,
            "feature_count": len(feature_cols)
        },
        "train_metrics": {
            "rmse": round(train_rmse, 4),
            "mae": round(train_mae, 4),
            "r2_score": round(train_r2, 4)
        },
        "test_metrics": {
            "rmse": round(test_rmse, 4),
            "mae": round(test_mae, 4),
            "r2_score": round(test_r2, 4)
        },
        "linear_regression_baseline": {
            "test_rmse": round(lr_test_rmse, 4),
            "test_mae": round(lr_test_mae, 4),
            "test_r2_score": round(lr_test_r2, 4),
            "xgb_rmse_improvement_pct": round(((lr_test_rmse - test_rmse) / lr_test_rmse) * 100, 2)
        },
        "top_10_feature_importances": [
            {"feature": row["feature"], "importance": round(float(row["importance"]), 6)}
            for row in top_10_features
        ]
    }

    return model, lr_model, metrics


def save_artifacts(model: xgb.XGBRegressor, feature_cols: List[str], metrics: Dict[str, Any]) -> None:
    """Save trained model, feature order list, and evaluation metrics."""
    joblib.dump(model, MODEL_ARTIFACT_PATH)
    logger.info("Saved model artifact to: %s", MODEL_ARTIFACT_PATH.resolve())

    joblib.dump(feature_cols, FEATURES_ARTIFACT_PATH)
    logger.info("Saved feature names to: %s", FEATURES_ARTIFACT_PATH.resolve())

    with open(METRICS_ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics JSON to: %s", METRICS_ARTIFACT_PATH.resolve())


def print_console_report(metrics: Dict[str, Any]) -> None:
    """Print beautifully formatted evaluation report to the console."""
    print("\n" + "=" * 80)
    print("🚀 FREIGHT RATE FORECASTING MODEL — EVALUATION REPORT")
    print("=" * 80)
    
    print("\n📊 DATASET & SPLIT SUMMARY:")
    print(f"  • Total Dataset Size   : {metrics['dataset_split']['total_records']:,} rows")
    print(f"  • Training Partition   : {metrics['dataset_split']['train_records']:,} rows ({metrics['dataset_split']['train_percentage']}%)")
    print(f"  • Test Partition       : {metrics['dataset_split']['test_records']:,} rows ({metrics['dataset_split']['test_percentage']}%)")
    print(f"  • Total Input Features : {metrics['dataset_split']['feature_count']}")

    print("\n📈 XGBOOST MODEL PERFORMANCE:")
    print("  [Training Set]")
    print(f"    • RMSE : ${metrics['train_metrics']['rmse']:.4f} / ton")
    print(f"    • MAE  : ${metrics['train_metrics']['mae']:.4f} / ton")
    print(f"    • R²   : {metrics['train_metrics']['r2_score']:.4f} ({metrics['train_metrics']['r2_score']*100:.2f}%)")
    print("  [Test Set]")
    print(f"    • RMSE : ${metrics['test_metrics']['rmse']:.4f} / ton")
    print(f"    • MAE  : ${metrics['test_metrics']['mae']:.4f} / ton")
    print(f"    • R²   : {metrics['test_metrics']['r2_score']:.4f} ({metrics['test_metrics']['r2_score']*100:.2f}%)")

    print("\n⚖️ BASELINE COMPARISON (Linear Regression):")
    print(f"  • Baseline Test RMSE : ${metrics['linear_regression_baseline']['test_rmse']:.4f} / ton")
    print(f"  • XGBoost Test RMSE  : ${metrics['test_metrics']['rmse']:.4f} / ton")
    print(f"  • Error Reduction    : {metrics['linear_regression_baseline']['xgb_rmse_improvement_pct']}% improvement over baseline")

    print("\n🏆 TOP 10 FEATURE IMPORTANCES (XGBoost):")
    print(f"  {'Rank':<5} {'Feature Name':<45} {'Importance':<12} {'Bar'}")
    print("  " + "-" * 75)
    for idx, item in enumerate(metrics["top_10_feature_importances"], 1):
        bar = "█" * int(item["importance"] * 40)
        print(f"  #{idx:<4} {item['feature']:<45} {item['importance']:<12.4f} {bar}")

    print("\n💾 SAVED ARTIFACTS:")
    print(f"  ✓ Model Pickle    : {MODEL_ARTIFACT_PATH.name}")
    print(f"  ✓ Features List   : {FEATURES_ARTIFACT_PATH.name}")
    print(f"  ✓ Metrics JSON    : {METRICS_ARTIFACT_PATH.name}")
    print("=" * 80 + "\n")


def main() -> None:
    """Main ML pipeline orchestrator."""
    logger.info("Starting ML Training Pipeline...")
    
    # 1. Load data from SQLite database
    df_raw = load_data_from_db()

    # 2. Data preparation, lag creation, noise injection, and one-hot encoding
    df_encoded, feature_cols = prepare_data(df_raw)

    # 3. Model training and evaluation
    model, lr_model, metrics = train_and_evaluate(df_encoded, feature_cols)

    # 4. Save artifacts
    save_artifacts(model, feature_cols, metrics)

    # 5. Print report
    print_console_report(metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Training pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
