"""
db_setup.py - Database Setup and CSV Ingestion for Logistics Platform
Ingests coking coal freight rates and port constraints CSVs into SQLite using SQLAlchemy.
"""

import sys
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData, Table, Column, inspect, text
from sqlalchemy.dialects.sqlite import INTEGER, REAL, TEXT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("db_setup")

# Database & File Configuration
BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "logistics_platform.db"
DB_URL = f"sqlite:///{DB_FILE.as_posix()}"

FREIGHT_CSV_NAME = "coking_coal_freight_rates_2021_2025 (1).csv"
PORT_CSV_NAME = "port_constraints.csv"


def resolve_file_path(filename: str, description: str) -> Path:
    """
    Search for a file in multiple candidate locations:
    1. Directory of db_setup.py
    2. Current working directory
    3. Parent directory
    4. User's Downloads directory
    """
    candidates = [
        BASE_DIR / filename,
        Path.cwd() / filename,
        BASE_DIR.parent / filename,
        Path.cwd().parent / filename,
        Path.home() / "Downloads" / filename,
        Path.home() / "Downloads" / "SIH_NEW" / filename,
        Path.home() / "Downloads" / "SIH_NEW" / "Antigravity_sih" / filename,
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            logger.info("Found %s at: %s", description, candidate.resolve())
            return candidate.resolve()
            
    # If not found anywhere, raise clear error with candidate paths searched
    searched_str = "\n  - ".join(str(p) for p in candidates[:4])
    error_msg = (
        f"Missing required file for {description}: '{filename}'.\n"
        f"Searched locations:\n  - {searched_str}"
    )
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def to_snake_case(name: str) -> str:
    """
    Standardize any column name to lowercase_snake_case.
    Handles spaces, CamelCase, hyphens, and punctuation.
    """
    s = str(name).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[\s\W]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def load_and_validate_csv(file_path: Path, description: str, expected_min_cols: int) -> pd.DataFrame:
    """
    Read and validate CSV structure.
    Fails fast with clear error messages if missing or malformed.
    """
    if not file_path.exists():
        error_msg = f"Missing required file for {description}: '{file_path.name}' at '{file_path}'"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if file_path.stat().st_size == 0:
        error_msg = f"File is empty: '{file_path.name}' has 0 bytes."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    df = None
    last_error = None
    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                skipinitialspace=True
            )
            break
        except Exception as exc:
            last_error = exc
            continue

    if df is None:
        error_msg = f"Malformed CSV error: Failed to parse '{file_path.name}'. Details: {last_error}"
        logger.error(error_msg)
        raise ValueError(error_msg) from last_error

    if df.empty:
        error_msg = f"Malformed CSV error: '{file_path.name}' contains headers but has 0 data rows."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(df.columns) < expected_min_cols:
        error_msg = (
            f"Malformed CSV error: '{file_path.name}' has only {len(df.columns)} columns; "
            f"expected at least {expected_min_cols}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Standardize column headers to lowercase_snake_case
    df.columns = [to_snake_case(col) for col in df.columns]
    logger.info("Successfully loaded '%s' (%d rows, %d columns)", file_path.name, len(df), len(df.columns))
    return df


def clean_text_value(val: Any) -> Optional[str]:
    """Clean text values, converting empty strings/NA to None (SQL NULL)."""
    if pd.isna(val) or val is None:
        return None
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in ("nan", "none", "null", "n/a", "na"):
        return None
    return val_str


def clean_real_value(val: Any) -> Optional[float]:
    """Clean numeric values, preserving empty cells as None (SQL NULL), not 0."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, str):
        val_str = val.strip()
        if val_str == "" or val_str.lower() in ("nan", "none", "null", "n/a", "na"):
            return None
        try:
            val = float(val_str)
        except ValueError as exc:
            raise ValueError(f"Cannot convert text '{val_str}' to REAL numeric value.") from exc
    
    if isinstance(val, (int, float, np.number)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    return None


def clean_boolean_value(val: Any) -> Optional[int]:
    """Clean boolean-like values into integer 0 or 1 (or NULL if missing)."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return 1 if val else 0
    if isinstance(val, (int, float, np.number)):
        if np.isnan(val):
            return None
        return 1 if int(val) != 0 else 0
    
    val_str = str(val).strip().lower()
    if val_str in ("1", "true", "t", "yes", "y"):
        return 1
    elif val_str in ("0", "false", "f", "no", "n"):
        return 0
    elif val_str in ("", "nan", "none", "null", "n/a", "na"):
        return None
    else:
        try:
            return 1 if int(float(val_str)) != 0 else 0
        except ValueError as exc:
            raise ValueError(f"Cannot parse boolean-like value: '{val}'") from exc


def clean_iso_date_value(val: Any) -> Optional[str]:
    """Parse and standardize date value to ISO YYYY-MM-DD string format (TEXT)."""
    if pd.isna(val) or val is None:
        return None
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in ("nan", "none", "null", "n/a", "na"):
        return None
    try:
        dt = pd.to_datetime(val_str)
        return dt.strftime("%Y-%m-%d")
    except Exception as exc:
        raise ValueError(f"Malformed date value '{val_str}'. Cannot parse to ISO format.") from exc


def define_schema(metadata: MetaData) -> Dict[str, Table]:
    """Define the exact relational schema for historical_freight and port_constraints."""
    
    historical_freight = Table(
        "historical_freight",
        metadata,
        Column("date", TEXT, nullable=True),
        Column("origin", TEXT, nullable=True),
        Column("destination", TEXT, nullable=True),
        Column("distance_nm", REAL, nullable=True),
        Column("bdi_index", REAL, nullable=True),
        Column("vlsfo_price", REAL, nullable=True),
        Column("is_monsoon", INTEGER, nullable=True),
        Column("freight_rate_usd", REAL, nullable=True),
    )

    port_constraints = Table(
        "port_constraints",
        metadata,
        Column("category", TEXT, nullable=True),
        Column("port", TEXT, nullable=True),
        Column("country_or_state", TEXT, nullable=True),
        Column("authority_or_region", TEXT, nullable=True),
        Column("type_or_commodity", TEXT, nullable=True),
        Column("max_loa_m", REAL, nullable=True),
        Column("max_beam_m", REAL, nullable=True),
        Column("max_draft_m", REAL, nullable=True),
        Column("max_dwt", REAL, nullable=True),
        Column("max_vessel_class", TEXT, nullable=True),
        Column("berths", REAL, nullable=True),
        Column("notes", TEXT, nullable=True),
    )

    return {
        "historical_freight": historical_freight,
        "port_constraints": port_constraints,
    }


def process_and_ingest(engine, tables: Dict[str, Table]) -> None:
    """Load CSVs, transform types, clean nulls, and ingest into SQLite."""
    
    # 1. Resolve and validate CSV paths
    freight_path = resolve_file_path(FREIGHT_CSV_NAME, "Historical Freight Rates")
    port_path = resolve_file_path(PORT_CSV_NAME, "Port Constraints")

    # 2. Load and validate raw CSVs
    logger.info("Loading freight rates from: %s", freight_path.name)
    df_freight = load_and_validate_csv(freight_path, "Historical Freight Rates", expected_min_cols=7)
    
    logger.info("Loading port constraints from: %s", port_path.name)
    df_ports = load_and_validate_csv(port_path, "Port Constraints", expected_min_cols=10)

    # 3. Transform & clean historical_freight data
    freight_records: List[Dict[str, Any]] = []
    for _, row in df_freight.iterrows():
        record = {
            "date": clean_iso_date_value(row.get("date")),
            "origin": clean_text_value(row.get("origin")),
            "destination": clean_text_value(row.get("destination")),
            "distance_nm": clean_real_value(row.get("distance_nm")),
            "bdi_index": clean_real_value(row.get("bdi_index")),
            "vlsfo_price": clean_real_value(row.get("vlsfo_price")),
            "is_monsoon": clean_boolean_value(row.get("is_monsoon")),
            "freight_rate_usd": clean_real_value(row.get("freight_rate_usd")),
        }
        freight_records.append(record)

    # 4. Transform & clean port_constraints data
    port_records: List[Dict[str, Any]] = []
    for _, row in df_ports.iterrows():
        record = {
            "category": clean_text_value(row.get("category")),
            "port": clean_text_value(row.get("port")),
            "country_or_state": clean_text_value(row.get("country_or_state")),
            "authority_or_region": clean_text_value(row.get("authority_or_region")),
            "type_or_commodity": clean_text_value(row.get("type_or_commodity")),
            "max_loa_m": clean_real_value(row.get("max_loa_m")),
            "max_beam_m": clean_real_value(row.get("max_beam_m")),
            "max_draft_m": clean_real_value(row.get("max_draft_m")),
            "max_dwt": clean_real_value(row.get("max_dwt")),
            "max_vessel_class": clean_text_value(row.get("max_vessel_class")),
            "berths": clean_real_value(row.get("berths")),
            "notes": clean_text_value(row.get("notes")),
        }
        port_records.append(record)

    # 5. Insert records using SQLAlchemy connection transaction
    with engine.begin() as conn:
        logger.info("Ingesting %d records into 'historical_freight'...", len(freight_records))
        conn.execute(tables["historical_freight"].insert(), freight_records)
        
        logger.info("Ingesting %d records into 'port_constraints'...", len(port_records))
        conn.execute(tables["port_constraints"].insert(), port_records)
    
    logger.info("Ingestion completed successfully.")


def print_database_summary(engine) -> None:
    """Query both tables and print row count, first 3 rows, and column schema."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    print("\n" + "=" * 80)
    print("DATABASE INSPECTION & VERIFICATION REPORT")
    print(f"Database Location: {DB_FILE}")
    print("=" * 80)

    with engine.connect() as conn:
        for table_name in ["historical_freight", "port_constraints"]:
            if table_name not in table_names:
                print(f"\n[ERROR] Table '{table_name}' was not found in the database!")
                continue

            print(f"\n--- TABLE: {table_name} ---")
            
            # Column names and types
            columns = inspector.get_columns(table_name)
            print("\nSchema (Column Names & Types):")
            for col in columns:
                nullable_str = "NULL" if col.get("nullable", True) else "NOT NULL"
                print(f"  - {col['name']:<22} {str(col['type']):<10} ({nullable_str})")

            # Row count
            count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
            print(f"\nTotal Row Count: {count_result}")

            # First 3 rows
            sample_query = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3"))
            rows = sample_query.fetchall()
            col_keys = list(sample_query.keys())

            print("\nFirst 3 Rows:")
            df_sample = pd.DataFrame(rows, columns=col_keys)
            print(df_sample.to_string(index=False))
            print("-" * 80)

    print("\n[SUCCESS] logistics_platform.db is verified and ready for downstream analytical queries.")
    print("=" * 80 + "\n")


def setup_database() -> None:
    """Main idempotent orchestration function."""
    logger.info("Starting database setup for: %s", DB_FILE.name)

    # Initialize SQLAlchemy engine
    engine = create_engine(DB_URL, echo=False)
    metadata = MetaData()

    # Define tables
    tables = define_schema(metadata)

    # Idempotence: Drop all existing tables and recreate afresh
    logger.info("Dropping existing tables (if any) to ensure idempotency...")
    metadata.drop_all(engine)
    
    logger.info("Creating fresh tables...")
    metadata.create_all(engine)

    # Ingest data
    process_and_ingest(engine, tables)

    # Inspect and verify output
    print_database_summary(engine)


if __name__ == "__main__":
    try:
        setup_database()
    except Exception as e:
        logger.error("Database setup failed: %s", e, exc_info=True)
        sys.exit(1)
