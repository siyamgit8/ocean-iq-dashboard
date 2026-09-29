"""
utils/predictor.py - Machine Learning Freight Rate Predictor and Forecaster
Loads trained XGBoost model and provides inference, 30-day time-series forecasting, and contract timing recommendations.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text

# Base Directory & Artifact Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    BASE_DIR / "freight_xgboost_model.pkl",
    Path.cwd() / "freight_xgboost_model.pkl",
    BASE_DIR.parent / "freight_xgboost_model.pkl",
]

FEATURES_CANDIDATES = [
    BASE_DIR / "model_features.pkl",
    Path.cwd() / "model_features.pkl",
    BASE_DIR.parent / "model_features.pkl",
]

DB_CANDIDATES = [
    BASE_DIR / "logistics_platform.db",
    Path.cwd() / "logistics_platform.db",
    BASE_DIR.parent / "logistics_platform.db",
]

METRICS_CANDIDATES = [
    BASE_DIR / "metrics.json",
    Path.cwd() / "metrics.json",
    BASE_DIR.parent / "metrics.json",
]

# Standard Route Distance Mapping (Nautical Miles)
ROUTE_DISTANCES: Dict[str, float] = {
    "Hay Point (Australia)": 4800.0,
    "Newcastle (Australia)": 5100.0,
    "Baltimore (USA)": 9500.0,
    "Hampton Roads (USA)": 9300.0,
    "Nacala (Mozambique)": 4300.0,
    "Richards Bay (South Africa)": 4300.0,
    "Taboneo (Indonesia)": 2800.0,
    "Kalimantan (Indonesia)": 2800.0,
    "Taman (Russia)": 6200.0,
    "Vostochny (Russia)": 5100.0,
}

# Origin Normalization Mapping to Model Training Categories
ORIGIN_NORMALIZER: Dict[str, str] = {
    "Hay Point (Australia)": "Hay Point (Australia)",
    "Newcastle (Australia)": "Hay Point (Australia)",
    "Baltimore (USA)": "Baltimore (USA)",
    "Hampton Roads (USA)": "Baltimore (USA)",
    "Nacala (Mozambique)": "Richards Bay (South Africa)",
    "Richards Bay (South Africa)": "Richards Bay (South Africa)",
    "Taboneo (Indonesia)": "Kalimantan (Indonesia)",
    "Kalimantan (Indonesia)": "Kalimantan (Indonesia)",
    "Taman (Russia)": "Baltimore (USA)",
    "Vostochny (Russia)": "Hay Point (Australia)",
}


def _resolve_file(candidates: List[Path], desc: str) -> Path:
    for c in candidates:
        if c.exists() and c.is_file() and c.stat().st_size > 0:
            return c.resolve()
    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Required artifact '{desc}' not found. Searched:\n  - {searched}")


class FreightPredictor:
    """Production ML Predictor for Maritime Coking Coal Freight Rates."""

    def __init__(self):
        model_path = _resolve_file(MODEL_CANDIDATES, "freight_xgboost_model.pkl")
        features_path = _resolve_file(FEATURES_CANDIDATES, "model_features.pkl")
        self.db_path = _resolve_file(DB_CANDIDATES, "logistics_platform.db")

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
        
        # Load metrics for uncertainty estimation
        self.rmse = 3.26
        try:
            metrics_path = _resolve_file(METRICS_CANDIDATES, "metrics.json")
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.rmse = data.get("test_metrics", {}).get("rmse", 3.26)
        except Exception:
            pass

        self.engine = create_engine(f"sqlite:///{self.db_path.as_posix()}", echo=False)

    def predict_rate(
        self,
        origin_port: str,
        destination_port: str,
        vessel_class: str = "Capesize",
        contract_type: str = "Spot",
        eval_date: Optional[datetime] = None,
        bdi_index: Optional[float] = None,
        vlsfo_price: Optional[float] = None,
        is_monsoon: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Predict point freight rate ($/ton) for a specific voyage.
        """
        if eval_date is None:
            eval_date = datetime.now()

        month = eval_date.month
        day_of_year = eval_date.timetuple().tm_yday
        if is_monsoon is None:
            is_monsoon = 1 if month in [6, 7, 8, 9] else 0

        # Defaults based on recent market indicators if not supplied
        if bdi_index is None:
            bdi_index = 1850.0
        if vlsfo_price is None:
            vlsfo_price = 540.0

        vlsfo_lag7 = vlsfo_price * 0.99
        bdi_lag7 = bdi_index * 0.985
        coal_price_usd = 250.0 + (bdi_index * 0.05)
        dist_nm = ROUTE_DISTANCES.get(origin_port, 4800.0)

        # Build feature dictionary
        row_dict: Dict[str, Any] = {
            "distance_nm": dist_nm,
            "bdi_index": bdi_index,
            "vlsfo_price": vlsfo_price,
            "is_monsoon": is_monsoon,
            "coal_price_usd": coal_price_usd,
            "port_congestion_origin": 2.5,
            "port_congestion_dest": 3.0,
            "month": month,
            "day_of_year": day_of_year,
            "vlsfo_price_lag7": vlsfo_lag7,
            "bdi_index_lag7": bdi_lag7,
        }

        # Initialize all one-hot columns to 0
        for feat in self.feature_names:
            if feat not in row_dict:
                row_dict[feat] = 0

        # Set specific active category indicators
        norm_origin = ORIGIN_NORMALIZER.get(origin_port, origin_port)
        origin_col = f"origin_{norm_origin}"
        if origin_col in self.feature_names:
            row_dict[origin_col] = 1

        dest_col = "destination_Indian East Coast (Paradip/Vizag/Gangavaram)"
        if dest_col in self.feature_names:
            row_dict[dest_col] = 1

        vessel_col = f"vessel_class_{vessel_class}"
        if vessel_col in self.feature_names:
            row_dict[vessel_col] = 1

        charter_col = f"charter_type_{contract_type}"
        if charter_col in self.feature_names:
            row_dict[charter_col] = 1

        # DataFrame in exact feature order
        X_df = pd.DataFrame([row_dict])[self.feature_names]
        predicted_val = float(self.model.predict(X_df)[0])
        
        # Adjust for Russian specific route distance ratio if applicable
        if "Russia" in origin_port:
            base_dist = ROUTE_DISTANCES.get(norm_origin, 4800.0)
            actual_dist = ROUTE_DISTANCES.get(origin_port, 5500.0)
            predicted_val = predicted_val * (0.65 + 0.35 * (actual_dist / base_dist))

        predicted_rate = max(5.0, round(predicted_val, 2))

        # Standard uncertainty range delta (e.g. ± $0.42 / ton standard error)
        uncertainty_delta = round(self.rmse * 0.13, 2)

        return {
            "predicted_rate_usd": predicted_rate,
            "uncertainty_delta": uncertainty_delta,
            "confidence_lower": round(max(4.0, predicted_rate - uncertainty_delta), 2),
            "confidence_upper": round(predicted_rate + uncertainty_delta, 2),
            "origin_port": origin_port,
            "destination_port": destination_port,
            "vessel_class": vessel_class,
            "contract_type": contract_type,
            "distance_nm": dist_nm,
            "bdi_index": bdi_index,
            "vlsfo_price": vlsfo_price,
            "is_monsoon": is_monsoon
        }

    def get_historical_and_forecast(
        self,
        origin_port: str,
        destination_port: str,
        days_history: int = 60,
        days_forecast: int = 30
    ) -> pd.DataFrame:
        """
        Fetch last 60 days of historical rates from database,
        and generate 30-day forward forecast with confidence bounds.
        """
        norm_origin = ORIGIN_NORMALIZER.get(origin_port, "Hay Point (Australia)")
        
        query = text("""
            SELECT date, freight_rate_usd as rate, bdi_index, vlsfo_price, is_monsoon
            FROM historical_freight
            WHERE origin LIKE :orig
            ORDER BY date DESC
            LIMIT :lim
        """)
        
        with self.engine.connect() as conn:
            hist_df = pd.read_sql(query, conn, params={"orig": f"%{norm_origin}%", "lim": days_history})

        if hist_df.empty:
            # Fallback query if specific origin not matched
            with self.engine.connect() as conn:
                hist_df = pd.read_sql(
                    text("SELECT date, freight_rate_usd as rate, bdi_index, vlsfo_price, is_monsoon FROM historical_freight ORDER BY date DESC LIMIT :lim"),
                    conn,
                    params={"lim": days_history}
                )

        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df = hist_df.sort_values("date", ascending=True).reset_index(drop=True)
        hist_df["type"] = "Historical"
        hist_df["upper_band"] = np.nan
        hist_df["lower_band"] = np.nan

        last_date = hist_df["date"].max() if not hist_df.empty else datetime(2025, 12, 30)
        last_bdi = hist_df["bdi_index"].iloc[-1] if not hist_df.empty else 1800.0
        last_vlsfo = hist_df["vlsfo_price"].iloc[-1] if not hist_df.empty else 540.0

        # Generate 30-day forward projection
        forecast_rows = []
        np.random.seed(42)
        
        curr_bdi = last_bdi
        curr_vlsfo = last_vlsfo
        
        for d in range(1, days_forecast + 1):
            f_date = last_date + timedelta(days=d)
            curr_bdi = max(800.0, curr_bdi + np.random.normal(0, 15))
            curr_vlsfo = max(350.0, curr_vlsfo + np.random.normal(0, 2))
            
            pred = self.predict_rate(
                origin_port=origin_port,
                destination_port=destination_port,
                eval_date=f_date,
                bdi_index=curr_bdi,
                vlsfo_price=curr_vlsfo
            )
            f_rate = pred["predicted_rate_usd"]
            
            # 15% confidence band
            band_spread = f_rate * 0.15
            forecast_rows.append({
                "date": f_date,
                "rate": f_rate,
                "bdi_index": curr_bdi,
                "vlsfo_price": curr_vlsfo,
                "is_monsoon": pred["is_monsoon"],
                "type": "Forecast",
                "upper_band": round(f_rate + band_spread, 2),
                "lower_band": round(max(3.0, f_rate - band_spread), 2)
            })

        forecast_df = pd.DataFrame(forecast_rows)
        
        # Connect the last historical point with forecast start for continuous line rendering
        if not hist_df.empty:
            bridge_point = hist_df.iloc[-1:].copy()
            bridge_point["type"] = "Forecast"
            bridge_point["upper_band"] = bridge_point["rate"] * 1.15
            bridge_point["lower_band"] = bridge_point["rate"] * 0.85
            forecast_df = pd.concat([bridge_point, forecast_df], ignore_index=True)

        combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
        return combined_df

    def evaluate_contract_timing_strategy(
        self,
        origin_port: str,
        destination_port: str,
        current_rate: float,
        cargo_volume_mt: float
    ) -> Dict[str, Any]:
        """
        Evaluate Spot vs Mid-Term Contract (3-6 month COA / Time Charter) based on 30-day forward momentum.
        """
        df_chart = self.get_historical_and_forecast(origin_port, destination_port, days_history=15, days_forecast=30)
        fore_df = df_chart[df_chart["type"] == "Forecast"]
        
        start_rate = fore_df["rate"].iloc[0] if not fore_df.empty else current_rate
        end_rate = fore_df["rate"].iloc[-1] if not fore_df.empty else current_rate
        avg_fore_rate = fore_df["rate"].mean() if not fore_df.empty else current_rate
        
        rate_change_pct = ((end_rate - start_rate) / max(start_rate, 0.01)) * 100.0
        
        # Mid-term COA typically offers a 4-8% stability discount from charterers
        mid_term_fixed_rate = current_rate * 0.95
        projected_spot_cost = avg_fore_rate * cargo_volume_mt
        projected_midterm_cost = mid_term_fixed_rate * cargo_volume_mt
        projected_savings = projected_spot_cost - projected_midterm_cost

        if rate_change_pct > 2.0:
            recommendation = "Lock in Mid-Term Contract (3–6 Months COA)"
            rec_type = "mid_term"
            badge = "🔥 Bullish Market (Rate Inflation Expected)"
            rationale = (
                f"Freight momentum is trending upward (+{rate_change_pct:.1f}% over 30 days). "
                f"Securing a 3-6 month Contract of Affreightment (COA) at current rates (~${mid_term_fixed_rate:.2f}/t) "
                f"hedges against spot price escalation, saving approx. ${abs(projected_savings):,.2f}."
            )
        elif rate_change_pct < -2.0:
            recommendation = "Maintain Spot Market Fixtures"
            rec_type = "spot"
            badge = "📉 Bearish Market (Softening Rates Expected)"
            rationale = (
                f"Freight rates are projected to soften ({rate_change_pct:.1f}% over 30 days). "
                f"Executing consecutive Spot charters allows SAIL/RINL to capture progressively lower voyage costs "
                f"without being locked into fixed premiums."
            )
        else:
            recommendation = "Hybrid Strategy (50% Index-Linked COA + 50% Spot)"
            rec_type = "hybrid"
            badge = "⚖️ Neutral / Stable Freight Trend"
            rationale = (
                f"Market momentum is flat ({rate_change_pct:+.1f}%). "
                f"A hybrid structure balances guaranteed loading slots with spot market flexibility."
            )

        return {
            "recommendation": recommendation,
            "rec_type": rec_type,
            "badge": badge,
            "rate_change_pct": round(rate_change_pct, 2),
            "current_spot_rate": current_rate,
            "avg_forecast_rate": round(avg_fore_rate, 2),
            "mid_term_fixed_rate": round(mid_term_fixed_rate, 2),
            "projected_spot_cost": round(projected_spot_cost, 2),
            "projected_midterm_cost": round(projected_midterm_cost, 2),
            "projected_savings_usd": round(projected_savings, 2),
            "rationale": rationale
        }
