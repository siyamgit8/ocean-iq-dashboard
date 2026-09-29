"""
utils/feasibility.py - Physical Port and Vessel Feasibility Module
Evaluates vessel draft, LOA, and beam against both Origin (loading) and Destination (discharge) port constraints.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

# Base Directory & Database Candidates
BASE_DIR = Path(__file__).resolve().parent.parent
DB_CANDIDATES = [
    BASE_DIR / "logistics_platform.db",
    Path.cwd() / "logistics_platform.db",
    BASE_DIR.parent / "logistics_platform.db",
    Path.cwd().parent / "logistics_platform.db",
]

# Standard Dry Bulk Vessel Dimensions
VESSEL_SPECS: Dict[str, Dict[str, Any]] = {
    "Handysize": {
        "capacity_mt": 35000,
        "draft_m": 8.5,
        "loa_m": 180.0,
        "beam_m": 28.0,
        "discount": 0.00,
        "daily_hire_usd": 12000,
        "description": "Small geared bulk carrier, high port accessibility, baseline rate"
    },
    "Supramax": {
        "capacity_mt": 55000,
        "draft_m": 11.5,
        "loa_m": 200.0,
        "beam_m": 32.0,
        "discount": 0.05,
        "daily_hire_usd": 14500,
        "description": "Mid-size bulk carrier, versatile geared vessel with 5% volume discount"
    },
    "Panamax": {
        "capacity_mt": 75000,
        "draft_m": 13.5,
        "loa_m": 225.0,
        "beam_m": 32.5,
        "discount": 0.10,
        "daily_hire_usd": 17000,
        "description": "Standard gearless coal carrier, optimized for major bulk terminals (10% discount)"
    },
    "Capesize": {
        "capacity_mt": 170000,
        "draft_m": 17.5,
        "loa_m": 290.0,
        "beam_m": 45.0,
        "discount": 0.15,
        "daily_hire_usd": 26000,
        "description": "Ultra-large deep-draft bulk carrier, lowest $/tonne baseline (15% discount)"
    }
}

# Port typical waiting days and turnaround factors
PORT_CONGESTION_DATA: Dict[str, Dict[str, Any]] = {
    "Paradip": {"avg_wait_days": 4.5, "congestion_index": 0.78, "berth_productivity_tpd": 35000, "status": "High Congestion"},
    "Haldia": {"avg_wait_days": 5.0, "congestion_index": 0.85, "berth_productivity_tpd": 18000, "status": "Severe Congestion / River Draft Delay"},
    "Visakhapatnam (Vizag)": {"avg_wait_days": 2.2, "congestion_index": 0.45, "berth_productivity_tpd": 40000, "status": "Moderate Congestion"},
    "Vizag": {"avg_wait_days": 2.2, "congestion_index": 0.45, "berth_productivity_tpd": 40000, "status": "Moderate Congestion"},
    "Gangavaram": {"avg_wait_days": 0.8, "congestion_index": 0.18, "berth_productivity_tpd": 65000, "status": "Fluid / Fast Turnaround"},
    "Dhamra": {"avg_wait_days": 1.2, "congestion_index": 0.25, "berth_productivity_tpd": 55000, "status": "Normal / Fast Turnaround"},
    "Gopalpur": {"avg_wait_days": 1.5, "congestion_index": 0.30, "berth_productivity_tpd": 30000, "status": "Normal Operation"},
}


def get_db_engine():
    """Locate logistics_platform.db and return a SQLAlchemy engine."""
    for candidate in DB_CANDIDATES:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return create_engine(f"sqlite:///{candidate.resolve().as_posix()}", echo=False)
            
    searched = "\n  - ".join(str(p) for p in DB_CANDIDATES)
    raise FileNotFoundError(f"Database 'logistics_platform.db' not found. Searched:\n  - {searched}")


def check_single_port_feasibility(port_name: str) -> Dict[str, Any]:
    """
    Check physical and navigational feasibility of all vessel classes for any given port (origin or destination).
    """
    engine = get_db_engine()
    query = text("SELECT category, port, country_or_state, max_draft_m, max_loa_m, max_beam_m, max_dwt, max_vessel_class, notes FROM port_constraints")
    
    with engine.connect() as conn:
        df_ports = pd.read_sql(query, conn)
        
    if df_ports.empty:
        raise ValueError("port_constraints table is empty in logistics_platform.db")

    target = port_name.strip().lower()
    
    # Clean target string (e.g. 'Baltimore (USA)' -> 'baltimore')
    base_target = target.split("(")[0].strip()
    
    # 1. Exact match
    matched = df_ports[df_ports["port"].str.strip().str.lower() == target]
    
    # 2. Base target match
    if matched.empty:
        matched = df_ports[df_ports["port"].str.strip().str.lower().str.startswith(base_target)]
        
    # 3. Substring match
    if matched.empty:
        matched = df_ports[df_ports["port"].str.strip().str.lower().str.contains(base_target, regex=False)]

    if matched.empty:
        # Return fallback with standard defaults if port is unknown
        return {
            "port_name": port_name,
            "category": "unknown",
            "country_or_state": "International",
            "port_constraints": {
                "max_draft_m": 16.5,
                "max_loa_m": 300.0,
                "max_beam_m": 50.0,
                "max_dwt": 180000.0,
                "notes": "Standard international dry bulk deep-water specifications."
            },
            "vessel_status": {
                v: {"feasible": True, "reasons": [], "specs": spec}
                for v, spec in VESSEL_SPECS.items()
            },
            "allowed_vessels": list(VESSEL_SPECS.keys()),
            "blocked_vessels": []
        }

    row = matched.iloc[0]
    matched_port_name = str(row["port"])
    category = str(row["category"])
    country = str(row["country_or_state"])
    max_draft = float(row["max_draft_m"]) if pd.notna(row["max_draft_m"]) else None
    max_loa = float(row["max_loa_m"]) if pd.notna(row["max_loa_m"]) else None
    max_beam = float(row["max_beam_m"]) if pd.notna(row["max_beam_m"]) else None
    max_dwt = float(row["max_dwt"]) if pd.notna(row["max_dwt"]) else None
    notes = str(row["notes"]) if pd.notna(row["notes"]) else ""

    vessel_status: Dict[str, Dict[str, Any]] = {}
    allowed_vessels: List[str] = []
    blocked_vessels: List[str] = []

    for v_name, spec in VESSEL_SPECS.items():
        reasons = []
        
        # Check draft
        if max_draft is not None and spec["draft_m"] > max_draft:
            reasons.append(f"draft {spec['draft_m']:.1f}m > port max {max_draft:.1f}m")
            
        # Check LOA
        if max_loa is not None and spec["loa_m"] > max_loa:
            reasons.append(f"LOA {spec['loa_m']:.0f}m > port max {max_loa:.0f}m")
            
        # Check beam
        if max_beam is not None and spec["beam_m"] > max_beam:
            reasons.append(f"beam {spec['beam_m']:.1f}m > port max {max_beam:.1f}m")

        is_feasible = (len(reasons) == 0)
        vessel_status[v_name] = {
            "feasible": is_feasible,
            "reasons": reasons,
            "specs": spec
        }
        
        if is_feasible:
            allowed_vessels.append(v_name)
        else:
            blocked_vessels.append(v_name)

    return {
        "port_name": matched_port_name,
        "category": category,
        "country_or_state": country,
        "port_constraints": {
            "max_draft_m": max_draft,
            "max_loa_m": max_loa,
            "max_beam_m": max_beam,
            "max_dwt": max_dwt,
            "notes": notes
        },
        "vessel_status": vessel_status,
        "allowed_vessels": allowed_vessels,
        "blocked_vessels": blocked_vessels
    }


def check_feasibility(destination_port: str) -> Dict[str, Any]:
    """Compatibility wrapper for destination port feasibility check."""
    return check_single_port_feasibility(destination_port)


def check_dual_route_feasibility(origin_port: str, destination_port: str) -> Dict[str, Any]:
    """
    Evaluate end-to-end navigational feasibility across BOTH origin loading port and destination discharge port.
    A vessel is only route-feasible if it complies with both origin and destination restrictions.
    """
    orig_res = check_single_port_feasibility(origin_port)
    dest_res = check_single_port_feasibility(destination_port)

    route_vessel_status: Dict[str, Dict[str, Any]] = {}
    route_allowed: List[str] = []
    route_blocked: List[str] = []

    for v_name in VESSEL_SPECS:
        orig_v = orig_res["vessel_status"][v_name]
        dest_v = dest_res["vessel_status"][v_name]
        
        all_reasons = []
        if not orig_v["feasible"]:
            all_reasons.extend([f"Origin ({orig_res['port_name']}): {r}" for r in orig_v["reasons"]])
        if not dest_v["feasible"]:
            all_reasons.extend([f"Destination ({dest_res['port_name']}): {r}" for r in dest_v["reasons"]])
            
        feasible = (len(all_reasons) == 0)
        route_vessel_status[v_name] = {
            "feasible": feasible,
            "reasons": all_reasons,
            "specs": VESSEL_SPECS[v_name]
        }
        
        if feasible:
            route_allowed.append(v_name)
        else:
            route_blocked.append(v_name)

    return {
        "origin": orig_res,
        "destination": dest_res,
        "route_vessel_status": route_vessel_status,
        "route_allowed_vessels": route_allowed,
        "route_blocked_vessels": route_blocked
    }


def calculate_demurrage_and_idle_risk(
    destination_port: str,
    vessel_class: str,
    fleet_count: int,
    cargo_volume_mt: float
) -> Dict[str, Any]:
    """
    Calculate financial demurrage exposure and vessel idle time costs based on port waiting times.
    """
    dest_clean = destination_port.split("(")[0].strip()
    cong = PORT_CONGESTION_DATA.get(dest_clean, PORT_CONGESTION_DATA.get(destination_port, {
        "avg_wait_days": 2.0,
        "congestion_index": 0.40,
        "berth_productivity_tpd": 40000,
        "status": "Moderate Operation"
    }))

    spec = VESSEL_SPECS.get(vessel_class, VESSEL_SPECS["Panamax"])
    daily_hire = spec.get("daily_hire_usd", 17000)
    
    # Demurrage is typically calculated at 1.0x to 1.25x the daily charter hire rate
    daily_demurrage_rate = daily_hire * 1.15
    
    # Free laytime allowed under standard charterparty (typically 3.0 days for loading/unloading)
    allowed_laytime_days = 3.0
    
    # Discharge duration based on berth productivity
    tpd = cong["berth_productivity_tpd"]
    discharge_days = (cargo_volume_mt / max(tpd, 10000)) / max(fleet_count, 1)
    
    total_port_stay_days = cong["avg_wait_days"] + discharge_days
    excess_idle_days = max(0.0, total_port_stay_days - allowed_laytime_days)
    
    # Demurrage cost per vessel and total fleet demurrage
    demurrage_per_vessel = excess_idle_days * daily_demurrage_rate
    total_demurrage_exposure = demurrage_per_vessel * fleet_count
    
    # Diversion savings potential (comparing against fluid deep-water Gangavaram)
    gangavaram_wait = PORT_CONGESTION_DATA["Gangavaram"]["avg_wait_days"]
    gangavaram_discharge = (cargo_volume_mt / PORT_CONGESTION_DATA["Gangavaram"]["berth_productivity_tpd"]) / max(fleet_count, 1)
    gangavaram_stay = gangavaram_wait + gangavaram_discharge
    gangavaram_excess = max(0.0, gangavaram_stay - allowed_laytime_days)
    gangavaram_demurrage = gangavaram_excess * daily_demurrage_rate * fleet_count
    
    diversion_savings = max(0.0, total_demurrage_exposure - gangavaram_demurrage)

    return {
        "destination_port": destination_port,
        "port_status": cong["status"],
        "avg_wait_days": cong["avg_wait_days"],
        "discharge_days": round(discharge_days, 1),
        "total_port_stay_days": round(total_port_stay_days, 1),
        "excess_idle_days": round(excess_idle_days, 1),
        "daily_demurrage_rate_usd": daily_demurrage_rate,
        "total_demurrage_exposure_usd": round(total_demurrage_exposure, 2),
        "demurrage_cost_per_tonne": round(total_demurrage_exposure / max(cargo_volume_mt, 1), 2),
        "gangavaram_diversion_savings_usd": round(diversion_savings, 2)
    }
