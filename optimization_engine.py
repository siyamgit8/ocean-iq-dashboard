"""
optimization_engine.py - Mixed-Integer Linear Programming (MILP) Vessel Chartering Optimizer
Phase 3: The Prescriptive Engine
Uses PuLP and SQLAlchemy to find the cost-optimal vessel chartering fleet subject to physical port and vessel constraints.
"""

import sys
import os
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure clean UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import pandas as pd
import pulp
from sqlalchemy import create_engine, text, inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("optimization_engine")

# Database File Path Resolution
BASE_DIR = Path(__file__).resolve().parent
DB_CANDIDATES = [
    BASE_DIR / "logistics_platform.db",
    Path.cwd() / "logistics_platform.db",
    BASE_DIR.parent / "logistics_platform.db",
    Path.cwd().parent / "logistics_platform.db",
    Path.home() / "Downloads" / "SIH_NEW" / "Antigravity_sih" / "logistics_platform.db",
    Path.home() / "Downloads" / "SIH_NEW" / "logistics_platform.db",
]

# ═══════════════════════════════════════════════════════════════════════════════
# VESSEL SPECIFICATIONS (Standard Dry Bulk Shipping Classes)
# Source: BIMCO / Clarkson Research / Port and Vessel Feasibility Standards
# ═══════════════════════════════════════════════════════════════════════════════
VESSEL_SPECS: Dict[str, Dict[str, Any]] = {
    "Handysize": {
        "capacity_mt": 35000,
        "draft_m": 8.5,
        "loa_m": 180.0,
        "beam_m": 28.0,
        "discount": 0.00,  # Baseline rate (0% economy of scale discount)
        "description": "Small geared bulk carrier, high port accessibility, no volume discount"
    },
    "Supramax": {
        "capacity_mt": 55000,
        "draft_m": 11.5,
        "loa_m": 200.0,
        "beam_m": 32.0,
        "discount": 0.05,  # 5% economy of scale discount
        "description": "Mid-size bulk carrier, versatile geared vessel with 5% volume discount"
    },
    "Panamax": {
        "capacity_mt": 75000,
        "draft_m": 13.5,
        "loa_m": 225.0,
        "beam_m": 32.5,
        "discount": 0.10,  # 10% economy of scale discount
        "description": "Standard gearless coal carrier, optimized for major bulk terminals (10% discount)"
    },
    "Capesize": {
        "capacity_mt": 170000,
        "draft_m": 17.5,
        "loa_m": 290.0,
        "beam_m": 45.0,
        "discount": 0.15,  # 15% economy of scale discount
        "description": "Ultra-large deep-draft bulk carrier, lowest $/tonne baseline (15% discount)"
    }
}


def get_db_engine():
    """Locate logistics_platform.db and return a SQLAlchemy engine."""
    for candidate in DB_CANDIDATES:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return create_engine(f"sqlite:///{candidate.resolve().as_posix()}", echo=False)
            
    searched = "\n  - ".join(str(p) for p in DB_CANDIDATES[:4])
    raise FileNotFoundError(f"Database 'logistics_platform.db' not found. Searched:\n  - {searched}")


def fetch_port_constraints(destination_port: str) -> Dict[str, Any]:
    """
    Fetch max_draft_m, max_loa_m, max_beam_m from port_constraints in database.
    Raises ValueError if destination port is not found.
    """
    engine = get_db_engine()
    query = text("SELECT port, max_draft_m, max_loa_m, max_beam_m, max_dwt, notes FROM port_constraints")
    
    with engine.connect() as conn:
        df_ports = pd.read_sql(query, conn)
        
    if df_ports.empty:
        raise ValueError("port_constraints table is empty in logistics_platform.db")

    target = destination_port.strip().lower()
    
    # 1. Exact match (case-insensitive)
    matched = df_ports[df_ports["port"].str.strip().str.lower() == target]
    
    # 2. If no exact match, try partial / alias match (e.g. 'Vizag' in 'Visakhapatnam (Vizag)')
    if matched.empty:
        matched = df_ports[df_ports["port"].str.strip().str.lower().str.contains(target, regex=False)]

    if matched.empty:
        available_ports = ", ".join(f"'{p}'" for p in df_ports["port"].tolist())
        raise ValueError(
            f"Destination port '{destination_port}' not found in port_constraints database.\n"
            f"Available ports: {available_ports}"
        )

    row = matched.iloc[0]
    return {
        "matched_port_name": str(row["port"]),
        "max_draft_m": float(row["max_draft_m"]) if pd.notna(row["max_draft_m"]) else None,
        "max_loa_m": float(row["max_loa_m"]) if pd.notna(row["max_loa_m"]) else None,
        "max_beam_m": float(row["max_beam_m"]) if pd.notna(row["max_beam_m"]) else None,
        "max_dwt": float(row["max_dwt"]) if pd.notna(row["max_dwt"]) else None,
        "notes": str(row["notes"]) if pd.notna(row["notes"]) else ""
    }


def check_historical_feasibility(destination_port: str, vessel_class: str) -> bool:
    """
    Check historical_freight table for recorded infeasibility flags for (vessel_class, destination_port).
    Returns False if explicitly recorded as is_feasible = 0, otherwise True.
    """
    engine = get_db_engine()
    inspector = inspect(engine)
    
    if "historical_freight" not in inspector.get_table_names():
        return True
        
    cols = [c["name"] for c in inspector.get_columns("historical_freight")]
    if "is_feasible" not in cols or "vessel_class" not in cols:
        return True

    query = text("""
        SELECT is_feasible 
        FROM historical_freight 
        WHERE LOWER(destination) LIKE :dest AND LOWER(vessel_class) = :vclass
        LIMIT 10
    """)
    
    with engine.connect() as conn:
        res = conn.execute(query, {
            "dest": f"%{destination_port.lower()}%",
            "vclass": vessel_class.lower()
        }).fetchall()
        
    if res and all(r[0] == 0 for r in res):
        return False
    return True


def run_charter_optimization(
    cargo_volume_mt: float,
    origin_port: str,
    destination_port: str,
    predicted_freight_rate_usd: float
) -> Dict[str, Any]:
    """
    Solve Mixed-Integer Linear Programming (MILP) vessel chartering optimization problem.
    
    Parameters:
        cargo_volume_mt (float): Total metric tonnes of coal to transport.
        origin_port (str): Origin loading port.
        destination_port (str): Discharge destination port.
        predicted_freight_rate_usd (float): ML predicted baseline freight rate ($/ton).
        
    Returns:
        dict: Optimization results containing status, fleet composition, capacity, slack, and costs.
    """
    if cargo_volume_mt <= 0:
        raise ValueError(f"Cargo volume must be positive. Received: {cargo_volume_mt}")
    if predicted_freight_rate_usd <= 0:
        raise ValueError(f"Predicted freight rate must be positive. Received: {predicted_freight_rate_usd}")

    # 1. Fetch physical port limits
    port_info = fetch_port_constraints(destination_port)
    matched_port = port_info["matched_port_name"]
    max_draft = port_info["max_draft_m"]
    max_loa = port_info["max_loa_m"]
    max_beam = port_info["max_beam_m"]

    # 2. Build PuLP MILP Model
    prob = pulp.LpProblem("Vessel_Chartering_Optimization", pulp.LpMinimize)
    
    n_vars: Dict[str, pulp.LpVariable] = {}
    blocked_vessels: List[Dict[str, Any]] = []
    blocked_names: List[str] = []

    # Constraint 1: Physical Port Limits & Practical Upper Bounds
    for v_name, spec in VESSEL_SPECS.items():
        reasons = []
        
        # Check draft constraint
        if max_draft is not None and spec["draft_m"] > max_draft:
            reasons.append(f"draft {spec['draft_m']}m > port max {max_draft}m")
            
        # Check LOA constraint
        if max_loa is not None and spec["loa_m"] > max_loa:
            reasons.append(f"LOA {spec['loa_m']}m > port max {max_loa}m")
            
        # Check beam constraint
        if max_beam is not None and spec["beam_m"] > max_beam:
            reasons.append(f"beam {spec['beam_m']}m > port max {max_beam}m")
            
        # Check historical feasibility flag
        if not check_historical_feasibility(destination_port, v_name):
            reasons.append("historically flagged as infeasible for this route")

        # Constraint 3: Practical Upper Bound ceil(2 * cargo_volume / capacity)
        upper_bound = math.ceil(2.0 * cargo_volume_mt / spec["capacity_mt"])

        if reasons:
            # Block this vessel class
            n_vars[v_name] = pulp.LpVariable(f"n_{v_name}", lowBound=0, upBound=0, cat=pulp.LpInteger)
            blocked_names.append(v_name)
            blocked_vessels.append({
                "vessel_class": v_name,
                "reasons": reasons
            })
        else:
            # Vessel is eligible
            n_vars[v_name] = pulp.LpVariable(f"n_{v_name}", lowBound=0, upBound=upper_bound, cat=pulp.LpInteger)

    # Constraint 2: Total Cargo Capacity Requirement (sum(n[v] * capacity[v]) >= cargo_volume_mt)
    prob += (
        pulp.lpSum([n_vars[v] * VESSEL_SPECS[v]["capacity_mt"] for v in VESSEL_SPECS]) >= cargo_volume_mt,
        "Total_Cargo_Requirement"
    )

    # Objective: Minimize Total Charter Cost
    # rate[v] = predicted_freight_rate_usd * (1 - discount[v])
    # cost[v] = n[v] * capacity[v] * rate[v]
    charter_costs = []
    for v_name, spec in VESSEL_SPECS.items():
        vessel_rate = predicted_freight_rate_usd * (1.0 - spec["discount"])
        vessel_unit_cost = spec["capacity_mt"] * vessel_rate
        charter_costs.append(n_vars[v_name] * vessel_unit_cost)
        
    prob += pulp.lpSum(charter_costs), "Total_Charter_Cost_USD"

    # Solve using CBC solver silently with 30s timeout
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    prob.solve(solver)
    
    status_str = pulp.LpStatus[prob.status].lower()

    # 3. Format Output
    if status_str == "optimal":
        fleet_dict = {v: int(round(n_vars[v].varValue or 0)) for v in VESSEL_SPECS}
        total_capacity = sum(fleet_dict[v] * VESSEL_SPECS[v]["capacity_mt"] for v in VESSEL_SPECS)
        
        # Calculate total cost
        total_cost = sum(
            fleet_dict[v] * VESSEL_SPECS[v]["capacity_mt"] * predicted_freight_rate_usd * (1.0 - VESSEL_SPECS[v]["discount"])
            for v in VESSEL_SPECS
        )
        slack = total_capacity - cargo_volume_mt
        cost_per_tonne = total_cost / cargo_volume_mt

        result = {
            "status": "optimal",
            "origin_port": origin_port,
            "destination_port": matched_port,
            "cargo_volume_mt": cargo_volume_mt,
            "predicted_freight_rate_usd": predicted_freight_rate_usd,
            "port_constraints": {
                "max_draft_m": max_draft,
                "max_loa_m": max_loa,
                "max_beam_m": max_beam
            },
            "blocked_vessels": blocked_names,
            "blocked_reasons": blocked_vessels,
            "fleet": fleet_dict,
            "total_capacity_mt": total_capacity,
            "slack_mt": slack,
            "total_cost_usd": round(total_cost, 2),
            "cost_per_tonne": round(cost_per_tonne, 2)
        }
    else:
        # Infeasible case
        result = {
            "status": "infeasible",
            "origin_port": origin_port,
            "destination_port": matched_port,
            "cargo_volume_mt": cargo_volume_mt,
            "predicted_freight_rate_usd": predicted_freight_rate_usd,
            "port_constraints": {
                "max_draft_m": max_draft,
                "max_loa_m": max_loa,
                "max_beam_m": max_beam
            },
            "blocked_vessels": blocked_names,
            "blocked_reasons": blocked_vessels,
            "fleet": {v: 0 for v in VESSEL_SPECS},
            "total_capacity_mt": 0,
            "slack_mt": -cargo_volume_mt,
            "total_cost_usd": 0.0,
            "cost_per_tonne": 0.0,
            "suggested_alternatives": [
                "All vessel classes blocked by port draft or dimensions.",
                "Consider deep-water transshipment via Gangavaram (max draft 18.2m) or Paradip (max draft 16.5m).",
                "Utilize ship-to-ship (STS) lighterage at Sagar / Sandheads anchorage before shallow river transit."
            ]
        }

    return result


def print_optimization_result(case_num: int, result: Dict[str, Any]) -> None:
    """Format and print optimization output report to console."""
    print(f"\n{'━'*80}")
    print(f"📦 TEST CASE {case_num}: {result['cargo_volume_mt']:,} MT → {result['destination_port']} (Baseline Rate: ${result['predicted_freight_rate_usd']:.2f}/ton)")
    print(f"{'━'*80}")
    
    constraints = result["port_constraints"]
    draft_str = f"{constraints['max_draft_m']}m" if constraints['max_draft_m'] else "No limit"
    loa_str = f"{constraints['max_loa_m']}m" if constraints['max_loa_m'] else "No limit"
    beam_str = f"{constraints['max_beam_m']}m" if constraints['max_beam_m'] else "No limit"
    
    print(f"⚓ Port Physical Constraints : Draft: {draft_str} | LOA: {loa_str} | Beam: {beam_str}")
    
    if result["blocked_vessels"]:
        print(f"🚫 Blocked Vessels ({len(result['blocked_vessels'])})      : {', '.join(result['blocked_vessels'])}")
        for b in result.get("blocked_reasons", []):
            print(f"   • {b['vessel_class']}: {'; '.join(b['reasons'])}")
    else:
        print("✅ Blocked Vessels           : None (All vessel classes permitted)")

    print(f"\n🎯 Optimization Status       : {result['status'].upper()}")

    if result["status"] == "optimal":
        fleet_str = ", ".join(f"{count}x {v}" for v, count in result["fleet"].items() if count > 0)
        print(f"🚢 Recommended Fleet         : {fleet_str if fleet_str else '0 vessels'}")
        print(f"📈 Total Capacity Provided   : {result['total_capacity_mt']:,} MT (Cargo: {result['cargo_volume_mt']:,} MT)")
        print(f"⚖️ Vessel Capacity Slack     : {result['slack_mt']:,} MT")
        print(f"💵 Total Charter Cost        : ${result['total_cost_usd']:,.2f}")
        print(f"🏷️ Effective Cost per Tonne  : ${result['cost_per_tonne']:.2f} / ton")
    else:
        print("\n❌ Infeasibility Diagnosis & Alternative Recommendations:")
        for alt in result.get("suggested_alternatives", []):
            print(f"   ⚠ {alt}")


def run_all_test_cases():
    """Execute the 5 required operations research test cases."""
    test_cases = [
        {"vol": 150000, "origin": "Hay Point (Australia)", "dest": "Paradip", "rate": 20.0, "desc": "150,000 MT to Paradip (Capesize blocked by 16.5m draft)"},
        {"vol": 150000, "origin": "Hay Point (Australia)", "dest": "Haldia", "rate": 20.0, "desc": "150,000 MT to Haldia (Only Handysize allowed due to 8.5m draft)"},
        {"vol": 150000, "origin": "Hay Point (Australia)", "dest": "Gangavaram", "rate": 20.0, "desc": "150,000 MT to Gangavaram (Capesize allowed, 18.2m draft)"},
        {"vol": 5000,   "origin": "Hay Point (Australia)", "dest": "Paradip", "rate": 20.0, "desc": "5,000 MT to Paradip (Tiny cargo — edge case)"},
        {"vol": 500000, "origin": "Hay Point (Australia)", "dest": "Paradip", "rate": 20.0, "desc": "500,000 MT to Paradip (Large cargo — multi-vessel fleet)"},
    ]

    print("\n" + "═" * 80)
    print("🚢 VESSEL CHARTERING MILP OPTIMIZATION ENGINE — TEST SUITE")
    print("═" * 80)

    for idx, tc in enumerate(test_cases, 1):
        res = run_charter_optimization(
            cargo_volume_mt=tc["vol"],
            origin_port=tc["origin"],
            destination_port=tc["dest"],
            predicted_freight_rate_usd=tc["rate"]
        )
        print_optimization_result(idx, res)

    print("\n" + "═" * 80)
    print("✅ ALL 5 TEST CASES EXECUTED SUCCESSFULLY.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    run_all_test_cases()
