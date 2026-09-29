"""
utils/optimizer.py - Mixed-Integer Linear Programming (MILP) Fleet Optimizer Module
Exposes run_charter_optimization for the DSS dashboard.
"""

import math
from typing import Any, Dict, List, Optional

import pulp
from utils.feasibility import check_feasibility, VESSEL_SPECS


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

    # 1. Fetch physical feasibility and port limits
    feasibility = check_feasibility(destination_port)
    port_name = feasibility["port_name"]
    port_constraints = feasibility["port_constraints"]
    vessel_status = feasibility["vessel_status"]
    blocked_vessels = feasibility["blocked_vessels"]

    # 2. Build PuLP MILP Model
    prob = pulp.LpProblem("Vessel_Chartering_Optimization", pulp.LpMinimize)
    
    n_vars: Dict[str, pulp.LpVariable] = {}
    blocked_reasons: List[Dict[str, Any]] = []

    for v_name, status in vessel_status.items():
        spec = status["specs"]
        upper_bound = math.ceil(2.0 * cargo_volume_mt / spec["capacity_mt"])

        if not status["feasible"]:
            n_vars[v_name] = pulp.LpVariable(f"n_{v_name}", lowBound=0, upBound=0, cat=pulp.LpInteger)
            blocked_reasons.append({
                "vessel_class": v_name,
                "reasons": status["reasons"]
            })
        else:
            n_vars[v_name] = pulp.LpVariable(f"n_{v_name}", lowBound=0, upBound=upper_bound, cat=pulp.LpInteger)

    # Constraint 2: Total Cargo Capacity Requirement (sum(n[v] * capacity[v]) >= cargo_volume_mt)
    prob += (
        pulp.lpSum([n_vars[v] * VESSEL_SPECS[v]["capacity_mt"] for v in VESSEL_SPECS]) >= cargo_volume_mt,
        "Total_Cargo_Requirement"
    )

    # Objective: Minimize Total Charter Cost
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

        # Human-readable fleet summary string (e.g. "2x Panamax")
        fleet_components = [f"{count}x {v}" for v, count in fleet_dict.items() if count > 0]
        fleet_summary_str = ", ".join(fleet_components) if fleet_components else "0 vessels"

        result = {
            "status": "optimal",
            "origin_port": origin_port,
            "destination_port": port_name,
            "cargo_volume_mt": cargo_volume_mt,
            "predicted_freight_rate_usd": predicted_freight_rate_usd,
            "port_constraints": port_constraints,
            "blocked_vessels": blocked_vessels,
            "blocked_reasons": blocked_reasons,
            "fleet": fleet_dict,
            "fleet_summary": fleet_summary_str,
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
            "destination_port": port_name,
            "cargo_volume_mt": cargo_volume_mt,
            "predicted_freight_rate_usd": predicted_freight_rate_usd,
            "port_constraints": port_constraints,
            "blocked_vessels": blocked_vessels,
            "blocked_reasons": blocked_reasons,
            "fleet": {v: 0 for v in VESSEL_SPECS},
            "fleet_summary": "Infeasible",
            "total_capacity_mt": 0,
            "slack_mt": -cargo_volume_mt,
            "total_cost_usd": 0.0,
            "cost_per_tonne": 0.0,
            "suggested_alternatives": [
                "All vessel classes are blocked by port physical limits.",
                "Consider transshipment via deep-water ports like Gangavaram (18.2m draft) or Paradip (16.5m draft).",
                "Utilize ship-to-ship (STS) lighterage transfer at Sagar / Sandheads anchorage."
            ]
        }

    return result
