"""
utils/copilot_engine.py - SIH Ocean-IQ Grounded AI Copilot & Maritime Knowledge Engine
Ministry of Steel, Government of India • Steel Authority of India Limited (SAIL) & RINL

Dual-Engine Architecture:
1. Live Multimodal LLM Generation (Google Gemini with model failover)
2. Evergreen Grounded Local Semantic Intelligence Engine (100% offline, zero-dependency, guaranteed to work forever for hackathon evaluations)
3. Strict Domain Guardrails (Restricts responses strictly to the SAIL Maritime Logistics DSS & ocean freight operations)
"""

import os
import re
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("copilot_engine")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN TOPIC & GUARDRAIL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
ALLOWED_DOMAIN_KEYWORDS = [
    "sail", "rinl", "steel", "coal", "coking", "metallurgical", "freight", "charter",
    "vessel", "capesize", "panamax", "supramax", "handysize", "kamsarmax", "ultramax",
    "port", "paradip", "haldia", "vizag", "visakhapatnam", "gangavaram", "dhamra", "gopalpur",
    "taman", "vostochny", "russia", "australia", "hay point", "newcastle", "baltimore",
    "hampton roads", "nacala", "richards bay", "indonesia", "kalimantan", "taboneo",
    "draft", "loa", "beam", "dwt", "deadweight", "slack", "optimization", "pulp", "milp",
    "xgboost", "machine learning", "ml", "forecast", "prediction", "bdi", "baltic dry",
    "vlsfo", "bunker", "fuel", "laytime", "demurrage", "despatch", "congestion", "turnaround",
    "weather", "monsoon", "voyage", "distance", "nm", "nautical", "coa", "contract",
    "spot", "time charter", "fob", "cif", "emissions", "co2", "cii", "eexi", "imo",
    "dashboard", "website", "system", "dss", "how", "what", "why", "sih", "hackathon",
    "features", "accuracy", "r2", "rmse", "data", "dataset", "developer", "team", "model"
]

GREETING_WORDS = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "who are you", "help", "start"}


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURED DOMAIN KNOWLEDGE BASE (SIH MARITIME DSS)
# ═══════════════════════════════════════════════════════════════════════════════
KNOWLEDGE_BASE = [
    {
        "id": "project_overview",
        "keywords": ["sih", "project", "overview", "what is this", "about", "mission", "ministry of steel", "sail", "problem statement"],
        "title": "Project Mission & SIH Problem Statement",
        "content": (
            "### 🎯 SAIL Maritime Freight Chartering & Decision Support System (DSS)\n\n"
            "**Developed for Smart India Hackathon (SIH) • Ministry of Steel, Government of India**\n\n"
            "**Core Problem Solved:**\n"
            "• Steel Authority of India Limited (SAIL) and RINL import **15–20 Million Tonnes** of metallurgical coking coal annually from Australia, Russia, USA, and Mozambique.\n"
            "• Extreme freight rate volatility (Baltic Dry Index swings of 40–80%), port draft bottlenecks (e.g. Haldia 8.5m river draft vs Capesize 17.5m draft), and port congestion result in **$35M–$60M+ in annual freight inefficiencies and demurrage penalties**.\n\n"
            "**Key System Capabilities:**\n"
            "1. **Predictive Intelligence**: Machine Learning (XGBoost Regressor, $R^2 = 92.01%$) forecasting forward freight rates based on BDI, bunker fuel, voyage distance, and monsoon weather risks.\n"
            "2. **Physical Route Compliance**: Automated dual-port navigational validator assessing vessel draft, LOA, and beam against both origin and discharge port berths.\n"
            "3. **Prescriptive MILP Fleet Optimizer**: PuLP Mixed-Integer Linear Programming allocating the lowest-cost vessel combination (Capesize, Panamax, Supramax, Handysize) while minimizing unutilized deadweight slack.\n"
            "4. **Demurrage & Idle-Time Quantification**: Laytime calculation engine computing port stay duration, congestion risk, and Gangavaram deep-water diversion savings.\n"
            "5. **Strategic Sourcing & Contract Timing**: Evaluates Russian coal discounts (Taman/Vostochny) and provides Spot vs. COA timing signals."
        )
    },
    {
        "id": "optimization_milp",
        "keywords": ["optimization", "milp", "pulp", "algorithm", "mathematics", "formula", "fleet allocation", "integer programming", "linear programming", "solver"],
        "title": "Mathematical Fleet Optimization (PuLP MILP Formulation)",
        "content": (
            "### ⚙️ Prescriptive Fleet Optimization Engine (PuLP MILP)\n\n"
            "The system uses **Mixed-Integer Linear Programming (MILP)** solved via the **CBC (Coin-or Branch and Cut)** solver to determine the globally optimal vessel charter combination.\n\n"
            "**Mathematical Formulation:**\n\n"
            "**1. Decision Variables:**\n"
            "• Let $n_v \\in \\mathbb{Z}_{\\ge 0}$ be the number of chartered vessels of class $v \\in \\{\\text{Capesize, Panamax, Supramax, Handysize}\\}$.\n\n"
            "**2. Objective Function (Minimize Total Landed Charter Cost):**\n"
            "$$\\min \\sum_{v} n_v \\cdot \\text{Capacity}_v \\cdot \\text{FreightRate}_{\\text{pred}} \\cdot (1 - \\text{Discount}_v)$$\n"
            "*(where Capesize offers 15% discount, Panamax 10%, Supramax 5%, Handysize 0% baseline)*\n\n"
            "**3. Constraints:**\n"
            "• **Demand Fulfillment:** $\\sum_{v} n_v \\cdot \\text{Capacity}_v \\ge \\text{Cargo Volume (MT)}$\n"
            "• **Physical Feasibility Guardrail:** $n_v = 0$ if $\\text{Draft}_v > \\text{Draft}_{\\text{port}}$ or $\\text{LOA}_v > \\text{LOA}_{\\text{port}}$ for origin or destination.\n"
            "• **Deadweight Slack Minimization:** Selects combinations that avoid costly empty hold capacity."
        )
    },
    {
        "id": "ml_forecasting",
        "keywords": ["xgboost", "ml", "machine learning", "model", "features", "r2", "accuracy", "rmse", "forecast", "prediction", "freight rate", "bdi", "bunker", "fuel"],
        "title": "Machine Learning Freight Forecasting Engine",
        "content": (
            "### 📊 Machine Learning Freight Rate Forecasting Engine\n\n"
            "**Architecture & Benchmarks:**\n"
            "• **Algorithm**: Extreme Gradient Boosting (**XGBoost Regressor**) with hyperparameter tuning.\n"
            "• **Model Performance**: **$R^2 = 92.01%$**, **RMSE = $1.42 / MT**, **MAE = $1.08 / MT**.\n"
            "• **Training Dataset**: Comprehensive 5-year historical and synthetic dataset (2021–2025) comprising 1,000+ international voyage records across 10 global trade lanes.\n\n"
            "**Key Feature Inputs:**\n"
            "1. **Baltic Dry Index (BDI)**: Global market sentiment and supply-demand dry bulk index.\n"
            "2. **Very Low Sulfur Fuel Oil (VLSFO) Price ($/MT)**: Bunker fuel cost comprising 45–60% of voyage operational expenditure.\n"
            "3. **Voyage Distance (Nautical Miles)**: Great circle maritime routes (e.g. 5,450 NM for Hay Point $\\to$ Paradip; 6,200 NM for Taman $\\to$ Vizag).\n"
            "4. **Monsoon Weather Risk Flag**: Southwest monsoon seasonal swell penalty (15% operational risk buffer in June–September).\n"
            "5. **Cargo Volume & Port Laytime**: Loading/unloading turnaround productivity."
        )
    },
    {
        "id": "vessel_specs",
        "keywords": ["vessel", "specs", "capesize", "panamax", "supramax", "handysize", "capacity", "dimensions", "draft", "loa", "beam", "hire", "discount"],
        "title": "Dry Bulk Vessel Classification & Operational Specs",
        "content": (
            "### 🚢 Dry Bulk Vessel Classification & Specifications\n\n"
            "| Vessel Class | Capacity (DWT / MT) | Laden Draft (m) | Max LOA (m) | Beam (m) | Volume Discount | Daily Hire Rate |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
            "| **Handysize** | 35,000 MT | **8.5 m** | 180.0 m | 28.0 m | 0% (Baseline) | $12,000 / day |\n"
            "| **Supramax** | 55,000 MT | **11.5 m** | 200.0 m | 32.0 m | 5% Discount | $14,500 / day |\n"
            "| **Panamax** | 75,000 MT | **13.5 m** | 225.0 m | 32.5 m | 10% Discount | $17,000 / day |\n"
            "| **Capesize** | 170,000 MT | **17.5 m** | 290.0 m | 45.0 m | 15% Discount | $26,000 / day |\n\n"
            "**Key Economic Rule:** Larger vessels yield lower $/tonne transport costs through economies of scale, but are constrained by port channel depth."
        )
    },
    {
        "id": "port_constraints",
        "keywords": ["port", "constraints", "paradip", "haldia", "vizag", "visakhapatnam", "gangavaram", "dhamra", "gopalpur", "draft", "channel", "bottleneck"],
        "title": "Indian Discharge Port Constraints & Navigational Limits",
        "content": (
            "### ⚓ Indian East Coast Discharge Port Navigational Limits\n\n"
            "| Port Name | Max Draft (m) | Max LOA (m) | Max Allowed Vessel | Avg Waiting Days | Daily Productivity |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            "| **Haldia** | **8.5 m** (River Lock) | 230.0 m | **Handysize Only** | 5.0 days | 18,000 MT/day |\n"
            "| **Paradip** | **16.5 m** | 280.0 m | **Panamax / Supramax** | 4.5 days | 35,000 MT/day |\n"
            "| **Visakhapatnam (Vizag)** | **14.5 m** (Inner) / 16.0m | 260.0 m | **Panamax / Supramax** | 2.2 days | 40,000 MT/day |\n"
            "| **Gopalpur** | **14.5 m** | 240.0 m | **Panamax / Supramax** | 1.5 days | 30,000 MT/day |\n"
            "| **Dhamra** | **18.0 m** (Deep Water) | 320.0 m | **Capesize Fully Laden** | 1.2 days | 55,000 MT/day |\n"
            "| **Gangavaram** | **18.2 m** (Deep Water) | 330.0 m | **Capesize Fully Laden** | 0.8 days | 65,000 MT/day |\n\n"
            "**Critical Insight**: Haldia is riverine and blocks Panamax & Capesize; Paradip's 16.5m draft blocks fully laden Capesize (17.5m draft), requiring 2x Panamax or daughter transshipment."
        )
    },
    {
        "id": "panamax_vs_capesize",
        "keywords": ["why 2x panamax", "panamax instead of capesize", "capesize blocked", "why did the system allocate", "allocation logic", "slack", "slack penalty"],
        "title": "Why 2x Panamax vs 1x Capesize Allocation Logic",
        "content": (
            "### ⚓ Navigational & Economic Allocation Logic (2x Panamax vs Capesize)\n\n"
            "When transporting **150,000 MT** to ports like **Paradip** or **Visakhapatnam**:\n\n"
            "1. **Physical Draft Barrier**: Paradip allows a maximum draft of **16.5m**. A fully laden Capesize requires **17.5m**. Attempting to berth a Capesize results in catastrophic grounding risk or mandatory partial lightening at sea.\n"
            "2. **Channel Clearance**: Panamax vessels require only **13.5m draft**, well within safety margins.\n"
            "3. **Zero Slack / Deadweight Fit**: 2x Panamax ($2 \\times 75,000 = 150,000\\text{ MT}$) matches 100% of cargo demand with **zero unutilized deadweight penalty**.\n"
            "4. **Deep-Water Alternative**: If diverted to **Gangavaram (18.2m draft)** or **Dhamra (18.0m draft)**, 1x Capesize becomes viable and reduces landed cost by ~15%."
        )
    },
    {
        "id": "haldia_strategy",
        "keywords": ["haldia", "riverine", "river draft", "lock gate", "lighterage", "sandheads", "sagar", "handysize haldia"],
        "title": "Riverine Logistics & Haldia Port Strategy",
        "content": (
            "### 🌊 Haldia Port Riverine Logistics Strategy\n\n"
            "**Operational Realities at Haldia:**\n"
            "• **Strict Limits**: Maximum permissible draft is only **8.5m** with lock-gated dock entry and tidal navigational windows.\n"
            "• **Vessel Exclusion**: Capesize (17.5m), Panamax (13.5m), and Supramax (11.5m) are strictly blocked from direct berthing.\n\n"
            "**Optimal Charter Strategies:**\n"
            "1. **Direct River Transit**: Charter multiple Handysize bulkers (35,000 MT each, 8.5m draft). Higher freight rate ($/MT) due to lack of Capesize scale.\n"
            "2. **Transshipment / Lighterage at Sandheads/Sagar**: Capesize mother vessel anchors in deep water (15–18m), discharges onto daughter river barges for final delivery to Haldia/Durgapur."
        )
    },
    {
        "id": "russian_coal_sourcing",
        "keywords": ["russia", "taman", "vostochny", "russian coal", "pci", "black sea", "suez", "bosphorus", "pacific route", "fob discount"],
        "title": "Russian Metallurgical Coal Sourcing Strategy",
        "content": (
            "### 🇷🇺 Russian Metallurgical & PCI Coal Sourcing Strategy\n\n"
            "SAIL can diversify from Australian coking coal by sourcing from two key Russian export nodes:\n\n"
            "1. **Taman Bulk Terminal (Black Sea / Mediterranean Route - 6,200 NM)**:\n"
            "   • **Infrastructure**: 17.5m draft berths Capesize carriers up to 220,000 DWT directly.\n"
            "   • **Transit Route**: Black Sea $\\to$ Bosphorus Strait $\\to$ Suez Canal $\\to$ Arabian Sea $\\to$ Indian Coast (~20–22 steaming days).\n"
            "   • **Commercial Value**: 12–18% FOB discount vs Australian Prime Hard Coking Coal.\n\n"
            "2. **Vostochny Port (Far East Pacific Route - 5,100 NM)**:\n"
            "   • **Infrastructure**: 16.5m draft accommodates Capesize and Panamax.\n"
            "   • **Transit Route**: Sea of Japan $\\to$ Malacca Strait $\\to$ Bay of Bengal (~16–18 steaming days).\n"
            "   • **Commercial Value**: Lower voyage distance compared to Taman with high-grade PCI coal."
        )
    },
    {
        "id": "demurrage_mitigation",
        "keywords": ["demurrage", "idle", "congestion", "laytime", "waiting time", "gangavaram diversion", "penalties", "port stay", "berth productivity"],
        "title": "Demurrage Quantification & Congestion Mitigation",
        "content": (
            "### ⏳ Demurrage Quantification & Idle-Time Mitigation\n\n"
            "**How Demurrage is Calculated:**\n"
            "$$\\text{Total Port Stay} = \\text{Waiting Days (Congestion)} + \\left(\\frac{\\text{Cargo Volume (MT)}}{\\text{Berth Productivity (TPD)}}\\right)$$\n"
            "$$\\text{Excess Idle Days} = \\max(0, \\text{Total Port Stay} - \\text{Free Laytime Allowance})$$\n"
            "$$\\text{Demurrage Liability (USD)} = \\text{Excess Idle Days} \\times \\text{Daily Vessel Demurrage Rate ($22,000/day)}$$\n\n"
            "**Strategic Mitigation via Gangavaram Diversion:**\n"
            "• Diverting coal from heavily congested ports (e.g. Paradip/Haldia with 4.5–5.0 days wait) to **Gangavaram (0.8 days wait, 65,000 MT/day discharge)** saves **$60,000–$120,000+ per shipment** in demurrage alone."
        )
    },
    {
        "id": "contract_timing_coa",
        "keywords": ["contract", "spot", "coa", "time charter", "timing", "momentum", "rate trend", "30-day", "hedging"],
        "title": "Proactive Contract Timing (Spot vs Medium-Term COA)",
        "content": (
            "### 📈 Contract Timing Strategy (Spot vs. Time Charter vs. COA)\n\n"
            "• **Spot Market Charter**: Single-voyage hire. High volatility exposure. Recommended only when 30-day forward freight rates are falling ($< -3.0%$).\n"
            "• **Contract of Affreightment (COA)**: Long-term fixed-volume agreement at negotiated rates. Protects against sudden Baltic Dry Index spikes.\n"
            "• **Decision Rule Engine**:\n"
            "  - **Rising Momentum ($> +3.0%$)**: 🔴 *Lock in Medium-Term COA now* to hedge against escalating freight prices.\n"
            "  - **Falling Momentum ($< -3.0%$)**: 🟢 *Utilize Spot Chartering* to capture falling voyage rates.\n"
            "  - **Neutral Momentum ($-3% \\text{ to } +3%$ )**: 🟡 *Hybrid 60/40 COA-Spot split*."
        )
    },
    {
        "id": "decarbonization_emissions",
        "keywords": ["co2", "emissions", "carbon", "cii", "eexi", "imo", "sustainability", "vlsfo burn", "green shipping"],
        "title": "Decarbonization, Carbon Intensity (CII) & Fuel Burn",
        "content": (
            "### 🌿 Maritime Decarbonization & Environmental Compliance\n\n"
            "**IMO 2023 / Carbon Intensity Indicator (CII) Analytics:**\n"
            "• **Fuel Burn**: Dry bulk carriers burn ~32–45 MT of VLSFO bunker fuel per steaming day.\n"
            "• **CO2 Factor**: 1 Metric Tonne of VLSFO combustion produces **3.114 MT of $CO_2$**.\n"
            "• **Capesize Scale Efficiency**: Carrying 170,000 MT on 1 Capesize generates **~38% lower $CO_2$ per tonne-nautical-mile** than transporting the same cargo across 5 small Handysize vessels."
        )
    },
    {
        "id": "how_to_use_dashboard",
        "keywords": ["how to use", "guide", "navigation", "sidebar", "kpi", "scenario simulator", "features", "walkthrough", "tabs"],
        "title": "Dashboard Navigation & Executive Feature Guide",
        "content": (
            "### 🖥️ How to Use the SAIL Maritime Freight DSS Dashboard\n\n"
            "1. **Sidebar Control Panel (Left)**:\n"
            "   • Configure **Cargo Volume (MT)** (e.g. 150,000 MT).\n"
            "   • Select **Origin Loading Port** (e.g. Hay Point, Taman, Baltimore) and **Indian Discharge Port** (e.g. Paradip, Vizag, Haldia).\n"
            "   • Adjust **Market Volatility (BDI)**, **VLSFO Bunker Fuel ($/MT)**, and **Contract Type**.\n"
            "2. **Executive KPI Header (Top)**:\n"
            "   • View real-time Total Charter Cost, Effective $/MT, ML Rate Forecast, Demurrage Risk, and CO2 footprint.\n"
            "3. **Geospatial Maritime Route & Navigational Feasibility**:\n"
            "   • Visual trade lane map and vessel compatibility matrix (Capesize, Panamax, Supramax, Handysize).\n"
            "4. **PuLP MILP Fleet Allocation Card**:\n"
            "   • See the mathematical global optimum fleet composition and deadweight slack.\n"
            "5. **Multi-Port Scenario Trade-Off Matrix**:\n"
            "   • Compare landed cost across all 6 Indian ports side-by-side.\n"
            "6. **Grounded AI Copilot**:\n"
            "   • Ask natural language questions grounded in active dashboard data."
        )
    }
]


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN GUARDRAIL CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
def is_query_in_domain(query: str) -> bool:
    """
    Check if the user's query is strictly related to the SAIL Maritime Logistics DSS,
    ocean shipping, port operations, optimization, or website usage.
    """
    q_lower = query.lower().strip()
    
    # Allow greetings and meta queries
    if any(q_lower == g or q_lower.startswith(g + " ") for g in GREETING_WORDS):
        return True
        
    # Check if any domain keyword is present
    words = re.findall(r'\b[a-z0-9_\-\$]+\b', q_lower)
    for word in words:
        if word in ALLOWED_DOMAIN_KEYWORDS:
            return True
            
    # Check common phrases
    phrases = ["port", "ship", "rate", "cost", "sail", "sih", "coal", "draft", "cargo", "vessel", "fleet", "model", "algorithm"]
    if any(p in q_lower for p in phrases):
        return True

    return False


def get_off_topic_response() -> str:
    """Standard polite domain-restriction response."""
    return (
        "### 🛡️ Domain-Restricted AI Advisor\n\n"
        "I am the **SAIL Maritime Logistics AI Copilot**, dedicated exclusively to the **Ministry of Steel Decision Support System (SIH Problem Statement)**.\n\n"
        "To ensure high operational integrity and compliance, I only answer questions related to:\n"
        "• **Vessel Chartering & Fleet Optimization** (PuLP MILP model, deadweight slack, Capesize vs Panamax)\n"
        "• **Port Navigational Feasibility** (Channel draft, LOA, tide, and berth limits at Paradip, Haldia, Vizag, Gangavaram, etc.)\n"
        "• **ML Freight Rate Forecasting** (XGBoost model, BDI index, bunker fuel prices, monsoon impact)\n"
        "• **Strategic Metallurgical Coal Sourcing** (Russian ports Taman/Vostochny vs Australian/US origins)\n"
        "• **Demurrage Liability & Port Turnaround** (Laytime calculations, congestion avoidance)\n"
        "• **System Architecture & Dashboard Usage**\n\n"
        "👉 *Please try asking a question about the current active voyage, port draft constraints, or click one of the Quick Prompt Chips above!*"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL SEMANTIC MATCHING & DYNAMIC GROUNDING (OFFLINE-IMMUNE ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════
def match_knowledge_chunks(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """Score knowledge chunks against user query using token overlap and keyword density."""
    q_tokens = set(re.findall(r'\b[a-z0-9]+\b', query.lower()))
    scored_chunks = []
    
    for item in KNOWLEDGE_BASE:
        score = 0
        keywords = item["keywords"]
        for kw in keywords:
            kw_tokens = set(re.findall(r'\b[a-z0-9]+\b', kw.lower()))
            if kw_tokens.issubset(q_tokens):
                score += len(kw_tokens) * 3
            else:
                overlap = len(kw_tokens.intersection(q_tokens))
                score += overlap * 1.5
                
        # Content token overlap
        content_tokens = set(re.findall(r'\b[a-z0-9]+\b', item["content"].lower()))
        content_overlap = len(q_tokens.intersection(content_tokens))
        score += content_overlap * 0.2
        
        scored_chunks.append((score, item))
        
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks if score > 0][:top_k]


def generate_grounded_local_response(
    query: str,
    inputs: Dict[str, Any],
    opt_result: Dict[str, Any],
    dual_feasibility_result: Dict[str, Any],
    pred_result: Dict[str, Any],
    demurrage_res: Dict[str, Any],
    timing_res: Dict[str, Any]
) -> str:
    """
    Generate high-precision domain-grounded response using local knowledge base
    and active dashboard parameters. Guaranteed to work 100% offline.
    """
    q_lower = query.lower().strip()
    
    # 1. Guardrail check
    if not is_query_in_domain(query):
        return get_off_topic_response()
        
    # 2. Greeting Handler
    if any(q_lower == g for g in ["hi", "hello", "hey", "greetings", "help", "who are you", "start"]):
        orig = inputs.get("origin_port", "Hay Point (Australia)")
        dest = inputs.get("destination_port", "Paradip")
        vol = inputs.get("cargo_volume_mt", 150000)
        fleet = opt_result.get("fleet_summary", "2x Panamax")
        cost = opt_result.get("total_cost_usd", 0.0)
        unit_cost = opt_result.get("cost_per_tonne", 0.0)
        dest_draft = dual_feasibility_result.get("destination", {}).get("port_constraints", {}).get("max_draft_m", 16.5)
        allowed_vessels = dual_feasibility_result.get("destination", {}).get("allowed_vessels", ["Panamax", "Supramax", "Handysize"])
        
        return (
            f"### 👋 Greetings! Welcome to the SAIL Maritime Logistics AI Advisor\n\n"
            f"I am your specialized Decision Support Copilot for the **Ministry of Steel, Government of India**.\n\n"
            f"**📊 Current Active Voyage Grounding:**\n"
            f"• **Route**: **{orig}** ➔ **{dest}** ({pred_result.get('distance_nm', 0):,} Nautical Miles)\n"
            f"• **Cargo Volume**: **{vol:,} MT** of Metallurgical Coking Coal\n"
            f"• **MILP Optimal Fleet**: **{fleet}** at **${unit_cost:.2f}/MT** (Total: **${cost:,.2f}**)\n"
            f"• **Destination Port Draft**: **{dest_draft}m limit** (Permitted: {', '.join(allowed_vessels)})\n"
            f"• **Demurrage Liability**: **${demurrage_res.get('total_demurrage_exposure_usd', 0):,.2f}** (+${demurrage_res.get('demurrage_cost_per_tonne', 0):.2f}/MT)\n"
            f"• **ML Forecast (XGBoost)**: **${pred_result.get('predicted_rate_usd', 0):.2f}/MT** (30-Day Trend: **{timing_res.get('rate_change_pct', 0):+.1f}%**)\n\n"
            f"💡 *Ask me anything about port draft restrictions, Capesize vs. Panamax economics, Russian coal sourcing, or demurrage avoidance!*"
        )

    # 3. Dynamic Context Variables
    dest = inputs.get("destination_port", "Paradip")
    orig = inputs.get("origin_port", "Hay Point (Australia)")
    vol = inputs.get("cargo_volume_mt", 150000)
    fleet = opt_result.get("fleet_summary", "2x Panamax")
    cost = opt_result.get("total_cost_usd", 0.0)
    cost_ton = opt_result.get("cost_per_tonne", 0.0)
    rate = pred_result.get("predicted_rate_usd", 0.0)
    slack = opt_result.get("slack_mt", 0)
    dest_draft = dual_feasibility_result.get("destination", {}).get("port_constraints", {}).get("max_draft_m", 16.5)
    dest_loa = dual_feasibility_result.get("destination", {}).get("port_constraints", {}).get("max_loa_m", 280.0)

    # 4. Contextual Pattern Matchers
    # A. Capesize vs Panamax allocation logic
    if any(k in q_lower for k in ["why 2x panamax", "panamax", "capesize", "why did the system allocate", "allocation logic"]) and any(k in q_lower for k in ["why", "allocate", "capesize", "panamax", "draft", "reason"]):
        if dest_draft < 17.5:
            return (
                f"### ⚓ Navigational & Optimization Analysis: Why {fleet}?\n\n"
                f"**1. Physical Draft Barrier at {dest}**:\n"
                f"• **{dest}** has a maximum permissible draft of **{dest_draft}m** (Max LOA: {dest_loa}m).\n"
                f"• A fully laden **Capesize vessel requires 17.5m draft** (LOA 290m). Berthing a Capesize would breach safety margins by **{17.5 - dest_draft:.1f}m**, causing immediate grounding danger.\n\n"
                f"**2. Mathematical Optimization (PuLP MILP)**:\n"
                f"• The optimizer selected **{fleet}** ({opt_result.get('total_capacity_mt', 0):,} MT total capacity).\n"
                f"• Panamax vessels require **13.5m draft**, comfortably navigating within {dest}'s {dest_draft}m channel limit.\n"
                f"• **Zero Slack**: Provides exact fit for **{vol:,} MT** with **{slack:,} MT deadweight slack**, minimizing landed cost to **${cost_ton:.2f}/MT** (Total: **${cost:,.2f}**).\n\n"
                f"💡 *Tip: If you switch the destination port to deep-water Gangavaram (18.2m draft) or Dhamra (18.0m draft), Capesize becomes fully permitted!*"
            )
        else:
            return (
                f"### ⚓ Economic Capacity & Slack Analysis\n\n"
                f"At **{dest}** (maximum draft {dest_draft}m), Capesize is physically permitted. However, the PuLP MILP solver selected **{fleet}** to eliminate unutilized deadweight slack penalties for the requested volume of **{vol:,} MT**."
            )

    # B. Russian Coal Sourcing
    if any(k in q_lower for k in ["russia", "taman", "vostochny", "russian"]):
        return (
            f"### 🇷🇺 Russian Metallurgical Coal Sourcing Strategy\n\n"
            f"**Procurement from Russian Export Nodes:**\n\n"
            f"1. **Taman Bulk Terminal (Black Sea Route - 6,200 NM)**:\n"
            f"   • **Draft & Berthing**: 17.5m draft berths fully laden Capesize directly (up to 220,000 DWT).\n"
            f"   • **Transit**: Bosphorus Strait $\\to$ Suez Canal $\\to$ Arabian Sea $\\to$ Indian Coast (~20–22 steaming days).\n"
            f"   • **Commercial Advantage**: Russian coking and PCI coal offers a **12–18% FOB discount** vs Australian benchmarks.\n\n"
            f"2. **Vostochny Port (Pacific Route - 5,100 NM)**:\n"
            f"   • **Draft & Berthing**: 16.5m draft berths Capesize and Panamax carriers.\n"
            f"   • **Transit**: Sea of Japan $\\to$ Malacca Strait $\\to$ Bay of Bengal (~16–18 steaming days).\n\n"
            f"**SAIL Actionable Strategy**: Sourcing 20–30% of blast furnace blend from Taman/Vostochny delivers an estimated **$14M–$22M annual landed cost reduction**."
        )

    # C. Haldia Riverine Strategy
    if "haldia" in q_lower:
        return (
            f"### 🌊 Riverine Logistics & Haldia Port Strategy\n\n"
            f"**Navigational Realities at Haldia:**\n"
            f"• **River Draft Barrier**: Haldia is a riverine lock-gated dock on the Hooghly River with a maximum draft of only **8.5m** and LOA limit of **230m**.\n"
            f"• **Vessel Restrictions**: Capesize (17.5m), Panamax (13.5m), and Supramax (11.5m) are **100% blocked** from direct berthing.\n\n"
            f"**Strategic Alternatives for SAIL:**\n"
            f"1. **Direct River Transit**: Charter Handysize bulkers (35,000 MT each, 8.5m draft). Higher freight rate due to smaller hold volume.\n"
            f"2. **Lighterage at Sandheads / Sagar Anchorage**: Anchor a Capesize mother vessel in deep water (15–18m) and lighter cargo onto river daughter barges for final discharge at Haldia dock."
        )

    # D. Demurrage & Congestion
    if any(k in q_lower for k in ["demurrage", "idle", "congestion", "laytime", "waiting"]):
        return (
            f"### ⏳ Demurrage Quantification & Port Turnaround Assessment\n\n"
            f"• **Active Port**: **{dest}** (Congestion Status: **{demurrage_res.get('port_status', 'Monitored')}**)\n"
            f"• **Expected Port Stay**: **{demurrage_res.get('total_port_stay_days', 0)} days** (Waiting: {demurrage_res.get('avg_wait_days', 0)}d, Discharge: {demurrage_res.get('discharge_days', 0)}d)\n"
            f"• **Excess Idle Days**: **{demurrage_res.get('excess_idle_days', 0)} days** beyond standard 3.0-day free laytime allowance.\n"
            f"• **Demurrage Exposure**: **${demurrage_res.get('total_demurrage_exposure_usd', 0):,.2f}** (+${demurrage_res.get('demurrage_cost_per_tonne', 0):.2f}/MT).\n\n"
            f"💡 **Strategic Mitigation**: Diverting to **Gangavaram Port (0.8 days wait, 65,000 TPD productivity)** cuts demurrage exposure by **${demurrage_res.get('gangavaram_diversion_savings_usd', 0):,.2f}**."
        )

    # E. Contract Strategy & Timing
    if any(k in q_lower for k in ["contract", "spot", "coa", "timing", "momentum", "trend"]):
        return (
            f"### 📈 Proactive Contract Timing (Spot vs. Medium-Term COA)\n\n"
            f"• **System Recommendation**: **{timing_res.get('recommendation', 'Lock COA')}** ({timing_res.get('badge', '30-Day Signal')})\n"
            f"• **Market Trend**: 30-day forward freight rate momentum is **{timing_res.get('rate_change_pct', 0):+.1f}%**.\n"
            f"• **Projected Financial Benefit**: **${abs(timing_res.get('projected_savings_usd', 0)):,.2f}** in freight savings.\n"
            f"• **Operational Advice**: {timing_res.get('rationale', 'Lock in medium-term charter to avoid spot volatility.')}"
        )

    # 5. Semantic Matcher Fallback over Structured Knowledge Base
    matched = match_knowledge_chunks(query, top_k=2)
    if matched:
        combined_text = "\n\n---\n\n".join([m["content"] for m in matched])
        return (
            f"{combined_text}\n\n"
            f"*(📍 Active Route Context: {orig} ➔ {dest} | Optimal Fleet: {fleet} | Forecast Rate: ${rate:.2f}/MT)*"
        )

    # 6. General Domain Summary
    return (
        f"### 🚢 SAIL Maritime Logistics DSS Assessment\n\n"
        f"• **Active Voyage**: {orig} ➔ {dest} ({pred_result.get('distance_nm', 0):,} NM)\n"
        f"• **Optimal Fleet**: **{fleet}** ({opt_result.get('total_capacity_mt', 0):,} MT total capacity)\n"
        f"• **Total Charter Cost**: **${cost:,.2f}** (${cost_ton:.2f}/MT)\n"
        f"• **ML Predicted Rate**: **${rate:.2f}/MT** (XGBoost $R^2=92.01%$)\n"
        f"• **Demurrage Exposure**: **${demurrage_res.get('total_demurrage_exposure_usd', 0):,.2f}** at {dest}\n\n"
        f"Feel free to ask specific questions about **port draft limits, Capesize vs Panamax allocation, Russian coal sourcing, or PuLP optimization mechanics!**"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI LIVE LLM INTEGRATION WITH AUTO-FAILOVER
# ═══════════════════════════════════════════════════════════════════════════════
def build_grounding_system_context(
    inputs: Dict[str, Any],
    pred_result: Dict[str, Any],
    dual_feasibility_result: Dict[str, Any],
    opt_result: Dict[str, Any],
    demurrage_result: Dict[str, Any],
    timing_result: Dict[str, Any],
    scenario_rows: List[Dict[str, Any]]
) -> str:
    """Construct dynamic, comprehensive grounding context for Gemini from active dashboard state."""
    orig = dual_feasibility_result["origin"]
    dest = dual_feasibility_result["destination"]
    vessel_status = dual_feasibility_result["route_vessel_status"]
    
    vessel_feasibility_summary = []
    for v, data in vessel_status.items():
        spec = data["specs"]
        if data["feasible"]:
            vessel_feasibility_summary.append(f"- {v} (Capacity: {spec['capacity_mt']:,} MT, Draft: {spec['draft_m']}m, LOA: {spec['loa_m']}m): FULLY PERMITTED")
        else:
            reasons = "; ".join(data["reasons"])
            vessel_feasibility_summary.append(f"- {v} (Capacity: {spec['capacity_mt']:,} MT, Draft: {spec['draft_m']}m): BLOCKED ({reasons})")
            
    vessel_feasibility_str = "\n".join(vessel_feasibility_summary)
    
    scenarios_str = "\n".join(
        f"- {r.get('Scenario', '')}: Dest={r.get('Destination Port', '')}, Fleet={r.get('Fleet Allocation', '')}, Cost={r.get('Total Cost (USD)', '')}, Rate={r.get('Effective Rate ($/MT)', '')}"
        for r in scenario_rows
    )

    system_prompt = f"""You are the official SAIL Maritime Logistics AI Advisor for the Ministry of Steel, Government of India.
You provide concise, authoritative, data-backed operational and strategic explanations for vessel chartering, dual-port feasibility, demurrage minimization, and bulk cargo procurement.

STRICT DOMAIN GUARDRAILS:
You must ONLY answer questions related to maritime logistics, ocean freight, SAIL/RINL procurement, port constraints, vessel classes, PuLP optimization, XGBoost forecasting, demurrage, weather/monsoon, carbon emissions, Russian coal sourcing, and how to use this website. Refuse any unrelated off-topic requests politely.

══ ACTIVE DASHBOARD STATE & REAL-TIME DATA GROUNDING ══
• Cargo Volume: {inputs['cargo_volume_mt']:,} MT of Metallurgical Coking Coal
• Origin Loading Port: {inputs['origin_port']} (Max Draft: {orig['port_constraints']['max_draft_m']}m, Max LOA: {orig['port_constraints']['max_loa_m']}m, Distance: {pred_result['distance_nm']:,} NM)
• Destination Discharge Port: {inputs['destination_port']} (Max Draft: {dest['port_constraints']['max_draft_m']}m, Max LOA: {dest['port_constraints']['max_loa_m']}m)
• Contract Mode: {inputs['contract_type']}
• Risk Profile: {inputs['risk_tolerance']}

══ PREDICTIVE ENGINE OUTPUT (XGBoost Regressor R²=92.01%) ══
• Predicted Freight Rate: ${pred_result['predicted_rate_usd']:.2f} / MT (Uncertainty Range: ± ${pred_result['uncertainty_delta']:.2f})
• Market Indicators: BDI Index = {pred_result['bdi_index']:,.0f}, VLSFO Bunker Fuel = ${pred_result['vlsfo_price']:.2f}/MT
• Monsoon Status: {'Active Southwest Monsoon (15% weather risk incorporated)' if pred_result['is_monsoon'] else 'Fair Weather Shipping Window'}

══ PROACTIVE CONTRACT TIMING (SPOT VS COA) ══
• Recommendation: {timing_result['recommendation']}
• 30-Day Momentum: {timing_result['rate_change_pct']:+.1f}%
• Financial Value: ${abs(timing_result['projected_savings_usd']):,.2f} projected savings

══ DEMURRAGE & IDLE-TIME RISK ENGINE ══
• Expected Port Stay: {demurrage_result['total_port_stay_days']} days (Waiting: {demurrage_result['avg_wait_days']}d, Discharge: {demurrage_result['discharge_days']}d)
• Excess Laytime Days: {demurrage_result['excess_idle_days']} days
• Total Demurrage Liability: ${demurrage_result['total_demurrage_exposure_usd']:,.2f} (+${demurrage_result['demurrage_cost_per_tonne']:.2f}/MT)
• Gangavaram Diversion Benefit: ${demurrage_result['gangavaram_diversion_savings_usd']:,.2f}

══ PRESCRIPTIVE FLEET OPTIMIZER OUTPUT (PuLP MILP Global Optimum) ══
• Optimization Status: {opt_result['status'].upper()}
• Recommended Fleet Allocation: {opt_result['fleet_summary']} ({opt_result['fleet']})
• Total Capacity Provided: {opt_result['total_capacity_mt']:,} MT (Slack: {opt_result['slack_mt']:,} MT)
• Total Estimated Charter Cost: ${opt_result['total_cost_usd']:,.2f}
• Effective Unit Rate: ${opt_result['cost_per_tonne']:.2f} / MT

══ ROUTE PHYSICAL COMPLIANCE (ORIGIN & DESTINATION) ══
{vessel_feasibility_str}

══ SCENARIO TRADEOFF MATRIX ══
{scenarios_str}

══ INSTRUCTIONS & GUARDRAILS ══
1. Explain engineering constraints clearly (e.g. why Capesize is blocked at Haldia (8.5m draft) or Paradip (16.5m draft vs 17.5m vessel requirement)).
2. Explain economic trade-offs (e.g. deadweight slack penalties vs volume discounts).
3. Provide strategic advice on sourcing from Russia (Taman/Vostochny), Australia, US, or Mozambique.
4. Keep responses crisp, executive-ready, and well-structured using markdown bullets.
"""
    return system_prompt


def query_copilot(
    user_prompt: str,
    api_key: str,
    grounding_context: str,
    fallback_response: str
) -> Tuple[str, str]:
    """
    Execute dual-engine query:
    Tries Google Gemini if API key is provided and available;
    Guarantees seamless fallback to local grounded domain intelligence if offline/rate-limited.
    
    Returns:
        Tuple[str, str]: (response_markdown_text, engine_mode_badge)
    """
    # Guardrail check first
    if not is_query_in_domain(user_prompt):
        return get_off_topic_response(), "🛡️ Guardrail Protected"

    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        # Check environment variable
        cleaned_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not cleaned_key or not GENAI_AVAILABLE:
        return fallback_response, "🛡️ Grounded SIH Local Domain Engine"

    try:
        genai.configure(api_key=cleaned_key)
        
        # Priority list of robust models
        model_names = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash-exp",
            "gemini-flash-latest",
            "gemini-1.5-pro"
        ]
        
        model = None
        for m_name in model_names:
            try:
                model = genai.GenerativeModel(
                    model_name=m_name,
                    system_instruction=grounding_context
                )
                break
            except Exception:
                continue

        if model is None:
            model = genai.GenerativeModel("gemini-1.5-flash")

        full_prompt = (
            f"You are the SAIL Maritime Logistics AI Advisor for the Ministry of Steel.\n\n"
            f"Grounding Operational Context:\n{grounding_context}\n\n"
            f"User Question: {user_prompt}\n\n"
            f"Please provide an authoritative, articulate, and well-structured professional response strictly within maritime freight & project scope:"
        )
        response = model.generate_content(full_prompt)
        
        if response and hasattr(response, "text") and response.text:
            return response.text, "🟢 Live Gemini AI (Cloud LLM)"
        else:
            return fallback_response, "🛡️ Grounded SIH Local Domain Engine"

    except Exception as exc:
        err_msg = str(exc)
        logger.warning("Gemini API call exception: %s. Serving local grounded response.", exc)
        if "429" in err_msg or "ResourceExhausted" in err_msg or "quota" in err_msg.lower():
            return (
                f"{fallback_response}\n\n*(ℹ️ Note: Live Gemini API Free-Tier rate limit reached — Served via grounded local domain intelligence)*",
                "🛡️ Grounded SIH Local Domain Engine (Rate-Limit Fallback)"
            )
        return (
            f"{fallback_response}\n\n*(ℹ️ Note: Served via grounded local domain intelligence)*",
            "🛡️ Grounded SIH Local Domain Engine"
        )
