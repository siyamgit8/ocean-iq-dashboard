"""
dashboard.py - SAIL Maritime Freight Chartering Decision Support System (DSS)
Ministry of Steel, Government of India • Steel Authority of India Limited (SAIL) & RINL
Executive Command Center for Intelligent Freight Rate Forecasting, Dual-Port Feasibility,
Prescriptive MILP Fleet Optimization, Demurrage Risk Quantification, and GenAI Logistics Copilot.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Add parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Google Generative AI for Copilot
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Import backend modules
from utils.predictor import FreightPredictor, ROUTE_DISTANCES
from utils.feasibility import (
    check_feasibility,
    check_single_port_feasibility,
    check_dual_route_feasibility,
    calculate_demurrage_and_idle_risk,
    VESSEL_SPECS,
    PORT_CONGESTION_DATA
)
from utils.optimizer import run_charter_optimization
from utils.copilot_engine import (
    generate_grounded_local_response,
    build_grounding_system_context,
    query_copilot,
    is_query_in_domain,
    get_off_topic_response,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("dashboard")

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SAIL Maritime Freight Chartering DSS | Ministry of Steel",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# PORT GEOGRAPHIC COORDINATES FOR INTERACTIVE MARITIME MAP
# ═══════════════════════════════════════════════════════════════════════════════
PORT_COORDS: Dict[str, Dict[str, Any]] = {
    "Hay Point (Australia)": {"lat": -21.28, "lon": 149.30, "type": "Origin", "country": "Australia", "draft": 17.5},
    "Newcastle (Australia)": {"lat": -32.92, "lon": 151.78, "type": "Origin", "country": "Australia", "draft": 16.2},
    "Baltimore (USA)": {"lat": 39.29, "lon": -76.61, "type": "Origin", "country": "USA", "draft": 15.2},
    "Hampton Roads (USA)": {"lat": 36.95, "lon": -76.33, "type": "Origin", "country": "USA", "draft": 15.2},
    "Nacala (Mozambique)": {"lat": -14.54, "lon": 40.67, "type": "Origin", "country": "Mozambique", "draft": 14.0},
    "Richards Bay (South Africa)": {"lat": -28.78, "lon": 32.04, "type": "Origin", "country": "South Africa", "draft": 17.5},
    "Kalimantan (Indonesia)": {"lat": -3.32, "lon": 114.59, "type": "Origin", "country": "Indonesia", "draft": 16.0},
    "Taboneo (Indonesia)": {"lat": -3.70, "lon": 114.47, "type": "Origin", "country": "Indonesia", "draft": 18.0},
    "Taman (Russia)": {"lat": 45.13, "lon": 36.68, "type": "Origin", "country": "Russia", "draft": 17.5},
    "Vostochny (Russia)": {"lat": 42.74, "lon": 133.08, "type": "Origin", "country": "Russia", "draft": 16.5},
    "Paradip": {"lat": 20.31, "lon": 86.61, "type": "Destination", "country": "India (Odisha)", "draft": 16.5},
    "Visakhapatnam (Vizag)": {"lat": 17.68, "lon": 83.21, "type": "Destination", "country": "India (Andhra Pradesh)", "draft": 14.5},
    "Gangavaram": {"lat": 17.62, "lon": 83.23, "type": "Destination", "country": "India (Andhra Pradesh)", "draft": 18.2},
    "Dhamra": {"lat": 20.80, "lon": 86.97, "type": "Destination", "country": "India (Odisha)", "draft": 18.0},
    "Haldia": {"lat": 22.02, "lon": 88.06, "type": "Destination", "country": "India (West Bengal)", "draft": 8.5},
    "Gopalpur": {"lat": 19.31, "lon": 84.97, "type": "Destination", "country": "India (Odisha)", "draft": 14.5},
}

# ═══════════════════════════════════════════════════════════════════════════════
# ULTRA-PREMIUM CINEMATIC COMMAND CENTER DESIGN SYSTEM (CSS & ANIMATIONS)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    :root {
        --cyan: #22D3EE;
        --violet: #A855F7;
        --lime: #A3E635;
        --amber: #FBBF24;
        --pink: #F472B6;
    }

    /* Base Typography */
    html, body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* App Background: Space Navy with Radial Glow */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(34, 211, 238, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.12) 0%, transparent 45%),
                    radial-gradient(circle at 50% 0%, #0d1330 0%, #04050c 100%) !important;
    }

    /* Top Command Center Hero Header */
    .hero-header {
        background: linear-gradient(135deg, rgba(14, 20, 48, 0.88) 0%, rgba(7, 10, 26, 0.96) 100%);
        border: 1px solid rgba(34, 211, 238, 0.35);
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.7), 0 0 25px -5px rgba(34, 211, 238, 0.15);
        backdrop-filter: blur(16px);
    }
    
    /* Telemetry Badges with Pulsing Animation */
    .telemetry-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(163, 230, 53, 0.12);
        color: #A3E635;
        border: 1px solid rgba(163, 230, 53, 0.4);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 0 10px rgba(163, 230, 53, 0.2);
    }
    
    .telemetry-badge-blue {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 211, 238, 0.12);
        color: #22D3EE;
        border: 1px solid rgba(34, 211, 238, 0.4);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 0 10px rgba(34, 211, 238, 0.2);
    }

    /* Executive KPI Glass Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(14, 20, 48, 0.8) 0%, rgba(7, 10, 26, 0.92) 100%) !important;
        border: 1px solid rgba(34, 211, 238, 0.25) !important;
        border-radius: 16px !important;
        padding: 18px 22px !important;
        box-shadow: 0 10px 28px -6px rgba(0, 0, 0, 0.5) !important;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        border-color: rgba(34, 211, 238, 0.65) !important;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.7), 0 0 20px rgba(34, 211, 238, 0.25) !important;
    }

    div[data-testid="stMetric"] label {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        color: #8892B0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 800 !important;
        color: #22D3EE !important;
    }

    /* Glassmorphism Section Containers */
    .glass-card {
        background: linear-gradient(135deg, rgba(14, 20, 48, 0.75) 0%, rgba(7, 10, 26, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.5);
    }
    
    /* Preset Buttons Bar */
    .preset-container {
        background: rgba(7, 10, 26, 0.75);
        border: 1px solid rgba(34, 211, 238, 0.2);
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }

    /* Feasibility Infographic Cards */
    .vessel-card {
        background: rgba(10, 15, 36, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 18px;
        height: 100%;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .vessel-card:hover {
        transform: translateY(-3px);
    }

    .vessel-pass {
        border-left: 4px solid #A3E635;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(163, 230, 53, 0.08);
    }

    .vessel-fail {
        border-left: 4px solid #F472B6;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(244, 114, 182, 0.08);
    }

    /* Waterline Bar Indicator */
    .waterline-container {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 6px;
        height: 10px;
        width: 100%;
        margin: 12px 0 8px 0;
        overflow: hidden;
        position: relative;
    }

    .waterline-fill-pass {
        background: linear-gradient(90deg, #A3E635, #4ADE80);
        height: 100%;
        border-radius: 6px;
    }

    .waterline-fill-fail {
        background: linear-gradient(90deg, #F472B6, #EF4444);
        height: 100%;
        border-radius: 6px;
    }

    /* Badge Pills */
    .port-pill {
        display: inline-block;
        background: rgba(34, 211, 238, 0.12);
        color: #22D3EE;
        border: 1px solid rgba(34, 211, 238, 0.35);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }

    /* Custom Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(7, 10, 26, 0.75) !important;
        padding: 8px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(34, 211, 238, 0.2) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 10px 22px !important;
        border-radius: 10px !important;
        color: #8892B0 !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 50%, #7E22CE 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45) !important;
        border: 1px solid rgba(34, 211, 238, 0.4) !important;
    }

    /* Primary Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284C7 0%, #22D3EE 50%, #A855F7 100%) !important;
        border: 1px solid rgba(34, 211, 238, 0.6) !important;
        color: #FFFFFF !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        padding: 0.7rem 1.6rem !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 22px rgba(2, 132, 199, 0.5) !important;
        transition: all 0.25s ease !important;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369A1 0%, #38BDF8 50%, #C084FC 100%) !important;
        box-shadow: 0 8px 30px rgba(2, 132, 199, 0.7) !important;
        transform: translateY(-2px) !important;
    }

    /* Sidebar Theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070a1a 0%, #04050c 100%) !important;
        border-right: 1px solid rgba(34, 211, 238, 0.15) !important;
    }

    /* Copilot Terminal Container */
    .copilot-card {
        background: linear-gradient(135deg, rgba(10, 15, 36, 0.9) 0%, rgba(18, 24, 56, 0.95) 100%);
        border: 1px solid rgba(34, 211, 238, 0.35);
        border-radius: 18px;
        padding: 26px;
        box-shadow: 0 14px 40px -8px rgba(0, 0, 0, 0.6);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND INITIALIZATION (Cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_predictor_engine() -> FreightPredictor:
    """Initialize and cache the Machine Learning Freight Predictor."""
    return FreightPredictor()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: USER CONFIGURATION PANEL
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> Dict[str, Any]:
    """Render the sidebar input widgets and handle preset parameter overrides."""
    st.sidebar.markdown("## 🚢 SAIL Logistics Command")
    st.sidebar.caption("Ministry of Steel • Government of India")
    st.sidebar.markdown("---")

    origin_options = [
        "Hay Point (Australia)",
        "Newcastle (Australia)",
        "Baltimore (USA)",
        "Hampton Roads (USA)",
        "Nacala (Mozambique)",
        "Richards Bay (South Africa)",
        "Kalimantan (Indonesia)",
        "Taboneo (Indonesia)",
        "Taman (Russia)",
        "Vostochny (Russia)"
    ]

    dest_options = ["Paradip", "Visakhapatnam (Vizag)", "Gangavaram", "Dhamra", "Haldia", "Gopalpur"]

    # Initialize default session state if not already set
    if "cargo_vol_input" not in st.session_state:
        st.session_state["cargo_vol_input"] = 150000
    if "origin_select" not in st.session_state:
        st.session_state["origin_select"] = "Hay Point (Australia)"
    if "dest_select" not in st.session_state:
        st.session_state["dest_select"] = "Paradip"
    if "contract_radio" not in st.session_state:
        st.session_state["contract_radio"] = "Spot"

    # Ensure selected options are valid
    if st.session_state["origin_select"] not in origin_options:
        st.session_state["origin_select"] = origin_options[0]
    if st.session_state["dest_select"] not in dest_options:
        st.session_state["dest_select"] = dest_options[0]

    cargo_volume_mt = st.sidebar.number_input(
        "Cargo Parcel Volume (MT)",
        min_value=5000,
        max_value=2000000,
        step=5000,
        key="cargo_vol_input",
        help="Total tonnage of metallurgical coking coal cargo to be chartered."
    )

    origin_port = st.sidebar.selectbox(
        "Origin Loading Port (Overseas)",
        options=origin_options,
        key="origin_select",
        help="Overseas bulk export terminal where metallurgical coal parcel is loaded."
    )

    destination_port = st.sidebar.selectbox(
        "Destination Discharge Port (East Coast India)",
        options=dest_options,
        key="dest_select",
        help="Designated Indian East Coast port for vessel berthing and blast furnace logistics."
    )

    contract_type = st.sidebar.radio(
        "Contract Mode",
        options=["Spot", "Time Charter / COA"],
        horizontal=True,
        key="contract_radio",
        help="Select between immediate single-voyage Spot fixture or periodic Time Charter / COA."
    )

    risk_tolerance = st.sidebar.select_slider(
        "Procurement Risk Profile",
        options=["Conservative", "Balanced", "Aggressive"],
        value="Balanced",
        help="Conservative prioritizes berth clearance & zero slack; Aggressive prioritizes lowest spot rate."
    )

    # Silently resolve Google Gemini API Key from Streamlit Secrets or Environment Variable
    secret_key = ""
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            secret_key = str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    api_key_backend = secret_key or os.environ.get("GEMINI_API_KEY", "")

    get_rec_btn = st.sidebar.button(
        "🔍 Run DSS Optimization Pipeline",
        type="primary",
        use_container_width=True
    )

    st.sidebar.markdown(
        """
        <div style='font-size: 0.75rem; color: #94A3B8; text-align: center; margin-top: 1.5rem;'>
            ML Core: XGBoost Regressor (R² = 92.01%)<br>
            Prescriptive: PuLP Mixed-Integer LP<br>
            Origins: Australia, USA, Russia, Mozambique, Indonesia<br>
            Demurrage & COA Risk Engine Active
        </div>
        """,
        unsafe_allow_html=True
    )

    return {
        "cargo_volume_mt": cargo_volume_mt,
        "origin_port": origin_port,
        "destination_port": destination_port,
        "contract_type": "Spot" if "Spot" in contract_type else "Time Charter",
        "risk_tolerance": risk_tolerance,
        "gemini_api_key": api_key_backend,
        "submitted": get_rec_btn
    }



# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 1: INTERACTIVE GLOWING GEOSPATIAL SHIPPING MAP (PLOTLY GEO)
# ═══════════════════════════════════════════════════════════════════════════════
def render_geospatial_shipping_map(origin_port: str, destination_port: str, distance_nm: float) -> None:
    """Render interactive dark geospatial world map showing glowing voyage routes and port limits."""
    
    orig_info = PORT_COORDS.get(origin_port, {"lat": 0, "lon": 0, "draft": 16.5, "country": "International"})
    dest_info = PORT_COORDS.get(destination_port, {"lat": 20.31, "lon": 86.61, "draft": 16.5, "country": "India"})
    
    fig = go.Figure()

    # 1. Background Port Markers (All Global Nodes)
    all_lats = [info["lat"] for info in PORT_COORDS.values()]
    all_lons = [info["lon"] for info in PORT_COORDS.values()]
    all_texts = [
        f"<b>{name}</b><br>Type: {info['type']}<br>Region: {info['country']}<br>Max Draft: {info['draft']}m"
        for name, info in PORT_COORDS.items()
    ]
    
    fig.add_trace(go.Scattergeo(
        lon=all_lons,
        lat=all_lats,
        mode="markers",
        marker=dict(
            size=7,
            color="rgba(148, 163, 184, 0.6)",
            line=dict(width=1, color="rgba(255, 255, 255, 0.4)")
        ),
        text=all_texts,
        hoverinfo="text",
        name="Global Coal Terminals"
    ))

    # 2. Active Voyage Curved / Great-Circle Route (Glowing Cyan Line)
    fig.add_trace(go.Scattergeo(
        lon=[orig_info["lon"], dest_info["lon"]],
        lat=[orig_info["lat"], dest_info["lat"]],
        mode="lines",
        line=dict(width=3.5, color="#00F2FE"),
        name=f"Active Voyage ({distance_nm:,.0f} NM)",
        hoverinfo="skip"
    ))

    # 3. Active Origin Marker (Glowing Orange Pin)
    fig.add_trace(go.Scattergeo(
        lon=[orig_info["lon"]],
        lat=[orig_info["lat"]],
        mode="markers+text",
        marker=dict(size=14, color="#F97316", symbol="circle", line=dict(width=2, color="#FFFFFF")),
        text=[f"🛫 {origin_port}"],
        textposition="top center",
        name="Origin Loading Port",
        hoverinfo="text",
        hovertext=f"<b>Origin Port: {origin_port}</b><br>Max Draft: {orig_info['draft']}m<br>Region: {orig_info['country']}"
    ))

    # 4. Active Destination Marker (Glowing Emerald Pin)
    fig.add_trace(go.Scattergeo(
        lon=[dest_info["lon"]],
        lat=[dest_info["lat"]],
        mode="markers+text",
        marker=dict(size=14, color="#10B981", symbol="diamond", line=dict(width=2, color="#FFFFFF")),
        text=[f"🛬 {destination_port}"],
        textposition="bottom center",
        name="Destination Discharge Port",
        hoverinfo="text",
        hovertext=f"<b>Destination Port: {destination_port}</b><br>Max Draft: {dest_info['draft']}m<br>Region: {dest_info['country']}"
    ))

    # Layout Aesthetics
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.0)",
        plot_bgcolor="rgba(15, 23, 42, 0.0)",
        margin=dict(l=0, r=0, t=10, b=10),
        height=360,
        showlegend=False,
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="#1E293B",
            showocean=True,
            oceancolor="#0B0F19",
            showlakes=True,
            lakecolor="#0B0F19",
            showcountries=True,
            countrycolor="rgba(148, 163, 184, 0.2)",
            coastlinecolor="rgba(148, 163, 184, 0.3)",
            bgcolor="rgba(15, 23, 42, 0.0)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2: VISUAL WATERLINE DRAFT FEASIBILITY CARDS
# ═══════════════════════════════════════════════════════════════════════════════
def render_waterline_feasibility_cards(dual_res: Dict[str, Any]) -> None:
    """Render visual vessel silhouette cards with draft water-depth progress meters."""
    dest_draft = dual_res["destination"]["port_constraints"]["max_draft_m"] or 16.5
    dest_name = dual_res["destination"]["port_name"]
    
    col_a, col_b, col_c, col_d = st.columns(4)
    cols = [col_a, col_b, col_c, col_d]
    vessels = ["Handysize", "Supramax", "Panamax", "Capesize"]

    for col, v_name in zip(cols, vessels):
        v_data = dual_res["route_vessel_status"][v_name]
        spec = v_data["specs"]
        feasible = v_data["feasible"]
        draft = spec["draft_m"]
        
        # Calculate percentage of port depth utilized
        pct = min(100.0, (draft / max(dest_draft, 8.0)) * 100.0)
        
        with col:
            if feasible:
                st.markdown(
                    f"""
                    <div class="vessel-card vessel-pass">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.1rem; font-weight: 800; color: #10B981;">✅ {v_name}</span>
                            <span style="font-size: 0.78rem; font-weight: 700; color: #34D399; background: rgba(16, 185, 129, 0.15); padding: 2px 8px; border-radius: 6px;">PERMITTED</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px;">Capacity: <b>{spec['capacity_mt']:,} MT</b> | LOA: <b>{spec['loa_m']:.0f}m</b></div>
                        <div class="waterline-container">
                            <div class="waterline-fill-pass" style="width: {pct:.1f}%;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #CBD5E1;">
                            <span>Draft: <b>{draft}m</b></span>
                            <span>Port Max: <b>{dest_draft}m</b></span>
                        </div>
                        <div style="font-size: 0.75rem; color: #10B981; margin-top: 8px; font-weight: 600;">✓ Safe Under-Keel Clearance (UKC)</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                reasons_html = "<br>• ".join(v_data["reasons"])
                st.markdown(
                    f"""
                    <div class="vessel-card vessel-fail">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.1rem; font-weight: 800; color: #EF4444;">❌ {v_name}</span>
                            <span style="font-size: 0.78rem; font-weight: 700; color: #F87171; background: rgba(239, 68, 68, 0.15); padding: 2px 8px; border-radius: 6px;">BLOCKED</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px;">Capacity: <b>{spec['capacity_mt']:,} MT</b> | LOA: <b>{spec['loa_m']:.0f}m</b></div>
                        <div class="waterline-container">
                            <div class="waterline-fill-fail" style="width: 100%;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #F87171;">
                            <span>Draft: <b>{draft}m</b></span>
                            <span>Port Max: <b>{dest_draft}m</b></span>
                        </div>
                        <div style="font-size: 0.73rem; color: #FCA5A5; margin-top: 8px; line-height: 1.3;"><b>Grounding Risk:</b><br>• {reasons_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 3: ONE-CLICK QUICK DEMO PRESETS (TOOLBAR FOR JUDGES)
# ═══════════════════════════════════════════════════════════════════════════════
def render_preset_toolbar() -> None:
    """Render quick scenario presets toolbar allowing 1-click demonstration of key use cases."""
    st.markdown("**⚡ One-Click Executive Demo Scenarios (Click to test live):**")
    
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    with p_col1:
        if st.button("🇦🇺 Australia ➔ Paradip (150k MT)", use_container_width=True, key="preset_btn_au"):
            st.session_state["cargo_vol_input"] = 150000
            st.session_state["origin_select"] = "Hay Point (Australia)"
            st.session_state["dest_select"] = "Paradip"
            st.session_state["contract_radio"] = "Spot"
            st.rerun()

    with p_col2:
        if st.button("🇷🇺 Russia ➔ Gangavaram (170k MT)", use_container_width=True, key="preset_btn_ru"):
            st.session_state["cargo_vol_input"] = 170000
            st.session_state["origin_select"] = "Taman (Russia)"
            st.session_state["dest_select"] = "Gangavaram"
            st.session_state["contract_radio"] = "Time Charter / COA"
            st.rerun()

    with p_col3:
        if st.button("🇺🇸 USA ➔ Haldia River (75k MT)", use_container_width=True, key="preset_btn_us"):
            st.session_state["cargo_vol_input"] = 75000
            st.session_state["origin_select"] = "Hampton Roads (USA)"
            st.session_state["dest_select"] = "Haldia"
            st.session_state["contract_radio"] = "Spot"
            st.rerun()

    with p_col4:
        if st.button("🇮🇩 Indonesia ➔ Vizag (55k MT)", use_container_width=True, key="preset_btn_id"):
            st.session_state["cargo_vol_input"] = 55000
            st.session_state["origin_select"] = "Kalimantan (Indonesia)"
            st.session_state["dest_select"] = "Visakhapatnam (Vizag)"
            st.session_state["contract_radio"] = "Spot"
            st.rerun()



# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTROLLER & APPLICATION FLOW
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # 1. Render Sidebar Inputs
    inputs = render_sidebar()

    # Top Command Center Hero Header
    st.markdown(
        """
        <div class="hero-header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                        🏛️ Ministry of Steel • Government of India
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">
                        SAIL Maritime Freight Chartering & Decision Support Command Center
                    </div>
                    <div style="font-size: 0.92rem; color: #94A3B8; margin-top: 4px;">
                        Intelligent Freight Forecasting, Dual-Port Navigational Feasibility, Prescriptive MILP Fleet Optimization, and GenAI Copilot
                    </div>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="telemetry-badge">🟢 ML PIPELINE CONNECTED</span>
                    <span class="telemetry-badge-blue">⚡ PU-LP SOLVER: READY</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Render One-Click Demo Presets Toolbar
    render_preset_toolbar()

    # 2. Trigger Pipeline Execution
    try:
        predictor = load_predictor_engine()
        
        # ML Inference
        pred_result = predictor.predict_rate(
            origin_port=inputs["origin_port"],
            destination_port=inputs["destination_port"],
            contract_type=inputs["contract_type"]
        )

        # Dual-Port Route Feasibility Check (Origin + Destination)
        dual_feasibility_result = check_dual_route_feasibility(
            origin_port=inputs["origin_port"],
            destination_port=inputs["destination_port"]
        )

        # Prescriptive MILP Fleet Optimization
        opt_result = run_charter_optimization(
            cargo_volume_mt=float(inputs["cargo_volume_mt"]),
            origin_port=inputs["origin_port"],
            destination_port=inputs["destination_port"],
            predicted_freight_rate_usd=pred_result["predicted_rate_usd"]
        )

        # Demurrage & Idle Time Risk Calculation
        primary_vessel = "Panamax"
        fleet_count = 1
        if opt_result["status"] == "optimal":
            for v, c in opt_result["fleet"].items():
                if c > 0:
                    primary_vessel = v
                    fleet_count = c
                    break
                    
        demurrage_res = calculate_demurrage_and_idle_risk(
            destination_port=inputs["destination_port"],
            vessel_class=primary_vessel,
            fleet_count=fleet_count,
            cargo_volume_mt=float(inputs["cargo_volume_mt"])
        )

        # Proactive Contract Timing (Spot vs Mid-Term COA)
        timing_res = predictor.evaluate_contract_timing_strategy(
            origin_port=inputs["origin_port"],
            destination_port=inputs["destination_port"],
            current_rate=pred_result["predicted_rate_usd"],
            cargo_volume_mt=float(inputs["cargo_volume_mt"])
        )

        # Interactive Visual Execution Feedback
        if inputs["submitted"]:
            st.toast(f"⚡ DSS Pipeline Executed: Allocated {opt_result['fleet_summary']} for {inputs['origin_port']} ➔ {inputs['destination_port']}!", icon="🚢")

        # Top Executive KPI Cards Row
        st.markdown('<div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 12px;">📊 Active Strategic Logistics Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rate = pred_result["predicted_rate_usd"]
            delta = pred_result["uncertainty_delta"]
            st.metric(
                label="Predicted Base Freight Rate",
                value=f"${rate:.2f} / MT",
                delta=f"± ${delta:.2f} uncertainty band",
                delta_color="off"
            )

        with col2:
            if opt_result["status"] == "optimal":
                fleet_display = opt_result["fleet_summary"]
                capacity_display = f"{opt_result['total_capacity_mt']:,} MT ({opt_result['slack_mt']:,} slack)"
            else:
                fleet_display = "Infeasible"
                capacity_display = "All vessels blocked"
                
            st.metric(
                label="Optimal Fleet Charter",
                value=fleet_display,
                delta=capacity_display,
                delta_color="normal" if opt_result["status"] == "optimal" else "inverse"
            )

        with col3:
            if opt_result["status"] == "optimal":
                total_cost_str = f"${opt_result['total_cost_usd']:,.2f}"
                unit_cost_str = f"${opt_result['cost_per_tonne']:.2f} / ton effective"
            else:
                total_cost_str = "N/A"
                unit_cost_str = "Alternative required"
                
            st.metric(
                label="Total Ocean Freight Cost",
                value=total_cost_str,
                delta=unit_cost_str,
                delta_color="off"
            )

        with col4:
            dem_cost = demurrage_res["total_demurrage_exposure_usd"]
            dem_unit = demurrage_res["demurrage_cost_per_tonne"]
            st.metric(
                label="Expected Demurrage Risk",
                value=f"${dem_cost:,.0f}",
                delta=f"+${dem_unit:.2f}/t idle exposure",
                delta_color="inverse" if dem_cost > 10000 else "normal"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════════════
        # 4-TAB EXECUTIVE COCKPIT WORKSPACE
        # ═══════════════════════════════════════════════════════════════════════
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ 1. Geospatial Routes & Dual-Port Feasibility",
            "📈 2. Predictive Freight Forecaster",
            "📐 3. Prescriptive Fleet Optimizer & Landed Cost",
            "⏳ 4. Demurrage & Weather Risk Radar"
        ])

        # ─── TAB 1: GEOSPATIAL & DUAL PORT FEASIBILITY ───────────────────────
        with tab1:
            st.markdown("### 🗺️ Global Sourcing Lanes & Navigational Feasibility")
            
            # Interactive Map
            render_geospatial_shipping_map(
                origin_port=inputs["origin_port"],
                destination_port=inputs["destination_port"],
                distance_nm=pred_result["distance_nm"]
            )
            
            orig = dual_feasibility_result["origin"]
            dest = dual_feasibility_result["destination"]
            c_orig = orig["port_constraints"]
            c_dest = dest["port_constraints"]
            
            st.markdown(
                f"""
                <div style='margin: 16px 0 12px 0;'>
                    <span class='port-pill'>🛫 <b>Origin Terminal:</b> {orig['port_name']} ({orig['country_or_state']}) | Max Draft: {c_orig['max_draft_m']}m | Max LOA: {c_orig['max_loa_m']}m</span>
                    <span class='port-pill'>🛬 <b>Discharge Terminal:</b> {dest['port_name']} ({dest['country_or_state']}) | Max Draft: {c_dest['max_draft_m']}m | Max LOA: {c_dest['max_loa_m']}m</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Waterline Infographic Cards
            render_waterline_feasibility_cards(dual_feasibility_result)

        # ─── TAB 2: PREDICTIVE FREIGHT FORECASTER ────────────────────────────
        with tab2:
            st.markdown("### 📈 Machine Learning Rate Forecast (XGBoost R²=92.01%)")
            
            # Proactive Timing Card
            col_t1, col_t2 = st.columns([1.8, 1.2])
            with col_t1:
                badge_color = "#F87171" if timing_res["rec_type"] == "mid_term" else "#34D399"
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size: 1.15rem; font-weight: 700; color: #38BDF8; margin-bottom: 8px;">
                            🎯 Strategic Contract Recommendation: {timing_res['recommendation']}
                        </div>
                        <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 12px;">
                            {timing_res['rationale']}
                        </div>
                        <div style="display: flex; gap: 16px; font-size: 0.85rem; color: #94A3B8;">
                            <span>• 30-Day Momentum: <b style="color: {badge_color};">{timing_res['rate_change_pct']:+.1f}%</b></span>
                            <span>• Spot Rate: <b style="color: #F8FAFC;">${timing_res['current_spot_rate']:.2f}/t</b></span>
                            <span>• Mid-Term COA: <b style="color: #F8FAFC;">${timing_res['mid_term_fixed_rate']:.2f}/t</b></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_t2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="text-align: center;">
                        <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Procurement Financial Impact</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: {'#10B981' if timing_res['projected_savings_usd'] > 0 else '#38BDF8'}; margin: 8px 0;">
                            ${abs(timing_res['projected_savings_usd']):,.2f}
                        </div>
                        <div style="font-size: 0.82rem; color: #CBD5E1;">
                            {'Estimated cost avoidance by locking in mid-term COA vs spot inflation' if timing_res['projected_savings_usd'] > 0 else 'Spot chartering preserves liquidity and captures softening rates'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 30-Day Forecast Plotly Chart
            df_chart = predictor.get_historical_and_forecast(inputs["origin_port"], inputs["destination_port"], days_history=60, days_forecast=30)
            hist_df = df_chart[df_chart["type"] == "Historical"]
            fore_df = df_chart[df_chart["type"] == "Forecast"]

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=hist_df["date"], y=hist_df["rate"], mode="lines", name="60-Day Historical Actuals",
                line=dict(color="#38BDF8", width=2.5)
            ))
            fig_f.add_trace(go.Scatter(
                x=fore_df["date"], y=fore_df["upper_band"], mode="lines", name="Upper Band (+15%)",
                line=dict(color="rgba(249, 115, 22, 0)"), showlegend=False, hoverinfo="skip"
            ))
            fig_f.add_trace(go.Scatter(
                x=fore_df["date"], y=fore_df["lower_band"], mode="lines", name="Confidence Band (±15%)",
                line=dict(color="rgba(249, 115, 22, 0)"), fill="tonexty", fillcolor="rgba(249, 115, 22, 0.18)", hoverinfo="skip"
            ))
            fig_f.add_trace(go.Scatter(
                x=fore_df["date"], y=fore_df["rate"], mode="lines", name="30-Day ML Forecast",
                line=dict(color="#F97316", width=3.0, dash="dash")
            ))
            fig_f.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=380,
                xaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)", title="Timeline (Date)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)", title="Freight Rate ($ / MT)", ticksuffix=" $/t"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_f, use_container_width=True)

        # ─── TAB 3: FLEET OPTIMIZER & SCENARIOS ──────────────────────────────
        with tab3:
            st.markdown("### 📐 Mixed-Integer Linear Programming (MILP) Fleet Allocation")
            
            # Fleet Composition summary
            f_col1, f_col2 = st.columns([1.5, 1.5])
            with f_col1:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size: 1.1rem; font-weight: 700; color: #38BDF8; margin-bottom: 8px;">⚓ Global Optimum Fleet Composition</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #10B981; margin-bottom: 6px;">{opt_result['fleet_summary']}</div>
                        <div style="font-size: 0.88rem; color: #CBD5E1;">Total Capacity Provided: <b>{opt_result['total_capacity_mt']:,} MT</b> | Deadweight Slack: <b>{opt_result['slack_mt']:,} MT</b></div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px;">Solver: <b>PuLP CBC Mixed-Integer Optimizer</b> (Global Mathematical Optimum)</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with f_col2:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size: 1.1rem; font-weight: 700; color: #38BDF8; margin-bottom: 8px;">💰 Financial Expenditure Analysis</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC; margin-bottom: 6px;">${opt_result['total_cost_usd']:,.2f}</div>
                        <div style="font-size: 0.88rem; color: #CBD5E1;">Effective Unit Landed Rate: <b>${opt_result['cost_per_tonne']:.2f} / Metric Tonne</b></div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px;">Includes economy of scale discounts & cargo capacity utilization penalties.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Scenario Matrix Table
            st.markdown("#### 🔄 Alternative Scenario Comparison Matrix")
            scenarios = [
                {"name": "1. Charter Now (Base Spot Plan)", "dest": inputs["destination_port"], "vol": float(inputs["cargo_volume_mt"]), "rf": 1.00},
                {"name": "2. Wait 7 Days (Capture Dip)", "dest": inputs["destination_port"], "vol": float(inputs["cargo_volume_mt"]), "rf": 0.965},
                {"name": "3. Via Alternative Deep Port (Gangavaram)", "dest": "Gangavaram", "vol": float(inputs["cargo_volume_mt"]), "rf": 0.95},
                {"name": "4. Split Voyages (2x 50% Batches)", "dest": inputs["destination_port"], "vol": float(inputs["cargo_volume_mt"]) / 2.0, "rf": 1.02}
            ]

            rows = []
            best_scenario = None
            min_eff = float("inf")

            for s in scenarios:
                adj_rate = pred_result["predicted_rate_usd"] * s["rf"]
                s_opt = run_charter_optimization(s["vol"], inputs["origin_port"], s["dest"], adj_rate)
                
                if s_opt["status"] == "optimal":
                    mult = 2.0 if "Split" in s["name"] else 1.0
                    cost = s_opt["total_cost_usd"] * mult
                    cap = s_opt["total_capacity_mt"] * mult
                    slack = cap - float(inputs["cargo_volume_mt"])
                    eff = cost / float(inputs["cargo_volume_mt"])
                    fleet_s = f"2x ({s_opt['fleet_summary']})" if "Split" in s["name"] else s_opt["fleet_summary"]
                else:
                    cost, eff, slack, fleet_s = np.nan, np.nan, np.nan, "Infeasible"

                rows.append({
                    "Scenario": s["name"],
                    "Destination Port": s_opt["destination_port"],
                    "Fleet Allocation": fleet_s,
                    "Total Cost (USD)": f"${cost:,.2f}" if pd.notna(cost) else "N/A",
                    "Effective Rate ($/MT)": f"${eff:.2f}" if pd.notna(eff) else "N/A",
                    "Capacity Slack (MT)": f"{slack:,.0f}" if pd.notna(slack) else "N/A",
                    "_raw_cost": cost,
                    "_raw_eff": eff
                })

                if pd.notna(eff) and eff < min_eff:
                    min_eff = eff
                    best_scenario = s["name"]

            st.dataframe(pd.DataFrame(rows)[["Scenario", "Destination Port", "Fleet Allocation", "Total Cost (USD)", "Effective Rate ($/MT)", "Capacity Slack (MT)"]], use_container_width=True, hide_index=True)

            if best_scenario:
                base_c = rows[0]["_raw_cost"]
                best_c = min(r["_raw_cost"] for r in rows if pd.notna(r["_raw_cost"]))
                savings = base_c - best_c
                if savings > 100:
                    st.success(f"💡 **Recommended Strategy**: **{best_scenario}** yields the lowest landed cost (${min_eff:.2f}/ton), saving **${savings:,.2f}** compared to immediate base chartering.")

        # ─── TAB 4: DEMURRAGE & RISK RADAR ───────────────────────────────────
        with tab4:
            st.markdown("### ⏳ Port Demurrage, Queue Delays & Weather Risk Intelligence")
            
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                st.metric("Port Queue Status", demurrage_res["port_status"].split("/")[0].strip(), f"{demurrage_res['avg_wait_days']} days anchor wait", delta_color="off")
            with r_col2:
                st.metric("Total Port Stay", f"{demurrage_res['total_port_stay_days']} Days", f"{demurrage_res['discharge_days']}d discharge + {demurrage_res['avg_wait_days']}d wait", delta_color="off")
            with r_col3:
                st.metric("Excess Laytime Penalty", f"{demurrage_res['excess_idle_days']} Idle Days", f"Daily Hire: ${demurrage_res['daily_demurrage_rate_usd']:,.0f}/day", delta_color="inverse" if demurrage_res["excess_idle_days"] > 0 else "normal")
            with r_col4:
                st.metric("Gangavaram Diversion Benefit", f"${demurrage_res['gangavaram_diversion_savings_usd']:,.0f}", "vs. Gangavaram fast turnaround", delta_color="normal")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Operational Risk Alerts
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                if inputs["destination_port"] in ["Paradip", "Haldia"]:
                    st.error(f"🚨 **High Congestion ({inputs['destination_port']})**: Queue wait ~4.5–5.0d. Consider pre-berthing clearance or alternate deep discharge at Gangavaram.")
                else:
                    st.success(f"🟢 **Fluid Congestion ({inputs['destination_port']})**: Turnaround index within safe limits.")
            with col_w2:
                bdi_val = pred_result.get("bdi_index", 1850)
                if bdi_val > 2200 or inputs["risk_tolerance"] == "Conservative":
                    st.warning(f"⚠️ **Market Volatility (BDI: {bdi_val:,.0f})**: Momentum elevated. Lock in Time Charter or hedge bunker fuel.")
                else:
                    st.success(f"🟢 **Stable BDI Index ({bdi_val:,.0f})**: Freight market volatility moderate.")
            with col_w3:
                month = datetime.now().month
                if month in [6, 7, 8, 9]:
                    st.warning("🌧️ **Active Monsoon Season (SW Monsoon)**: Swell delays expected in Bay of Bengal (15% weather risk multiplier applied).")
                else:
                    st.success("☀️ **Fair Weather Shipping Window**: Optimal loading and discharge productivity expected.")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION H: DOCKED GENAI LOGISTICS COPILOT (DUAL-ENGINE SIH ADVISOR)
        # ═══════════════════════════════════════════════════════════════════════
        st.markdown("<br><hr>", unsafe_allow_html=True)
        
        copilot_head_col1, copilot_head_col2 = st.columns([3, 1])
        with copilot_head_col1:
            st.markdown("### 🤖 SAIL Logistics AI Copilot — Strategic Natural Language Advisor")
            st.markdown(
                """
                <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 10px;">
                    Conversational Decision Support grounded in live XGBoost ML predictions, dual-port draft feasibility, and PuLP MILP fleet optimization.
                </div>
                """,
                unsafe_allow_html=True
            )
        with copilot_head_col2:
            st.markdown("<div style='text-align: right; padding-top: 10px;'>", unsafe_allow_html=True)
            if st.button("🧹 Clear Conversation", key="clear_copilot_chat"):
                st.session_state.copilot_messages = []
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("💬 Open Conversational Command Terminal", expanded=True):
            # Engine Mode Status Banner
            api_key_active = bool(inputs.get("gemini_api_key", "").strip())
            if api_key_active:
                st.markdown("<div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.85rem; color: #10B981;'>🟢 <b>Live Multimodal LLM Engine Connected</b> &bull; Domain-Grounded SIH Project Intelligence</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.85rem; color: #38BDF8;'>🛡️ <b>Evergreen Grounded Domain Engine Active</b> &bull; 100% Offline-Immune &bull; Zero-Dependency Evaluation Ready</div>", unsafe_allow_html=True)


            # Initialize chat history
            if "copilot_messages" not in st.session_state or not st.session_state.copilot_messages:
                st.session_state.copilot_messages = [
                    {
                        "role": "assistant",
                        "badge": "🛡️ SIH System Assistant",
                        "content": (
                            "👋 **Greetings! I am your SAIL Maritime Logistics AI Advisor for the Ministry of Steel.**\n\n"
                            "I am strictly grounded in your active dashboard parameters, mathematical fleet optimizations, and ocean freight logistics.\n\n"
                            "**🎯 Quick Demo Questions (Click below or type your own question):**\n"
                            "1. *Why did the system allocate 2x Panamax instead of a single Capesize for this cargo?*\n"
                            "2. *How does sourcing from Russian ports (Taman / Vostochny) compare to Australian origins?*\n"
                            "3. *What is our expected demurrage liability at this discharge port and how can we mitigate it?*\n"
                            "4. *What are the riverine navigation constraints at Haldia port?*\n"
                            "5. *How does the PuLP Mixed-Integer Linear Programming (MILP) model formulate the fleet cost?*\n"
                            "6. *What machine learning architecture is used for freight rate forecasting?*"
                        )
                    }
                ]

            # Quick Prompt Action Chips - Row 1
            st.markdown("**⚡ Quick Prompt Chips (Click to Ask):**")
            chip_col1, chip_col2, chip_col3 = st.columns(3)
            
            selected_chip = None
            with chip_col1:
                if st.button("🚢 1. 2x Panamax vs Capesize?", use_container_width=True, key="chip_panamax"):
                    selected_chip = "Why did the system allocate 2x Panamax instead of a single Capesize for this cargo?"
            with chip_col2:
                if st.button("🌊 2. Haldia River Constraints?", use_container_width=True, key="chip_haldia"):
                    selected_chip = "What are the riverine navigation constraints and vessel restrictions at Haldia port?"
            with chip_col3:
                if st.button("🇷🇺 3. Russian Coal Sourcing?", use_container_width=True, key="chip_russia"):
                    selected_chip = "How does sourcing metallurgical coal from Russian ports (Taman / Vostochny) compare to Australian origins?"

            # Quick Prompt Action Chips - Row 2
            chip_col4, chip_col5, chip_col6 = st.columns(3)
            with chip_col4:
                if st.button("⏳ 4. Demurrage & Gangavaram?", use_container_width=True, key="chip_demurrage"):
                    selected_chip = "What is our expected demurrage liability at this discharge port and how can we mitigate it via Gangavaram?"
            with chip_col5:
                if st.button("📊 5. XGBoost ML Architecture?", use_container_width=True, key="chip_xgboost"):
                    selected_chip = "What machine learning architecture and feature inputs are used for freight rate forecasting?"
            with chip_col6:
                if st.button("⚙️ 6. PuLP MILP Mathematical Model?", use_container_width=True, key="chip_milp"):
                    selected_chip = "How does the PuLP Mixed-Integer Linear Programming (MILP) model formulate the vessel allocation problem?"

            st.markdown("---")

            # Render conversation history
            for msg in st.session_state.copilot_messages:
                avatar = "🧑‍💼" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
                    if "badge" in msg and msg["badge"] and msg["role"] == "assistant":
                        st.caption(f"Engine: {msg['badge']}")

            # Construct dynamic grounding system context
            grounding_context = build_grounding_system_context(
                inputs=inputs,
                pred_result=pred_result,
                dual_feasibility_result=dual_feasibility_result,
                opt_result=opt_result,
                demurrage_result=demurrage_res,
                timing_result=timing_res,
                scenario_rows=rows
            )

            # Chat Input Box
            user_query = st.chat_input("Ask a logistics, draft, demurrage, ML, or chartering question...")
            query_to_process = selected_chip or user_query

            if query_to_process:
                st.session_state.copilot_messages.append({"role": "user", "content": query_to_process})
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(query_to_process)

                fallback_ans = generate_grounded_local_response(
                    query=query_to_process,
                    inputs=inputs,
                    opt_result=opt_result,
                    dual_feasibility_result=dual_feasibility_result,
                    pred_result=pred_result,
                    demurrage_res=demurrage_res,
                    timing_res=timing_res
                )

                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Analyzing operational constraints & generating strategic advice..."):
                        ai_reply, engine_badge = query_copilot(
                            user_prompt=query_to_process,
                            api_key=inputs.get("gemini_api_key", ""),
                            grounding_context=grounding_context,
                            fallback_response=fallback_ans
                        )
                        st.markdown(ai_reply)
                        st.caption(f"Engine: {engine_badge}")

                st.session_state.copilot_messages.append({
                    "role": "assistant",
                    "content": ai_reply,
                    "badge": engine_badge
                })

    except Exception as exc:
        st.error(f"❌ An error occurred while executing the decision support engine: {exc}")
        logger.error("Dashboard error: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
