"""
utils package for SAIL Freight Chartering Decision Support System.
"""

from utils.copilot_engine import (
    generate_grounded_local_response,
    build_grounding_system_context,
    query_copilot,
    is_query_in_domain,
    get_off_topic_response,
)
