from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    defaults = {
        "openai_api_key": "",
        "qdrant_mode": "local",
        "qdrant_url": "",
        "qdrant_api_key": "",
        "qdrant_local_path": "qdrant_local_data",
        "embeddings": None,
        "llm": None,
        "router_llm": None,
        "qdrant_client": None,
        "collection_definitions": None,
        "collection_summary_embeddings": {},
        "last_query_result": None,
        "analytics_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
