from __future__ import annotations

import streamlit as st

from analytics import log_event, summarize_events
from backend import initialize_backend
from config import (
    SOURCE_RELIABILITY_OPTIONS,
    SUPPORTED_DOCUMENT_TYPES,
    SUPPORTED_LANGUAGES,
    SUPPORTED_MARKETS,
    SUPPORTED_VEHICLE_MODELS,
    load_collection_definitions,
)
from demo_seed import seed_demo_dataset
from fallback import web_fallback
from ingestion import process_document, store_chunks
from planner import build_query_plan
from retrieval import confidence_label, retrieve_hits
from router import decide_route, detect_query_context, score_stage1_candidates
from state import init_session_state
from synthesizer import synthesize_answer


def main() -> None:
    st.set_page_config(page_title="Global EV Knowledge Router", page_icon="🌍", layout="wide")
    init_session_state()

    if st.session_state.qdrant_local_path == "rag_tutorials/rag_database_routing/qdrant_local_data":
        st.session_state.qdrant_local_path = "qdrant_local_data"

    if st.session_state.collection_definitions is None:
        st.session_state.collection_definitions = load_collection_definitions()

    collections = st.session_state.collection_definitions
    _render_sidebar(collections)
    _render_header()

    if not _backend_ready():
        st.info("Enter your OpenAI and Qdrant credentials in the sidebar to activate ingestion and routing.")
        return

    _render_analytics_panel()
    _render_demo_loader(collections)
    _render_uploads(collections)
    _render_query_experience(collections)


def _render_sidebar(collections) -> None:
    with st.sidebar:
        st.title("Global EV Knowledge Router")
        st.caption("Enterprise-style automotive RAG router with metadata-aware ingestion and multi-collection answering.")

        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
        )
        st.session_state.qdrant_mode = st.radio(
            "Qdrant Mode",
            options=["local", "cloud"],
            horizontal=True,
            index=0 if st.session_state.qdrant_mode == "local" else 1,
        )

        if st.session_state.qdrant_mode == "local":
            st.session_state.qdrant_local_path = st.text_input(
                "Local Qdrant Path",
                value=st.session_state.qdrant_local_path or "qdrant_local_data",
                help="Local path used by embedded Qdrant. This is the easiest demo mode.",
            )
        else:
            st.session_state.qdrant_url = st.text_input(
                "Qdrant URL",
                value=st.session_state.qdrant_url,
                help="Example: https://your-cluster.qdrant.tech",
            )
            st.session_state.qdrant_api_key = st.text_input(
                "Qdrant API Key",
                value=st.session_state.qdrant_api_key,
                type="password",
            )

        if st.button("Connect / Refresh", use_container_width=True):
            _connect_backend(collections)

        if _backend_ready():
            st.success("Backend ready")
        else:
            st.warning("Backend not initialized")


def _connect_backend(collections) -> None:
    has_qdrant_config = (
        bool(st.session_state.qdrant_local_path or "qdrant_local_data")
        if st.session_state.qdrant_mode == "local"
        else bool(st.session_state.qdrant_url and st.session_state.qdrant_api_key)
    )

    if not (st.session_state.openai_api_key and has_qdrant_config):
        return

    try:
        embeddings, llm, router_llm, qdrant_client = initialize_backend(
            openai_api_key=st.session_state.openai_api_key,
            qdrant_mode=st.session_state.qdrant_mode,
            qdrant_url=st.session_state.qdrant_url,
            qdrant_api_key=st.session_state.qdrant_api_key,
            qdrant_local_path=st.session_state.qdrant_local_path,
            collections=collections,
        )
    except RuntimeError as exc:
        st.error(
            "Local Qdrant storage is locked by another running app instance. "
            "Close other Streamlit windows using this repo, or change `Local Qdrant Path` to a new folder name."
        )
        st.exception(exc)
        return

    st.session_state.embeddings = embeddings
    st.session_state.llm = llm
    st.session_state.router_llm = router_llm
    st.session_state.qdrant_client = qdrant_client
    st.session_state.collection_summary_embeddings = {
        collection_id: embeddings.embed_query(definition.summary)
        for collection_id, definition in collections.items()
    }


def _backend_ready() -> bool:
    return bool(
        st.session_state.embeddings
        and st.session_state.llm
        and st.session_state.router_llm
        and st.session_state.qdrant_client
    )


def _render_header() -> None:
    st.title("🌍 Global EV Knowledge Router")
    st.write(
        "Route EV support questions across manuals, charging guides, service FAQs, market policy, warranty terms, and specs — with evidence-backed synthesis."
    )
    st.caption("Tip: use local Qdrant mode for the easiest demo setup.")


def _render_analytics_panel() -> None:
    summary = summarize_events()
    metrics = st.columns(4)
    metrics[0].metric("Queries Logged", summary["total_queries"])
    metrics[1].metric("Fallback Rate", summary["fallback_rate"])
    metrics[2].metric("Top Collection", next(iter(summary["most_common_collections"].keys()), "n/a"))
    metrics[3].metric("Confidence Labels", len(summary["confidence_distribution"]))


def _render_demo_loader(collections) -> None:
    st.subheader("Demo Dataset")
    st.caption("Use the synthetic EV PDFs in `demo_content/` to seed a clean demo in one click.")
    if st.button("Load Demo Dataset", use_container_width=False):
        result = seed_demo_dataset(
            client=st.session_state.qdrant_client,
            embeddings=st.session_state.embeddings,
            collections=collections,
        )
        log_event("demo_seed", result)
        st.success(f"Loaded demo dataset with {result['total_chunks']} chunks.")
        st.write(result["documents"])


def _render_uploads(collections) -> None:
    st.subheader("Document Ingestion")
    st.caption("Upload PDFs into richly tagged collections so routing can use region, model, language, and document type metadata.")
    tabs = st.tabs([definition.display_name for definition in collections.values()])

    for (collection_id, definition), tab in zip(collections.items(), tabs):
        with tab:
            st.write(definition.description)
            market = st.selectbox(f"Market - {definition.display_name}", SUPPORTED_MARKETS, key=f"market_{collection_id}")
            vehicle_model = st.selectbox(f"Vehicle model - {definition.display_name}", SUPPORTED_VEHICLE_MODELS, key=f"vehicle_{collection_id}")
            language = st.selectbox(f"Language - {definition.display_name}", ["english", "chinese"], key=f"language_{collection_id}")
            document_type = st.selectbox(
                f"Document type - {definition.display_name}",
                options=definition.document_types + [item for item in SUPPORTED_DOCUMENT_TYPES if item not in definition.document_types],
                key=f"doc_type_{collection_id}",
            )
            model_year = st.text_input(f"Model year - {definition.display_name}", value="2025", key=f"year_{collection_id}")
            version = st.text_input(f"Version - {definition.display_name}", value="v1", key=f"version_{collection_id}")
            source_reliability = st.selectbox(
                f"Source reliability - {definition.display_name}",
                SOURCE_RELIABILITY_OPTIONS,
                key=f"reliability_{collection_id}",
            )

            files = st.file_uploader(
                f"Upload PDF files to {definition.display_name}",
                type="pdf",
                accept_multiple_files=True,
                key=f"upload_{collection_id}",
            )
            if files and st.button(f"Ingest into {definition.display_name}", key=f"ingest_button_{collection_id}"):
                total_chunks = 0
                for file in files:
                    chunks = process_document(
                        file,
                        metadata={
                            "collection_id": collection_id,
                            "market": market,
                            "vehicle_model": vehicle_model,
                            "language": language,
                            "document_type": document_type,
                            "model_year": model_year,
                            "version": version,
                            "source_reliability": source_reliability,
                        },
                    )
                    total_chunks += store_chunks(
                        client=st.session_state.qdrant_client,
                        embeddings=st.session_state.embeddings,
                        collection_name=definition.collection_name,
                        chunks=chunks,
                    )
                log_event("ingestion", {"collection_id": collection_id, "chunks_added": total_chunks})
                st.success(f"Added {total_chunks} chunks to {definition.display_name}.")


def _render_query_experience(collections) -> None:
    st.subheader("Ask a Question")
    question = st.text_area(
        "Enter an EV support or product question",
        placeholder="Compare charging guidance and warranty terms for Seal owners in Brazil.",
    )

    col1, col2, col3 = st.columns(3)
    selected_market = col1.selectbox("Preferred market", ["auto"] + SUPPORTED_MARKETS, index=0)
    selected_language = col2.selectbox("Response language", SUPPORTED_LANGUAGES, index=0)
    selected_model = col3.selectbox("Vehicle model hint", SUPPORTED_VEHICLE_MODELS, index=0)

    if question and st.button("Run Router", type="primary"):
        query_context = detect_query_context(question)
        if selected_market != "auto" and selected_market not in query_context.markets:
            query_context.markets.append(selected_market)
        if selected_language != "auto":
            query_context.language = selected_language
        if selected_model != "General" and selected_model not in query_context.vehicle_models:
            query_context.vehicle_models.append(selected_model)

        stage1_scores = score_stage1_candidates(
            question=question,
            query_context=query_context,
            embeddings=st.session_state.embeddings,
            collections=collections,
            collection_summary_embeddings=st.session_state.collection_summary_embeddings,
        )
        route_decision = decide_route(
            question=question,
            query_context=query_context,
            stage1_scores=stage1_scores,
            collections=collections,
            router_llm=st.session_state.router_llm,
        )
        planner = build_query_plan(question, query_context, route_decision, st.session_state.llm)
        hits = retrieve_hits(
            client=st.session_state.qdrant_client,
            embeddings=st.session_state.embeddings,
            collections=collections,
            route_decision=route_decision,
            query_context=query_context,
            planner_questions=planner.sub_questions,
        )
        label = confidence_label(route_decision, hits)

        if label == "low":
            answer = web_fallback(question, st.session_state.llm)
            used_fallback = True
        else:
            answer = synthesize_answer(
                question=question,
                route_decision=route_decision,
                query_context=query_context,
                planner=planner,
                hits=hits,
                collections=collections,
                llm=st.session_state.llm,
            )
            used_fallback = False

        log_event(
            "query",
            {
                "question": question,
                "primary_collection": route_decision.primary_collection,
                "secondary_collections": route_decision.secondary_collections,
                "confidence_label": label,
                "used_fallback": used_fallback,
            },
        )
        _render_query_result(route_decision, query_context, planner, hits, label, answer, used_fallback, collections)


def _render_query_result(route_decision, query_context, planner, hits, label, answer, used_fallback, collections) -> None:
    top = st.columns(4)
    top[0].metric("Primary Route", collections[route_decision.primary_collection].display_name)
    top[1].metric("Confidence", label)
    top[2].metric("Fallback Used", "yes" if used_fallback else "no")
    top[3].metric("Evidence Chunks", len(hits))

    with st.expander("Routing Details", expanded=True):
        st.write(
            {
                "route_reason": route_decision.reason,
                "stage1_scores": route_decision.stage1_scores,
                "language": query_context.language,
                "markets": query_context.markets,
                "vehicle_models": query_context.vehicle_models,
                "sub_questions": planner.sub_questions,
                "planner_rationale": planner.rationale,
            }
        )

    st.markdown("### Answer")
    if used_fallback:
        st.warning("This answer used external fallback because internal evidence confidence was low.")
    st.write(answer)

    st.markdown("### Evidence")
    if not hits:
        st.info("No internal evidence retrieved.")
        return
    for hit in hits[:6]:
        with st.container(border=True):
            st.write(
                f"**{collections[hit.collection_id].display_name}** | "
                f"{hit.metadata.get('source_file', 'unknown file')} | "
                f"page {hit.metadata.get('page_number', 'n/a')} | "
                f"market: {hit.metadata.get('market', 'n/a')} | score: {hit.score:.3f}"
            )
            st.caption(hit.content[:500])


if __name__ == "__main__":
    main()
