from __future__ import annotations

import json
import math
import re
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage

from config import CollectionDefinition, SUPPORTED_MARKETS, SUPPORTED_VEHICLE_MODELS
from models import QueryContext, RouteDecision


MARKET_ALIASES = {
    "brazil": ["brazil", "brazilian"],
    "thailand": ["thailand", "thai"],
    "hungary": ["hungary", "hungarian"],
    "vietnam": ["vietnam", "vietnamese"],
    "indonesia": ["indonesia", "indonesian"],
}


def detect_query_context(question: str) -> QueryContext:
    lowered = question.lower()
    language = "chinese" if re.search(r"[\u4e00-\u9fff]", question) else "english"
    markets = [market for market, aliases in MARKET_ALIASES.items() if any(alias in lowered for alias in aliases)]
    vehicle_models = [
        model for model in SUPPORTED_VEHICLE_MODELS if model != "General" and model.lower() in lowered
    ]
    complexity = "complex" if any(token in lowered for token in ["compare", "versus", "difference", "and", "plus"]) else "simple"
    requested_doc_types = []
    if any(token in lowered for token in ["charge", "charging", "charger"]):
        requested_doc_types.append("charging_guide")
    if any(token in lowered for token in ["warranty", "coverage"]):
        requested_doc_types.append("warranty_terms")
    if any(token in lowered for token in ["manual", "dashboard", "warning", "how to"]):
        requested_doc_types.append("owner_manual")

    return QueryContext(
        language=language,
        markets=markets,
        vehicle_models=vehicle_models,
        complexity=complexity,
        requested_doc_types=requested_doc_types,
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def score_stage1_candidates(
    question: str,
    query_context: QueryContext,
    embeddings,
    collections: Dict[str, CollectionDefinition],
    collection_summary_embeddings: Dict[str, list[float]],
) -> dict[str, float]:
    query_vector = embeddings.embed_query(question)
    lowered = question.lower()
    scores: dict[str, float] = {}

    for collection_id, definition in collections.items():
        semantic_score = _cosine_similarity(query_vector, collection_summary_embeddings[collection_id])
        keyword_hits = sum(1 for keyword in definition.default_keywords if keyword.lower() in lowered)
        keyword_score = min(keyword_hits * 0.08, 0.24)
        market_boost = 0.08 if any(m in definition.allowed_regions for m in query_context.markets) else 0.0
        doc_type_boost = 0.12 if any(doc in definition.document_types for doc in query_context.requested_doc_types) else 0.0
        priority_boost = definition.priority * 0.01
        scores[collection_id] = semantic_score + keyword_score + market_boost + doc_type_boost + priority_boost

    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def decide_route(
    question: str,
    query_context: QueryContext,
    stage1_scores: dict[str, float],
    collections: Dict[str, CollectionDefinition],
    router_llm,
) -> RouteDecision:
    top_candidates = list(stage1_scores.items())[:3]
    candidate_descriptions = [
        {
            "id": collection_id,
            "display_name": collections[collection_id].display_name,
            "description": collections[collection_id].description,
            "score": round(score, 4),
        }
        for collection_id, score in top_candidates
    ]

    system_prompt = (
        "You are a routing planner for a global EV knowledge platform. "
        "Choose the most suitable primary collection and optional secondary collections. "
        "Return valid JSON with keys: primary_collection, secondary_collections, reason, confidence, needs_multi_collection."
    )
    human_prompt = {
        "question": question,
        "language": query_context.language,
        "markets": query_context.markets,
        "vehicle_models": query_context.vehicle_models,
        "complexity": query_context.complexity,
        "candidates": candidate_descriptions,
    }

    try:
        response = router_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(human_prompt, ensure_ascii=False))]
        )
        payload = _extract_json_object(response.content)
        primary = payload.get("primary_collection") or top_candidates[0][0]
        secondary = [item for item in payload.get("secondary_collections", []) if item in collections and item != primary]
        confidence = float(payload.get("confidence", top_candidates[0][1]))
        needs_multi = bool(payload.get("needs_multi_collection", query_context.complexity == "complex"))
        reason = payload.get("reason", "Selected using stage 1 candidate scores and route synthesis.")
    except Exception:
        primary = top_candidates[0][0]
        secondary = [collection_id for collection_id, _ in top_candidates[1:2]] if query_context.complexity == "complex" else []
        confidence = top_candidates[0][1]
        needs_multi = query_context.complexity == "complex"
        reason = "Fell back to heuristic routing because structured router output was unavailable."

    return RouteDecision(
        primary_collection=primary,
        secondary_collections=secondary,
        reason=reason,
        confidence=round(min(max(confidence, 0.0), 1.0), 3),
        needs_multi_collection=needs_multi,
        stage1_scores=stage1_scores,
    )


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in router response")
    return json.loads(text[start : end + 1])
