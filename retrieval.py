from __future__ import annotations

from typing import Dict

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from config import CollectionDefinition, DEFAULT_TOP_K
from models import QueryContext, RetrievalHit, RouteDecision


def retrieve_hits(
    client: QdrantClient,
    embeddings,
    collections: Dict[str, CollectionDefinition],
    route_decision: RouteDecision,
    query_context: QueryContext,
    planner_questions: list[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalHit]:
    collection_ids = [route_decision.primary_collection]
    if route_decision.needs_multi_collection:
        collection_ids.extend(route_decision.secondary_collections)

    filtered_hits = _search_collections(
        client=client,
        embeddings=embeddings,
        collections=collections,
        collection_ids=collection_ids,
        query_context=query_context,
        planner_questions=planner_questions,
        top_k=top_k,
        use_filter=True,
    )
    if filtered_hits:
        return filtered_hits

    expanded_collection_ids = list(dict.fromkeys(collection_ids + list(route_decision.stage1_scores.keys())[:4]))
    expanded_hits = _search_collections(
        client=client,
        embeddings=embeddings,
        collections=collections,
        collection_ids=expanded_collection_ids,
        query_context=query_context,
        planner_questions=planner_questions,
        top_k=top_k,
        use_filter=True,
    )
    if expanded_hits:
        return expanded_hits

    return _search_collections(
        client=client,
        embeddings=embeddings,
        collections=collections,
        collection_ids=list(collections.keys()),
        query_context=query_context,
        planner_questions=planner_questions,
        top_k=top_k,
        use_filter=False,
    )


def _search_collections(
    client: QdrantClient,
    embeddings,
    collections: Dict[str, CollectionDefinition],
    collection_ids: list[str],
    query_context: QueryContext,
    planner_questions: list[str],
    top_k: int,
    use_filter: bool,
) -> list[RetrievalHit]:
    unique_hits: dict[str, RetrievalHit] = {}
    query_filter = _build_filter(query_context) if use_filter else None

    for sub_question in planner_questions:
        query_vector = embeddings.embed_query(sub_question)

        for collection_id in collection_ids:
            definition = collections[collection_id]
            search_results = client.search(
                collection_name=definition.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            for point in search_results:
                payload = point.payload or {}
                hit = RetrievalHit(
                    collection_id=collection_id,
                    score=_adjust_score(float(point.score), payload, query_context, definition.priority),
                    content=payload.get("content", ""),
                    metadata=payload,
                )
                dedupe_key = f"{payload.get('source_file')}::{payload.get('page_number')}::{payload.get('chunk_index')}"
                if dedupe_key not in unique_hits or unique_hits[dedupe_key].score < hit.score:
                    unique_hits[dedupe_key] = hit

    return sorted(unique_hits.values(), key=lambda hit: hit.score, reverse=True)[:8]


def _build_filter(query_context: QueryContext) -> Filter | None:
    must_conditions = []
    if query_context.markets:
        must_conditions.append(FieldCondition(key="market", match=MatchAny(any=query_context.markets + ["global"])))
    if query_context.vehicle_models:
        must_conditions.append(FieldCondition(key="vehicle_model", match=MatchAny(any=query_context.vehicle_models)))
    return Filter(must=must_conditions) if must_conditions else None


def _adjust_score(base_score: float, payload: dict, query_context: QueryContext, priority: int) -> float:
    score = base_score
    if payload.get("market") in query_context.markets:
        score += 0.08
    if payload.get("vehicle_model") in query_context.vehicle_models:
        score += 0.08
    if payload.get("language") == query_context.language:
        score += 0.05
    if payload.get("source_reliability") == "high":
        score += 0.05
    score += priority * 0.01
    return score


def confidence_label(route_decision: RouteDecision, hits: list[RetrievalHit]) -> str:
    if not hits:
        return "low"
    avg_hit_score = sum(hit.score for hit in hits[:3]) / min(len(hits), 3)
    top_hit_score = hits[0].score
    blended = (route_decision.confidence * 0.3) + (avg_hit_score * 0.7)
    if blended >= 0.82:
        return "high"
    if blended >= 0.58 or top_hit_score >= 0.75:
        return "medium"
    return "low"
