from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from config import CollectionDefinition
from models import QueryContext, QueryPlan, RetrievalHit, RouteDecision


def synthesize_answer(
    question: str,
    route_decision: RouteDecision,
    query_context: QueryContext,
    planner: QueryPlan,
    hits: list[RetrievalHit],
    collections: dict[str, CollectionDefinition],
    llm,
) -> str:
    evidence = []
    for index, hit in enumerate(hits[:8], start=1):
        evidence.append(
            {
                "rank": index,
                "collection": collections[hit.collection_id].display_name,
                "source_file": hit.metadata.get("source_file"),
                "market": hit.metadata.get("market"),
                "vehicle_model": hit.metadata.get("vehicle_model"),
                "page_number": hit.metadata.get("page_number"),
                "score": round(hit.score, 3),
                "content": hit.content[:1200],
            }
        )

    system_prompt = (
        "You are a global EV knowledge assistant. Use only the provided evidence. "
        "Answer in the user's language. Structure the answer with these headings: Short Answer, Evidence by Source, Regional Caveats, Uncertainty. "
        "If evidence is incomplete, say so clearly."
    )
    human_payload = {
        "question": question,
        "language": query_context.language,
        "markets": query_context.markets,
        "vehicle_models": query_context.vehicle_models,
        "route_reason": route_decision.reason,
        "query_plan": planner.sub_questions,
        "evidence": evidence,
    }
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=str(human_payload))]
    )
    return response.content
