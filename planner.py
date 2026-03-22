from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from models import QueryContext, QueryPlan, RouteDecision


def build_query_plan(question: str, query_context: QueryContext, route_decision: RouteDecision, llm) -> QueryPlan:
    if query_context.complexity != "complex":
        return QueryPlan(sub_questions=[question], rationale="Simple question; no decomposition needed.")

    system_prompt = (
        "You decompose complex EV support questions into concise retrieval-ready sub-questions. "
        "Return JSON with keys: sub_questions and rationale. Keep 2 to 4 sub-questions maximum."
    )
    human_payload = {
        "question": question,
        "route_primary": route_decision.primary_collection,
        "route_secondary": route_decision.secondary_collections,
        "markets": query_context.markets,
        "vehicle_models": query_context.vehicle_models,
    }

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(human_payload, ensure_ascii=False))]
        )
        payload = _extract_json_object(response.content)
        sub_questions = payload.get("sub_questions") or [question]
        rationale = payload.get("rationale", "Generated sub-questions for multi-collection retrieval.")
        return QueryPlan(sub_questions=sub_questions[:4], rationale=rationale)
    except Exception:
        return QueryPlan(
            sub_questions=[question],
            rationale="Planner fell back to the original question because structured decomposition was unavailable.",
        )


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])
