from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryContext:
    language: str = "english"
    markets: list[str] = field(default_factory=list)
    vehicle_models: list[str] = field(default_factory=list)
    complexity: str = "simple"
    requested_doc_types: list[str] = field(default_factory=list)


@dataclass
class RouteDecision:
    primary_collection: str
    secondary_collections: list[str]
    reason: str
    confidence: float
    needs_multi_collection: bool
    stage1_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    collection_id: str
    score: float
    content: str
    metadata: dict[str, Any]


@dataclass
class QueryPlan:
    sub_questions: list[str]
    rationale: str


@dataclass
class QueryResult:
    answer: str
    confidence_label: str
    used_fallback: bool
    route_decision: RouteDecision
    query_context: QueryContext
    hits: list[RetrievalHit]
    planner: QueryPlan
