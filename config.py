from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ANALYTICS_DIR = BASE_DIR / "analytics"
EVALS_DIR = BASE_DIR / "evals"
COLLECTIONS_CONFIG_PATH = DATA_DIR / "collections.json"
LOCAL_QDRANT_PATH = BASE_DIR / "qdrant_local_data"

CHAT_MODEL = "gpt-4o-mini"
ROUTER_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

DEFAULT_TOP_K = 5
DEFAULT_COLLECTION_CANDIDATES = 3
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180

SUPPORTED_LANGUAGES = ["auto", "english", "chinese"]
SUPPORTED_MARKETS = ["global", "brazil", "thailand", "hungary", "vietnam", "indonesia"]
SUPPORTED_VEHICLE_MODELS = [
    "General",
    "Seal",
    "Dolphin",
    "Atto 3",
    "Han",
    "Tang",
    "Seagull",
    "Yuan Up",
]
SUPPORTED_DOCUMENT_TYPES = [
    "owner_manual",
    "vehicle_specs",
    "service_faq",
    "charging_guide",
    "market_policy",
    "warranty_terms",
    "press_material",
]
SOURCE_RELIABILITY_OPTIONS = ["high", "medium", "low"]


@dataclass(frozen=True)
class CollectionDefinition:
    id: str
    display_name: str
    description: str
    collection_name: str
    priority: int
    document_types: list[str]
    allowed_regions: list[str]
    default_keywords: list[str]

    @property
    def summary(self) -> str:
        return (
            f"{self.display_name}: {self.description}. "
            f"Document types: {', '.join(self.document_types)}. "
            f"Regions: {', '.join(self.allowed_regions)}. "
            f"Keywords: {', '.join(self.default_keywords)}."
        )


def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    ANALYTICS_DIR.mkdir(exist_ok=True)
    EVALS_DIR.mkdir(exist_ok=True)


def load_collection_definitions() -> dict[str, CollectionDefinition]:
    ensure_runtime_dirs()
    raw = json.loads(COLLECTIONS_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        item["id"]: CollectionDefinition(
            id=item["id"],
            display_name=item["display_name"],
            description=item["description"],
            collection_name=item["collection_name"],
            priority=item["priority"],
            document_types=item["document_types"],
            allowed_regions=item["allowed_regions"],
            default_keywords=item["default_keywords"],
        )
        for item in raw
    }
