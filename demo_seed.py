from __future__ import annotations

from pathlib import Path

from config import BASE_DIR
from ingestion import process_pdf_bytes, store_chunks


DEMO_MANIFEST = [
    {
        "path": "demo_content/vehicle_specs/seal_global_specs_demo.pdf",
        "collection_id": "vehicle_specs",
        "market": "global",
        "vehicle_model": "Seal",
        "language": "english",
        "document_type": "vehicle_specs",
        "model_year": "2025",
        "version": "demo-v1",
        "source_reliability": "high",
    },
    {
        "path": "demo_content/charging_guides/dolphin_thailand_home_charging_demo.pdf",
        "collection_id": "charging_guides",
        "market": "thailand",
        "vehicle_model": "Dolphin",
        "language": "english",
        "document_type": "charging_guide",
        "model_year": "2025",
        "version": "demo-v1",
        "source_reliability": "high",
    },
    {
        "path": "demo_content/warranty_terms/seal_brazil_warranty_demo.pdf",
        "collection_id": "warranty_terms",
        "market": "brazil",
        "vehicle_model": "Seal",
        "language": "english",
        "document_type": "warranty_terms",
        "model_year": "2025",
        "version": "demo-v1",
        "source_reliability": "high",
    },
    {
        "path": "demo_content/owner_manuals/atto3_dashboard_warnings_demo.pdf",
        "collection_id": "owner_manuals",
        "market": "global",
        "vehicle_model": "Atto 3",
        "language": "english",
        "document_type": "owner_manual",
        "model_year": "2025",
        "version": "demo-v1",
        "source_reliability": "high",
    },
    {
        "path": "demo_content/service_faqs/global_service_troubleshooting_demo.pdf",
        "collection_id": "service_faqs",
        "market": "global",
        "vehicle_model": "General",
        "language": "english",
        "document_type": "service_faq",
        "model_year": "2025",
        "version": "demo-v1",
        "source_reliability": "high",
    },
    {
        "path": "demo_content/market_policy/hungary_home_charger_installation_demo.pdf",
        "collection_id": "market_policy",
        "market": "hungary",
        "vehicle_model": "General",
        "language": "english",
        "document_type": "market_policy",
        "model_year": "2025",
        "version": "demo-v1",
        "source_reliability": "high",
    },
]


def seed_demo_dataset(client, embeddings, collections: dict[str, object]) -> dict:
    total_chunks = 0
    ingested_documents = []

    for item in DEMO_MANIFEST:
        file_path = BASE_DIR / item["path"]
        if not file_path.exists():
            continue
        chunks = process_pdf_bytes(
            file_name=file_path.name,
            file_bytes=file_path.read_bytes(),
            metadata={key: value for key, value in item.items() if key not in {"path"}},
        )
        definition = collections[item["collection_id"]]
        chunk_count = store_chunks(
            client=client,
            embeddings=embeddings,
            collection_name=definition.collection_name,
            chunks=chunks,
        )
        total_chunks += chunk_count
        ingested_documents.append({"file": file_path.name, "collection": item["collection_id"], "chunks": chunk_count})

    return {"documents": ingested_documents, "total_chunks": total_chunks}
