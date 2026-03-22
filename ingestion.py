from __future__ import annotations

import tempfile
import uuid
from datetime import date
from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def process_document(file, metadata: dict) -> list[dict]:
    return process_pdf_bytes(file.name, file.getvalue(), metadata)


def process_pdf_bytes(file_name: str, file_bytes: bytes, metadata: dict) -> list[dict]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    normalized_chunks = []
    for index, chunk in enumerate(chunks):
        page_number = int(chunk.metadata.get("page", 0)) + 1
        payload = {
            "content": chunk.page_content,
            "source_file": file_name,
            "page_number": page_number,
            "chunk_index": index,
            "ingested_at": date.today().isoformat(),
            **metadata,
        }
        normalized_chunks.append(payload)

    return normalized_chunks


def store_chunks(
    client: QdrantClient,
    embeddings,
    collection_name: str,
    chunks: Iterable[dict],
) -> int:
    chunk_list = list(chunks)
    if not chunk_list:
        return 0

    vectors = embeddings.embed_documents([chunk["content"] for chunk in chunk_list])
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        for payload, vector in zip(chunk_list, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)
