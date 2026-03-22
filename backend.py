from __future__ import annotations

from typing import Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from config import CHAT_MODEL, EMBEDDING_MODEL, LOCAL_QDRANT_PATH, ROUTER_MODEL, VECTOR_SIZE, CollectionDefinition


def initialize_backend(
    openai_api_key: str,
    qdrant_mode: str,
    qdrant_url: str,
    qdrant_api_key: str,
    qdrant_local_path: str,
    collections: Dict[str, CollectionDefinition],
):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=openai_api_key)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=openai_api_key)
    router_llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0, api_key=openai_api_key)

    if qdrant_mode == "local":
        local_path = qdrant_local_path or str(LOCAL_QDRANT_PATH)
        client = QdrantClient(path=local_path)
    else:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    client.get_collections()

    for definition in collections.values():
        try:
            client.get_collection(definition.collection_name)
        except (UnexpectedResponse, ValueError):
            client.create_collection(
                collection_name=definition.collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    return embeddings, llm, router_llm, client
