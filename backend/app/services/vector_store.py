"""
Day 3-4: ChromaDB vector store wrapper.
Handles storing chunks (with embeddings + metadata) and, later,
querying them. Metadata stored per-chunk now (filename, page number)
sets up Week 3's RBAC filtering - we'll add role/sensitivity tags
to this same metadata dict then, without changing this file's shape.
"""
import chromadb
from typing import List

from app.config import CHROMA_DIR, CHROMA_COLLECTION_NAME
from app.services.document_processor import Chunk
from app.services.embedding_service import embed_texts

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def add_chunks(chunks: List[Chunk]) -> int:
    """
    Embed and store a list of Chunk objects in ChromaDB.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "filename": c.filename,
            "page_number": c.page_number,
            "chunk_index_on_page": c.chunk_index_on_page,
        }
        for c in chunks
    ]

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def get_collection_count() -> int:
    """Total number of chunks currently stored - useful for sanity checks."""
    return _collection.count()