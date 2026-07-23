"""
Day 3-4: ChromaDB vector store wrapper.
Handles storing chunks (with embeddings + metadata) and, later,
querying them. Metadata stored per-chunk now (filename, page number)
sets up Week 3's RBAC filtering - we'll add role/sensitivity tags
to this same metadata dict then, without changing this file's shape.
"""
import chromadb
from typing import List
from app.config import CHROMA_DIR, CHROMA_COLLECTION_NAME, CHROMA_SIMILARITY_SPACE
from app.services.document_processor import Chunk
from app.services.embedding_service import embed_texts

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": CHROMA_SIMILARITY_SPACE},
)

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
            "department": c.department,
            "sensitivity": c.sensitivity,
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

def query_similar_chunks(question_embedding: List[float], top_k: int,
                          allowed_sensitivities: List[str] = None) -> dict:
    
    where_filter = {"sensitivity": {"$in": allowed_sensitivities}} if allowed_sensitivities else None
    return _collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where=where_filter,
    )

def get_all_chunks(allowed_sensitivities: List[str] = None) -> dict:
    """
    Fetch every chunk currently stored (ids, documents, metadatas),
    optionally RBAC-filtered the same way as query_similar_chunks.
    Used to build the BM25 keyword index for hybrid search (Week 2) -
    filtering here too ensures BM25 can never surface a chunk the user
    isn't cleared to see, matching the vector search path.

    Design note: BM25 index is rebuilt from this fresh on every query
    rather than persisted separately. Fine for a project-scale document
    set; at production scale you'd maintain a persisted, incrementally
    updated keyword index (e.g. Elasticsearch/Whoosh) instead of rebuilding
    it per request.
    """
    where_filter = {"sensitivity": {"$in": allowed_sensitivities}} if allowed_sensitivities else None
    return _collection.get(include=["documents", "metadatas"], where=where_filter)