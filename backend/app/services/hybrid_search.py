from rank_bm25 import BM25Okapi
from typing import List

from app.services.vector_store import query_similar_chunks, get_all_chunks

RRF_K = 60  # standard smoothing constant from the RRF paper


def _tokenize(text: str) -> List[str]:
    """Simple lowercase whitespace tokenizer - sufficient for BM25 here."""
    return text.lower().split()


def _bm25_search(question: str, top_n: int, allowed_sensitivities: List[str] = None) -> List[str]:
    
    all_chunks = get_all_chunks(allowed_sensitivities)
    ids = all_chunks["ids"]
    documents = all_chunks["documents"]

    if not ids:
        return []

    tokenized_corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = _tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked[:top_n]]


def _vector_search(question_embedding: List[float], top_n: int,
                    allowed_sensitivities: List[str] = None) -> List[str]:
    
    results = query_similar_chunks(question_embedding, top_n, allowed_sensitivities)
    if not results["ids"] or not results["ids"][0]:
        return []
    return results["ids"][0]


def _reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = RRF_K) -> List[str]:
   
    fused_scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    fused_sorted = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in fused_sorted]


def hybrid_retrieve(question: str, question_embedding: List[float],
                     candidate_pool_size: int = 20,
                     allowed_sensitivities: List[str] = None) -> dict:
    """
    Run vector search + BM25 search, fuse via RRF, and return the fused
    candidate pool (chunk_ids in order) along with a lookup of chunk data
    (text + metadata) keyed by chunk_id.

    allowed_sensitivities (Day 17-18): if given, RBAC filtering is applied
    to BOTH underlying searches before fusion - a chunk outside the
    caller's clearance can never enter the candidate pool at all, let
    alone reach the LLM.
    """
    vector_ranked_ids = _vector_search(question_embedding, candidate_pool_size, allowed_sensitivities)
    bm25_ranked_ids = _bm25_search(question, candidate_pool_size, allowed_sensitivities)

    fused_ids = _reciprocal_rank_fusion([vector_ranked_ids, bm25_ranked_ids])
    fused_ids = fused_ids[:candidate_pool_size]

    all_chunks = get_all_chunks(allowed_sensitivities)
    chunk_lookup = {}
    for i, chunk_id in enumerate(all_chunks["ids"]):
        if chunk_id in fused_ids:
            chunk_lookup[chunk_id] = {
                "text": all_chunks["documents"][i],
                "metadata": all_chunks["metadatas"][i],
            }

    return {"ranked_ids": fused_ids, "chunk_lookup": chunk_lookup}

    