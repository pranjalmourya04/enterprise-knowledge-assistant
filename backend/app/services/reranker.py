from sentence_transformers import CrossEncoder
from typing import List, Dict
import functools

from app.config import RERANKER_MODEL_NAME


@functools.lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    """Loaded once and cached, same reasoning as the embedding model."""
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank(question: str, candidates: List[Dict]) -> List[Dict]:
    """
    candidates: list of dicts, each with at least a "text" key (plus
    whatever metadata/chunk_id the caller wants carried through).
    Returns the same dicts, sorted best-first, each with an added
    "rerank_score" key.
    """
    if not candidates:
        return []

    model = get_reranker_model()
    pairs = [(question, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)