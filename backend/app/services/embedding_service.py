"""
Day 3-4: Embedding service.
Wraps Sentence Transformers so the rest of the app doesn't touch the
model directly - keeps it swappable later without touching callers.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import functools

from app.config import EMBEDDING_MODEL_NAME


@functools.lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Loaded once and cached (model load is slow - a few seconds - so we
    don't want to reload it on every request).
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts, returns list of embedding vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()