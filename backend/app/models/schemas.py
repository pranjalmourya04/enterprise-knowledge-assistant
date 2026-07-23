from pydantic import BaseModel
from typing import List


class ChunkPreview(BaseModel):
    chunk_id: str
    page_number: int
    chunk_index_on_page: int
    text_preview: str  # truncated text, just for sanity-checking the response


class UploadResponse(BaseModel):
    filename: str
    pages_extracted: int
    chunks_created: int
    chunks_stored_in_vector_db: int
    total_chunks_in_collection: int
    sample_chunks: List[ChunkPreview]


class HealthResponse(BaseModel):
    status: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    filename: str
    page_number: int
    similarity_score: float


class QueryResponse(BaseModel):
    question: str
    retrieved_chunks: List[RetrievedChunk]

class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceUsed(BaseModel):
    filename: str
    page_number: int
    relevance_score: float  # cross-encoder score - relative ranking, not 0-1 bounded

class AskResponse(BaseModel):
    question: str
    rewritten_query: str  
    answer: str
    sources: List[SourceUsed]