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
    sample_chunks: List[ChunkPreview]


class HealthResponse(BaseModel):
    status: str
