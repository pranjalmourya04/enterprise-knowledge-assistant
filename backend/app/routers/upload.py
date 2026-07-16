"""
Day 1-2: PDF upload endpoint.

Flow: receive PDF -> save to disk -> extract text -> chunk it.
(Embedding + storing in ChromaDB added Day 3-4.)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from app.config import UPLOADS_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from app.services.document_processor import chunk_document
from app.services.vector_store import add_chunks, get_collection_count
from app.models.schemas import UploadResponse, ChunkPreview

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    # --- Validate extension ---
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # --- Read and validate size ---
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max is {MAX_UPLOAD_SIZE_MB}MB.",
        )

    # --- Save to disk ---
    save_path = UPLOADS_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(contents)

    # --- Extract + chunk ---
    try:
        chunks = chunk_document(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF. "
                   "It may be a scanned/image-only document (OCR not yet supported).",
        )

    pages_extracted = len({c.page_number for c in chunks})

    # --- Embed + store in ChromaDB ---
    try:
        stored_count = add_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed/store chunks: {e}")

    return UploadResponse(
        filename=file.filename,
        pages_extracted=pages_extracted,
        chunks_created=len(chunks),
        chunks_stored_in_vector_db=stored_count,
        total_chunks_in_collection=get_collection_count(),
        sample_chunks=[
            ChunkPreview(
                chunk_id=c.chunk_id,
                page_number=c.page_number,
                chunk_index_on_page=c.chunk_index_on_page,
                text_preview=(c.text[:200] + "...") if len(c.text) > 200 else c.text,
            )
            for c in chunks[:3]  # just show first 3 chunks as a sanity check
        ],
    )