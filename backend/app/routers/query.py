
from fastapi import APIRouter, HTTPException

from app.config import DEFAULT_TOP_K
from app.services.embedding_service import embed_texts
from app.services.vector_store import query_similar_chunks
from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    top_k = request.top_k if request.top_k > 0 else DEFAULT_TOP_K

    # Embed the question (embed_texts expects a list, we pass a single-item list)
    question_embedding = embed_texts([request.question])[0]

    results = query_similar_chunks(question_embedding, top_k)

    retrieved_chunks = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance  # cosine space: similarity = 1 - distance
            metadata = results["metadatas"][0][i]
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    filename=metadata["filename"],
                    page_number=metadata["page_number"],
                    similarity_score=round(similarity, 4),
                )
            )

    return QueryResponse(question=request.question, retrieved_chunks=retrieved_chunks)