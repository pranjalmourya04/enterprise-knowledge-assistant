from fastapi import APIRouter, HTTPException

from app.config import DEFAULT_TOP_K
from app.services.embedding_service import embed_texts
from app.services.vector_store import query_similar_chunks
from app.services.llm_service import generate_answer
from app.models.schemas import AskRequest, AskResponse, SourceUsed

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
async def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    top_k = request.top_k if request.top_k > 0 else DEFAULT_TOP_K

    question_embedding = embed_texts([request.question])[0]
    results = query_similar_chunks(question_embedding, top_k)

    if not results["ids"] or not results["ids"][0]:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Upload a document first.",
        )

    chunk_texts = results["documents"][0]
    sources = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = round(1 - distance, 4)
        metadata = results["metadatas"][0][i]
        sources.append(
            SourceUsed(
                filename=metadata["filename"],
                page_number=metadata["page_number"],
                similarity_score=similarity,
            )
        )

    try:
        answer = generate_answer(request.question, chunk_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return AskResponse(question=request.question, answer=answer, sources=sources)