from fastapi import APIRouter, HTTPException
from app.services.hybrid_search import hybrid_retrieve
from app.services.reranker import rerank
from app.services.query_rewriter import rewrite_query
from app.config import CANDIDATE_POOL_SIZE, FINAL_TOP_K
from app.services.embedding_service import embed_texts
from app.services.llm_service import generate_answer
from app.models.schemas import AskRequest, AskResponse, SourceUsed

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
async def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")


    retrieval_query = rewrite_query(request.question)

    question_embedding = embed_texts([retrieval_query])[0]

    hybrid_result = hybrid_retrieve(retrieval_query, question_embedding, CANDIDATE_POOL_SIZE)
    ranked_ids = hybrid_result["ranked_ids"]
    chunk_lookup = hybrid_result["chunk_lookup"]

    if not ranked_ids:
        raise HTTPException(status_code=404, detail="No documents found. Upload a document first.")

    candidates = [
        {
            "chunk_id": chunk_id,
            "text": chunk_lookup[chunk_id]["text"],
            "metadata": chunk_lookup[chunk_id]["metadata"],
        }
        for chunk_id in ranked_ids
    ]

    reranked = rerank(retrieval_query, candidates)
    top_chunks = reranked[:FINAL_TOP_K]

    chunk_texts = [c["text"] for c in top_chunks]
    sources = [
        SourceUsed(
            filename=c["metadata"]["filename"],
            page_number=c["metadata"]["page_number"],
            relevance_score=round(c["rerank_score"], 4),
        )
        for c in top_chunks
    ]

    try:
        answer = generate_answer(request.question, chunk_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return AskResponse(
        question=request.question,
        rewritten_query=retrieval_query,
        answer=answer,
        sources=sources,
    )