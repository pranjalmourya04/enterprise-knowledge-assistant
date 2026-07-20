from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are an internal enterprise knowledge assistant. Answer the user's "
    "question using ONLY the provided context chunks below. "
    "If the context does not contain enough information to answer the "
    "question, say clearly that you don't have enough information in the "
    "provided documents - do not make up an answer. "
    "Keep answers concise and factual."
)


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Given a question and a list of retrieved context chunk texts,
    call the LLM and return the generated answer text.
    """
    context_block = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the context above:"
    )

    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature - we want factual, not creative
        max_tokens=800,
    )

    return response.choices[0].message.content