from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY)

REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into clear, formal search queries for "
    "retrieving relevant passages from internal company documents. "
    "Keep the rewritten query short (under 20 words), preserve the "
    "original meaning and intent exactly, and use formal/policy-style "
    "vocabulary where appropriate. Return ONLY the rewritten query, "
    "nothing else - no preamble, no quotes, no explanation."
)


def rewrite_query(question: str) -> str:
    """
    Returns a rewritten version of the question optimized for retrieval.
    Falls back to the original question if the LLM call fails.
    """
    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception as e:
        print(f"[query_rewriter] Rewrite failed, falling back to original question: {e}")
        return question