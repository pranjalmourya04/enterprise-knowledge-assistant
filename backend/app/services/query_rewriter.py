from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

# Common words that don't carry meaning - excluded from the overlap check
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "how", "why",
    "does", "do", "did", "of", "for", "in", "on", "to", "and", "or",
    "this", "that", "it", "its", "with", "about", "explain", "explanation",
}

MIN_OVERLAP_RATIO = 0.2  # if fewer than 20% of meaningful words survive, distrust the rewrite


def _meaningful_words(text: str) -> set:
    return {w for w in text.lower().split() if w not in _STOPWORDS and len(w) > 2}


def _rewrite_is_trustworthy(original: str, rewritten: str) -> bool:
   
    original_words = _meaningful_words(original)
    rewritten_words = _meaningful_words(rewritten)

    if not original_words:
        return True  # nothing meaningful to compare against, allow it

    overlap = original_words & rewritten_words
    overlap_ratio = len(overlap) / len(original_words)
    return overlap_ratio >= MIN_OVERLAP_RATIO

_client = Groq(api_key=GROQ_API_KEY)

REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into clear, formal search queries for "
    "retrieving relevant passages from internal company documents. "
    "Keep the rewritten query short (under 20 words), preserve the "
    "original meaning and intent exactly, and use formal/policy-style "
    "vocabulary where appropriate. "
    "If the question uses a vague pronoun (like 'it', 'this', 'that') "
    "with no clear subject stated in the question itself, do NOT guess "
    "or invent a generic replacement - keep the original wording as-is "
    "instead of abstracting it away. "
    "Return ONLY the rewritten query, nothing else - no preamble, no "
    "quotes, no explanation."
)


def rewrite_query(question: str) -> str:
  
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

        if not rewritten:
            return question

        if not _rewrite_is_trustworthy(question, rewritten):
            print(f"[query_rewriter] Rewrite '{rewritten}' failed overlap guardrail "
                  f"against original '{question}' - falling back to original.")
            return question

        return rewritten
    except Exception as e:
        print(f"[query_rewriter] Rewrite failed, falling back to original question: {e}")
        return question