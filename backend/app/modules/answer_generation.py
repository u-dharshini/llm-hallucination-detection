"""
Answer Generation — Stage 1 of the pipeline.

Sends the user's question to an LLM (OpenAI or Groq, chosen via .env)
and returns the raw text answer. This stands in for "any real-world
chatbot interaction" per the project slides — it's the answer whose
claims we go on to verify.

Owner: P1
Location in repo: backend/app/modules/answer_generation.py
"""

from ..config import settings

# Both SDKs are imported lazily inside the functions below, so that a
# teammate who only has one of the two API keys set up doesn't get an
# import error for the other provider's package.


def _generate_with_openai(question: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=500,
        temperature=0.3,  # lower temperature = more factual/deterministic, less creative
    )
    return response.choices[0].message.content.strip()


def _generate_with_groq(question: str) -> str:
    from groq import Groq

    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "user", "content": question}
        ],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def generate_answer(question: str) -> str:
    """
    Main entry point. Sends `question` to whichever LLM provider is
    configured in .env (LLM_PROVIDER=openai or groq) and returns the
    answer text.

    Raises a RuntimeError with a clear message if the API call fails,
    rather than letting a raw SDK exception bubble up — this keeps the
    error readable for whoever is running the pipeline (and for tests).
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    provider = settings.LLM_PROVIDER.lower()

    try:
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )
            return _generate_with_openai(question)

        elif provider == "groq":
            if not settings.GROQ_API_KEY:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to your .env file."
                )
            return _generate_with_groq(question)

        else:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER '{provider}'. Use 'openai' or 'groq' in .env."
            )

    except Exception as e:
        # Wrap any SDK-specific error (timeout, rate limit, auth failure,
        # network issue) into one consistent, readable error for the
        # rest of the pipeline to handle.
        raise RuntimeError(f"Answer generation failed: {e}") from e
