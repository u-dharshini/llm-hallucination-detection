"""
Claim Extraction — Stage 2 of the pipeline.

Splits a raw LLM answer into individual atomic, checkable factual
statements. Example: "The Eiffel Tower was built in 1887 by Gustave
Eiffel" becomes two separate Claim objects — one about the build year,
one about who built it — so each can be independently verified in
Stage 3.

Strategy:
  1. Primary: ask an LLM to do the splitting (it understands grammar,
     coreference, and conjunctions much better than rule-based tools).
  2. Fallback: if the LLM call fails for any reason (no API key, network
     issue, bad JSON response), fall back to a simple sentence-splitter
     so the pipeline never completely breaks.

Owner: P2
Location in repo: backend/app/modules/claim_extraction.py
"""

import json
import re
from typing import List

from ..config import settings
from ..models.schemas import Claim


_EXTRACTION_PROMPT = """You are a fact-checking assistant. Break the following \
answer into a list of short, independent, atomic factual claims. Each claim \
should state exactly one fact and be understandable on its own (resolve \
pronouns like "he"/"it" into the actual subject).

Respond with ONLY a JSON array of strings, nothing else. No markdown, no \
explanation, no code fences.

Example:
Answer: "The Eiffel Tower was built in 1887 by Gustave Eiffel."
Response: ["The Eiffel Tower was built in 1887.", "Gustave Eiffel built the Eiffel Tower."]

Answer: "{answer}"
Response:"""


def _call_llm_for_extraction(answer: str) -> str:
    """Sends the extraction prompt to whichever provider is configured and
    returns the raw text response."""
    prompt = _EXTRACTION_PROMPT.format(answer=answer)
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,  # deterministic — we want consistent splitting, not creativity
        )
        return response.choices[0].message.content.strip()

    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'")


def _parse_claims_json(raw_text: str) -> List[str]:
    """
    Parses the LLM's JSON array response into a list of claim strings.
    Handles the common case where the model wraps the JSON in markdown
    code fences despite being told not to.
    """
    cleaned = raw_text.strip()
    # Strip ```json ... ``` or ``` ... ``` wrapping if the model added it anyway
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array of claim strings")

    return [str(item).strip() for item in parsed if str(item).strip()]


def _fallback_sentence_split(answer: str) -> List[str]:
    """
    Naive fallback: split on sentence-ending punctuation. Used only if
    the LLM-based extraction above fails for any reason, so the pipeline
    degrades gracefully instead of crashing.
    """
    # Split on '.', '!', '?' followed by a space or end of string
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_claims(answer: str) -> List[Claim]:
    """
    Main entry point. Takes a raw LLM answer and returns a list of
    Claim objects, each with a unique claim_id and claim_text set.
    Verdict/confidence/source/evidence fields are left empty here —
    they get filled in later by the verification stage (M3).
    """
    if not answer or not answer.strip():
        return []

    try:
        raw_response = _call_llm_for_extraction(answer)
        claim_texts = _parse_claims_json(raw_response)
        if not claim_texts:
            raise ValueError("LLM returned an empty claims list")
    except Exception:
        # LLM extraction failed (no API key, bad JSON, network issue, etc.)
        # — fall back to simple sentence splitting rather than crash.
        claim_texts = _fallback_sentence_split(answer)

    return [
        Claim(claim_id=f"c{i+1}", claim_text=text)
        for i, text in enumerate(claim_texts)
    ]
