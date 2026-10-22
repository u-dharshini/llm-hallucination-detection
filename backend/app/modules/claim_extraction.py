"""
Claim Extraction — Stage 2 of the pipeline.

Splits a raw LLM answer into individual atomic, checkable factual
statements. Example: "The Eiffel Tower was built in 1887 by Gustave
Eiffel" becomes two separate Claim objects — one about the build year,
one about who built it — so each can be independently verified in
Stage 3.

Strategy:
  1. Primary: ask an LLM to do the splitting (it understands grammar,
     coreference and conjunctions far better than rule-based tools).
  2. If the LLM returns an empty list, retry once with a stricter prompt.
  3. Fallback: if the LLM still fails (no API key, network issue, bad
     JSON, empty result), use a simple sentence splitter so the pipeline
     never completely breaks.

Owner: P2
Location in repo: backend/app/modules/claim_extraction.py
"""

import json
import re
from typing import List

from ..config import settings
from ..models.schemas import Claim


_SYSTEM_PROMPT = """You extract factual claims from text for a fact-checking system.

Rules:
- Break the text into short, independent, atomic factual claims. Each claim states exactly one fact.
- Make every claim understandable on its own: replace pronouns and vague references ("it", "the tower", "he") with the actual subject.
- Include facts that appear inside parentheses.
- Extract exactly what the text asserts. Do NOT correct, judge, verify or add information, even if a statement looks wrong.
- Never return an empty list if the text contains any factual statement.
- Respond with ONLY a JSON array of strings. No markdown, no code fences, no commentary.

Example input:
The Eiffel Tower was built in 1887 by Gustave Eiffel (a French engineer).
Example output:
["The Eiffel Tower was built in 1887.", "Gustave Eiffel built the Eiffel Tower.", "Gustave Eiffel was a French engineer."]"""

_USER_TEMPLATE = 'Text:\n\"\"\"{answer}\"\"\"\n\nJSON array of claims:'

_RETRY_NOTE = (
    "\n\nYour previous reply was an empty list. The text above DOES contain "
    "factual statements. List every one of them as a JSON array of strings."
)


def _strip_markdown(text: str) -> str:
    """
    Removes Markdown formatting (bold/italic markers, tables, headers,
    bullet and numbered lists, horizontal rules) that LLMs often add to
    answers. Claim extraction works much better on plain text.
    """
    cleaned = text

    # Remove bold/italic markers: **text** or *text* -> text
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)

    lines = cleaned.split("\n")
    rebuilt_lines = []
    for idx, line in enumerate(lines):
        stripped = line.strip()

        # Table separator rows like |---|---|
        if re.match(r"^\|?[\s\-:|]+\|?$", stripped) and "|" in stripped:
            continue

        # Horizontal rules: ---, ***, ___
        if re.match(r"^([-*_])\1{2,}$", stripped):
            continue

        # Headers: "## 1. Why the sky is blue" -> "Why the sky is blue."
        header_match = re.match(r"^#{1,6}\s*(.+)", stripped)
        if header_match:
            content = header_match.group(1).strip()
            # Strip a leading list number too, or it later splits into an
            # orphan "1." claim.
            numbered = re.match(r"^\d+\.\s*(.+)", content)
            if numbered:
                content = numbered.group(1).strip()
            if content and not re.match(r"^[\d.\s]*$", content):
                if not content.endswith((".", "!", "?", ":")):
                    content += "."
                rebuilt_lines.append(content)
            continue

        # Table rows: skip header rows (followed by a separator row),
        # turn data rows into "cell: cell: cell." sentences.
        if stripped.startswith("|") and stripped.endswith("|"):
            next_stripped = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
            next_is_separator = (
                bool(re.match(r"^\|?[\s\-:|]+\|?$", next_stripped)) and "|" in next_stripped
            )
            if next_is_separator:
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip()]
            if cells:
                rebuilt_lines.append(": ".join(cells) + ".")
            continue

        # Bullet / numbered list items: make each its own sentence.
        bullet_match = re.match(r"^(?:[-*•]|\d+\.)\s*(.+)", stripped)
        if bullet_match:
            content = bullet_match.group(1).strip()
            if content and not content.endswith((".", "!", "?")):
                content += "."
            if content:
                rebuilt_lines.append(content)
            continue

        rebuilt_lines.append(line)

    cleaned = "\n".join(rebuilt_lines)
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _call_llm_for_extraction(answer: str, strict: bool = False) -> str:
    """Sends the extraction prompt to whichever provider is configured and
    returns the raw text response. `strict=True` adds a note used when the
    first attempt came back empty."""
    user_content = _USER_TEMPLATE.format(answer=answer)
    if strict:
        user_content += _RETRY_NOTE

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'")

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        max_tokens=1000,
        temperature=0.0,  # deterministic — consistent splitting, not creativity
    )
    return (response.choices[0].message.content or "").strip()


def _parse_claims_json(raw_text: str) -> List[str]:
    """
    Parses the LLM's JSON array response into a list of claim strings.
    Handles the case where the model wraps the JSON in markdown fences.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array of claim strings")

    return [str(item).strip() for item in parsed if str(item).strip()]


def _fallback_sentence_split(answer: str) -> List[str]:
    """
    Naive fallback: split on sentence-ending punctuation. Used only if
    LLM-based extraction fails, so the pipeline degrades gracefully.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s.strip() for s in sentences if s.strip()]


def _extract_with_llm(cleaned_answer: str) -> List[str]:
    """Runs LLM extraction, retrying once with a stricter prompt if the
    first attempt returns an empty list. Raises if both attempts fail."""
    claims = _parse_claims_json(_call_llm_for_extraction(cleaned_answer))
    if not claims:
        claims = _parse_claims_json(_call_llm_for_extraction(cleaned_answer, strict=True))
    if not claims:
        raise ValueError("LLM returned an empty claims list twice")
    return claims


def extract_claims(answer: str) -> List[Claim]:
    """
    Main entry point. Takes a raw LLM answer and returns a list of
    Claim objects, each with a unique claim_id and claim_text set.
    Verdict/confidence/source/evidence fields are left empty here —
    they are filled in later by the verification stage (M3).
    """
    if not answer or not answer.strip():
        return []

    cleaned_answer = _strip_markdown(answer)

    try:
        claim_texts = _extract_with_llm(cleaned_answer)
    except Exception:
        # No API key, bad JSON, network issue, or empty result twice —
        # fall back to simple sentence splitting rather than crash.
        claim_texts = _fallback_sentence_split(cleaned_answer)

    # Remove exact duplicates while keeping order.
    seen = set()
    unique_texts = []
    for text in claim_texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    return [
        Claim(claim_id=f"c{i + 1}", claim_text=text)
        for i, text in enumerate(unique_texts)
    ]