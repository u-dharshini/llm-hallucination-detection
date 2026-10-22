"""
Evidence Retrieval — Stage 3 (part 1) of the pipeline.

Given a single Claim, independently retrieves supporting text from a
trusted external source. We do NOT rely on the LLM's own citations,
since it often gives none or fabricates them (the project's core
contribution).

How it works:
  1. Build a short search query from the claim's proper nouns.
  2. Look at the top few matching Wikipedia pages (not just the first).
  3. Pull a longer excerpt from each, split it into sentences, and score
     every sentence by how well it overlaps with the claim.
  4. Return ONLY the single best sentence across all pages. The NLI model
     (roberta-large-mnli) is trained on short sentence pairs, so a short,
     focused premise gives much clearer verdicts than a long paragraph.

Owner: P3
Location in repo: backend/app/modules/evidence_retrieval.py
"""

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests

from ..models.schemas import Claim


@dataclass
class Evidence:
    """Container for whatever text we found to check a claim against."""
    text: str
    source: str            # e.g. "Wikipedia", "Semantic Scholar", "None"
    url: Optional[str] = None


# Wikipedia expects a descriptive User-Agent on API requests. We call the
# API directly with `requests` because the old `wikipedia` PyPI package is
# unmaintained and breaks against Wikipedia's current API.
_HEADERS = {
    "User-Agent": "LLM-Hallucination-Detection-Project/1.0 (capstone project; contact: team@example.com)"
}

# Words that carry little meaning for matching a sentence to a claim.
_STOPWORDS = {
    "the", "and", "was", "were", "for", "with", "that", "this", "from", "his",
    "her", "its", "are", "has", "had", "have", "been", "who", "which", "into",
    "than", "then", "also", "but", "not", "can", "may",
    "between", "during", "about", "over", "after", "before", "their", "they",
}

MAX_PAGES_TO_CHECK = 3
MIN_SENTENCE_WORDS = 5
MAX_EVIDENCE_CHARS = 500

# Sentences that begin with a pronoun need the previous sentence for context.
_PRONOUN_START = re.compile(r"^(It|He|She|They|This|These|Its|His|Her|Their)\b")


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

def _extract_search_query(claim_text: str) -> str:
    """
    Builds a short, search-friendly query from a (often long) claim by
    pulling out capitalized proper-noun phrases, which are the words most
    likely to match a Wikipedia page title.

    A single capitalized word at the very start of the claim is ignored,
    because it is capitalized only because it begins the sentence
    (e.g. "Construction began in 1887..." should not search for "Construction").

    Falls back to the full claim text if no useful proper nouns are found.
    """
    pattern = r"\b[A-Z][a-zA-ZÀ-ÿ]*(?:\s+[A-Z][a-zA-ZÀ-ÿ]*)*\b"
    found = []
    for m in re.finditer(pattern, claim_text):
        phrase = m.group(0)
        is_single_word = " " not in phrase
        if m.start() == 0 and is_single_word:
            continue  # sentence-initial capital, not a real proper noun
        found.append(phrase)

    stop = {"The", "It", "So", "A", "An", "This", "That", "In", "On", "At"}
    useful = [p for p in found if p not in stop and len(p) > 1]

    if not useful:
        return claim_text

    seen = set()
    deduped = []
    for term in useful:
        if term not in seen:
            seen.add(term)
            deduped.append(term)

    return " ".join(deduped)[:150]


# ---------------------------------------------------------------------------
# Wikipedia access
# ---------------------------------------------------------------------------

def _wikipedia_search_titles(query: str, limit: int = MAX_PAGES_TO_CHECK) -> List[str]:
    """Search Wikipedia and return candidate page titles, best match first."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }
    response = requests.get(url, params=params, headers=_HEADERS, timeout=10)
    response.raise_for_status()
    results = response.json().get("query", {}).get("search", [])
    return [r["title"] for r in results]


def _wikipedia_get_extract(title: str, chars: int = 3000) -> Optional[Tuple[str, str]]:
    """
    Fetch a plain-text excerpt of a Wikipedia article.
    Returns (text, page_url), or None if the page doesn't exist.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "format": "json",
        "explaintext": 1,
        "exchars": chars,
        "redirects": 1,
    }
    response = requests.get(url, params=params, headers=_HEADERS, timeout=10)
    if response.status_code != 200:
        return None

    pages = response.json().get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            return None
        extract = page_data.get("extract")
        if extract and extract.strip():
            page_url = "https://en.wikipedia.org/wiki/" + requests.utils.quote(
                title.replace(" ", "_")
            )
            return extract.strip(), page_url
    return None


# ---------------------------------------------------------------------------
# Sentence selection
# ---------------------------------------------------------------------------

# Parentheticals holding pronunciations / phonetic spelling, e.g.
# "( EYE-fəl; French: Tour Eiffel [tuʁ ɛfɛl] )". They contain IPA characters
# or square brackets, and just add noise for the NLI model.
_PRONUNCIATION = re.compile(r"\s*\([^()]*[\[\]\u0250-\u02FF][^()]*\)")


def _split_sentences(text: str) -> List[str]:
    """Split an extract into sentences, dropping '== Section ==' headers
    and pronunciation guides."""
    text = _PRONUNCIATION.sub("", text)
    text = re.sub(r"={2,}\s*.*?\s*={2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


def _content_words(text: str) -> set:
    """Lowercased meaningful words (and numbers) from a piece of text."""
    return {
        w for w in re.findall(r"\w+", text.lower())
        if (len(w) > 2 and w not in _STOPWORDS) or w.isdigit()
    }


def _score_sentence(sentence: str, claim_words: set) -> float:
    """
    Scores how well a sentence matches a claim. Numbers (e.g. years) count
    double because they are the most specific facts, and the total is
    normalized by sentence length so long rambling sentences don't win
    just by containing more words.
    """
    sentence_words = _content_words(sentence)
    if len(sentence.split()) < MIN_SENTENCE_WORDS:
        return 0.0

    score = 0.0
    for w in claim_words & sentence_words:
        score += 2.0 if w.isdigit() else 1.0

    return score / math.sqrt(max(len(sentence.split()), 1))


def _best_sentence_for_claim(titles: List[str], claim_text: str) -> Optional[Evidence]:
    """
    Fetches an excerpt for each candidate page, scores every sentence
    against the claim, and returns the single best sentence overall.
    """
    claim_words = _content_words(claim_text)
    best_score = 0.0
    best: Optional[Evidence] = None
    fallback: Optional[Evidence] = None

    for title in titles:
        fetched = _wikipedia_get_extract(title)
        if not fetched:
            continue
        extract, page_url = fetched

        sentences = _split_sentences(extract)
        if not sentences:
            continue

        # Remember the first page's opening sentence in case nothing scores.
        if fallback is None:
            fallback = Evidence(
                text=sentences[0][:MAX_EVIDENCE_CHARS],
                source="Wikipedia",
                url=page_url,
            )

        for i, sentence in enumerate(sentences):
            score = _score_sentence(sentence, claim_words)
            if score > best_score:
                best_score = score
                text = sentence
                # A sentence starting with "It", "He", "This"... has no
                # subject on its own, which confuses the NLI model. Include
                # the sentence before it so the pronoun has a referent.
                if i > 0 and _PRONOUN_START.match(sentence):
                    text = sentences[i - 1] + " " + sentence
                best = Evidence(
                    text=text[:MAX_EVIDENCE_CHARS],
                    source="Wikipedia",
                    url=page_url,
                )

    return best or fallback


def _search_wikipedia(query: str, claim_text: str) -> Optional[Evidence]:
    """Search Wikipedia and return the best-matching sentence, or None."""
    try:
        titles = _wikipedia_search_titles(query)
        if not titles:
            return None
        return _best_sentence_for_claim(titles, claim_text)
    except Exception:
        # Network issues, rate limits, etc. — fail safe rather than crash.
        return None


# ---------------------------------------------------------------------------
# Semantic Scholar (for academic-citation claims)
# ---------------------------------------------------------------------------

def _search_semantic_scholar(query: str) -> Optional[Evidence]:
    """
    Search Semantic Scholar for a paper matching the claim. Useful for
    claims about research papers / academic citations, which is this
    project's specific focus area.
    """
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": 1,
            "fields": "title,abstract,url,year,authors",
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        papers = response.json().get("data", [])
        if not papers:
            return None

        paper = papers[0]
        abstract = paper.get("abstract")
        if not abstract:
            return None

        return Evidence(text=abstract, source="Semantic Scholar", url=paper.get("url"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def retrieve_evidence(claim: Claim) -> Evidence:
    """
    Given a Claim, retrieve the best available evidence.

    Strategy:
      1. If the claim sounds academic (paper/study/research), try
         Semantic Scholar first.
      2. Search Wikipedia with a short proper-noun query, check the top
         few pages, and keep the single best-matching sentence.
      3. If that finds nothing, retry Wikipedia with the full claim text.
      4. As a last resort, try Semantic Scholar.
      5. If nothing is found, return empty evidence so verification
         marks the claim Unverifiable.
    """
    academic_keywords = ("paper", "study", "research", "published", "journal", "et al")
    if any(k in claim.claim_text.lower() for k in academic_keywords):
        evidence = _search_semantic_scholar(claim.claim_text)
        if evidence:
            return evidence

    short_query = _extract_search_query(claim.claim_text)
    evidence = _search_wikipedia(short_query, claim.claim_text)
    if evidence:
        return evidence

    if short_query != claim.claim_text:
        evidence = _search_wikipedia(claim.claim_text, claim.claim_text)
        if evidence:
            return evidence

    evidence = _search_semantic_scholar(claim.claim_text)
    if evidence:
        return evidence

    return Evidence(text="", source="None", url=None)