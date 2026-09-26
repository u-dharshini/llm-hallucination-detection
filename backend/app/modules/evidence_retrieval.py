"""
Evidence Retrieval — Stage 3 (part 1) of the pipeline.

Given a single Claim, independently retrieves supporting text from a
trusted external source. We do NOT rely on the LLM's own citations,
since it often gives none or fabricates them (per the project's core
contribution).

Owner: P3
Location in repo: backend/app/modules/evidence_retrieval.py
"""

from dataclasses import dataclass
from typing import Optional

import requests

from ..models.schemas import Claim


@dataclass
class Evidence:
    """Container for whatever text we found to check a claim against."""
    text: str
    source: str            # e.g. "Wikipedia", "Semantic Scholar", "None"
    url: Optional[str] = None


# Wikipedia requires a descriptive User-Agent on all API requests, or it
# may reject/throttle the request. We call the API directly with
# `requests` instead of using the old, unmaintained `wikipedia` PyPI
# package, which is known to break against Wikipedia's current API
# (it fails to set a valid User-Agent and gets non-JSON error responses).
_HEADERS = {
    "User-Agent": "LLM-Hallucination-Detection-Project/1.0 (capstone project; contact: team@example.com)"
}


def _wikipedia_search_titles(query: str, limit: int = 3) -> list:
    """Search Wikipedia and return a list of candidate page titles."""
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
    data = response.json()
    results = data.get("query", {}).get("search", [])
    return [r["title"] for r in results]


def _wikipedia_get_summary(title: str) -> Optional[Evidence]:
    """Fetch the plain-text summary (intro section) of a Wikipedia page."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
    response = requests.get(url, headers=_HEADERS, timeout=10)
    if response.status_code != 200:
        return None

    data = response.json()
    extract = data.get("extract")
    if not extract:
        return None

    page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
    return Evidence(text=extract, source="Wikipedia", url=page_url)


def _search_wikipedia(query: str) -> Optional[Evidence]:
    """
    Search Wikipedia for a page matching the claim, and return its summary
    as evidence text. Returns None if nothing usable is found.
    """
    try:
        titles = _wikipedia_search_titles(query)
        for title in titles:
            evidence = _wikipedia_get_summary(title)
            if evidence:
                return evidence
        return None
    except Exception:
        # Network issues, rate limits, etc. — fail safe rather than crash the pipeline
        return None


def _search_semantic_scholar(query: str) -> Optional[Evidence]:
    """
    Search Semantic Scholar for an academic paper matching the claim.
    Useful for claims about research papers / academic citations,
    which is this project's specific focus area.
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
        data = response.json()

        papers = data.get("data", [])
        if not papers:
            return None

        paper = papers[0]
        abstract = paper.get("abstract")
        if not abstract:
            return None

        return Evidence(
            text=abstract,
            source="Semantic Scholar",
            url=paper.get("url"),
        )
    except Exception:
        return None


def retrieve_evidence(claim: Claim) -> Evidence:
    """
    Main entry point. Given a Claim, retrieve the best available evidence.

    Strategy:
      1. If the claim mentions a paper/study/research-sounding term, try
         Semantic Scholar first.
      2. Otherwise (or as a fallback), search Wikipedia.
      3. If nothing is found anywhere, return an "insufficient evidence"
         Evidence object — the verification stage will mark this claim
         as Unverifiable.
    """
    academic_keywords = ("paper", "study", "research", "published", "journal", "et al")
    text_lower = claim.claim_text.lower()

    if any(keyword in text_lower for keyword in academic_keywords):
        evidence = _search_semantic_scholar(claim.claim_text)
        if evidence:
            return evidence

    evidence = _search_wikipedia(claim.claim_text)
    if evidence:
        return evidence

    # Last resort: try Semantic Scholar even for non-academic-sounding claims,
    # in case Wikipedia had nothing
    evidence = _search_semantic_scholar(claim.claim_text)
    if evidence:
        return evidence

    return Evidence(text="", source="None", url=None)
