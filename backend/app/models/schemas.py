"""
Shared data schemas for the LLM Hallucination Detection pipeline.

Every module (Claim Extraction, Evidence Retrieval, Verification, Result Display)
imports and uses these models so the data passed between stages has a
consistent, agreed-upon shape.

Owner: P2 (Claim Extraction) — drafted here, approved by P3 and P5.
Location in repo: backend/app/models/schemas.py
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class VerdictLabel(str, Enum):
    """Possible verification outcomes for a single claim."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class Claim(BaseModel):
    """
    A single atomic, checkable statement extracted from an LLM's answer.

    Example: "Eiffel Tower was built in 1887 by Gustave Eiffel" becomes two
    Claim objects:
      1. claim_text="The Eiffel Tower was built in 1887"
      2. claim_text="Gustave Eiffel built the Eiffel Tower"
    """
    claim_id: str = Field(..., description="Unique id for this claim, e.g. 'c1', 'c2'")
    claim_text: str = Field(..., description="The extracted atomic factual statement")

    # Filled in by Evidence Retrieval + Verification (M3)
    verdict: Optional[VerdictLabel] = Field(
        default=None, description="Supported / Contradicted / Unverifiable"
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Model confidence score, 0-1"
    )
    source: Optional[str] = Field(
        default=None, description="Where evidence came from, e.g. 'Wikipedia', 'Semantic Scholar'"
    )
    evidence_snippet: Optional[str] = Field(
        default=None, description="The retrieved text used to verify this claim"
    )
    evidence_url: Optional[str] = Field(
        default=None, description="Link to the evidence source, if available"
    )


class AnswerRequest(BaseModel):
    """Incoming request from the frontend when a user asks a question."""
    question: str = Field(..., description="The user's natural-language question")


class VerificationResult(BaseModel):
    """
    Full result returned to the frontend after the pipeline runs:
    Answer Generation -> Claim Extraction -> Evidence Retrieval + Verification.
    """
    question: str
    raw_answer: str = Field(..., description="The original, unmodified LLM answer")
    claims: List[Claim] = Field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.claims if c.verdict == VerdictLabel.SUPPORTED)

    @property
    def contradicted_count(self) -> int:
        return sum(1 for c in self.claims if c.verdict == VerdictLabel.CONTRADICTED)

    @property
    def unverifiable_count(self) -> int:
        return sum(1 for c in self.claims if c.verdict == VerdictLabel.UNVERIFIABLE)
