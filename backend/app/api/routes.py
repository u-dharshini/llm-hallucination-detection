"""
backend/app/api/routes.py

Wires the full pipeline together:
  question -> M1 answer_generation -> M2 claim_extraction
           -> M3 evidence_retrieval + verification -> save to DB -> return

Uses safe stub fallbacks for M1/M2 so this runs end-to-end even before
those teammates push their real code. Once P1/P2 push real modules with
the function signatures below, these routes need zero changes.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.app.database import get_db
from backend.app.models.db_models import QueryRecord, ClaimRecord
from backend.app.config import settings

router = APIRouter(prefix="/api", tags=["pipeline"])


# ---------------------------------------------------------------------------
# Try to import real modules; fall back to stubs if a teammate hasn't pushed
# their part yet. This keeps /api/verify-answer working throughout dev.
# ---------------------------------------------------------------------------
try:
    from backend.app.modules.answer_generation import generate_answer
except ImportError:
    def generate_answer(question: str) -> str:
        return f"[STUB ANSWER] answer_generation.py not implemented yet. Question was: {question}"

try:
    from backend.app.modules.claim_extraction import extract_claims
except ImportError:
    def extract_claims(answer_text: str) -> list[str]:
        # naive fallback: treat each sentence as a claim
        return [s.strip() for s in answer_text.split(".") if s.strip()]

# M3 — your module, should already exist and work
from backend.app.modules.evidence_retrieval import retrieve_evidence
from backend.app.modules.verification import verify_claim


# ---------------------------------------------------------------------------
# Request / response schemas for this router
# (Full shared Claim schema lives in models/schemas.py, owned by P2)
# ---------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str


class ClaimResult(BaseModel):
    claim_text: str
    verdict: str
    confidence: float | None = None
    source: str | None = None
    evidence_snippet: str | None = None


class PipelineResponse(BaseModel):
    query_id: int
    question: str
    answer_text: str
    claims: list[ClaimResult]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/verify-answer", response_model=PipelineResponse)
def verify_answer(payload: QuestionRequest, db: Session = Depends(get_db)):
    """Run the full pipeline for a single question and persist the result."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    # 1. Generate answer (M1)
    answer_text = generate_answer(question)

    # 2. Extract atomic claims (M2)
    claim_texts = extract_claims(answer_text)
    if not claim_texts:
        raise HTTPException(status_code=422, detail="No claims could be extracted from the answer")

    # 3. Retrieve evidence + verify each claim independently (M3)
    results: list[ClaimResult] = []
    for claim_text in claim_texts:
        evidence = retrieve_evidence(claim_text)          # e.g. list of {source, snippet}
        verdict, confidence, best_evidence = verify_claim(claim_text, evidence)
        results.append(
            ClaimResult(
                claim_text=claim_text,
                verdict=verdict,
                confidence=confidence,
                source=best_evidence.get("source") if best_evidence else None,
                evidence_snippet=best_evidence.get("snippet") if best_evidence else None,
            )
        )

    # 4. Persist to DB
    query_record = QueryRecord(
        question=question,
        answer_text=answer_text,
        llm_provider=settings.DEFAULT_LLM_PROVIDER,
        llm_model=(
            settings.DEFAULT_MODEL_OPENAI
            if settings.DEFAULT_LLM_PROVIDER == "openai"
            else settings.DEFAULT_MODEL_GROQ
        ),
    )
    db.add(query_record)
    db.flush()  # get query_record.id before commit

    for r in results:
        db.add(
            ClaimRecord(
                query_id=query_record.id,
                claim_text=r.claim_text,
                verdict=r.verdict,
                confidence=r.confidence,
                source=r.source,
                evidence_snippet=r.evidence_snippet,
            )
        )
    db.commit()
    db.refresh(query_record)

    return PipelineResponse(
        query_id=query_record.id,
        question=question,
        answer_text=answer_text,
        claims=results,
    )


@router.get("/history")
def get_history(limit: int = 20, db: Session = Depends(get_db)):
    """Return the most recent pipeline runs, newest first (for M4 frontend)."""
    records = (
        db.query(QueryRecord)
        .order_by(QueryRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "query_id": rec.id,
            "question": rec.question,
            "answer_text": rec.answer_text,
            "created_at": rec.created_at.isoformat(),
            "claims": [
                {
                    "claim_text": c.claim_text,
                    "verdict": c.verdict,
                    "confidence": c.confidence,
                    "source": c.source,
                    "evidence_snippet": c.evidence_snippet,
                }
                for c in rec.claims
            ],
        }
        for rec in records
    ]


@router.get("/history/{query_id}")
def get_history_item(query_id: int, db: Session = Depends(get_db)):
    rec = db.query(QueryRecord).filter(QueryRecord.id == query_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="query_id not found")
    return {
        "query_id": rec.id,
        "question": rec.question,
        "answer_text": rec.answer_text,
        "created_at": rec.created_at.isoformat(),
        "claims": [
            {
                "claim_text": c.claim_text,
                "verdict": c.verdict,
                "confidence": c.confidence,
                "source": c.source,
                "evidence_snippet": c.evidence_snippet,
            }
            for c in rec.claims
        ],
    }
