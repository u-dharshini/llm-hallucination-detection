"""
backend/app/api/routes.py

Full pipeline:
  question -> M1 generate_answer -> M2 extract_claims
           -> M3 retrieve_evidence(Claim) -> verify_claim(Claim, Evidence)
           -> save to SQLite -> return

Matches the real signatures:
  retrieve_evidence(claim: Claim) -> Evidence
  verify_claim(claim: Claim, evidence: Evidence) -> Claim
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models.db_models import ClaimRecord, QueryRecord
from backend.app.models.schemas import Claim

router = APIRouter(prefix="/api", tags=["pipeline"])

MAX_CLAIMS = 10  # keeps a demo run fast; raise later if needed

# ---------------------------------------------------------------------------
# M1 / M2: use the real modules when they exist, otherwise fall back.
# ---------------------------------------------------------------------------
try:
    from backend.app.modules.answer_generation import generate_answer
except ImportError:
    def generate_answer(question: str) -> str:
        return f"[STUB ANSWER] answer_generation.py not implemented. Question: {question}"

try:
    from backend.app.modules.claim_extraction import extract_claims
except ImportError:
    def extract_claims(answer_text: str) -> list[str]:
        """Naive fallback until P2's real extractor lands: one claim per sentence."""
        text = answer_text.replace("**", "").replace("\n", " ")
        parts = [s.strip() for s in text.split(".")]
        return [p for p in parts if len(p) > 15]

# M3 (real code)
from backend.app.modules.evidence_retrieval import retrieve_evidence
from backend.app.modules.verification import verify_claim


# ---------------------------------------------------------------------------
# Schemas for this router
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
# Helpers
# ---------------------------------------------------------------------------
def _to_claim(item, idx: int) -> Claim:
    """Accept either a plain string or a Claim from the extractor."""
    if isinstance(item, Claim):
        return item
    text = item if isinstance(item, str) else getattr(item, "claim_text", str(item))
    return Claim(claim_id=f"c{idx}", claim_text=text)


def _verdict_str(verdict) -> str:
    if verdict is None:
        return "unverifiable"
    return getattr(verdict, "value", str(verdict))


def _serialize_query(rec: QueryRecord) -> dict:
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/verify-answer", response_model=PipelineResponse)
def verify_answer(payload: QuestionRequest, db: Session = Depends(get_db)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    # 1. Answer generation (M1)
    try:
        answer_text = generate_answer(question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {e}")

    # 2. Claim extraction (M2)
    raw_claims = extract_claims(answer_text)
    if not raw_claims:
        raise HTTPException(status_code=422, detail="No claims could be extracted")

    # 3. Evidence retrieval + verification (M3), one claim at a time
    results: list[ClaimResult] = []
    for idx, item in enumerate(raw_claims[:MAX_CLAIMS], start=1):
        claim = _to_claim(item, idx)
        try:
            evidence = retrieve_evidence(claim)
            verified = verify_claim(claim, evidence)
        except Exception as e:
            # One bad claim (network hiccup, etc.) shouldn't kill the whole run
            verified = claim
            verified.source = "error"
            verified.evidence_snippet = f"Verification failed: {e}"

        results.append(
            ClaimResult(
                claim_text=verified.claim_text,
                verdict=_verdict_str(verified.verdict),
                confidence=verified.confidence,
                source=verified.source,
                evidence_snippet=verified.evidence_snippet,
            )
        )

    # 4. Save to DB
    query_record = QueryRecord(
        question=question,
        answer_text=answer_text,
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
    )
    db.add(query_record)
    db.flush()  # get the id before commit

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
    records = (
        db.query(QueryRecord)
        .order_by(QueryRecord.created_at.desc())
        .limit(limit)
        .all()
    )
    return [_serialize_query(rec) for rec in records]


@router.get("/history/{query_id}")
def get_history_item(query_id: int, db: Session = Depends(get_db)):
    rec = db.query(QueryRecord).filter(QueryRecord.id == query_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="query_id not found")
    return _serialize_query(rec)
