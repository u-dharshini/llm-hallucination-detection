"""
FastAPI entrypoint for the LLM Hallucination Detection backend.

This wires together the four pipeline stages:
  1. Answer Generation      (P1 -> modules/answer_generation.py)
  2. Claim Extraction       (P2 -> modules/claim_extraction.py)
  3. Evidence + Verification(P3 -> modules/evidence_retrieval.py, verification.py)
  4. Result Display         (P4 -> frontend, consumes this API)

Owner: P5 (Backend Integration)
Location in repo: backend/app/main.py

Run locally with:
    uvicorn backend.app.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models.schemas import AnswerRequest, VerificationResult, Claim

# NOTE: these imports will start working once P1/P2/P3 fill in their files.
# Until then, this file uses safe fallback stubs (see the try/except below)
# so the API still runs and can be tested end-to-end with placeholder data.
try:
    from .modules.answer_generation import generate_answer
except ImportError:
    def generate_answer(question: str) -> str:
        return f"[stub answer] This is a placeholder answer to: {question}"

try:
    from .modules.claim_extraction import extract_claims
except ImportError:
    def extract_claims(answer: str):
        return [Claim(claim_id="c1", claim_text=answer)]

try:
    from .modules.evidence_retrieval import retrieve_evidence
    from .modules.verification import verify_claim
except ImportError:
    def retrieve_evidence(claim: Claim):
        return None

    def verify_claim(claim: Claim, evidence):
        from .models.schemas import VerdictLabel
        claim.verdict = VerdictLabel.UNVERIFIABLE
        claim.confidence = 0.0
        claim.source = "stub"
        return claim


app = FastAPI(title="LLM Hallucination Detection API")

# Allow the frontend (Streamlit/React, likely a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Simple check to confirm the API is running."""
    return {"status": "ok"}


@app.post("/verify", response_model=VerificationResult)
def verify_answer(request: AnswerRequest):
    """
    Full pipeline: question -> LLM answer -> claims -> evidence -> verdicts.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Stage 1: Answer Generation
    raw_answer = generate_answer(request.question)

    # Stage 2: Claim Extraction
    claims = extract_claims(raw_answer)

    # Stage 3: Evidence Retrieval + Verification
    verified_claims = []
    for claim in claims:
        evidence = retrieve_evidence(claim)
        verified_claim = verify_claim(claim, evidence)
        verified_claims.append(verified_claim)

    result = VerificationResult(
        question=request.question,
        raw_answer=raw_answer,
        claims=verified_claims,
    )

    # TODO (P5): save `result` to SQLite verification history table here

    return result
