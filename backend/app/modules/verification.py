"""
Verification — Stage 3 (part 2) of the pipeline.

Given a Claim and the Evidence retrieved for it, runs a Natural Language
Inference (NLI) model to decide whether the evidence SUPPORTS,
CONTRADICTS, or is silent on (UNVERIFIABLE) the claim.

Model: roberta-large-mnli (via HuggingFace transformers)
  - "ENTAILMENT"    -> evidence supports the claim      -> SUPPORTED
  - "CONTRADICTION" -> evidence contradicts the claim    -> CONTRADICTED
  - "NEUTRAL"        -> evidence doesn't confirm or deny -> UNVERIFIABLE

Owner: P3
Location in repo: backend/app/modules/verification.py
"""

import os

# Force transformers to use only the PyTorch backend and ignore any
# TensorFlow installation on this machine. Some environments have both
# frameworks installed for other projects, and TensorFlow's protobuf
# requirements can conflict with the ones this project needs — setting
# this before `transformers` is imported avoids that clash entirely.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from functools import lru_cache

from transformers import pipeline

from ..models.schemas import Claim, VerdictLabel
from .evidence_retrieval import Evidence


# Loading the model is slow (~seconds) and memory-heavy, so we load it
# once and reuse it for every claim, instead of reloading per-request.
@lru_cache(maxsize=1)
def _get_nli_pipeline():
    return pipeline(
        "text-classification",
        model="roberta-large-mnli",
        top_k=None,  # return scores for all 3 labels, not just the top one
    )


# roberta-large-mnli's raw output labels, mapped to our own Verdict enum
_LABEL_MAP = {
    "ENTAILMENT": VerdictLabel.SUPPORTED,
    "CONTRADICTION": VerdictLabel.CONTRADICTED,
    "NEUTRAL": VerdictLabel.UNVERIFIABLE,
}


def verify_claim(claim: Claim, evidence: Evidence) -> Claim:
    """
    Compares a claim against its evidence and fills in the claim's
    verdict, confidence, source, and evidence fields in place, then
    returns it.
    """
    # No evidence was found at all -> can't verify, don't call the model
    if not evidence.text.strip():
        claim.verdict = VerdictLabel.UNVERIFIABLE
        claim.confidence = 0.0
        claim.source = "Insufficient evidence"
        claim.evidence_snippet = None
        claim.evidence_url = None
        return claim

    nli = _get_nli_pipeline()

    # NLI models expect a (premise, hypothesis) pair.
    # premise = the evidence we trust; hypothesis = the claim we're checking.
    pair_input = f"{evidence.text}</s></s>{claim.claim_text}"

    try:
        results = nli(pair_input)
        # results looks like: [[{'label': 'ENTAILMENT', 'score': 0.94}, ...]]
        scores = results[0] if isinstance(results[0], list) else results
        best = max(scores, key=lambda x: x["score"])

        claim.verdict = _LABEL_MAP.get(best["label"].upper(), VerdictLabel.UNVERIFIABLE)
        claim.confidence = round(float(best["score"]), 2)
        claim.source = evidence.source
        claim.evidence_snippet = evidence.text[:500]  # keep it short for display
        claim.evidence_url = evidence.url

    except Exception:
        # If the model call fails for any reason, don't crash the whole
        # pipeline — mark this one claim as unverifiable instead.
        claim.verdict = VerdictLabel.UNVERIFIABLE
        claim.confidence = 0.0
        claim.source = "Verification error"
        claim.evidence_snippet = None
        claim.evidence_url = None

    return claim
