"""
Tests for evidence_retrieval.py and verification.py.

Run with:
    pytest backend/tests/test_verification.py -v

Note: these tests hit real external APIs (Wikipedia) and load a real
NLI model the first time they run, so they will be slow (the model
download alone can take a minute or two the first time). This is
expected for M3.
"""

import pytest

from backend.app.models.schemas import Claim, VerdictLabel
from backend.app.modules.evidence_retrieval import retrieve_evidence, Evidence
from backend.app.modules.verification import verify_claim


def test_supported_claim():
    """A true, unambiguous, well-documented fact should come back as SUPPORTED."""
    claim = Claim(claim_id="c1", claim_text="The Eiffel Tower is located in Paris, France")
    evidence = retrieve_evidence(claim)

    assert evidence.text != ""
    result = verify_claim(claim, evidence)

    assert result.verdict == VerdictLabel.SUPPORTED
    assert result.confidence > 0.5
    assert result.source == "Wikipedia"


def test_contradicted_claim():
    """A clearly false fact should come back as CONTRADICTED."""
    claim = Claim(claim_id="c2", claim_text="The Eiffel Tower is located in London, England")
    evidence = retrieve_evidence(claim)

    result = verify_claim(claim, evidence)

    assert result.verdict == VerdictLabel.CONTRADICTED


def test_no_evidence_found():
    """
    A nonsense claim with no matching evidence should be marked
    UNVERIFIABLE without crashing, and should not call the NLI model.
    """
    claim = Claim(claim_id="c3", claim_text="asdkjfh qwoeiruqwoeiru nonsense claim xyz123")
    evidence = Evidence(text="", source="None", url=None)

    result = verify_claim(claim, evidence)

    assert result.verdict == VerdictLabel.UNVERIFIABLE
    assert result.confidence == 0.0
    assert result.source == "Insufficient evidence"


def test_verify_claim_fills_all_fields():
    """After verification, every relevant field on the Claim should be set."""
    claim = Claim(claim_id="c4", claim_text="Water boils at 100 degrees Celsius at sea level")
    evidence = Evidence(
        text="Water boils at 100 degrees Celsius (212 F) at sea level atmospheric pressure.",
        source="Wikipedia",
        url="https://en.wikipedia.org/wiki/Boiling_point",
    )

    result = verify_claim(claim, evidence)

    assert result.verdict is not None
    assert result.confidence is not None
    assert result.source == "Wikipedia"
    assert result.evidence_snippet is not None