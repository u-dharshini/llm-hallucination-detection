"""
Mock data for frontend development (M4).

Standing in for the real pipeline until M1 (answer generation), M2 (claim
extraction) and M5 (API integration) are wired up. Each mock claim matches
the shared Claim schema so this module is a drop-in replacement for a real
API response — swap `get_mock_response()` for a call to P5's endpoint later
(see the TODO in streamlit_app.py).
"""

MOCK_RESPONSES = {
    "default": {
        "question": "What did the original Transformer paper propose?",
        "answer": (
            "The 2017 paper 'Attention Is All You Need' by Vaswani et al. "
            "introduced the Transformer architecture, which relies entirely "
            "on self-attention mechanisms rather than recurrence. It was "
            "trained on the WMT 2014 English-to-German dataset and achieved "
            "a BLEU score of 41.8, and the authors also claim it was the "
            "first model to achieve human-level translation accuracy."
        ),
        "claims": [
            {
                "claim_text": (
                    "The Transformer architecture relies entirely on "
                    "self-attention mechanisms rather than recurrence."
                ),
                "verdict": "Supported",
                "confidence": 0.94,
                "source": "Vaswani et al., 'Attention Is All You Need' (2017), arXiv:1706.03762",
                "evidence_snippet": (
                    "The Transformer, a model architecture eschewing recurrence "
                    "and instead relying entirely on an attention mechanism..."
                ),
            },
            {
                "claim_text": (
                    "It was trained on the WMT 2014 English-to-German dataset "
                    "and achieved a BLEU score of 41.8."
                ),
                "verdict": "Contradicted",
                "confidence": 0.81,
                "source": "Vaswani et al. (2017), Table 2",
                "evidence_snippet": (
                    "Our big model establishes a new single-model state-of-the-art "
                    "BLEU score of 28.4 on the WMT 2014 English-to-German task..."
                ),
            },
            {
                "claim_text": (
                    "The authors claim it was the first model to achieve "
                    "human-level translation accuracy."
                ),
                "verdict": "Unverifiable",
                "confidence": 0.52,
                "source": "No matching passage found",
                "evidence_snippet": "",
            },
        ],
    }
}


def get_mock_response(question: str) -> dict:
    """Return a mock pipeline response for a given question.

    TODO(M5 integration): replace this with a call to the real backend, e.g.
        response = requests.post(f"{API_BASE_URL}/analyze", json={"question": question})
        return response.json()
    The returned shape (question, answer, claims[]) should match whatever
    api/routes.py exposes, so keep this function's return contract in sync
    with P5's schema.
    """
    return MOCK_RESPONSES["default"]
