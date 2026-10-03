"""
Shared style constants for the hallucination-detection UI.

Verdict values must match the shared Claim schema (see backend/app/models/schemas.py):
    "Supported" | "Contradicted" | "Unverifiable"
"""

VERDICT_COLORS = {
    "Supported": {
        "bg": "#e6f4ea",
        "border": "#34a853",
        "text": "#1e7e34",
        "icon": "✅",
    },
    "Contradicted": {
        "bg": "#fdecea",
        "border": "#ea4335",
        "text": "#c5221f",
        "icon": "❌",
    },
    "Unverifiable": {
        "bg": "#fef7e0",
        "border": "#fbbc04",
        "text": "#a86400",
        "icon": "⚠️",
    },
}

DEFAULT_STYLE = {
    "bg": "#f1f3f4",
    "border": "#9aa0a6",
    "text": "#5f6368",
    "icon": "❔",
}


def get_verdict_style(verdict: str) -> dict:
    """Return the color/icon config for a verdict, falling back to a neutral style
    for anything unexpected (keeps the UI from breaking if a module sends a
    verdict string outside the three expected values)."""
    return VERDICT_COLORS.get(verdict, DEFAULT_STYLE)


PAGE_CSS = """
<style>
.claim-card {
    border-left: 5px solid;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.claim-card .claim-text {
    font-size: 1.02rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.claim-card .verdict-badge {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 999px;
    margin-bottom: 8px;
}
.claim-card .evidence-snippet {
    font-size: 0.9rem;
    font-style: italic;
    color: #3c4043;
    margin-top: 6px;
    border-top: 1px solid rgba(0,0,0,0.08);
    padding-top: 6px;
}
.claim-card .source-line {
    font-size: 0.78rem;
    color: #5f6368;
    margin-top: 4px;
}
.confidence-track {
    background: #e0e0e0;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.confidence-fill {
    height: 6px;
    border-radius: 4px;
}
</style>
"""
