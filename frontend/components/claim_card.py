"""Renders a single claim as a color-coded card."""

import streamlit as st

from .styles import get_verdict_style


def render_claim_card(claim: dict) -> None:
    """Render one Claim (claim_text, verdict, confidence, source,
    evidence_snippet) as a color-coded HTML card.

    Expects `claim` to follow the shared schema. Missing optional fields
    (source, evidence_snippet) degrade gracefully.
    """
    verdict = claim.get("verdict", "Unverifiable")
    style = get_verdict_style(verdict)
    confidence = claim.get("confidence")
    confidence_pct = f"{confidence * 100:.0f}%" if isinstance(confidence, (int, float)) else "—"

    evidence_html = ""
    if claim.get("evidence_snippet"):
        evidence_html = (
            f'<div class="evidence-snippet" style="color:#3c4043;">'
            f'"{claim["evidence_snippet"]}"</div>'
        )

    source_html = ""
    if claim.get("source"):
        source_html = (
            f'<div class="source-line" style="color:#5f6368;">'
            f'Source: {claim["source"]}</div>'
        )

    confidence_bar = ""
    if isinstance(confidence, (int, float)):
        confidence_bar = (
            '<div class="confidence-track">'
            f'<div class="confidence-fill" style="width:{confidence * 100:.0f}%; '
            f'background:{style["border"]};"></div></div>'
        )

    # NOTE: built as one unindented string on purpose. Streamlit's markdown
    # renderer follows CommonMark rules, where a line indented 4+ spaces is
    # treated as a code block even with unsafe_allow_html=True — a multi-line
    # indented f-string here silently breaks the styling.
    card_html = (
        f'<div class="claim-card" style="background:{style["bg"]}; border-color:{style["border"]};">'
        f'<div class="claim-text" style="color:#1a1a1a;">{claim.get("claim_text", "")}</div>'
        f'<span class="verdict-badge" style="background:{style["border"]}; color:white;">'
        f'{style["icon"]} {verdict} · {confidence_pct} confidence</span>'
        f'{confidence_bar}{evidence_html}{source_html}'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)
