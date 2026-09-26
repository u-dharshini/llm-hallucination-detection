"""
M4 — Frontend / Result Display

Currently wired to mock data (components/mock_data.py) so the UI can be
built and reviewed independently of M1/M2/M5. When P5's API is ready, swap
`get_mock_response()` for a real request — see the TODO below.
"""

import streamlit as st

from components.claim_card import render_claim_card
from components.mock_data import get_mock_response
from components.styles import PAGE_CSS

st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🔍",
    layout="centered",
)
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.title("🔍 LLM Hallucination Detector")
st.caption("Ask a question, get an LLM answer, and see each claim checked against independent evidence.")

with st.sidebar:
    st.subheader("Legend")
    st.markdown("✅ **Supported** — evidence backs the claim")
    st.markdown("❌ **Contradicted** — evidence disagrees with the claim")
    st.markdown("⚠️ **Unverifiable** — no matching evidence found")
    st.divider()
    st.caption("Frontend currently running on mock data (M1/M2/M5 not yet integrated).")

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What did the original Transformer paper propose?",
)
submitted = st.button("Analyze", type="primary")

if submitted and question.strip():
    with st.spinner("Generating answer and checking claims..."):
        # TODO(M5 integration): replace with a real backend call, e.g.
        #   import requests
        #   result = requests.post(f"{API_BASE_URL}/analyze", json={"question": question}).json()
        result = get_mock_response(question)

    st.subheader("LLM Answer")
    st.write(result["answer"])

    claims = result.get("claims", [])
    if claims:
        supported = sum(1 for c in claims if c["verdict"] == "Supported")
        contradicted = sum(1 for c in claims if c["verdict"] == "Contradicted")
        unverifiable = sum(1 for c in claims if c["verdict"] == "Unverifiable")

        st.subheader("Claim Breakdown")
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Supported", supported)
        col2.metric("❌ Contradicted", contradicted)
        col3.metric("⚠️ Unverifiable", unverifiable)

        st.divider()
        for claim in claims:
            render_claim_card(claim)
    else:
        st.info("No claims were extracted from this answer.")

elif submitted:
    st.warning("Please enter a question first.")
