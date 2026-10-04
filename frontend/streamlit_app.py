"""
Streamlit frontend for the LLM Hallucination Detection project.

Calls the real FastAPI backend (/verify) instead of static mock data,
and renders each claim with a color-coded verdict, confidence score,
source, and evidence snippet.

Owner: P4
Location in repo: frontend/streamlit_app.py

Run with:
    streamlit run frontend/streamlit_app.py

Requires the backend to be running separately:
    uvicorn backend.app.main:app --reload
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/verify-answer"

VERDICT_STYLE = {
    "supported": {"icon": "✅", "label": "Supported", "color": "#1e7e34", "bg": "#d4edda"},
    "contradicted": {"icon": "❌", "label": "Contradicted", "color": "#a71d2a", "bg": "#f8d7da"},
    "unverifiable": {"icon": "⚠️", "label": "Unverifiable", "color": "#8a6d00", "bg": "#fff3cd"},
}

st.set_page_config(page_title="LLM Hallucination Detector", page_icon="🔍", layout="centered")

st.title("🔍 LLM Hallucination Detector")
st.caption("Ask a question, get an LLM answer, and see each claim checked against independent evidence.")

question = st.text_input("Ask a question", placeholder="e.g. Who built the Eiffel Tower and when?")

if st.button("Verify Answer", type="primary") and question.strip():
    with st.spinner("Generating answer, extracting claims, and verifying evidence... this can take 10-20 seconds"):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=60)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the backend API. Make sure it's running with:\n\n"
                "`uvicorn backend.app.main:app --reload`"
            )
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"Backend returned an error: {e}")
            st.stop()

    st.subheader("LLM Answer")
    st.write(result.get("answer_text", "(no answer returned)"))

    claims = result.get("claims", [])

    if not claims:
        st.warning("No claims were extracted from this answer.")
    else:
        # Summary counts
        supported = sum(1 for c in claims if c.get("verdict") == "supported")
        contradicted = sum(1 for c in claims if c.get("verdict") == "contradicted")
        unverifiable = sum(1 for c in claims if c.get("verdict") == "unverifiable")

        st.subheader("Claim Breakdown")
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Supported", supported)
        col2.metric("❌ Contradicted", contradicted)
        col3.metric("⚠️ Unverifiable", unverifiable)

        st.divider()

        for claim in claims:
            verdict = claim.get("verdict", "unverifiable")
            style = VERDICT_STYLE.get(verdict, VERDICT_STYLE["unverifiable"])
            confidence = claim.get("confidence")
            confidence_pct = f"{round(confidence * 100)}%" if confidence is not None else "N/A"

            st.markdown(
                f"""
                <div style="border-left: 5px solid {style['color']}; background-color: {style['bg']};
                            padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                    <div style="font-weight: 600; margin-bottom: 4px; color: #1a1a1a;">{claim.get('claim_text', '')}</div>
                    <div style="color: {style['color']}; font-weight: 600;">
                        {style['icon']} {style['label']} · {confidence_pct} confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            evidence_snippet = claim.get("evidence_snippet")
            source = claim.get("source", "Unknown")
            evidence_url = claim.get("evidence_url")

            with st.expander("View evidence"):
                if evidence_snippet:
                    # Explicit colors here (not relying on Streamlit's theme
                    # default text color) so the evidence text stays readable
                    # in both light and dark mode, instead of blending into
                    # the background.
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; color: #1a1a1a;
                                    padding: 10px 14px; border-radius: 6px;
                                    font-style: italic; margin-bottom: 8px;">
                            {evidence_snippet}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color: #888;'>No matching evidence found.</span>",
                        unsafe_allow_html=True,
                    )

                if evidence_url:
                    st.markdown(
                        f"<span style='color: #1a1a1a;'><b>Source:</b> "
                        f"<a href='{evidence_url}' target='_blank'>{source}</a></span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<span style='color: #1a1a1a;'><b>Source:</b> {source}</span>",
                        unsafe_allow_html=True,
                    )

elif question.strip() == "" and st.session_state.get("_submitted"):
    st.warning("Please enter a question first.")