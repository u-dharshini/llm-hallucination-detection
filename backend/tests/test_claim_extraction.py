"""
Tests for claim_extraction.py.

Most tests mock the LLM call so they run instantly without a real API key.
Run with:
    python -m pytest backend/tests/test_claim_extraction.py -v
"""

from unittest.mock import patch, MagicMock

from backend.app.modules import claim_extraction as ce
from backend.app.modules.claim_extraction import (
    extract_claims,
    _fallback_sentence_split,
    _parse_claims_json,
    _strip_markdown,
)


def _make_fake_response(text: str):
    fake_message = MagicMock()
    fake_message.content = text
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    return fake_response


def test_extract_claims_splits_into_atomic_facts(monkeypatch):
    """A compound answer should be split into separate Claim objects by the LLM."""
    monkeypatch.setattr(
        "backend.app.modules.claim_extraction.settings.LLM_PROVIDER", "openai"
    )

    fake_json = '["The Eiffel Tower was built in 1887.", "Gustave Eiffel built the Eiffel Tower."]'
    fake_response = _make_fake_response(fake_json)

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        mock_openai_class.return_value = mock_client

        claims = extract_claims("The Eiffel Tower was built in 1887 by Gustave Eiffel.")

    assert len(claims) == 2
    assert claims[0].claim_id == "c1"
    assert claims[0].claim_text == "The Eiffel Tower was built in 1887."
    assert claims[1].claim_id == "c2"
    assert claims[1].claim_text == "Gustave Eiffel built the Eiffel Tower."


def test_extract_claims_handles_markdown_wrapped_json(monkeypatch):
    """Some models wrap JSON in ```json fences despite being told not to."""
    monkeypatch.setattr(
        "backend.app.modules.claim_extraction.settings.LLM_PROVIDER", "openai"
    )

    fake_json = '```json\n["Water boils at 100 degrees Celsius."]\n```'
    fake_response = _make_fake_response(fake_json)

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        mock_openai_class.return_value = mock_client

        claims = extract_claims("Water boils at 100 degrees Celsius.")

    assert len(claims) == 1
    assert claims[0].claim_text == "Water boils at 100 degrees Celsius."


def test_extract_claims_falls_back_on_llm_failure(monkeypatch):
    """If the LLM call raises an exception, fall back to sentence splitting."""
    monkeypatch.setattr(
        "backend.app.modules.claim_extraction.settings.LLM_PROVIDER", "openai"
    )

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_class.return_value = mock_client

        claims = extract_claims("The sky is blue. Water is wet.")

    assert len(claims) == 2
    assert claims[0].claim_text == "The sky is blue."
    assert claims[1].claim_text == "Water is wet."


def test_retries_once_when_llm_returns_empty_list(monkeypatch):
    """Regression: the LLM sometimes replies '[]'. We retry once with a
    stricter prompt instead of immediately falling back."""
    calls = []

    def fake_call(answer, strict=False):
        calls.append(strict)
        return "[]" if not strict else '["The sky is blue."]'

    monkeypatch.setattr(ce, "_call_llm_for_extraction", fake_call)

    claims = extract_claims("The sky is blue today. It is sunny.")

    assert [c.claim_text for c in claims] == ["The sky is blue."]
    assert calls == [False, True]


def test_falls_back_when_llm_returns_empty_list_twice(monkeypatch):
    """If the retry is also empty, use the sentence-split fallback."""
    monkeypatch.setattr(ce, "_call_llm_for_extraction", lambda a, strict=False: "[]")

    claims = extract_claims("The sky is blue. Water is wet.")

    assert [c.claim_text for c in claims] == ["The sky is blue.", "Water is wet."]


def test_duplicate_claims_are_removed(monkeypatch):
    monkeypatch.setattr(
        ce, "_call_llm_for_extraction",
        lambda a, strict=False: '["Same fact.", "Same fact.", "Other fact."]',
    )
    claims = extract_claims("anything")
    assert [c.claim_text for c in claims] == ["Same fact.", "Other fact."]


def test_empty_answer_returns_empty_list():
    """An empty answer should return an empty claims list, not an error."""
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_fallback_sentence_split_directly():
    result = _fallback_sentence_split("First fact. Second fact! Third fact?")
    assert result == ["First fact.", "Second fact!", "Third fact?"]


def test_parse_claims_json_rejects_non_list():
    import pytest
    with pytest.raises(ValueError):
        _parse_claims_json('{"not": "a list"}')


def test_strip_markdown_handles_headers_tables_bullets_and_rules():
    """Regression: Markdown formatting used to leak into claims
    (stray '##', '---', '|', '1.' fragments, table column headers)."""
    md = (
        "**The Eiffel Tower** was built by **Gustave Eiffel**.\n\n"
        "## 1. Why the sky looks blue\n"
        "- **Sunlight is white** – It contains all colors.\n\n"
        "---\n\n"
        "| Step | What happens |\n"
        "|------|--------------|\n"
        "| Entry | Sunlight enters |\n"
    )
    out = _strip_markdown(md)

    assert "*" not in out and "|" not in out and "#" not in out and "---" not in out
    assert "The Eiffel Tower was built by Gustave Eiffel." in out
    assert "Why the sky looks blue." in out          # header kept, number dropped
    assert "Sunlight is white – It contains all colors." in out
    assert "Entry: Sunlight enters." in out           # table data row kept
    assert "Step: What happens" not in out            # table header row dropped