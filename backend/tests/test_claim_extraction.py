"""
Tests for claim_extraction.py.

Most tests mock the LLM call so they run instantly without needing a
real API key. One test checks the fallback sentence-splitter directly,
since that path doesn't need any mocking at all.

Run with:
    python -m pytest backend/tests/test_claim_extraction.py -v
"""

from unittest.mock import patch, MagicMock

from backend.app.modules.claim_extraction import (
    extract_claims,
    _fallback_sentence_split,
    _parse_claims_json,
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
    """Some models wrap JSON in ```json fences despite being told not to — handle it."""
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

    # Fallback should still produce usable claims
    assert len(claims) == 2
    assert claims[0].claim_text == "The sky is blue."
    assert claims[1].claim_text == "Water is wet."


def test_empty_answer_returns_empty_list():
    """An empty answer should return an empty claims list, not an error."""
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_fallback_sentence_split_directly():
    """The fallback splitter itself should break on sentence boundaries."""
    result = _fallback_sentence_split("First fact. Second fact! Third fact?")
    assert result == ["First fact.", "Second fact!", "Third fact?"]


def test_parse_claims_json_rejects_non_list():
    """A JSON object instead of a list should raise, triggering the fallback upstream."""
    import pytest
    with pytest.raises(ValueError):
        _parse_claims_json('{"not": "a list"}')
