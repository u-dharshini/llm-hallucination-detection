"""
Tests for answer_generation.py.

These tests MOCK the OpenAI/Groq API calls, so they run instantly and
don't need a real API key or internet access. This is intentional —
you should never depend on a live paid API call inside a test suite.

Run with:
    python -m pytest backend/tests/test_answer_generation.py -v
"""

from unittest.mock import patch, MagicMock

import pytest

from backend.app.modules.answer_generation import generate_answer


def _make_fake_openai_response(text: str):
    """Builds a fake response object shaped like the real OpenAI SDK's response."""
    fake_message = MagicMock()
    fake_message.content = text
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    return fake_response


def test_generate_answer_returns_text(monkeypatch):
    """A normal question should return the (mocked) model's text answer."""
    monkeypatch.setattr(
        "backend.app.modules.answer_generation.settings.LLM_PROVIDER", "openai"
    )
    monkeypatch.setattr(
        "backend.app.modules.answer_generation.settings.OPENAI_API_KEY", "fake-key"
    )

    fake_response = _make_fake_openai_response("The Eiffel Tower was built in 1887.")

    with patch("openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        mock_openai_class.return_value = mock_client

        result = generate_answer("Who built the Eiffel Tower?")

    assert result == "The Eiffel Tower was built in 1887."


def test_empty_question_raises_error():
    """An empty or whitespace-only question should be rejected before calling any API."""
    with pytest.raises(ValueError):
        generate_answer("")

    with pytest.raises(ValueError):
        generate_answer("   ")


def test_missing_api_key_raises_clear_error(monkeypatch):
    """If no API key is configured, the error message should say so clearly."""
    monkeypatch.setattr(
        "backend.app.modules.answer_generation.settings.LLM_PROVIDER", "openai"
    )
    monkeypatch.setattr(
        "backend.app.modules.answer_generation.settings.OPENAI_API_KEY", ""
    )

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        generate_answer("Any question")


def test_unknown_provider_raises_error(monkeypatch):
    """An invalid LLM_PROVIDER value in .env should fail with a clear message."""
    monkeypatch.setattr(
        "backend.app.modules.answer_generation.settings.LLM_PROVIDER", "not-a-real-provider"
    )

    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        generate_answer("Any question")
