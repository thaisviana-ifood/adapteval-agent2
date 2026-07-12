"""Tests for the DeepSeek integration in the shared LLM client"""

import os
from unittest.mock import MagicMock, AsyncMock

import httpx
import openai
import pytest

from src.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from src.shared.llm.client import LLMClient


def make_chat_response(text: str) -> MagicMock:
    """Build a fake openai.chat.completions.create response"""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    return response


def make_api_error(message: str = "boom") -> openai.APIError:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return openai.APIError(message, request, body=None)


def make_rate_limit_error(message: str = "rate limited") -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError(message, response=response, body=None)


def skip_sleep(coro) -> None:
    """Stand-in for asyncio.run that discards the sleep instead of awaiting it"""
    coro.close()


class TestLLMClientConfiguration:
    """Confirm the client is wired to DeepSeek via src.config / .env"""

    def test_defaults_come_from_deepseek_config(self):
        client = LLMClient()
        assert client.model == LLM_MODEL == "deepseek-v4-pro"
        assert LLM_API_KEY  # loaded from DEEP_SEEK_API_KEY in .env
        assert str(client.client.base_url).rstrip("/") == LLM_BASE_URL.rstrip("/")
        assert str(client.async_client.base_url).rstrip("/") == LLM_BASE_URL.rstrip("/")

    def test_base_url_points_at_deepseek(self):
        client = LLMClient()
        assert "deepseek.com" in str(client.client.base_url)


class TestLLMClientCall:
    """Test the synchronous call() path against a mocked DeepSeek response"""

    def test_call_returns_response_text(self):
        client = LLMClient(api_key="test-key")
        client.client.chat.completions.create = MagicMock(
            return_value=make_chat_response("hello from deepseek")
        )

        result = client.call("ping")

        assert result == "hello from deepseek"
        assert client.call_count == 1
        assert client.last_call_time is not None

    def test_call_uses_configured_model_and_messages(self):
        client = LLMClient(api_key="test-key", model="deepseek-v4-pro")
        create_mock = MagicMock(return_value=make_chat_response("ok"))
        client.client.chat.completions.create = create_mock

        client.call("what is 2+2?", system_prompt="be concise")

        _, kwargs = create_mock.call_args
        assert kwargs["model"] == "deepseek-v4-pro"
        assert kwargs["messages"] == [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "what is 2+2?"},
        ]

    def test_call_without_system_prompt_omits_system_message(self):
        client = LLMClient(api_key="test-key")
        create_mock = MagicMock(return_value=make_chat_response("ok"))
        client.client.chat.completions.create = create_mock

        client.call("hi")

        _, kwargs = create_mock.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_call_retries_on_rate_limit_then_succeeds(self, monkeypatch):
        client = LLMClient(api_key="test-key")
        client.client.chat.completions.create = MagicMock(
            side_effect=[make_rate_limit_error(), make_chat_response("recovered")]
        )
        monkeypatch.setattr("asyncio.run", skip_sleep)

        result = client.call("ping", max_retries=3)

        assert result == "recovered"
        assert client.client.chat.completions.create.call_count == 2

    def test_call_raises_after_exhausting_retries(self, monkeypatch):
        client = LLMClient(api_key="test-key")
        client.client.chat.completions.create = MagicMock(
            side_effect=make_api_error("persistent failure")
        )
        monkeypatch.setattr("asyncio.run", skip_sleep)

        with pytest.raises(openai.APIError):
            client.call("ping", max_retries=2)

        assert client.client.chat.completions.create.call_count == 2


class TestLLMClientAsyncCall:
    """Test the asynchronous acall() path"""

    @pytest.mark.asyncio
    async def test_acall_returns_response_text(self):
        client = LLMClient(api_key="test-key")
        client.async_client.chat.completions.create = AsyncMock(
            return_value=make_chat_response("async hello")
        )

        result = await client.acall("ping")

        assert result == "async hello"
        assert client.call_count == 1


class TestLLMClientStats:
    def test_get_stats_reports_model_and_call_count(self):
        client = LLMClient(api_key="test-key", model="deepseek-v4-pro")
        client.client.chat.completions.create = MagicMock(
            return_value=make_chat_response("ok")
        )

        client.call("ping")
        stats = client.get_stats()

        assert stats["model"] == "deepseek-v4-pro"
        assert stats["total_calls"] == 1
        assert stats["last_call_time"] is not None


@pytest.mark.skipif(
    not os.getenv("RUN_DEEPSEEK_LIVE_TESTS"),
    reason="Set RUN_DEEPSEEK_LIVE_TESTS=1 to hit the real DeepSeek API",
)
class TestLLMClientLiveIntegration:
    """Optional end-to-end check against the real DeepSeek API (opt-in, costs quota)"""

    def test_live_call(self):
        client = LLMClient()
        result = client.call("Reply with exactly the word: pong")
        assert isinstance(result, str)
        assert len(result) > 0
