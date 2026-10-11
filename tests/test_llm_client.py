"""Tests for the DeepSeek integration in the shared LLM client

Every client built here passes `fallbacks=()` unless the test is about
fallbacks: with the configured chain left on, a test that mocks only the
primary handle would let the next target make a real API call.
"""

import os
from unittest.mock import MagicMock, AsyncMock

import httpx
import openai
import pytest

from src.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from src.shared.llm.client import LLMClient

FALLBACK = {
    "name": "spare",
    "model": "spare-model",
    "api_key": "spare-key",
    "base_url": "https://api.spare.test/v1",
}


def make_chat_response(text: str) -> MagicMock:
    """Build a fake openai.chat.completions.create response"""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    return response


def make_api_error(message: str = "boom") -> openai.APIError:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return openai.APIError(message, request, body=None)


def make_status_error(status: int, message: str = "boom") -> openai.APIStatusError:
    """Build the SDK error class that goes with an HTTP status"""
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(status, request=request)
    error_class = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        403: openai.PermissionDeniedError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
    }[status]
    return error_class(message, response=response, body=None)


def make_rate_limit_error(message: str = "rate limited") -> openai.RateLimitError:
    return make_status_error(429, message)


def client_with_fallback(**kwargs) -> LLMClient:
    """A client whose primary and fallback handles are both mocked"""
    client = LLMClient(api_key="test-key", fallbacks=[FALLBACK], **kwargs)
    # Touching the property builds the fallback's handle once, so the test
    # and the client mock the same object.
    client._targets[1]._sync_client = MagicMock()
    return client


def skip_sleep(seconds) -> None:
    """Stand-in for time.sleep that skips the actual wait"""
    return None


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """Retries are tested for behavior, not for how long they wait"""
    monkeypatch.setattr("time.sleep", skip_sleep)


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
        client = LLMClient(api_key="test-key", fallbacks=())
        client.client.chat.completions.create = MagicMock(
            return_value=make_chat_response("hello from deepseek")
        )

        result = client.call("ping")

        assert result == "hello from deepseek"
        assert client.call_count == 1
        assert client.last_call_time is not None

    def test_call_uses_configured_model_and_messages(self):
        client = LLMClient(
            api_key="test-key", model="deepseek-v4-pro", fallbacks=()
        )
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
        client = LLMClient(api_key="test-key", fallbacks=())
        create_mock = MagicMock(return_value=make_chat_response("ok"))
        client.client.chat.completions.create = create_mock

        client.call("hi")

        _, kwargs = create_mock.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_call_retries_on_rate_limit_then_succeeds(self):
        client = LLMClient(api_key="test-key", fallbacks=())
        client.client.chat.completions.create = MagicMock(
            side_effect=[make_rate_limit_error(), make_chat_response("recovered")]
        )

        result = client.call("ping", max_retries=3)

        assert result == "recovered"
        assert client.client.chat.completions.create.call_count == 2
        assert client.retry_count == 1

    def test_call_raises_after_exhausting_retries(self):
        client = LLMClient(api_key="test-key", fallbacks=())
        client.client.chat.completions.create = MagicMock(
            side_effect=make_api_error("persistent failure")
        )

        with pytest.raises(openai.APIError):
            client.call("ping", max_retries=2)

        assert client.client.chat.completions.create.call_count == 2


class TestLLMClientRetryPolicy:
    """Which failures earn another attempt, and which do not"""

    @pytest.mark.parametrize("status", [429, 500])
    def test_transient_statuses_are_retried(self, status):
        client = LLMClient(api_key="test-key", fallbacks=())
        client.client.chat.completions.create = MagicMock(
            side_effect=make_status_error(status)
        )

        with pytest.raises(openai.APIStatusError):
            client.call("ping", max_retries=3)

        assert client.client.chat.completions.create.call_count == 3

    def test_a_malformed_request_is_neither_retried_nor_passed_on(self):
        """Every provider rejects a bad request, so replaying it only burns
        quota -- and hides the bug behind three identical failures"""
        client = client_with_fallback()
        client.client.chat.completions.create = MagicMock(
            side_effect=make_status_error(400, "invalid 'messages'")
        )
        spare = MagicMock()
        client._targets[1].sync_client.chat.completions.create = spare

        with pytest.raises(openai.BadRequestError):
            client.call("ping", max_retries=3)

        assert client.client.chat.completions.create.call_count == 1
        assert spare.call_count == 0

    def test_an_unknown_model_400_moves_to_the_next_target(self):
        """DeepSeek reports an unknown model as 400, not 404 -- that is a
        provider that cannot serve the request, not a bad request"""
        client = client_with_fallback(model="model-that-does-not-exist")
        client.client.chat.completions.create = MagicMock(
            side_effect=make_status_error(
                400,
                "The supported API model names are deepseek-flash, "
                "deepseek-v4-pro, but you passed model-that-does-not-exist.",
            )
        )
        client._targets[1].sync_client.chat.completions.create = MagicMock(
            return_value=make_chat_response("from the spare")
        )

        assert client.call("ping") == "from the spare"
        assert client.client.chat.completions.create.call_count == 1

    @pytest.mark.parametrize("status", [401, 403, 404])
    def test_a_provider_fault_skips_retries(self, status):
        """A bad key or an unknown model will not fix itself; the chain moves
        on after one attempt instead of backing off three times"""
        client = LLMClient(api_key="test-key", fallbacks=())
        client.client.chat.completions.create = MagicMock(
            side_effect=make_status_error(status)
        )

        with pytest.raises(openai.APIStatusError):
            client.call("ping", max_retries=3)

        assert client.client.chat.completions.create.call_count == 1

    def test_a_model_quirk_does_not_spend_an_attempt(self):
        """Adapting the request and retrying is free: otherwise two quirks
        would exhaust a three-attempt budget before the first real try"""
        client = LLMClient(api_key="test-key", fallbacks=())
        client.client.chat.completions.create = MagicMock(
            side_effect=[
                make_status_error(400, "use 'max_completion_tokens' instead"),
                make_status_error(400, "'temperature' is not supported"),
                make_rate_limit_error(),
                make_chat_response("adapted"),
            ]
        )

        result = client.call("ping", max_retries=2)

        assert result == "adapted"
        _, kwargs = client.client.chat.completions.create.call_args
        assert kwargs["max_completion_tokens"] == client.max_tokens
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs

    def test_retry_after_header_is_honored(self):
        """The provider's own number beats our backoff curve"""
        client = LLMClient(api_key="test-key", fallbacks=())
        request = httpx.Request("POST", "https://api.deepseek.com/v1")
        error = openai.RateLimitError(
            "slow down",
            response=httpx.Response(
                429, request=request, headers={"retry-after": "7"}
            ),
            body=None,
        )

        assert client._retry_delay(0, error) == 7.0

    def test_backoff_grows_and_is_jittered(self):
        client = LLMClient(api_key="test-key", fallbacks=())
        error = make_api_error()

        first = [client._retry_delay(0, error) for _ in range(20)]
        second = [client._retry_delay(1, error) for _ in range(20)]

        assert all(0.5 <= delay <= 1.0 for delay in first)
        assert all(1.0 <= delay <= 2.0 for delay in second)
        # Jitter: identical waits would make a parallel pool retry in lockstep
        assert len(set(first)) > 1


class TestLLMClientFallback:
    """Falling over to the next provider in the chain"""

    def test_fallback_answers_when_the_primary_is_exhausted(self):
        client = client_with_fallback()
        client.client.chat.completions.create = MagicMock(
            side_effect=make_rate_limit_error()
        )
        client._targets[1].sync_client.chat.completions.create = MagicMock(
            return_value=make_chat_response("from the spare")
        )

        result = client.call("ping", max_retries=2)

        assert result == "from the spare"
        assert client.client.chat.completions.create.call_count == 2
        assert client.fallback_count == 1

    def test_the_fallback_is_called_with_its_own_model(self):
        client = client_with_fallback()
        client.client.chat.completions.create = MagicMock(
            side_effect=make_status_error(401)
        )
        create_mock = MagicMock(return_value=make_chat_response("ok"))
        client._targets[1].sync_client.chat.completions.create = create_mock

        client.call("ping")

        _, kwargs = create_mock.call_args
        assert kwargs["model"] == "spare-model"

    def test_quirks_are_not_carried_across_providers(self):
        """What the primary rejects says nothing about the fallback"""
        client = client_with_fallback()
        client.client.chat.completions.create = MagicMock(
            side_effect=[
                make_status_error(400, "use 'max_completion_tokens' instead"),
                make_status_error(500),
                make_status_error(500),
            ]
        )
        create_mock = MagicMock(return_value=make_chat_response("ok"))
        client._targets[1].sync_client.chat.completions.create = create_mock

        client.call("ping", max_retries=2)

        _, kwargs = create_mock.call_args
        assert kwargs["max_tokens"] == client.max_tokens
        assert client._targets[0].max_tokens_param == "max_completion_tokens"

    def test_the_last_error_is_raised_once_the_chain_is_exhausted(self):
        """Not a generic RuntimeError: the caller needs the real cause"""
        client = client_with_fallback()
        client.client.chat.completions.create = MagicMock(
            side_effect=make_rate_limit_error()
        )
        client._targets[1].sync_client.chat.completions.create = MagicMock(
            side_effect=make_status_error(500, "spare is down too")
        )

        with pytest.raises(openai.InternalServerError, match="spare is down"):
            client.call("ping", max_retries=2)

    def test_fallbacks_can_be_turned_off(self):
        client = LLMClient(api_key="test-key", fallbacks=())

        assert client.get_stats()["fallback_chain"] == []

    def test_a_fallback_that_repeats_the_primary_is_dropped(self):
        """Retrying the same model at the same endpoint is not a fallback"""
        client = LLMClient(
            api_key="test-key",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
            fallbacks=[
                {
                    "name": "same",
                    "model": "deepseek-v4-pro",
                    "api_key": "other-key",
                    "base_url": "https://api.deepseek.com/",
                },
                FALLBACK,
            ],
        )

        assert client.get_stats()["fallback_chain"] == ["spare/spare-model"]

    def test_an_uncredentialed_fallback_is_dropped(self):
        client = LLMClient(
            api_key="test-key",
            fallbacks=[{"name": "empty", "model": "m", "api_key": "", "base_url": "u"}],
        )

        assert client.get_stats()["fallback_chain"] == []


class TestLLMClientAsyncCall:
    """Test the asynchronous acall() path"""

    @pytest.mark.asyncio
    async def test_acall_returns_response_text(self):
        client = LLMClient(api_key="test-key", fallbacks=())
        client.async_client.chat.completions.create = AsyncMock(
            return_value=make_chat_response("async hello")
        )

        result = await client.acall("ping")

        assert result == "async hello"
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_acall_retries_then_falls_back(self, monkeypatch):
        async def no_sleep(seconds):
            return None

        monkeypatch.setattr("asyncio.sleep", no_sleep)
        client = LLMClient(api_key="test-key", fallbacks=[FALLBACK])
        client.async_client.chat.completions.create = AsyncMock(
            side_effect=make_rate_limit_error()
        )
        client._targets[1]._async_client = MagicMock()
        client._targets[1].async_client.chat.completions.create = AsyncMock(
            return_value=make_chat_response("async spare")
        )

        result = await client.acall("ping", max_retries=2)

        assert result == "async spare"
        assert client.fallback_count == 1


class TestLLMClientStats:
    def test_get_stats_reports_model_and_call_count(self):
        client = LLMClient(
            api_key="test-key", model="deepseek-v4-pro", fallbacks=()
        )
        client.client.chat.completions.create = MagicMock(
            return_value=make_chat_response("ok")
        )

        client.call("ping")
        stats = client.get_stats()

        assert stats["model"] == "deepseek-v4-pro"
        assert stats["total_calls"] == 1
        assert stats["last_call_time"] is not None

    def test_stats_name_the_model_that_actually_answered(self):
        """`model` is what was asked for; a fallback makes them differ"""
        client = client_with_fallback(model="deepseek-v4-pro")
        client.client.chat.completions.create = MagicMock(
            side_effect=make_status_error(404)
        )
        client._targets[1].sync_client.chat.completions.create = MagicMock(
            return_value=make_chat_response("ok")
        )

        client.call("ping")
        stats = client.get_stats()

        assert stats["model"] == "deepseek-v4-pro"
        assert stats["calls_by_model"] == {"spare-model": 1}
        assert stats["fallback_calls"] == 1


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
