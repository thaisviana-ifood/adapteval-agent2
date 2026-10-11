"""LLM client wrapper with retry, fallback and rate-limit handling

One call goes through a chain of targets: the configured model first, then
each provider in `LLM_FALLBACK_PROVIDERS`. Every target gets its own retry
budget (`LLM_MAX_RETRIES`), and the chain moves on only when a target has
run out of attempts or cannot serve the request at all.

What is retried, and what is not, is the whole design -- retrying the wrong
error wastes a quota and hides a bug:

  - Retried on the same target: rate limits (429, honoring Retry-After),
    timeouts, dropped connections, and 5xx/408/409/425. These are transient
    by definition, with exponential backoff plus jitter so a parallel batch
    does not retry in lockstep.
  - Not retried, straight to the next target: a bad key (401), a forbidden
    key (403) and an unknown model (404). The request is fine; this provider
    cannot serve it, and waiting will not change that.
  - Not retried at all: 400/422. A malformed request is malformed for every
    provider, so it is raised immediately rather than replayed three times
    against each one. Two kinds of 400 are read differently: the "this model
    does not support that parameter" ones `_adapt_to_model_quirk` knows,
    which are fixed and retried for free (no attempt spent), and a complaint
    about the model name itself (DeepSeek answers an unknown model with 400,
    not 404), which moves to the next target.

A fallback answers with a different vendor's model. That is the right trade
for a pipeline stage that needs an answer and the wrong one for the jury
(independence is the point) and for an accuracy measurement (the report
names one model), so both construct their clients with `fallbacks=()`.
Every switch is logged, and `get_stats()` reports which models actually
answered, so a silent substitution cannot go unnoticed.
"""

import asyncio
import random
import time
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from datetime import datetime
import openai
from langfuse.openai import AsyncOpenAI, OpenAI

from src.config import (
    LLM_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_FALLBACK_PROVIDERS,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_RETRY_BASE_DELAY,
    LLM_RETRY_MAX_DELAY,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)
from src.shared.llm.langfuse_client import get_langfuse_client
from src.shared.logger import get_logger

logger = get_logger(__name__)

# Status codes worth another attempt against the same provider.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Status codes that condemn the provider, not the request.
_PROVIDER_FAULT_STATUS = frozenset({401, 403, 404})

# What to do with a failed attempt.
_ADAPT = "adapt"  # a known model quirk: fix the request and retry for free
_RETRY = "retry"  # transient: back off and try the same target again
_NEXT = "next"  # this target cannot serve the request: fall back
_FATAL = "fatal"  # the request itself is broken: raise now


class _Target:
    """One model the client can call, with its own SDK handles and quirks

    Quirks are per-model (`max_tokens` vs `max_completion_tokens`, whether a
    custom temperature is accepted), so they live here rather than on the
    client -- what the primary model rejects says nothing about what a
    fallback accepts.

    A fallback's handles are built on first use: configuring a fallback
    costs nothing until something actually fails over to it.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: int,
        name: str,
        is_fallback: bool,
        sync_client: Optional[Any] = None,
        async_client: Optional[Any] = None,
    ):
        self.model = model
        self.name = name
        self.is_fallback = is_fallback
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._sync_client = sync_client
        self._async_client = async_client

        # Lazily detected from the API's own 400s and cached per model.
        self.max_tokens_param = "max_tokens"
        self.omit_temperature = False

    @property
    def sync_client(self) -> Any:
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._sync_client

    @property
    def async_client(self) -> Any:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._async_client

    @property
    def identity(self) -> Tuple[str, str]:
        """What makes two targets the same call: model plus endpoint"""
        return (self.model, self._base_url.rstrip("/"))

    def __str__(self) -> str:
        return f"{self.name}/{self.model}" if self.name else self.model


class LLMClient:
    """Unified wrapper for LLM API calls with retry and rate-limit handling"""

    def __init__(
        self,
        model: str = LLM_MODEL,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: int = LLM_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
        fallbacks: Optional[Sequence[Mapping[str, str]]] = None,
    ):
        """
        Args:
            model/api_key/base_url: The primary target.
            temperature/max_tokens/timeout: Request parameters, shared by
                every target in the chain.
            max_retries: Attempts per target (not per call) before the chain
                moves on. Overridable per call.
            fallbacks: Providers to try after the primary, as
                {"name", "model", "api_key", "base_url"} mappings. Defaults
                to `LLM_FALLBACK_PROVIDERS`; pass `()` to require this exact
                model, which is what the jury and the accuracy scripts do.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # Ensure the shared Langfuse client is registered (with the right host)
        # before any traced OpenAI call is made.
        get_langfuse_client()
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        primary = _Target(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            name="primary",
            is_fallback=False,
            # The primary's handles are the public `client`/`async_client`,
            # so patching either one (tests do) reaches the real call path.
            sync_client=self.client,
            async_client=self.async_client,
        )
        self._targets = [primary] + self._build_fallbacks(
            LLM_FALLBACK_PROVIDERS if fallbacks is None else fallbacks,
            primary,
        )

        self.call_count = 0
        self.retry_count = 0
        self.fallback_count = 0
        self.calls_by_model: Counter = Counter()
        self.last_call_time = None

    def _build_fallbacks(
        self, providers: Sequence[Mapping[str, str]], primary: _Target
    ) -> List[_Target]:
        """Turn configured providers into targets, skipping useless ones"""
        targets: List[_Target] = []
        seen = {primary.identity}

        for provider in providers:
            model = provider.get("model") or ""
            api_key = provider.get("api_key") or ""
            base_url = provider.get("base_url") or ""
            if not model or not api_key or not base_url:
                continue

            target = _Target(
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout=self.timeout,
                name=provider.get("name") or "fallback",
                is_fallback=True,
            )
            # Falling back to the model that just failed is not a fallback.
            if target.identity in seen:
                continue
            seen.add(target.identity)
            targets.append(target)

        if targets:
            logger.debug(
                f"LLM fallback chain after {primary.model}: "
                f"{[str(target) for target in targets]}"
            )
        return targets

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def _adapt_to_model_quirk(self, target: _Target, error: Exception) -> bool:
        """Recognize a handful of 'unsupported parameter for this model' 400s
        and adjust future requests to work around them.

        Returns True if the error was recognized (caller should retry
        immediately), False otherwise (caller should treat it as a normal
        API error).
        """
        message = str(error)

        if target.max_tokens_param == "max_tokens" and "max_completion_tokens" in message:
            logger.info(
                f"Model '{target.model}' rejects 'max_tokens'; "
                "switching to 'max_completion_tokens'"
            )
            target.max_tokens_param = "max_completion_tokens"
            return True

        if (
            not target.omit_temperature
            and "'temperature'" in message
            and "not support" in message
        ):
            logger.info(
                f"Model '{target.model}' rejects a custom 'temperature'; "
                "omitting it and using the model's default"
            )
            target.omit_temperature = True
            return True

        return False

    def _classify_failure(self, target: _Target, error: Exception) -> str:
        """Decide what a failed attempt means: _ADAPT/_RETRY/_NEXT/_FATAL"""
        if isinstance(error, openai.APIStatusError):
            status = getattr(error, "status_code", None)

            if status in (400, 422) or isinstance(
                error, (openai.BadRequestError, openai.UnprocessableEntityError)
            ):
                if self._adapt_to_model_quirk(target, error):
                    return _ADAPT
                # Not every 400 is our fault: DeepSeek answers an unknown
                # model name with 400, not 404 ("The supported API model
                # names are ..., but you passed ..."), and that is precisely
                # a provider that cannot serve the request.
                if self._names_an_unknown_model(target, error):
                    return _NEXT
                return _FATAL

            if status in _RETRYABLE_STATUS or isinstance(
                error, (openai.RateLimitError, openai.InternalServerError)
            ):
                return _RETRY

            if status in _PROVIDER_FAULT_STATUS:
                return _NEXT

            # An unmapped status: the provider answered something we do not
            # model, so try the next one rather than hammering this one.
            return _NEXT

        if isinstance(error, openai.APIConnectionError):
            # Covers APITimeoutError.
            return _RETRY

        if isinstance(error, openai.APIError):
            # Base-class APIError carries no status (the SDK raises it for
            # malformed/unknown responses); it may be transient, so retry.
            return _ADAPT if self._adapt_to_model_quirk(target, error) else _RETRY

        return _FATAL

    @staticmethod
    def _names_an_unknown_model(target: _Target, error: Exception) -> bool:
        """Does this 4xx complain about the model rather than the request?

        Read narrowly on purpose: the message has to be about a model *and*
        either name the one that was sent or say it is not supported.
        Anything looser would turn a malformed-request bug into a silent
        tour of every provider.
        """
        message = str(error).lower()
        if "model" not in message:
            return False
        return target.model.lower() in message or any(
            phrase in message
            for phrase in (
                "not exist",
                "not found",
                "not support",
                "unsupported",
                "unknown model",
                "invalid model",
            )
        )

    def _retry_delay(self, attempt: int, error: Exception) -> float:
        """Exponential backoff with jitter, or the provider's own Retry-After

        Jitter matters because these calls run in parallel pools: without it
        every worker that hit the same rate limit retries in the same
        instant and trips it again.
        """
        retry_after = self._retry_after_seconds(error)
        if retry_after is not None:
            return min(retry_after, LLM_RETRY_MAX_DELAY)

        window = min(LLM_RETRY_BASE_DELAY * (2 ** attempt), LLM_RETRY_MAX_DELAY)
        return random.uniform(window / 2, window)

    @staticmethod
    def _retry_after_seconds(error: Exception) -> Optional[float]:
        """The Retry-After the provider asked for, if it sent a usable one"""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        try:
            return max(float(headers.get("retry-after")), 0.0)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Request building and reading
    # ------------------------------------------------------------------

    def _request_kwargs(
        self, target: _Target, messages: list, **kwargs: Any
    ) -> dict:
        """Build chat.completions.create kwargs, applying any detected quirks"""
        request_kwargs = {
            "model": target.model,
            "messages": messages,
            target.max_tokens_param: self.max_tokens,
        }
        if not target.omit_temperature:
            request_kwargs["temperature"] = self.temperature
        request_kwargs.update(kwargs)
        return request_kwargs

    @staticmethod
    def _messages(prompt: str, system_prompt: Optional[str]) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _content_of(self, target: _Target, response: Any) -> str:
        """Pull the message text out of a completion, warning if the budget
        was exhausted before the model wrote any of it.

        A reasoning model spends hidden reasoning tokens before emitting
        visible text, so an undersized max_tokens comes back as
        finish_reason="length" with empty/truncated content rather than as an
        API error. Callers that only see "" can't tell that apart from a model
        that genuinely had nothing to say, so name the cause here.
        """
        choice = response.choices[0]
        content = choice.message.content

        if choice.finish_reason == "length":
            reasoning_tokens = getattr(
                getattr(response.usage, "completion_tokens_details", None),
                "reasoning_tokens",
                None,
            )
            logger.warning(
                f"Model '{target.model}' hit the {self.max_tokens}-token limit "
                f"before finishing its response"
                + (
                    f" ({reasoning_tokens} of them spent on reasoning)"
                    if reasoning_tokens
                    else ""
                )
                + f"; output is {'empty' if not content else 'truncated'}. "
                "Raise the max_tokens budget for this stage."
            )

        return content

    def _record_success(self, target: _Target) -> None:
        self.call_count += 1
        self.calls_by_model[target.model] += 1
        self.last_call_time = datetime.now()
        if target.is_fallback:
            self.fallback_count += 1
            logger.warning(
                f"LLM call answered by fallback {target} "
                f"instead of '{self.model}'"
            )
        logger.debug(f"LLM call successful. Total calls: {self.call_count}")

    def _give_up(self, last_error: Optional[Exception], max_retries: int) -> None:
        """Raise the failure that ended the chain, never a generic one"""
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"Failed to get LLM response after {max_retries} attempts "
            f"on {len(self._targets)} target(s)"
        )

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous LLM call with retry, fallback and rate-limit handling

        Args:
            prompt: User message/prompt
            system_prompt: System instruction (optional)
            max_retries: Attempts per target (default: LLM_MAX_RETRIES)
            **kwargs: Additional parameters to pass to the API

        Returns:
            str: The generated response text

        Raises:
            The last error seen, once every target has been exhausted, or
            immediately for a request no provider could accept.
        """
        messages = self._messages(prompt, system_prompt)
        retries = self.max_retries if max_retries is None else max_retries
        last_error: Optional[Exception] = None

        for target in self._targets:
            attempt = 0
            while attempt < retries:
                try:
                    logger.debug(
                        f"LLM call to {target}, attempt {attempt + 1}/{retries}"
                    )
                    response = target.sync_client.chat.completions.create(
                        **self._request_kwargs(target, messages, **kwargs)
                    )
                    self._record_success(target)
                    return self._content_of(target, response)

                except Exception as error:
                    decision = self._classify_failure(target, error)
                    if decision == _ADAPT:
                        continue  # not an attempt: the request was wrong

                    last_error = error
                    if decision == _FATAL:
                        logger.error(f"LLM call to {target} rejected: {error}")
                        raise
                    if decision == _NEXT:
                        logger.warning(
                            f"Target {target} cannot serve this request "
                            f"({error}); trying the next one"
                        )
                        break

                    attempt += 1
                    if attempt >= retries:
                        logger.warning(
                            f"Target {target} failed {retries} attempt(s): "
                            f"{error}"
                        )
                        break

                    delay = self._retry_delay(attempt - 1, error)
                    self.retry_count += 1
                    logger.warning(
                        f"LLM call to {target} failed ({error}); retrying in "
                        f"{delay:.1f}s"
                    )
                    time.sleep(delay)

        self._give_up(last_error, retries)

    async def acall(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Asynchronous LLM call with retry, fallback and rate-limit handling

        Args:
            prompt: User message/prompt
            system_prompt: System instruction (optional)
            max_retries: Attempts per target (default: LLM_MAX_RETRIES)
            **kwargs: Additional parameters to pass to the API

        Returns:
            str: The generated response text
        """
        messages = self._messages(prompt, system_prompt)
        retries = self.max_retries if max_retries is None else max_retries
        last_error: Optional[Exception] = None

        for target in self._targets:
            attempt = 0
            while attempt < retries:
                try:
                    logger.debug(
                        f"Async LLM call to {target}, "
                        f"attempt {attempt + 1}/{retries}"
                    )
                    response = await target.async_client.chat.completions.create(
                        **self._request_kwargs(target, messages, **kwargs)
                    )
                    self._record_success(target)
                    return self._content_of(target, response)

                except Exception as error:
                    decision = self._classify_failure(target, error)
                    if decision == _ADAPT:
                        continue

                    last_error = error
                    if decision == _FATAL:
                        logger.error(f"LLM call to {target} rejected: {error}")
                        raise
                    if decision == _NEXT:
                        logger.warning(
                            f"Target {target} cannot serve this request "
                            f"({error}); trying the next one"
                        )
                        break

                    attempt += 1
                    if attempt >= retries:
                        logger.warning(
                            f"Target {target} failed {retries} attempt(s): "
                            f"{error}"
                        )
                        break

                    delay = self._retry_delay(attempt - 1, error)
                    self.retry_count += 1
                    logger.warning(
                        f"Async LLM call to {target} failed ({error}); "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        self._give_up(last_error, retries)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM calls

        `model` is the model that was asked for; `calls_by_model` is what
        actually answered, which differ as soon as a fallback steps in.
        """
        return {
            "total_calls": self.call_count,
            "last_call_time": self.last_call_time,
            "model": self.model,
            "calls_by_model": dict(self.calls_by_model),
            "fallback_calls": self.fallback_count,
            "retries": self.retry_count,
            "fallback_chain": [
                str(target) for target in self._targets if target.is_fallback
            ],
        }
