"""LLM client wrapper with retry and rate-limit handling"""

import asyncio
import time
from typing import Any, Optional
from datetime import datetime
import openai
from langfuse.openai import AsyncOpenAI, OpenAI

from src.config import (
    LLM_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
)
from src.shared.llm.langfuse_client import get_langfuse_client
from src.shared.logger import get_logger

logger = get_logger(__name__)


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
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Ensure the shared Langfuse client is registered (with the right host)
        # before any traced OpenAI call is made.
        get_langfuse_client()
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.call_count = 0
        self.last_call_time = None
        # Some models (e.g. newer OpenAI reasoning models) reject params this
        # client would otherwise always send -- 'max_tokens' has to become
        # 'max_completion_tokens', 'temperature' has to be omitted entirely.
        # Which quirks a given model/provider has isn't knowable up front, so
        # they're detected lazily from the API's own 400s and cached here.
        self._max_tokens_param = "max_tokens"
        self._omit_temperature = False

    def _adapt_to_model_quirk(self, error: Exception) -> bool:
        """Recognize a handful of 'unsupported parameter for this model' 400s
        and adjust future requests to work around them.

        Returns True if the error was recognized (caller should retry
        immediately), False otherwise (caller should treat it as a normal
        API error).
        """
        message = str(error)

        if self._max_tokens_param == "max_tokens" and "max_completion_tokens" in message:
            logger.info(
                f"Model '{self.model}' rejects 'max_tokens'; "
                "switching to 'max_completion_tokens'"
            )
            self._max_tokens_param = "max_completion_tokens"
            return True

        if (
            not self._omit_temperature
            and "'temperature'" in message
            and "not support" in message
        ):
            logger.info(
                f"Model '{self.model}' rejects a custom 'temperature'; "
                "omitting it and using the model's default"
            )
            self._omit_temperature = True
            return True

        return False

    def _request_kwargs(self, messages: list, **kwargs: Any) -> dict:
        """Build chat.completions.create kwargs, applying any detected quirks"""
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            self._max_tokens_param: self.max_tokens,
        }
        if not self._omit_temperature:
            request_kwargs["temperature"] = self.temperature
        request_kwargs.update(kwargs)
        return request_kwargs

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous LLM call with retry logic and rate-limit handling

        Args:
            prompt: User message/prompt
            system_prompt: System instruction (optional)
            max_retries: Number of retry attempts on failure
            **kwargs: Additional parameters to pass to the API

        Returns:
            str: The generated response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                logger.debug(f"LLM call attempt {attempt + 1}/{max_retries}")

                response = self.client.chat.completions.create(
                    **self._request_kwargs(messages, **kwargs)
                )

                self.call_count += 1
                self.last_call_time = datetime.now()

                logger.debug(
                    f"LLM call successful. Total calls: {self.call_count}"
                )
                return response.choices[0].message.content

            except openai.RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limited. Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)

            except openai.APIError as e:
                if self._adapt_to_model_quirk(e):
                    continue
                if attempt < max_retries - 1:
                    logger.warning(f"API error on attempt {attempt + 1}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise

        raise RuntimeError(
            f"Failed to get LLM response after {max_retries} attempts"
        )

    async def acall(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> str:
        """
        Asynchronous LLM call with retry logic and rate-limit handling

        Args:
            prompt: User message/prompt
            system_prompt: System instruction (optional)
            max_retries: Number of retry attempts on failure
            **kwargs: Additional parameters to pass to the API

        Returns:
            str: The generated response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                logger.debug(f"Async LLM call attempt {attempt + 1}/{max_retries}")

                response = await self.async_client.chat.completions.create(
                    **self._request_kwargs(messages, **kwargs)
                )

                self.call_count += 1
                self.last_call_time = datetime.now()

                logger.debug(
                    f"Async LLM call successful. Total calls: {self.call_count}"
                )
                return response.choices[0].message.content

            except openai.RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limited. Waiting {wait_time}s before retry..."
                )
                await asyncio.sleep(wait_time)

            except openai.APIError as e:
                if self._adapt_to_model_quirk(e):
                    continue
                if attempt < max_retries - 1:
                    logger.warning(f"API error on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise

        raise RuntimeError(
            f"Failed to get LLM response after {max_retries} attempts"
        )

    def get_stats(self) -> dict:
        """Get statistics about LLM calls"""
        return {
            "total_calls": self.call_count,
            "last_call_time": self.last_call_time,
            "model": self.model,
        }
