"""LLM client wrapper with retry and rate-limit handling"""

import asyncio
from typing import Any, Optional
from datetime import datetime
import anthropic

from src.config import (
    LLM_MODEL,
    LLM_API_KEY,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
)
from src.shared.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified wrapper for LLM API calls with retry and rate-limit handling"""

    def __init__(
        self,
        model: str = LLM_MODEL,
        api_key: str = LLM_API_KEY,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: int = LLM_TIMEOUT,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
        self.call_count = 0
        self.last_call_time = None

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
        for attempt in range(max_retries):
            try:
                logger.debug(f"LLM call attempt {attempt + 1}/{max_retries}")

                messages = [{"role": "user", "content": prompt}]

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=messages,
                    system=system_prompt,
                    **kwargs,
                )

                self.call_count += 1
                self.last_call_time = datetime.now()

                logger.debug(
                    f"LLM call successful. Total calls: {self.call_count}"
                )
                return response.content[0].text

            except anthropic.RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limited. Waiting {wait_time}s before retry..."
                )
                asyncio.run(asyncio.sleep(wait_time))

            except anthropic.APIError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API error on attempt {attempt + 1}: {e}")
                    asyncio.run(asyncio.sleep(2 ** attempt))
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
        for attempt in range(max_retries):
            try:
                logger.debug(f"Async LLM call attempt {attempt + 1}/{max_retries}")

                messages = [{"role": "user", "content": prompt}]

                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=messages,
                    system=system_prompt,
                    **kwargs,
                )

                self.call_count += 1
                self.last_call_time = datetime.now()

                logger.debug(
                    f"Async LLM call successful. Total calls: {self.call_count}"
                )
                return response.content[0].text

            except anthropic.RateLimitError as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Rate limited. Waiting {wait_time}s before retry..."
                )
                await asyncio.sleep(wait_time)

            except anthropic.APIError as e:
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
