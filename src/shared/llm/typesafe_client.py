"""Client for the Typesafe structured Q&A evaluation API

Typesafe's endpoint (see https://docs.typesafe.ai/introduction/quickstart)
takes a block of text ("state") plus a set of typed questions -- yes/no
("noul"), multiple-choice ("choice"), or a 0-N scale ("score") -- and
returns a structured answer with a confidence/probability distribution per
question. This is not an OpenAI-compatible chat-completions API (different
path, different request/response shape), so it gets its own small client
instead of going through LLMClient.
"""

import time
from typing import Any, Dict

import httpx

from src.shared.logger import get_logger

logger = get_logger(__name__)


class TypesafeClient:
    """Minimal client for the Typesafe structured-evaluation API"""

    def __init__(self, model: str, api_key: str, base_url: str, timeout: int = 60):
        self.model = model
        self.base_url = base_url
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def ask(
        self,
        state: str,
        questions: Dict[str, Dict[str, Any]],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Submit `state` plus typed `questions`, return the parsed response body"""
        payload = {"model": self.model, "state": state, "questions": questions}

        for attempt in range(max_retries):
            try:
                logger.debug(f"Typesafe call attempt {attempt + 1}/{max_retries}")
                response = self._client.post(self.base_url, json=payload)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429 or status >= 500:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(
                            f"Typesafe error {status} on attempt {attempt + 1}. "
                            f"Waiting {wait_time}s before retry..."
                        )
                        time.sleep(wait_time)
                        continue
                logger.error(f"Typesafe request failed: {e} -- {e.response.text}")
                raise

            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Typesafe network error on attempt {attempt + 1}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Typesafe failed after {max_retries} attempts: {e}")
                    raise

        raise RuntimeError(f"Failed to get Typesafe response after {max_retries} attempts")
