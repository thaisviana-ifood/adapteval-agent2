"""Shared plumbing for the LLM classifiers used by context analysis

Every classifier in this project has the same shape: format a
PromptManager-managed prompt, call the LLM with it, pull a JSON object out of
the response, and degrade to a non-LLM default if any of those steps fail --
recording the prompt's name in a `classifier_failures` list rather than
raising, so one unavailable field never takes down the whole analysis.

`JSONClassifier` owns that loop, and the module-level `clean_*` helpers
normalize the values a classifier hands back (which are model output, so they
may be the wrong type, out of range, duplicated or blank).
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.config import CLASSIFIER_LLM_MAX_TOKENS, CLASSIFIER_LLM_TEMPERATURE
from src.shared.llm.client import LLMClient
from src.shared.llm.prompts import PromptManager
from src.shared.logger import get_logger

logger = get_logger(__name__)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class JSONClassifier:
    """Runs one prompt at a time and returns its parsed JSON object"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        self.llm_client = llm_client or LLMClient(
            max_tokens=CLASSIFIER_LLM_MAX_TOKENS,
            temperature=CLASSIFIER_LLM_TEMPERATURE,
        )
        self.prompt_manager = prompt_manager or PromptManager()

    def call(
        self,
        template_name: str,
        trace_name: str,
        failures: List[str],
        **variables: Any,
    ) -> Optional[Dict[str, Any]]:
        """Run one classifier prompt and return its parsed JSON object

        Returns None (and appends `template_name` to `failures`) if the prompt
        is missing, the call fails, or the response isn't a parseable JSON
        object.
        """
        prompt = self.prompt_manager.format_prompt(template_name, **variables)
        if not prompt:
            logger.warning(f"Prompt '{template_name}' unavailable")
            failures.append(template_name)
            return None

        try:
            response_text = self.llm_client.call(
                prompt,
                name=trace_name,
                langfuse_prompt=self.prompt_manager.get_prompt_client(
                    template_name
                ),
            )
        except Exception as e:
            logger.warning(f"Classifier '{template_name}' call failed: {e}")
            failures.append(template_name)
            return None

        parsed = self.parse_json_object(response_text, template_name)
        if parsed is None:
            failures.append(template_name)
        return parsed

    @staticmethod
    def parse_json_object(
        response_text: Optional[str], template_name: str
    ) -> Optional[Dict[str, Any]]:
        """Extract the JSON object from a classifier response"""
        if not response_text:
            logger.warning(f"Classifier '{template_name}' returned nothing")
            return None

        match = _JSON_OBJECT_PATTERN.search(response_text)
        if not match:
            logger.warning(
                f"Classifier '{template_name}' returned no JSON object"
            )
            return None

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse '{template_name}' JSON: {e}")
            return None

        if not isinstance(parsed, dict):
            logger.warning(
                f"Classifier '{template_name}' returned "
                f"{type(parsed).__name__}, expected an object"
            )
            return None

        return parsed


def clean_string_list(value: Any, limit: int) -> List[str]:
    """Normalize a classifier's string list: de-duplicated, capped, no blanks"""
    if not isinstance(value, list):
        return []

    cleaned: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, (str, int, float)):
            continue
        text = str(item).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        cleaned.append(text)
        if len(cleaned) >= limit:
            break

    return cleaned


def clean_score(value: Any, default: float) -> float:
    """Coerce a classifier's 0-1 score, clamping out-of-range values"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(score, 0.0), 1.0)


def clean_confidence(value: Any) -> float:
    """Coerce a classifier's confidence; absent confidence means 0.0"""
    return clean_score(value, default=0.0)
