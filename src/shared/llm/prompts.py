"""Unified prompt management system, backed by Langfuse Prompt Management"""

import re
from typing import Any, Dict, List, Optional

from src.shared.llm.langfuse_client import get_langfuse_client
from src.shared.logger import get_logger

logger = get_logger(__name__)

_VARIABLE_PATTERN = re.compile(r"\{\{(.*?)\}\}")

# Local defaults, used as the Langfuse "fallback" and to seed Langfuse on first use.
# Variables use Langfuse's double-curly-brace syntax, e.g. {{conversation}}.
_DEFAULT_PROMPTS: Dict[str, str] = {
    "context_analysis": """Analyze the following conversation context and provide insights about:
1. Structural patterns (turn sequence, speaker patterns)
2. Semantic content (main topics, key phrases, sentiment)
3. Complexity indicators (number of topics, semantic diversity)
4. Intent analysis (user goals, assistant objectives)

Conversation:
{{conversation}}

Rule: regardless of the conversation's original language, write the extracted topics in English.

Provide a structured analysis.""",
    "rule_generation": """Based on the context analysis, generate evaluation rules for assessing responses:

Context Analysis:
{{context_analysis}}

Task: {{task}}

Generate rules for:
1. Task-specific classification criteria
2. Success and failure conditions
3. Context-aware evaluation criteria
4. Minimum heuristic checks

Provide a JSON-formatted set of rules.""",
    "jury_evaluation": """Evaluate the provided response using the following rules and context:

Rules:
{{rules}}

Response to Evaluate:
{{response}}

Context:
{{context}}

Provide your evaluation with:
1. Score (0-10)
2. Confidence level (0-1)
3. Reasoning
4. Strengths
5. Weaknesses""",
    "metrics_aggregation": """Aggregate the following jury evaluations into a final assessment:

Evaluations:
{{evaluations}}

Generate:
1. Weighted average score
2. Confidence estimation
3. Final recommendation
4. Areas of agreement/disagreement""",
    "dynamic_criteria_generation": """You are designing evaluation criteria for an AI response.

Context Analysis:
{{context_summary}}

Objectives:
{{objectives_summary}}

Existing Criteria (do not repeat or duplicate these):
{{existing_criteria}}

Propose additional evaluation criteria that are specific to this context and objectives, and that complement (do not overlap with) the existing criteria above.

Rules:
- Each criterion MUST be a strict boolean (yes/no) check.
- Phrase each "description" as a question that can only be answered True or False.
- Propose at most 3 criteria. Return fewer, or an empty array, if nothing meaningful is missing.
- Do not restate or rephrase an existing criterion.

Return ONLY a JSON array, with no surrounding text, in this exact form:
[{"name": "Short Name", "description": "Does the response ...?"}]""",
    "llm_evaluation": """Evaluate the following response against the given criteria.

Response to Evaluate:
{{response}}

Evaluation Criteria:
{{criteria_text}}

Context:
- Task Type: {{task_type}}
- Complexity: {{complexity_score}}

Provide your evaluation in the following format:
1. Overall Score (0-10):
2. Confidence (0-1):
3. Strengths:
4. Weaknesses:
5. Reasoning:""",
}


def _compile_locally(template: str, variables: Dict[str, Any]) -> str:
    """Substitute {{variable}} placeholders without calling out to Langfuse"""

    def replace(match: "re.Match[str]") -> str:
        key = match.group(1).strip()
        if key not in variables:
            return match.group(0)
        value = variables[key]
        return "" if value is None else str(value)

    return _VARIABLE_PATTERN.sub(replace, template)


class PromptManager:
    """Manages prompt templates via Langfuse Prompt Management, with local fallback defaults"""

    def __init__(self):
        self._local_defaults: Dict[str, str] = dict(_DEFAULT_PROMPTS)
        self._langfuse = get_langfuse_client()

    def get_prompt_client(self, name: str) -> Optional[Any]:
        """Fetch a prompt client for `name` from Langfuse, seeding it there on first use"""
        if self._langfuse is None:
            return None

        fallback = self._local_defaults.get(name)
        try:
            prompt = self._langfuse.get_prompt(
                name,
                fallback=fallback,
                max_retries=1,
                fetch_timeout_seconds=3,
            )
        except Exception as e:
            logger.warning(f"Could not fetch prompt '{name}' from Langfuse: {e}")
            return None

        if prompt.is_fallback and fallback is not None:
            # Prompt doesn't exist in Langfuse yet: publish it so it becomes
            # visible and editable there going forward.
            try:
                self._langfuse.create_prompt(
                    name=name, prompt=fallback, labels=["production"], type="text"
                )
                logger.debug(f"Seeded prompt '{name}' in Langfuse")
            except Exception as e:
                logger.warning(f"Could not seed prompt '{name}' in Langfuse: {e}")

        return prompt

    def get_template(self, template_name: str) -> Optional[str]:
        """Get the raw prompt text for a template, preferring the Langfuse-managed version"""
        prompt = self.get_prompt_client(template_name)
        if prompt is not None:
            return prompt.prompt

        template = self._local_defaults.get(template_name)
        if not template:
            logger.warning(f"Template '{template_name}' not found")
        return template

    def register_template(self, name: str, template: str) -> None:
        """Register a new prompt template locally and publish it to Langfuse"""
        self._local_defaults[name] = template
        if self._langfuse is not None:
            try:
                self._langfuse.create_prompt(
                    name=name, prompt=template, labels=["production"], type="text"
                )
            except Exception as e:
                logger.warning(f"Could not publish prompt '{name}' to Langfuse: {e}")
        logger.debug(f"Registered template: {name}")

    def format_prompt(self, template_name: str, **variables: Any) -> Optional[str]:
        """Format a prompt template with provided variables"""
        prompt = self.get_prompt_client(template_name)
        if prompt is not None:
            try:
                formatted = prompt.compile(**variables)
                logger.debug(f"Formatted prompt: {template_name}")
                return formatted
            except Exception as e:
                logger.error(f"Failed to format prompt '{template_name}': {e}")
                return None

        template = self._local_defaults.get(template_name)
        if not template:
            logger.warning(f"Template '{template_name}' not found")
            return None

        formatted = _compile_locally(template, variables)
        logger.debug(f"Formatted prompt: {template_name}")
        return formatted

    def list_templates(self) -> List[str]:
        """List all available (locally known) templates"""
        return list(self._local_defaults.keys())
