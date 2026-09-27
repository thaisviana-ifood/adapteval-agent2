"""Unified prompt management system, backed by Langfuse Prompt Management"""

import re
from typing import Any, Dict, List, Optional

from src.shared.llm.langfuse_client import get_langfuse_client
from src.shared.logger import get_logger

logger = get_logger(__name__)

_VARIABLE_PATTERN = re.compile(r"\{\{(.*?)\}\}")

# Local defaults, used as the Langfuse "fallback" and to seed Langfuse on first use.
# Variables use Langfuse's double-curly-brace syntax, e.g. {{conversation}}.
#
# v2: every template below states its output contract explicitly (a fixed
# JSON schema or a fixed labeled-line format) instead of a numbered list of
# topics to cover, and tells the model to check criteria/rules individually
# before scoring rather than judging holistically. This is meant to reduce
# parsing failures and partial-credit drift, without changing variable names
# or output shapes that downstream code (regexes, json.loads) depends on.
_DEFAULT_PROMPTS: Dict[str, str] = {
    "context_analysis": """You are analyzing a conversation between a user and an AI assistant, to support downstream evaluation of the assistant's response.

Conversation:
{{conversation}}

Analyze the conversation along four dimensions and return your findings as a single JSON object with exactly these keys:

1. "structural": turn-taking patterns.
   - "total_turns" (int), "user_turns" (int), "assistant_turns" (int)
   - "speaker_pattern" (list of "user"/"assistant", collapsing consecutive repeats)
   - "turn_distribution_ratio" (float: user_turns / assistant_turns, 0 if assistant_turns is 0)

2. "semantic": what the conversation is about.
   - "topics" (list of short strings)
   - "key_phrases" (list of short strings)
   - "sentiment_score" (float, -1 negative to 1 positive)

3. "complexity": how demanding the conversation is.
   - "complexity_score" (float, 0-1)
   - "technical_terms_count" (int)
   - "question_count" (int)

4. "intent": what each party is trying to achieve.
   - "user_intents" (list of short verb-based labels, e.g. "fix", "explain", "create")
   - "assistant_objectives" (list of short strings)
   - "primary_intent" (string: the single most important user intent, or "unknown")

Rules:
- Base every field only on what is present in the conversation; do not invent details.
- Regardless of the conversation's original language, write "topics", "key_phrases", "user_intents" and "assistant_objectives" in English.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "rule_generation": """Based on the context analysis below, generate evaluation rules for assessing an AI assistant's response to this task: {{task}}

Context Analysis:
{{context_analysis}}

Return a single JSON object with exactly these keys:
- "classification_criteria": short strings describing what distinguishes this task type from others.
- "success_conditions": concrete, checkable conditions a good response must satisfy.
- "failure_conditions": concrete, checkable conditions that would make a response unacceptable.
- "context_aware_criteria": a list of {"name": str, "description": str} objects, each a strict boolean (yes/no) check specific to this context -- not a generic criterion like "is relevant" that a fixed criterion already covers elsewhere.
- "heuristic_checks": short strings naming cheap, non-LLM checks worth running (e.g. "response is non-empty", "no unresolved placeholders").

Rules:
- Every "context_aware_criteria" description must be phrasable as a yes/no question.
- Do not repeat the same idea across multiple lists.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "jury_evaluation": """You are an impartial juror evaluating an AI assistant's response against a fixed set of rules.

Rules:
{{rules}}

Response to Evaluate:
{{response}}

Context:
{{context}}

Check the response against every rule above before scoring. Do not give credit for a rule that is not clearly satisfied.

Respond in EXACTLY this format, with no extra commentary before or after:

Score: <single number from 0 to 10, no range, no fraction>
Confidence: <single number from 0 to 1>
Strengths:
- <bullet, tied to a specific rule>
Weaknesses:
- <bullet, tied to a specific rule>
Reasoning: <2-4 sentences justifying the score by referencing which rules passed or failed>""",
    "metrics_aggregation": """Aggregate the jury evaluations below into a single final assessment.

Evaluations:
{{evaluations}}

Return a single JSON object with exactly these keys:
- "weighted_average_score": float, 0-10, weighting each evaluation by its own confidence rather than a plain average.
- "confidence": float, 0-1 -- your confidence in the aggregated score (lower when evaluators disagree).
- "recommendation": one of "approve", "approve_with_notes", "review_required", "reject".
- "agreement_areas": short strings naming points where the evaluations agree.
- "disagreement_areas": short strings naming points where the evaluations diverge, and why.

Rules:
- If evaluations disagree by more than 3 points, "recommendation" must be "review_required" or "reject".
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "dynamic_criteria_generation": """You are designing evaluation criteria for an AI response.

Context Analysis:
{{context_summary}}

Objectives:
{{objectives_summary}}

Existing Criteria (do not repeat or duplicate these):
{{existing_criteria}}

Propose additional evaluation criteria that are specific to this context and objectives, and that complement (do not overlap with) the existing criteria above. Favor criteria that catch failure modes particular to this task -- not generic qualities like "helpfulness" or "clarity" that a fixed criterion already covers elsewhere.

Rules:
- Each criterion MUST be a strict boolean (yes/no) check with a single unambiguous answer -- no "partially" or "somewhat".
- Phrase each "description" as a question that can only be answered True or False.
- Propose at most 3 criteria. Return fewer, or an empty array, if nothing meaningful is missing.
- Do not restate, rephrase, or narrow an existing criterion.
- "name" must be 1-4 words, in Title Case.

Example of a good criterion, for a task that asks the assistant to cite sources:
{"name": "Citations Present", "description": "Does the response include at least one citation for its factual claims?"}

Return ONLY a JSON array, with no surrounding text, in this exact form:
[{"name": "Short Name", "description": "Does the response ...?"}]""",
    "llm_evaluation": """You are an impartial evaluator scoring an AI assistant's response.

Task Context:
- Task Type: {{task_type}}
- Complexity: {{complexity_score}}

Response to Evaluate:
{{response}}

Evaluation Criteria (each is a strict yes/no check):
{{criteria_text}}

Check the response against every criterion above before scoring. A response that fails a criterion cannot receive full marks; do not give partial credit for a failed criterion.

Respond in EXACTLY this format, with no extra commentary before or after:

Score: <single number from 0 to 10, no range, no fraction>
Confidence: <single number from 0 to 1>
Strengths:
- <bullet, tied to a specific criterion>
Weaknesses:
- <bullet, tied to a specific criterion>
Reasoning: <2-4 sentences explaining the score, referencing which criteria passed or failed>""",
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


def get_default_prompts() -> Dict[str, str]:
    """Return a copy of the local default prompt templates (the current
    "v2" text), keyed by template name. Used to push these templates to
    Langfuse as new prompt versions -- see scripts/push_prompts_to_langfuse.py.
    """
    return dict(_DEFAULT_PROMPTS)


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
