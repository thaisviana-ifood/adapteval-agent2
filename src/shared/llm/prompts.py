"""Unified prompt management system"""

from typing import Dict, Optional
from string import Template

from src.shared.logger import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manages prompt templates and formatting"""

    def __init__(self):
        self.templates: Dict[str, str] = {}
        self._load_default_prompts()

    def _load_default_prompts(self) -> None:
        """Load default prompt templates"""
        self.templates = {
            "context_analysis": """Analyze the following conversation context and provide insights about:
1. Structural patterns (turn sequence, speaker patterns)
2. Semantic content (main topics, key phrases, sentiment)
3. Complexity indicators (number of topics, semantic diversity)
4. Intent analysis (user goals, assistant objectives)

Conversation:
$conversation

Provide a structured analysis.""",
            "rule_generation": """Based on the context analysis, generate evaluation rules for assessing responses:

Context Analysis:
$context_analysis

Task: $task

Generate rules for:
1. Task-specific classification criteria
2. Success and failure conditions
3. Context-aware evaluation criteria
4. Minimum heuristic checks

Provide a JSON-formatted set of rules.""",
            "jury_evaluation": """Evaluate the provided response using the following rules and context:

Rules:
$rules

Response to Evaluate:
$response

Context:
$context

Provide your evaluation with:
1. Score (0-10)
2. Confidence level (0-1)
3. Reasoning
4. Strengths
5. Weaknesses""",
            "metrics_aggregation": """Aggregate the following jury evaluations into a final assessment:

Evaluations:
$evaluations

Generate:
1. Weighted average score
2. Confidence estimation
3. Final recommendation
4. Areas of agreement/disagreement""",
        }
        logger.debug(f"Loaded {len(self.templates)} default prompt templates")

    def get_template(self, template_name: str) -> Optional[str]:
        """Get a prompt template by name"""
        template = self.templates.get(template_name)
        if not template:
            logger.warning(f"Template '{template_name}' not found")
        return template

    def register_template(self, name: str, template: str) -> None:
        """Register a new prompt template"""
        self.templates[name] = template
        logger.debug(f"Registered template: {name}")

    def format_prompt(
        self, template_name: str, **variables: str
    ) -> Optional[str]:
        """Format a prompt template with provided variables"""
        template = self.get_template(template_name)
        if not template:
            return None

        try:
            prompt = Template(template)
            formatted = prompt.substitute(variables)
            logger.debug(f"Formatted prompt: {template_name}")
            return formatted
        except KeyError as e:
            logger.error(f"Missing variable for template '{template_name}': {e}")
            return None

    def list_templates(self) -> list:
        """List all available templates"""
        return list(self.templates.keys())
