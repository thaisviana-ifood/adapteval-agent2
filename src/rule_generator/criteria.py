"""Context-aware criteria generation

Every criterion produced here is boolean: it is a strict yes/no check
(`criterion_type: "boolean"`) with a description phrased as a question that
can only be answered True or False, so downstream evaluators score pass/fail
rather than a graded scale.
"""

import json
import re
from typing import Dict, Any, List, Optional

from src.shared.logger import get_logger
from src.shared.llm.client import LLMClient
from src.shared.llm.prompts import PromptManager

logger = get_logger(__name__)

_MAX_DYNAMIC_CRITERIA = 3
_JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


class CriteriaGenerator:
    """Generate evaluation criteria from context and objectives"""

    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_manager = PromptManager()

    def generate(
        self,
        context: Dict[str, Any],
        objectives: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate context-aware evaluation criteria

        Args:
            context: Context analysis results
            objectives: Success objectives

        Returns:
            Evaluation criteria
        """
        base_criteria = self._extract_base_criteria(context)
        contextual_criteria = self._generate_contextual_criteria(
            context, objectives
        )
        dynamic_criteria = self._generate_dynamic_criteria(
            context, objectives, base_criteria + contextual_criteria
        )
        weighted_criteria = self._assign_weights(
            base_criteria + contextual_criteria + dynamic_criteria
        )

        criteria = {
            "base_criteria": base_criteria,
            "contextual_criteria": contextual_criteria,
            "dynamic_criteria": dynamic_criteria,
            "weighted_criteria": weighted_criteria,
            "total_criteria_count": len(weighted_criteria),
        }

        logger.debug(f"Generated {len(weighted_criteria)} criteria")
        return criteria

    def _extract_base_criteria(self, context: Dict[str, Any]) -> List[Dict]:
        """Extract base evaluation criteria"""
        criteria = [
            {
                "name": "Relevance",
                "description": "Is the response relevant to the query?",
                "criterion_type": "boolean",
                "weight": 1.0,
            },
            {
                "name": "Correctness",
                "description": "Is the information in the response factually correct?",
                "criterion_type": "boolean",
                "weight": 1.0,
            },
            {
                "name": "Clarity",
                "description": "Is the response clear and understandable?",
                "criterion_type": "boolean",
                "weight": 0.8,
            },
            {
                "name": "Completeness",
                "description": "Does the response address all aspects of the request?",
                "criterion_type": "boolean",
                "weight": 0.9,
            },
        ]

        return criteria

    def _generate_contextual_criteria(
        self,
        context: Dict[str, Any],
        objectives: Dict[str, Any],
    ) -> List[Dict]:
        """Generate criteria specific to this context"""
        contextual = []

        # Based on complexity
        complexity = context.get("complexity", {}).get(
            "complexity_score", 5
        )
        if complexity > 7:
            contextual.append(
                {
                    "name": "Technical Depth",
                    "description": (
                        "Does the response address the technical "
                        "complexity of the request appropriately?"
                    ),
                    "criterion_type": "boolean",
                    "weight": 1.0,
                }
            )

        # Based on user intents
        intents = (
            context.get("intent", {})
            .get("user_intents", [])
        )
        if "fix" in intents:
            contextual.append(
                {
                    "name": "Problem Solving",
                    "description": (
                        "Does the response effectively solve the "
                        "stated problem?"
                    ),
                    "criterion_type": "boolean",
                    "weight": 1.2,
                }
            )

        # Based on must-haves (from ObjectiveDefinition.get_objectives's
        # "success_criteria", or define_success's "must_haves")
        must_haves = objectives.get(
            "success_criteria", objectives.get("must_haves", [])
        )
        for criterion in must_haves:
            contextual.append(
                {
                    "name": criterion,
                    "description": f"Does the response satisfy: {criterion}?",
                    "criterion_type": "boolean",
                    "weight": 1.0,
                }
            )

        return contextual

    def _generate_dynamic_criteria(
        self,
        context: Dict[str, Any],
        objectives: Dict[str, Any],
        existing_criteria: List[Dict],
    ) -> List[Dict]:
        """Use the LLM to propose complementary, context-specific criteria
        not already covered by the base/contextual ones. Always boolean.
        """
        prompt = self.prompt_manager.format_prompt(
            "dynamic_criteria_generation",
            context_summary=self._summarize_context(context),
            objectives_summary=self._summarize_objectives(objectives),
            existing_criteria=self._summarize_existing(existing_criteria),
        )
        if not prompt:
            return []

        try:
            response_text = self.llm_client.call(
                prompt,
                name="rule_generator.dynamic_criteria",
                langfuse_prompt=self.prompt_manager.get_prompt_client(
                    "dynamic_criteria_generation"
                ),
            )
        except Exception as e:
            logger.warning(f"Dynamic criteria generation failed: {e}")
            return []

        existing_names = {c["name"].strip().lower() for c in existing_criteria}
        return self._parse_dynamic_criteria(response_text, existing_names)

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Compact summary of the signals relevant to criteria generation"""
        summary = {
            "topics": context.get("semantic", {}).get("topics", []),
            "user_intents": context.get("intent", {}).get("user_intents", []),
            "complexity_score": context.get("complexity", {}).get(
                "complexity_score"
            ),
        }
        return json.dumps(summary, default=str)

    def _summarize_objectives(self, objectives: Dict[str, Any]) -> str:
        """Compact summary of the objectives relevant to criteria generation"""
        summary = {
            "primary_objective": objectives.get("primary_objective")
            or objectives.get("primary"),
            "secondary_objectives": objectives.get("secondary_objectives")
            or objectives.get("secondary", []),
            "success_criteria": objectives.get("success_criteria")
            or objectives.get("must_haves", []),
        }
        return json.dumps(summary, default=str)

    def _summarize_existing(self, existing_criteria: List[Dict]) -> str:
        if not existing_criteria:
            return "None"
        return "\n".join(f"- {c['name']}" for c in existing_criteria)

    def _parse_dynamic_criteria(
        self, response_text: Optional[str], existing_names: set
    ) -> List[Dict]:
        """Parse the LLM's JSON array response into boolean criteria dicts"""
        if not response_text:
            return []

        match = _JSON_ARRAY_PATTERN.search(response_text)
        if not match:
            logger.warning(
                "Dynamic criteria generation returned no JSON array"
            )
            return []

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse dynamic criteria JSON: {e}")
            return []

        dynamic = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            if not name or not description:
                continue
            if name.lower() in existing_names:
                continue

            existing_names.add(name.lower())
            dynamic.append(
                {
                    "name": name,
                    "description": description,
                    "criterion_type": "boolean",
                    "weight": 1.0,
                }
            )
            if len(dynamic) >= _MAX_DYNAMIC_CRITERIA:
                break

        return dynamic

    def _assign_weights(self, criteria: List[Dict]) -> List[Dict]:
        """Assign normalized weights to criteria"""
        total_weight = sum(c.get("weight", 1.0) for c in criteria)

        weighted = []
        for criterion in criteria:
            weight = criterion.get("weight", 1.0)
            normalized_weight = weight / total_weight if total_weight > 0 else 1.0 / len(criteria)

            weighted.append(
                {
                    **criterion,
                    "weight": weight,
                    "normalized_weight": round(
                        normalized_weight, 4
                    ),
                }
            )

        return weighted
