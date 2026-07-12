"""Context-aware criteria generation"""

from typing import Dict, Any, List

from src.shared.logger import get_logger
from src.shared.llm.client import LLMClient

logger = get_logger(__name__)


class CriteriaGenerator:
    """Generate evaluation criteria from context and objectives"""

    def __init__(self):
        self.llm_client = LLMClient()

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
        weighted_criteria = self._assign_weights(
            base_criteria + contextual_criteria
        )

        criteria = {
            "base_criteria": base_criteria,
            "contextual_criteria": contextual_criteria,
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
                "description": "Response is relevant to the query",
                "weight": 1.0,
            },
            {
                "name": "Correctness",
                "description": "Information is factually correct",
                "weight": 1.0,
            },
            {
                "name": "Clarity",
                "description": "Response is clear and understandable",
                "weight": 0.8,
            },
            {
                "name": "Completeness",
                "description": "Response addresses all aspects",
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
                        "Addresses technical complexity appropriately"
                    ),
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
                        "Effectively solves the stated problem"
                    ),
                    "weight": 1.2,
                }
            )

        # Based on must-haves
        for criterion in objectives.get("must_haves", []):
            contextual.append(
                {
                    "name": criterion,
                    "description": f"Satisfies: {criterion}",
                    "weight": 1.0,
                }
            )

        return contextual

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
