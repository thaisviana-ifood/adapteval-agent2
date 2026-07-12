"""Final score calculation and consolidation"""

from typing import Dict, Any, List

from src.shared.logger import get_logger
from src.shared.utils import calculate_weighted_average

logger = get_logger(__name__)


class FinalScoreCalculator:
    """Calculate final evaluation scores with context awareness"""

    def __init__(self):
        self.component_weights = {
            "jury_score": 0.5,
            "heuristic_score": 0.2,
            "accuracy_score": 0.2,
            "context_adjustment": 0.1,
        }

    def calculate_final_score(
        self,
        jury_evaluation: Dict[str, Any],
        heuristic_check: Dict[str, Any],
        accuracy_metrics: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate final consolidated score

        Args:
            jury_evaluation: Results from jury panel
            heuristic_check: Heuristic check results
            accuracy_metrics: Accuracy from annotated data
            context: Context information

        Returns:
            Final evaluation score
        """
        # Extract component scores
        jury_score = jury_evaluation.get("average_score", 5.0)
        jury_confidence = jury_evaluation.get("confidence", 0.5)

        heuristic_score = self._heuristic_to_score(heuristic_check)

        accuracy_score = self._accuracy_to_score(accuracy_metrics)

        context_adjustment = self._calculate_context_adjustment(
            context
        )

        # Calculate weighted final score
        final_score = calculate_weighted_average(
            [
                jury_score,
                heuristic_score,
                accuracy_score,
                context_adjustment,
            ],
            [
                self.component_weights["jury_score"],
                self.component_weights["heuristic_score"],
                self.component_weights["accuracy_score"],
                self.component_weights["context_adjustment"],
            ],
        )

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            jury_evaluation,
            heuristic_check,
            accuracy_metrics,
        )

        result = {
            "final_score": round(final_score, 2),
            "overall_confidence": round(overall_confidence, 2),
            "jury_contribution": jury_score,
            "heuristic_contribution": heuristic_score,
            "accuracy_contribution": accuracy_score,
            "context_adjustment": context_adjustment,
            "component_breakdown": {
                "jury": {
                    "score": jury_score,
                    "weight": self.component_weights["jury_score"],
                },
                "heuristic": {
                    "score": heuristic_score,
                    "weight": self.component_weights["heuristic_score"],
                },
                "accuracy": {
                    "score": accuracy_score,
                    "weight": self.component_weights["accuracy_score"],
                },
                "context": {
                    "score": context_adjustment,
                    "weight": self.component_weights["context_adjustment"],
                },
            },
        }

        logger.debug(f"Final score calculated: {final_score:.2f}")
        return result

    def _heuristic_to_score(
        self, heuristic_check: Dict[str, Any]
    ) -> float:
        """Convert heuristic pass rate to score"""
        pass_rate = heuristic_check.get("pass_rate", 0.5)
        return pass_rate * 10

    def _accuracy_to_score(
        self, accuracy_metrics: Dict[str, Any]
    ) -> float:
        """Convert accuracy metrics to score"""
        accuracy = accuracy_metrics.get("accuracy", 0.5)
        reference = accuracy_metrics.get("reference_score", 5.0)

        # Score based on accuracy relative to reference
        return reference + (accuracy - 0.5) * 2

    def _calculate_context_adjustment(
        self, context: Dict[str, Any]
    ) -> float:
        """Calculate context-based score adjustment"""
        adjustment = 5.0  # Neutral baseline

        # Adjust based on complexity
        complexity = (
            context.get("complexity", {})
            .get("complexity_score", 5) / 10
        )
        adjustment += (complexity - 0.5) * 2

        # Cap adjustment
        return min(max(adjustment, 0), 10)

    def _calculate_overall_confidence(
        self,
        jury_evaluation: Dict[str, Any],
        heuristic_check: Dict[str, Any],
        accuracy_metrics: Dict[str, Any],
    ) -> float:
        """Calculate overall confidence in final score"""
        jury_conf = jury_evaluation.get("confidence", 0.5)
        heuristic_conf = heuristic_check.get("pass_rate", 0.5)
        accuracy_conf = accuracy_metrics.get("confidence", 0.5)

        # Weight by component importance
        overall = (
            jury_conf * 0.5
            + heuristic_conf * 0.2
            + accuracy_conf * 0.3
        )

        return min(max(overall, 0), 1.0)

    def generate_report(
        self,
        final_score: Dict[str, Any],
        evaluation_id: str = "",
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report

        Args:
            final_score: Final score calculation result
            evaluation_id: Optional evaluation identifier

        Returns:
            Formatted evaluation report
        """
        report = {
            "evaluation_id": evaluation_id,
            "final_score": final_score["final_score"],
            "confidence": final_score["overall_confidence"],
            "recommendation": self._score_to_recommendation(
                final_score["final_score"]
            ),
            "summary": {
                "strength": self._identify_strengths(final_score),
                "weakness": self._identify_weaknesses(final_score),
            },
            "detailed_breakdown": final_score[
                "component_breakdown"
            ],
        }

        logger.debug(f"Report generated for {evaluation_id}")
        return report

    def _score_to_recommendation(self, score: float) -> str:
        """Convert score to recommendation"""
        if score >= 8:
            return "Excellent - Approved"
        elif score >= 7:
            return "Good - Approved with minor notes"
        elif score >= 5:
            return "Acceptable - Approved"
        elif score >= 3:
            return "Needs improvement - Review required"
        else:
            return "Poor - Revision needed"

    def _identify_strengths(
        self, final_score: Dict[str, Any]
    ) -> List[str]:
        """Identify evaluation strengths"""
        strengths = []
        breakdown = final_score.get("component_breakdown", {})

        for component, data in breakdown.items():
            if data.get("score", 5) >= 7:
                strengths.append(f"Strong {component} evaluation")

        return strengths if strengths else ["Baseline standards met"]

    def _identify_weaknesses(
        self, final_score: Dict[str, Any]
    ) -> List[str]:
        """Identify evaluation weaknesses"""
        weaknesses = []
        breakdown = final_score.get("component_breakdown", {})

        for component, data in breakdown.items():
            if data.get("score", 5) < 5:
                weaknesses.append(
                    f"Weak {component} performance"
                )

        return (
            weaknesses
            if weaknesses
            else ["No significant weaknesses"]
        )

    def set_component_weight(
        self, component: str, weight: float
    ) -> None:
        """Set weight for a component"""
        if component in self.component_weights:
            self.component_weights[component] = weight
            logger.debug(f"Updated {component} weight to {weight}")
        else:
            logger.warning(f"Unknown component: {component}")

    def get_weights(self) -> Dict[str, float]:
        """Get current component weights"""
        return self.component_weights.copy()
