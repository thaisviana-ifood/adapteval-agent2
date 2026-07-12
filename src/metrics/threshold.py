"""Threshold calculation for decision-making"""

from typing import Dict, Any, List

from src.shared.logger import get_logger

logger = get_logger(__name__)


class ThresholdCalculator:
    """Calculate and apply thresholds for evaluation decisions"""

    def __init__(self, default_threshold: float = 0.7):
        self.default_threshold = default_threshold
        self.adaptive_thresholds: Dict[str, float] = {}
        self.threshold_history: List[Dict] = []

    def calculate_threshold(
        self,
        context: Dict[str, Any],
        criteria_count: int,
        jury_size: int,
    ) -> float:
        """
        Calculate appropriate threshold

        Args:
            context: Context information
            criteria_count: Number of evaluation criteria
            jury_size: Size of evaluating jury

        Returns:
            Recommended threshold
        """
        # Base threshold
        threshold = self.default_threshold

        # Adjust based on complexity
        complexity = (
            context.get("complexity", {})
            .get("complexity_score", 5) / 10
        )
        threshold += (complexity * 0.1)

        # Adjust based on jury size
        jury_factor = min(jury_size / 5, 1.0)
        threshold += (jury_factor * 0.05)

        # Adjust based on criteria count
        criteria_factor = min(criteria_count / 10, 1.0)
        threshold += (criteria_factor * 0.05)

        # Cap between 0.5 and 1.0
        threshold = min(max(threshold, 0.5), 1.0)

        logger.debug(
            f"Calculated threshold: {threshold:.2f} "
            f"(complexity: {complexity:.2f}, "
            f"jury: {jury_size}, criteria: {criteria_count})"
        )

        return round(threshold, 2)

    def apply_threshold(
        self, score: float, threshold: float
    ) -> Dict[str, Any]:
        """
        Apply threshold to make decision

        Args:
            score: Evaluation score
            threshold: Decision threshold

        Returns:
            Decision and reasoning
        """
        decision = {
            "score": score,
            "threshold": threshold,
            "normalized_score": score / 10,
            "passed": False,
            "margin": 0.0,
            "confidence": 0.0,
        }

        if score / 10 >= threshold:
            decision["passed"] = True

        decision["margin"] = abs((score / 10) - threshold)

        # Confidence based on distance from threshold
        if decision["margin"] < 0.1:
            decision["confidence"] = 0.5
        elif decision["margin"] < 0.3:
            decision["confidence"] = 0.75
        else:
            decision["confidence"] = 0.95

        logger.debug(
            f"Threshold applied: score={score}, "
            f"threshold={threshold}, passed={decision['passed']}"
        )

        return decision

    def set_adaptive_threshold(
        self, criterion_name: str, threshold: float
    ) -> None:
        """
        Set threshold specific to a criterion

        Args:
            criterion_name: Name of criterion
            threshold: Threshold value
        """
        self.adaptive_thresholds[criterion_name] = threshold
        logger.debug(
            f"Set adaptive threshold for {criterion_name}: "
            f"{threshold}"
        )

    def get_adaptive_threshold(
        self, criterion_name: str
    ) -> float:
        """Get threshold for specific criterion"""
        return self.adaptive_thresholds.get(
            criterion_name, self.default_threshold
        )

    def calculate_pass_rate(
        self,
        scores: List[float],
        threshold: float,
    ) -> Dict[str, Any]:
        """
        Calculate pass rate for a set of scores

        Args:
            scores: List of scores
            threshold: Pass threshold

        Returns:
            Pass rate statistics
        """
        if not scores:
            return {"pass_rate": 0.0, "passing_count": 0}

        normalized = [s / 10 for s in scores]
        passing = sum(1 for s in normalized if s >= threshold)
        pass_rate = passing / len(scores)

        return {
            "pass_rate": round(pass_rate, 2),
            "passing_count": passing,
            "total_count": len(scores),
            "threshold": threshold,
            "average_score": round(sum(scores) / len(scores), 2),
        }

    def recommend_threshold(
        self,
        historical_scores: List[float],
        target_pass_rate: float = 0.7,
    ) -> float:
        """
        Recommend threshold based on historical data

        Args:
            historical_scores: Previous scores
            target_pass_rate: Desired pass rate

        Returns:
            Recommended threshold
        """
        if not historical_scores:
            return self.default_threshold

        sorted_scores = sorted(historical_scores)
        index = int(len(sorted_scores) * (1 - target_pass_rate))
        index = max(0, min(index, len(sorted_scores) - 1))

        recommended = sorted_scores[index] / 10

        logger.info(
            f"Recommended threshold: {recommended:.2f} "
            f"for {target_pass_rate*100:.0f}% pass rate"
        )

        return round(recommended, 2)

    def track_threshold_application(
        self,
        score: float,
        threshold: float,
        decision: bool,
        context: str = "",
    ) -> None:
        """Track threshold decisions for analysis"""
        record = {
            "score": score,
            "threshold": threshold,
            "decision": decision,
            "context": context,
        }
        self.threshold_history.append(record)

    def get_threshold_stats(self) -> Dict[str, Any]:
        """Get statistics about threshold applications"""
        if not self.threshold_history:
            return {
                "total_decisions": 0,
                "pass_rate": 0.0,
            }

        passed = sum(
            1 for record in self.threshold_history
            if record["decision"]
        )

        return {
            "total_decisions": len(self.threshold_history),
            "passed_decisions": passed,
            "pass_rate": round(
                passed / len(self.threshold_history), 2
            ),
            "avg_score": round(
                sum(r["score"] for r in self.threshold_history)
                / len(self.threshold_history),
                2,
            ),
        }
