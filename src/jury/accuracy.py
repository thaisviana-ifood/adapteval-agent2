"""Accuracy calculation based on annotated data"""

from typing import Dict, Any, List, Optional

from src.shared.logger import get_logger
from src.shared.utils import calculate_weighted_average

logger = get_logger(__name__)


class AccuracyCalculator:
    """Calculate accuracy metrics from annotated data"""

    def __init__(self):
        self.annotated_data: List[Dict] = []
        self.accuracy_metrics = {}

    def load_annotated_data(self, data: List[Dict]) -> None:
        """
        Load annotated reference data

        Args:
            data: List of annotated examples with scores
        """
        self.annotated_data = data
        logger.info(
            f"Loaded {len(data)} annotated examples"
        )

    def calculate_accuracy(
        self,
        predicted_score: float,
        criteria: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate accuracy metrics

        Args:
            predicted_score: Score predicted by evaluators
            criteria: Evaluation criteria used

        Returns:
            Accuracy metrics and confidence
        """
        if not self.annotated_data:
            logger.warning("No annotated data loaded")
            return {"accuracy": 0.5, "confidence": 0.0}

        # Compare to annotated data
        reference_scores = [
            item.get("score", 5.0)
            for item in self.annotated_data
        ]

        if not reference_scores:
            return {"accuracy": 0.5, "confidence": 0.0}

        avg_reference = sum(reference_scores) / len(
            reference_scores
        )

        # Calculate difference
        difference = abs(predicted_score - avg_reference)
        max_diff = 10.0

        accuracy = max(0, 1 - (difference / max_diff))

        metrics = {
            "predicted_score": predicted_score,
            "reference_score": avg_reference,
            "difference": round(difference, 2),
            "accuracy": round(accuracy, 4),
            "confidence": self._calculate_confidence(
                accuracy, len(self.annotated_data)
            ),
        }

        logger.debug(f"Accuracy metrics: {metrics}")
        return metrics

    def calculate_criterion_accuracy(
        self,
        criterion_name: str,
        predicted_value: Any,
    ) -> Dict[str, Any]:
        """
        Calculate accuracy for a specific criterion

        Args:
            criterion_name: Name of criterion
            predicted_value: Predicted value

        Returns:
            Accuracy metrics for criterion
        """
        # Filter annotated data for this criterion
        relevant_data = [
            item
            for item in self.annotated_data
            if criterion_name in item.get("criteria", {})
        ]

        if not relevant_data:
            return {
                "criterion": criterion_name,
                "accuracy": 0.5,
                "confidence": 0.0,
            }

        reference_values = [
            item["criteria"][criterion_name]
            for item in relevant_data
        ]

        # Calculate accuracy based on value type
        if isinstance(predicted_value, (int, float)):
            accuracy = self._numeric_accuracy(
                predicted_value, reference_values
            )
        else:
            accuracy = self._categorical_accuracy(
                predicted_value, reference_values
            )

        return {
            "criterion": criterion_name,
            "predicted": predicted_value,
            "reference_avg": (
                sum(reference_values) / len(reference_values)
                if reference_values
                else None
            ),
            "accuracy": round(accuracy, 4),
            "confidence": self._calculate_confidence(
                accuracy, len(relevant_data)
            ),
        }

    def _numeric_accuracy(
        self,
        predicted: float,
        references: List[float],
    ) -> float:
        """Calculate accuracy for numeric values"""
        avg_ref = sum(references) / len(references)
        std_dev = self._calculate_std_dev(references)

        if std_dev == 0:
            return 1.0 if predicted == avg_ref else 0.0

        z_score = abs(predicted - avg_ref) / std_dev
        accuracy = 1.0 / (1.0 + z_score)

        return min(max(accuracy, 0.0), 1.0)

    def _categorical_accuracy(
        self,
        predicted: Any,
        references: List[Any],
    ) -> float:
        """Calculate accuracy for categorical values"""
        match_count = sum(
            1 for ref in references if ref == predicted
        )
        return match_count / len(references) if references else 0.0

    def _calculate_confidence(
        self, accuracy: float, sample_size: int
    ) -> float:
        """
        Calculate confidence in accuracy estimate

        Args:
            accuracy: Calculated accuracy
            sample_size: Size of reference sample

        Returns:
            Confidence score (0-1)
        """
        # Confidence increases with sample size and accuracy
        sample_confidence = min(sample_size / 10, 1.0)
        accuracy_confidence = accuracy

        return (sample_confidence + accuracy_confidence) / 2

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(
            values
        )
        return variance ** 0.5

    def compare_to_baseline(
        self,
        score: float,
        baseline_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Compare score to baseline

        Args:
            score: Score to evaluate
            baseline_scores: Baseline scores for comparison

        Returns:
            Comparison metrics
        """
        if not baseline_scores:
            return {"percentile": 50.0, "relative_performance": 0.0}

        baseline_avg = sum(baseline_scores) / len(
            baseline_scores
        )
        baseline_std = self._calculate_std_dev(baseline_scores)

        percentile = (
            sum(
                1 for s in baseline_scores if s <= score
            )
            / len(baseline_scores)
        ) * 100

        if baseline_std > 0:
            relative = (score - baseline_avg) / baseline_std
        else:
            relative = 0.0

        return {
            "score": score,
            "baseline_mean": round(baseline_avg, 2),
            "baseline_std": round(baseline_std, 2),
            "percentile": round(percentile, 1),
            "relative_performance": round(relative, 2),
        }
