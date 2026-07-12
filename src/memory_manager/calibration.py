"""Calibration with annotated data"""

from typing import List, Dict, Any

from src.shared.logger import get_logger
from src.shared.utils import calculate_average

logger = get_logger(__name__)


class CalibrationManager:
    """Manage calibration against annotated data"""

    def __init__(self):
        self.annotated_data: List[Dict] = []
        self.calibration_metrics: Dict[str, Any] = {}
        self.calibration_history: List[Dict] = []

    def load_annotated_data(
        self, data: List[Dict]
    ) -> None:
        """
        Load annotated reference data

        Args:
            data: List of annotated examples
        """
        self.annotated_data = data
        logger.info(
            f"Loaded {len(data)} annotated examples for calibration"
        )

    def calibrate(
        self,
        predicted_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Calibrate predictions against annotated data

        Args:
            predicted_scores: List of predicted scores

        Returns:
            Calibration results
        """
        if not self.annotated_data:
            logger.warning(
                "No annotated data available for calibration"
            )
            return {"calibrated": False, "reason": "no_data"}

        reference_scores = [
            item.get("score", 5.0) for item in self.annotated_data
        ]

        if not reference_scores:
            return {"calibrated": False, "reason": "no_scores"}

        # Calculate metrics
        metrics = self._calculate_calibration_metrics(
            predicted_scores, reference_scores
        )

        # Calculate adjustment factors
        adjustments = self._calculate_adjustments(
            predicted_scores, reference_scores
        )

        result = {
            "calibrated": True,
            "metrics": metrics,
            "adjustments": adjustments,
            "sample_size": len(reference_scores),
        }

        self.calibration_metrics = metrics
        self.calibration_history.append(result)

        logger.debug(f"Calibration complete: {metrics}")
        return result

    def _calculate_calibration_metrics(
        self,
        predicted: List[float],
        reference: List[float],
    ) -> Dict[str, float]:
        """Calculate calibration metrics"""
        if not predicted or not reference:
            return {}

        # Mean Absolute Error
        mae = calculate_average(
            [abs(p - r) for p, r in zip(predicted, reference)]
        )

        # Root Mean Square Error
        rmse = (
            sum((p - r) ** 2 for p, r in zip(predicted, reference))
            / len(predicted)
        ) ** 0.5

        # Bias
        bias = calculate_average(
            [p - r for p, r in zip(predicted, reference)]
        )

        # Correlation
        pred_mean = calculate_average(predicted)
        ref_mean = calculate_average(reference)

        covariance = sum(
            (p - pred_mean) * (r - ref_mean)
            for p, r in zip(predicted, reference)
        ) / len(predicted)

        pred_std = (
            sum((p - pred_mean) ** 2 for p in predicted)
            / len(predicted)
        ) ** 0.5
        ref_std = (
            sum((r - ref_mean) ** 2 for r in reference)
            / len(reference)
        ) ** 0.5

        correlation = (
            covariance / (pred_std * ref_std)
            if pred_std * ref_std > 0
            else 0
        )

        return {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "bias": round(bias, 3),
            "correlation": round(correlation, 3),
        }

    def _calculate_adjustments(
        self,
        predicted: List[float],
        reference: List[float],
    ) -> Dict[str, float]:
        """Calculate adjustment factors for predictions"""
        if not predicted or not reference:
            return {}

        # Calculate range adjustment
        pred_range = max(predicted) - min(predicted)
        ref_range = max(reference) - min(reference)

        range_factor = (
            ref_range / pred_range if pred_range > 0 else 1.0
        )

        # Calculate offset
        offset = calculate_average(reference) - calculate_average(
            predicted
        )

        return {
            "offset": round(offset, 2),
            "range_factor": round(range_factor, 2),
            "scale": round(range_factor, 2),
        }

    def apply_calibration(
        self, score: float
    ) -> float:
        """
        Apply calibration adjustments to a score

        Args:
            score: Raw score

        Returns:
            Calibrated score
        """
        if not self.calibration_metrics:
            return score

        adjustments = self.calibration_metrics

        # Apply offset and scale
        offset = adjustments.get("offset", 0)
        scale = adjustments.get("scale", 1.0)

        calibrated = (score - offset) * scale

        # Keep within bounds
        return min(max(calibrated, 0), 10)

    def evaluate_calibration_quality(
        self,
    ) -> Dict[str, Any]:
        """Evaluate quality of current calibration"""
        if not self.calibration_metrics:
            return {"quality": "uncalibrated"}

        metrics = self.calibration_metrics

        mae = metrics.get("mae", 5.0)
        correlation = metrics.get("correlation", 0.0)

        # Determine quality level
        if mae < 0.5 and correlation > 0.9:
            quality = "excellent"
        elif mae < 1.0 and correlation > 0.8:
            quality = "good"
        elif mae < 2.0 and correlation > 0.6:
            quality = "acceptable"
        else:
            quality = "poor"

        return {
            "quality": quality,
            "mae": mae,
            "correlation": correlation,
            "recommendation": (
                "Model is well-calibrated"
                if quality in ["excellent", "good"]
                else "Consider recalibration"
            ),
        }

    def recalibrate_if_needed(
        self,
        new_data: List[Dict],
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Check if recalibration is needed and perform if necessary

        Args:
            new_data: New annotated data
            threshold: Quality threshold for triggering recalibration

        Returns:
            Recalibration result
        """
        current_quality = self.evaluate_calibration_quality()

        if (
            current_quality.get("quality") == "poor"
            or current_quality.get("mae", 5.0) > (10 - threshold)
        ):
            logger.info("Triggering recalibration...")
            self.load_annotated_data(new_data)
            new_scores = [
                d.get("score", 5.0) for d in new_data
            ]
            return self.calibrate(new_scores)
        else:
            return {
                "recalibrated": False,
                "reason": "quality_acceptable",
            }

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get comprehensive calibration report"""
        quality = self.evaluate_calibration_quality()

        return {
            "current_quality": quality,
            "metrics": self.calibration_metrics,
            "sample_size": len(self.annotated_data),
            "calibration_count": len(self.calibration_history),
            "last_calibration": (
                self.calibration_history[-1]
                if self.calibration_history
                else None
            ),
        }
