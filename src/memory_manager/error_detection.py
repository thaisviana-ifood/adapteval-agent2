"""Error detection and analysis"""

from typing import List, Dict, Any

from src.shared.logger import get_logger

logger = get_logger(__name__)


class ErrorDetector:
    """Detect and analyze errors in evaluations"""

    def __init__(self):
        self.error_patterns: List[Dict] = []
        self.error_categories = [
            "missing_criteria",
            "inconsistent_scoring",
            "out_of_range_values",
            "conflicting_evaluations",
            "timeout_errors",
            "format_errors",
        ]

    def detect_errors(
        self,
        evaluation: Dict[str, Any],
        criteria: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect errors in evaluation

        Args:
            evaluation: Evaluation result to check
            criteria: Expected criteria

        Returns:
            List of detected errors
        """
        errors = []

        # Check for missing criteria
        errors.extend(self._check_missing_criteria(evaluation, criteria))

        # Check for invalid values
        errors.extend(self._check_invalid_values(evaluation))

        # Check for inconsistencies
        errors.extend(self._check_inconsistencies(evaluation))

        # Check for format issues
        errors.extend(self._check_format_issues(evaluation))

        if errors:
            logger.warning(
                f"Found {len(errors)} errors in evaluation"
            )

        return errors

    def _check_missing_criteria(
        self,
        evaluation: Dict,
        criteria: Dict,
    ) -> List[Dict]:
        """Check for missing evaluation criteria"""
        errors = []

        required_fields = [
            "final_score",
            "confidence",
            "reasoning",
        ]

        for field in required_fields:
            if field not in evaluation:
                errors.append(
                    {
                        "category": "missing_criteria",
                        "field": field,
                        "message": f"Missing required field: {field}",
                        "severity": "high",
                    }
                )

        return errors

    def _check_invalid_values(
        self, evaluation: Dict
    ) -> List[Dict]:
        """Check for invalid or out-of-range values"""
        errors = []

        # Check score range
        if "final_score" in evaluation:
            score = evaluation["final_score"]
            if not isinstance(score, (int, float)):
                errors.append(
                    {
                        "category": "format_errors",
                        "field": "final_score",
                        "message": "Score must be numeric",
                        "severity": "high",
                        "value": score,
                    }
                )
            elif not 0 <= score <= 10:
                errors.append(
                    {
                        "category": "out_of_range_values",
                        "field": "final_score",
                        "message": f"Score {score} out of range [0, 10]",
                        "severity": "high",
                        "value": score,
                    }
                )

        # Check confidence range
        if "confidence" in evaluation:
            conf = evaluation["confidence"]
            if not isinstance(conf, (int, float)):
                errors.append(
                    {
                        "category": "format_errors",
                        "field": "confidence",
                        "message": "Confidence must be numeric",
                        "severity": "high",
                        "value": conf,
                    }
                )
            elif not 0 <= conf <= 1:
                errors.append(
                    {
                        "category": "out_of_range_values",
                        "field": "confidence",
                        "message": f"Confidence {conf} out of range [0, 1]",
                        "severity": "high",
                        "value": conf,
                    }
                )

        return errors

    def _check_inconsistencies(
        self, evaluation: Dict
    ) -> List[Dict]:
        """Check for inconsistent evaluations"""
        errors = []

        # Check if confidence matches score clarity
        if (
            "final_score" in evaluation
            and "confidence" in evaluation
        ):
            score = evaluation["final_score"]
            conf = evaluation["confidence"]

            # Low confidence for very clear scores (near 0 or 10)
            if (score < 1 or score > 9) and conf < 0.5:
                errors.append(
                    {
                        "category": "inconsistent_scoring",
                        "message": (
                            "Low confidence for clear-cut score"
                        ),
                        "severity": "medium",
                        "score": score,
                        "confidence": conf,
                    }
                )

        return errors

    def _check_format_issues(
        self, evaluation: Dict
    ) -> List[Dict]:
        """Check for format/structure issues"""
        errors = []

        # Check reasoning exists and is non-empty
        if "reasoning" in evaluation:
            reasoning = evaluation["reasoning"]
            if not reasoning or len(str(reasoning).strip()) == 0:
                errors.append(
                    {
                        "category": "format_errors",
                        "field": "reasoning",
                        "message": "Reasoning cannot be empty",
                        "severity": "medium",
                    }
                )

        return errors

    def analyze_error_patterns(
        self, error_list: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in errors

        Args:
            error_list: List of detected errors

        Returns:
            Error pattern analysis
        """
        if not error_list:
            return {"patterns": [], "summary": "No errors detected"}

        # Group by category
        by_category: Dict[str, int] = {}
        for error in error_list:
            category = error.get("category", "unknown")
            by_category[category] = by_category.get(category, 0) + 1

        # Identify most common
        most_common = max(
            by_category.items(), key=lambda x: x[1]
        ) if by_category else ("none", 0)

        analysis = {
            "total_errors": len(error_list),
            "by_category": by_category,
            "most_common": most_common[0],
            "high_severity_count": sum(
                1
                for e in error_list
                if e.get("severity") == "high"
            ),
            "summary": f"Found {len(error_list)} errors, "
            f"most common: {most_common[0]}",
        }

        return analysis

    def suggest_fixes(
        self, error: Dict
    ) -> str:
        """Suggest fix for an error"""
        category = error.get("category")
        field = error.get("field")

        suggestions = {
            "missing_criteria": (
                f"Add required field: {field}"
            ),
            "out_of_range_values": (
                f"Validate {field} is within expected range"
            ),
            "inconsistent_scoring": (
                "Review score and confidence values for consistency"
            ),
            "format_errors": (
                f"Check format of {field}"
            ),
        }

        return suggestions.get(
            category, "Please review this error"
        )

    def record_error_pattern(
        self, category: str, details: Dict
    ) -> None:
        """Record error pattern for future analysis"""
        pattern = {
            "category": category,
            "details": details,
            "frequency": self._count_pattern(category, details),
        }

        self.error_patterns.append(pattern)
        logger.debug(f"Recorded error pattern: {category}")

    def _count_pattern(
        self, category: str, details: Dict
    ) -> int:
        """Count occurrences of error pattern"""
        count = sum(
            1
            for p in self.error_patterns
            if p["category"] == category
        )
        return count
