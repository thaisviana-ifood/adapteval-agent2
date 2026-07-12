"""Heuristic checks for quick evaluation"""

from typing import Dict, Any, List

from src.shared.logger import get_logger

logger = get_logger(__name__)


class HeuristicChecker:
    """Fast heuristic checks before detailed evaluation"""

    def __init__(self):
        self.heuristics = {
            "minimum_length": {
                "description": "Response has minimum required length",
                "check_fn": self._check_minimum_length,
            },
            "no_empty_response": {
                "description": "Response is not empty",
                "check_fn": self._check_not_empty,
            },
            "coherence": {
                "description": "Response is coherent",
                "check_fn": self._check_coherence,
            },
            "relevance": {
                "description": "Response mentions relevant keywords",
                "check_fn": self._check_relevance,
            },
        }

    def check_all(
        self,
        response: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Run all heuristic checks

        Args:
            response: The response to check
            query: The original query/context

        Returns:
            Dict with heuristic results
        """
        results = {
            "passed_checks": [],
            "failed_checks": [],
            "overall_pass": True,
        }

        for heuristic_name, heuristic_info in self.heuristics.items():
            passed = heuristic_info["check_fn"](response, query)
            if passed:
                results["passed_checks"].append(heuristic_name)
            else:
                results["failed_checks"].append(heuristic_name)
                results["overall_pass"] = False

        results["check_count"] = len(self.heuristics)
        results["pass_rate"] = (
            len(results["passed_checks"])
            / len(self.heuristics)
        )

        logger.debug(f"Heuristic check results: {results}")
        return results

    def _check_minimum_length(
        self, response: str, query: str
    ) -> bool:
        """Check if response has minimum length"""
        min_length = max(len(query) // 2, 20)
        return len(response) >= min_length

    def _check_not_empty(self, response: str, query: str) -> bool:
        """Check if response is not empty"""
        return len(response.strip()) > 0

    def _check_coherence(self, response: str, query: str) -> bool:
        """Check if response is coherent"""
        # Simple coherence: has multiple sentences and proper punctuation
        sentences = response.split(".")
        if len(sentences) < 2:
            return False

        # Check for common coherence issues
        incoherence_markers = ["[error]", "[failed]", "sorry", "cannot"]
        if any(
            marker in response.lower()
            for marker in incoherence_markers
        ):
            return False

        return True

    def _check_relevance(
        self, response: str, query: str
    ) -> bool:
        """Check if response is relevant to query"""
        # Extract key terms from query
        query_words = set(query.lower().split())

        # Check if response contains some query terms
        response_words = set(response.lower().split())
        overlap = query_words & response_words

        # At least 30% of query words should appear in response
        threshold = max(len(query_words) * 0.3, 1)
        return len(overlap) >= threshold

    def check_specific(
        self, heuristic_name: str, response: str, query: str
    ) -> bool:
        """
        Run a specific heuristic check

        Args:
            heuristic_name: Name of the heuristic
            response: Response to check
            query: Original query

        Returns:
            Whether the check passed
        """
        if heuristic_name not in self.heuristics:
            logger.warning(f"Unknown heuristic: {heuristic_name}")
            return True

        check_fn = self.heuristics[heuristic_name]["check_fn"]
        return check_fn(response, query)

    def list_heuristics(self) -> List[str]:
        """List all available heuristics"""
        return list(self.heuristics.keys())

    def get_heuristic_info(self, heuristic_name: str) -> Dict[str, Any]:
        """Get information about a specific heuristic"""
        return self.heuristics.get(heuristic_name, {})
