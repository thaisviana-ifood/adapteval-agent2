"""Jury composition and voting mechanisms"""

from typing import Dict, Any, List

from src.shared.logger import get_logger
from src.shared.utils import calculate_weighted_average

logger = get_logger(__name__)


class JuryComposition:
    """Manage jury composition and aggregation"""

    def __init__(self, min_jurors: int = 3):
        self.min_jurors = min_jurors
        self.jurors: List[Dict[str, Any]] = []
        self.voting_weights: Dict[str, float] = {}

    def add_juror(
        self,
        juror_id: str,
        juror_type: str,
        expertise_level: float = 1.0,
    ) -> None:
        """
        Add a juror to the panel

        Args:
            juror_id: Unique ID for juror
            juror_type: Type of juror (llm, human, rules-based)
            expertise_level: Expertise level (0-1)
        """
        juror = {
            "id": juror_id,
            "type": juror_type,
            "expertise": expertise_level,
            "votes": [],
        }
        self.jurors.append(juror)
        self._recalculate_weights()

        logger.debug(
            f"Added juror {juror_id} ({juror_type}) "
            f"with expertise {expertise_level}"
        )

    def add_vote(
        self,
        juror_id: str,
        score: float,
        confidence: float = 0.5,
    ) -> bool:
        """
        Record a vote from a juror

        Args:
            juror_id: Juror ID
            score: Score (0-10)
            confidence: Confidence in score (0-1)

        Returns:
            Success status
        """
        for juror in self.jurors:
            if juror["id"] == juror_id:
                vote = {"score": score, "confidence": confidence}
                juror["votes"].append(vote)
                logger.debug(
                    f"Vote recorded from {juror_id}: "
                    f"score={score}, confidence={confidence}"
                )
                return True

        logger.warning(f"Juror not found: {juror_id}")
        return False

    def aggregate_votes(self) -> Dict[str, Any]:
        """
        Aggregate votes from all jurors

        Args:
            Uses all votes from current panel

        Returns:
            Aggregation results
        """
        if len(self.jurors) < self.min_jurors:
            logger.warning(
                f"Not enough jurors ({len(self.jurors)} "
                f"< {self.min_jurors})"
            )

        # Extract scores and weights
        scores = []
        weights = []
        confidences = []

        for juror in self.jurors:
            if juror["votes"]:
                latest_vote = juror["votes"][-1]
                scores.append(latest_vote["score"])
                confidences.append(
                    latest_vote.get("confidence", 0.5)
                )
                weights.append(
                    self.voting_weights.get(juror["id"], 1.0)
                )

        if not scores:
            return {
                "aggregate_score": 5.0,
                "confidence": 0.0,
                "voter_count": 0,
            }

        # Calculate weighted average
        weighted_score = calculate_weighted_average(scores, weights)

        # Calculate agreement level
        agreement = self._calculate_agreement(scores)

        # Average confidence
        avg_confidence = sum(confidences) / len(confidences)

        result = {
            "aggregate_score": round(weighted_score, 2),
            "individual_scores": scores,
            "weights": weights,
            "confidence": round(
                (avg_confidence + agreement) / 2, 2
            ),
            "agreement_level": round(agreement, 2),
            "voter_count": len(scores),
            "consensus": agreement > 0.7,
        }

        logger.debug(f"Vote aggregation: {result}")
        return result

    def _recalculate_weights(self) -> None:
        """Recalculate voting weights based on expertise"""
        total_expertise = sum(
            j.get("expertise", 1.0) for j in self.jurors
        )

        if total_expertise == 0:
            total_expertise = len(self.jurors)

        for juror in self.jurors:
            weight = juror.get("expertise", 1.0) / total_expertise
            self.voting_weights[juror["id"]] = weight

        logger.debug(
            f"Updated voting weights: {self.voting_weights}"
        )

    def _calculate_agreement(self, scores: List[float]) -> float:
        """
        Calculate agreement level among jurors

        Returns:
            Agreement score (0-1)
        """
        if len(scores) < 2:
            return 1.0

        # Calculate standard deviation
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(
            scores
        )
        std_dev = variance ** 0.5

        # Convert std_dev to agreement (lower std = higher agreement)
        max_std = 5.0  # Maximum possible std_dev for 0-10 range
        agreement = max(0, 1 - (std_dev / max_std))

        return agreement

    def get_jury_stats(self) -> Dict[str, Any]:
        """Get statistics about jury composition"""
        return {
            "juror_count": len(self.jurors),
            "total_votes": sum(
                len(j.get("votes", [])) for j in self.jurors
            ),
            "juror_types": [j["type"] for j in self.jurors],
            "expertise_levels": [
                j.get("expertise", 1.0) for j in self.jurors
            ],
            "avg_expertise": (
                sum(j.get("expertise", 1.0) for j in self.jurors)
                / len(self.jurors)
                if self.jurors
                else 0.0
            ),
        }
