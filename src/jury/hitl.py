"""Human-in-the-loop evaluation system"""

from typing import Dict, Any, Optional
from datetime import datetime

from src.shared.logger import get_logger

logger = get_logger(__name__)


class HumanInTheLoop:
    """Manage human review and feedback in evaluation"""

    def __init__(self):
        self.pending_reviews: Dict[str, Dict] = {}
        self.feedback_history: list = []

    def request_review(
        self,
        evaluation_id: str,
        response: str,
        criteria: Dict[str, Any],
        automated_score: float,
        reason: str = "Low confidence in automated evaluation",
    ) -> Dict[str, Any]:
        """
        Request human review of evaluation

        Args:
            evaluation_id: Unique evaluation ID
            response: Response being evaluated
            criteria: Evaluation criteria
            automated_score: Score from automated evaluation
            reason: Reason for requesting review

        Returns:
            Review request details
        """
        review_request = {
            "evaluation_id": evaluation_id,
            "response": response,
            "criteria": criteria,
            "automated_score": automated_score,
            "reason": reason,
            "requested_at": datetime.utcnow().isoformat(),
            "status": "pending",
            "human_score": None,
            "human_feedback": None,
        }

        self.pending_reviews[evaluation_id] = review_request
        logger.info(
            f"Human review requested for evaluation: "
            f"{evaluation_id} - {reason}"
        )

        return review_request

    def submit_feedback(
        self,
        evaluation_id: str,
        human_score: float,
        feedback: str,
        approved: bool = True,
    ) -> Dict[str, Any]:
        """
        Submit human feedback on evaluation

        Args:
            evaluation_id: Evaluation ID
            human_score: Human-assigned score
            feedback: Textual feedback
            approved: Whether human approves automated score

        Returns:
            Updated evaluation record
        """
        if evaluation_id not in self.pending_reviews:
            logger.warning(
                f"No pending review for: {evaluation_id}"
            )
            return {}

        review = self.pending_reviews[evaluation_id]
        review["human_score"] = human_score
        review["human_feedback"] = feedback
        review["approved"] = approved
        review["reviewed_at"] = datetime.utcnow().isoformat()
        review["status"] = "reviewed"

        # Calculate discrepancy
        discrepancy = abs(
            review["automated_score"] - human_score
        )
        review["discrepancy"] = discrepancy
        review["significant_discrepancy"] = discrepancy > 2.0

        self.feedback_history.append(review)

        logger.info(
            f"Human feedback submitted for: {evaluation_id} "
            f"(automated: {review['automated_score']}, "
            f"human: {human_score})"
        )

        return review

    def get_pending_reviews(self) -> Dict[str, Dict]:
        """Get all pending review requests"""
        return {
            k: v
            for k, v in self.pending_reviews.items()
            if v["status"] == "pending"
        }

    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get summary of human feedback

        Returns:
            Summary statistics
        """
        if not self.feedback_history:
            return {
                "total_reviews": 0,
                "average_discrepancy": 0.0,
                "approval_rate": 0.0,
                "significant_discrepancies": 0,
            }

        approved = sum(
            1 for f in self.feedback_history if f.get("approved")
        )
        significant_disc = sum(
            1
            for f in self.feedback_history
            if f.get("significant_discrepancy")
        )
        avg_discrepancy = (
            sum(f.get("discrepancy", 0) for f in self.feedback_history)
            / len(self.feedback_history)
        )

        return {
            "total_reviews": len(self.feedback_history),
            "approved_count": approved,
            "approval_rate": (
                approved / len(self.feedback_history)
                if self.feedback_history
                else 0.0
            ),
            "average_discrepancy": round(avg_discrepancy, 2),
            "significant_discrepancies": significant_disc,
        }

    def should_request_review(
        self,
        confidence: float,
        confidence_threshold: float = 0.7,
        score: float = 5.0,
    ) -> bool:
        """
        Determine if human review should be requested

        Args:
            confidence: Confidence score (0-1)
            confidence_threshold: Threshold for automatic approval
            score: Predicted score

        Returns:
            Whether review should be requested
        """
        # Request review if confidence is low
        if confidence < confidence_threshold:
            return True

        # Request review if score is borderline (around 5)
        if 4.5 <= score <= 5.5:
            return confidence < (confidence_threshold + 0.2)

        return False

    def get_review_queue(self) -> list:
        """Get list of reviews awaiting input"""
        pending = self.get_pending_reviews()
        return [
            {
                "evaluation_id": eid,
                "requested_at": review["requested_at"],
                "reason": review["reason"],
            }
            for eid, review in pending.items()
        ]

    def get_evaluation_with_feedback(
        self, evaluation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get evaluation with human feedback if available

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Evaluation record with feedback, or None
        """
        for review in self.feedback_history:
            if review["evaluation_id"] == evaluation_id:
                return review

        return None
