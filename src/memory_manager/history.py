"""History tracking for metrics and responses"""

from typing import List, Dict, Any
from datetime import datetime

from src.shared.logger import get_logger

logger = get_logger(__name__)


class HistoryTracker:
    """Track history of evaluations and metrics"""

    def __init__(self):
        self.evaluation_history: List[Dict] = []
        self.metrics_history: List[Dict] = []
        self.response_times: List[float] = []

    def record_evaluation(
        self,
        evaluation_id: str,
        task_type: str,
        final_score: float,
        confidence: float,
        components: Dict[str, float],
    ) -> None:
        """
        Record evaluation in history

        Args:
            evaluation_id: Unique evaluation ID
            task_type: Type of task evaluated
            final_score: Final evaluation score
            confidence: Overall confidence
            components: Component scores
        """
        record = {
            "evaluation_id": evaluation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "task_type": task_type,
            "final_score": final_score,
            "confidence": confidence,
            "components": components,
        }

        self.evaluation_history.append(record)
        logger.debug(f"Recorded evaluation: {evaluation_id}")

    def record_metrics(
        self,
        metric_name: str,
        value: float,
        tags: Dict[str, str] = None,
    ) -> None:
        """
        Record metric value

        Args:
            metric_name: Name of metric
            value: Metric value
            tags: Optional tags for grouping
        """
        record = {
            "metric": metric_name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": tags or {},
        }

        self.metrics_history.append(record)
        logger.debug(
            f"Recorded metric: {metric_name} = {value}"
        )

    def record_response_time(self, duration_seconds: float) -> None:
        """Record response processing time"""
        self.response_times.append(duration_seconds)

    def get_evaluation_history(
        self, limit: int = 100
    ) -> List[Dict]:
        """Get recent evaluation history"""
        return self.evaluation_history[-limit:]

    def get_metrics_history(
        self, metric_name: str = None, limit: int = 100
    ) -> List[Dict]:
        """Get metrics history"""
        if metric_name:
            filtered = [
                m
                for m in self.metrics_history
                if m["metric"] == metric_name
            ]
            return filtered[-limit:]
        return self.metrics_history[-limit:]

    def get_average_score_by_task(self) -> Dict[str, float]:
        """Get average score by task type"""
        task_scores: Dict[str, List[float]] = {}

        for record in self.evaluation_history:
            task = record["task_type"]
            score = record["final_score"]

            if task not in task_scores:
                task_scores[task] = []
            task_scores[task].append(score)

        return {
            task: sum(scores) / len(scores)
            for task, scores in task_scores.items()
        }

    def get_performance_trends(
        self, window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze performance trends

        Args:
            window_size: Size of sliding window

        Returns:
            Trend analysis
        """
        if len(self.evaluation_history) < window_size:
            return {}

        recent = self.evaluation_history[-window_size:]
        older = self.evaluation_history[-2 * window_size : -window_size]

        if not older:
            return {}

        recent_avg = (
            sum(r["final_score"] for r in recent) / len(recent)
        )
        older_avg = (
            sum(r["final_score"] for r in older) / len(older)
        )

        trend = "improving" if recent_avg > older_avg else "declining"
        change = round(recent_avg - older_avg, 2)

        return {
            "trend": trend,
            "change": change,
            "recent_avg": round(recent_avg, 2),
            "previous_avg": round(older_avg, 2),
        }

    def get_response_time_stats(self) -> Dict[str, float]:
        """Get response time statistics"""
        if not self.response_times:
            return {}

        times = self.response_times
        return {
            "avg_time": round(sum(times) / len(times), 2),
            "min_time": round(min(times), 2),
            "max_time": round(max(times), 2),
            "total_requests": len(times),
        }

    def clear_old_history(self, days: int = 30) -> None:
        """Clear history older than specified days"""
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        initial_count = len(self.evaluation_history)
        self.evaluation_history = [
            record
            for record in self.evaluation_history
            if datetime.fromisoformat(record["timestamp"])
            > cutoff_date
        ]
        removed = initial_count - len(self.evaluation_history)

        logger.info(
            f"Removed {removed} evaluation records older than "
            f"{days} days"
        )

    def export_history(self) -> Dict[str, Any]:
        """Export all history"""
        return {
            "evaluations": self.evaluation_history,
            "metrics": self.metrics_history,
            "response_times": self.response_times,
            "exported_at": datetime.utcnow().isoformat(),
        }
