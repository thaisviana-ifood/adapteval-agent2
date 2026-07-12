"""Tests for memory management module"""

import pytest
from src.memory_manager import (
    CacheManager,
    HistoryTracker,
    ErrorDetector,
    CalibrationManager,
)


class TestCacheManager:
    """Test cache management"""

    def test_cache_operations(self):
        """Test basic cache operations"""
        cache = CacheManager(max_size_mb=100, ttl_hours=1)

        # Set and get
        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")
        assert result == {"data": "value1"}

        # Miss
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = CacheManager()
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert stats["hits"] >= 1


class TestHistoryTracker:
    """Test history tracking"""

    def test_record_evaluation(self):
        """Test recording evaluations"""
        tracker = HistoryTracker()

        tracker.record_evaluation(
            "eval_001",
            "question_answering",
            8.5,
            0.9,
            {"jury": 8.5, "heuristic": 8.0},
        )

        history = tracker.get_evaluation_history()
        assert len(history) > 0
        assert history[0]["evaluation_id"] == "eval_001"

    def test_get_average_score_by_task(self):
        """Test averaging by task type"""
        tracker = HistoryTracker()

        tracker.record_evaluation(
            "eval_001",
            "qa",
            8.5,
            0.9,
            {},
        )
        tracker.record_evaluation(
            "eval_002",
            "qa",
            7.5,
            0.8,
            {},
        )

        avgs = tracker.get_average_score_by_task()
        assert "qa" in avgs
        assert avgs["qa"] == 8.0


class TestErrorDetector:
    """Test error detection"""

    def test_detect_errors(self):
        """Test error detection"""
        detector = ErrorDetector()

        # Create evaluation with errors
        eval_with_errors = {
            "final_score": 15,  # Out of range
            "confidence": 1.5,  # Out of range
            "reasoning": "",  # Empty
        }

        errors = detector.detect_errors(eval_with_errors, {})
        assert len(errors) > 0


class TestCalibrationManager:
    """Test calibration"""

    def test_calibration_init(self):
        """Test calibration initialization"""
        calibrator = CalibrationManager()
        assert calibrator is not None

    def test_load_annotated_data(self):
        """Test loading annotated data"""
        calibrator = CalibrationManager()

        data = [
            {"score": 8.0},
            {"score": 7.5},
            {"score": 9.0},
        ]

        calibrator.load_annotated_data(data)
        assert len(calibrator.annotated_data) == 3


if __name__ == "__main__":
    pytest.main([__file__])
