"""Tests for shared services layer"""

import pytest
from src.shared.logger import get_logger
from src.shared.llm.prompts import PromptManager
from src.shared.utils import (
    safe_json_loads,
    safe_json_dumps,
    calculate_weighted_average,
)


class TestLogger:
    """Test logger functionality"""

    def test_get_logger(self):
        """Test logger initialization"""
        logger = get_logger(__name__)
        assert logger is not None
        assert logger.name == __name__


class TestPromptManager:
    """Test prompt management"""

    def test_prompt_manager_init(self):
        """Test PromptManager initialization"""
        pm = PromptManager()
        assert pm is not None
        assert len(pm.list_templates()) > 0

    def test_get_template(self):
        """Test getting a template"""
        pm = PromptManager()
        template = pm.get_template("context_analysis")
        assert template is not None
        assert "$" in template

    def test_format_prompt(self):
        """Test prompt formatting"""
        pm = PromptManager()
        formatted = pm.format_prompt(
            "context_analysis", conversation="test"
        )
        assert formatted is not None
        assert "test" in formatted


class TestUtils:
    """Test utility functions"""

    def test_safe_json_loads(self):
        """Test safe JSON loading"""
        result = safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}

        result = safe_json_loads("invalid json", default={})
        assert result == {}

    def test_safe_json_dumps(self):
        """Test safe JSON dumping"""
        data = {"key": "value"}
        result = safe_json_dumps(data)
        assert "key" in result
        assert "value" in result

    def test_calculate_weighted_average(self):
        """Test weighted average calculation"""
        values = [10.0, 20.0, 30.0]
        weights = [1.0, 2.0, 3.0]

        avg = calculate_weighted_average(values, weights)
        assert avg > 0
        assert avg <= 30


if __name__ == "__main__":
    pytest.main([__file__])
