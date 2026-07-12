"""Utility functions and helpers"""

import json
from typing import Any, Dict, Optional, List
from datetime import datetime

from src.shared.logger import get_logger

logger = get_logger(__name__)


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """Safely serialize object to JSON string"""
    try:
        return json.dumps(obj, indent=indent, default=str)
    except Exception as e:
        logger.error(f"Failed to serialize to JSON: {e}")
        return "{}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def timestamp_now() -> str:
    """Get current ISO timestamp"""
    return datetime.utcnow().isoformat()


def parse_conversation_turns(
    conversation: str, delimiter: str = "\n\n"
) -> List[Dict[str, str]]:
    """
    Parse conversation string into structured turns

    Args:
        conversation: Raw conversation text
        delimiter: Turn delimiter (default: double newline)

    Returns:
        List of {role, content} dicts
    """
    turns = []
    parts = conversation.split(delimiter)

    for part in parts:
        if not part.strip():
            continue

        # Try to identify role (User/Assistant)
        if part.strip().startswith("User:"):
            turns.append(
                {"role": "user", "content": part.replace("User:", "").strip()}
            )
        elif part.strip().startswith("Assistant:"):
            turns.append(
                {
                    "role": "assistant",
                    "content": part.replace("Assistant:", "").strip(),
                }
            )
        else:
            # Default to assistant if role unclear
            turns.append({"role": "assistant", "content": part.strip()})

    return turns


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary for display"""
    lines = ["Metrics:"]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def calculate_average(values: List[float]) -> float:
    """Calculate average of numeric values"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_weighted_average(
    values: List[float], weights: Optional[List[float]] = None
) -> float:
    """Calculate weighted average"""
    if not values:
        return 0.0

    if weights is None:
        weights = [1.0] * len(values)

    if len(values) != len(weights):
        logger.warning(
            f"Value and weight lengths don't match: {len(values)} vs {len(weights)}"
        )
        return calculate_average(values)

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    return sum(v * w for v, w in zip(values, weights)) / total_weight
