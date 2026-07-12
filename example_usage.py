"""
Example usage of the Adaptive Jury Agent

This script demonstrates how to use the system to evaluate a response
based on multi-turn conversation context.
"""

import asyncio
from src.main import AdaptiveJuryAgent
from src.shared.logger import get_logger
from src.shared.utils import safe_json_dumps

logger = get_logger(__name__)


async def example_basic_evaluation():
    """Basic evaluation example"""
    logger.info("=== Basic Evaluation Example ===")

    # Initialize agent
    agent = AdaptiveJuryAgent()

    # Define conversation and response
    conversation = """
User: Can you explain what machine learning is?"""