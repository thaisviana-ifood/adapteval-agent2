"""Process-wide Langfuse client, shared by prompt management and LLM call tracing"""

from typing import Optional

from langfuse import Langfuse

from src.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
from src.shared.logger import get_logger

logger = get_logger(__name__)

_client: Optional[Langfuse] = None
_init_attempted = False


def get_langfuse_client() -> Optional[Langfuse]:
    """Return the shared Langfuse client, creating it on first call.

    Returns None when Langfuse credentials are not configured, so prompt
    management and LLM call tracing degrade gracefully to local-only behavior.
    """
    global _client, _init_attempted

    if _init_attempted:
        return _client

    _init_attempted = True
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        logger.warning(
            "Langfuse credentials not configured; prompt management and LLM "
            "call tracing are disabled"
        )
        return None

    try:
        _client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    except Exception as e:
        logger.warning(f"Could not initialize Langfuse client: {e}")
        _client = None

    return _client
