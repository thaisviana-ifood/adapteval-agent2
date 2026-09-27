"""Process-wide Langfuse LangChain/LangGraph callback handler

Mirrors langfuse_client.py's lazy, degrade-gracefully pattern: when Langfuse
credentials aren't configured the handler is simply None and callers skip
attaching it, so LangGraph execution is unaffected either way.
"""

from typing import Optional

from langfuse.langchain import CallbackHandler

from src.shared.llm.langfuse_client import get_langfuse_client
from src.shared.logger import get_logger

logger = get_logger(__name__)

_handler: Optional[CallbackHandler] = None
_init_attempted = False


def get_langfuse_callback_handler() -> Optional[CallbackHandler]:
    """Return the shared Langfuse CallbackHandler for LangGraph/LangChain runs

    Per https://langfuse.com/integrations/frameworks/langgraph: pass the
    returned handler as `config={"callbacks": [handler]}` on every
    graph.invoke()/stream() call, and wrap the call in
    `langfuse.propagate_attributes(...)` to attach session_id/user_id/tags.

    Returns None when Langfuse credentials are not configured.
    """
    global _handler, _init_attempted

    if _init_attempted:
        return _handler

    _init_attempted = True
    if get_langfuse_client() is None:
        return None

    try:
        _handler = CallbackHandler()
    except Exception as e:
        logger.warning(f"Could not initialize Langfuse callback handler: {e}")
        _handler = None

    return _handler
