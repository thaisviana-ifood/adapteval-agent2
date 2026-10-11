"""Configuration and environment variables management"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ANNOTATED_DATA_DIR = DATA_DIR / "annotated"
OUTPUT_DATA_DIR = DATA_DIR / "output"

# Ensure data directories exist
for directory in [RAW_DATA_DIR, ANNOTATED_DATA_DIR, OUTPUT_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# LLM Configuration (DeepSeek, OpenAI-compatible API)
# This is the model used by the deep agent's own conversational front-end
# (src/agent/deep_agent.py), not by the jury -- see JURY_PROVIDERS below.
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-v4-pro")
LLM_API_KEY: str = os.getenv("DEEP_SEEK_API_KEY", "")
LLM_BASE_URL: str = os.getenv("DEEP_SEEK_BASE_URL", "https://api.deepseek.com")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))

# Retry policy, applied per model by src/shared/llm/client.py. Attempts are
# spaced by exponential backoff with jitter (and by the provider's own
# Retry-After when it sends one), capped at LLM_RETRY_MAX_DELAY so one stuck
# call cannot stall a batch for minutes. Only failures a retry can fix are
# retried -- see the client's docstring for which.
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
LLM_RETRY_MAX_DELAY: float = float(os.getenv("LLM_RETRY_MAX_DELAY", "30.0"))

# Every LLM provider this project has credentials for, by name. The jury
# takes one juror per provider and the shared client takes its fallback
# chain from the same place, so a vendor is configured once.
PROVIDERS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "model": os.getenv("LLM_MODEL_DEEP_SEEK", "deepseek-v4-pro"),
        "api_key": os.getenv("DEEP_SEEK_API_KEY", ""),
        "base_url": os.getenv("DEEP_SEEK_BASE_URL", "https://api.deepseek.com"),
    },
    "openai": {
        "model": os.getenv("LLM_MODEL_OPEN_AI", "gpt-4o"),
        "api_key": os.getenv("OPEN_AI_API_KEY", ""),
        "base_url": os.getenv("OPEN_AI_BASE_URL", "https://api.openai.com/v1"),
    },
    "typesafe": {
        "model": os.getenv("LLM_MODEL_TYPESAFE", ""),
        "api_key": os.getenv("TYPESAFE_API_KEY", ""),
        "base_url": os.getenv("TYPESAFE_BASE_URL", ""),
    },
}

# Providers that speak the OpenAI chat-completions API, and so can stand in
# for one another inside LLMClient. Typesafe cannot: it has its own wire
# format and its own client (src/shared/llm/typesafe_client.py), which is
# why it is a juror but never a fallback.
OPENAI_COMPATIBLE_PROVIDERS = frozenset({"deepseek", "openai"})

# Jury Configuration -- one LLM provider per juror, so the panel is never
# just one vendor's model grading itself. Each entry is passed as the model
# parameters for one juror in src.jury.evaluators.EvaluatorPanel.
JURY_PROVIDERS: List[Dict[str, str]] = [
    {"name": name, **provider} for name, provider in PROVIDERS.items()
]

# Fallback chain for LLMClient: provider names, in order, tried when the
# primary model has exhausted its retries or cannot serve the request at all
# (bad key, model not found, provider down). Set LLM_FALLBACK_PROVIDERS= to
# an empty value to disable fallbacks entirely.
#
# A fallback answers with a different vendor's model, which is the right
# trade for a pipeline stage that just needs an answer and the wrong one for
# the jury (whose independence is the point) and for an accuracy measurement
# (whose report names one model) -- both of those pass fallbacks=() instead.
LLM_FALLBACK_PROVIDER_NAMES: List[str] = [
    name.strip().lower()
    for name in os.getenv("LLM_FALLBACK_PROVIDERS", "openai").split(",")
    if name.strip()
]
LLM_FALLBACK_PROVIDERS: List[Dict[str, str]] = [
    {"name": name, **PROVIDERS[name]}
    for name in LLM_FALLBACK_PROVIDER_NAMES
    if name in PROVIDERS
    and name in OPENAI_COMPATIBLE_PROVIDERS
    # A provider with no key or no model would fail on every call, so it is
    # not a fallback -- it is a slower way to report the same failure.
    and PROVIDERS[name]["api_key"]
    and PROVIDERS[name]["model"]
]

# Jurors judge every rubric criterion independently in one call, and a
# reasoning model (e.g. deepseek-v4-pro) spends part of this budget on
# hidden reasoning tokens before it writes any visible verdict -- with the
# default LLM_MAX_TOKENS that reasoning alone can exhaust the budget and
# truncate/empty the output on a rubric with many criteria. Give jurors a
# much larger budget than other pipeline stages need.
JURY_LLM_MAX_TOKENS: int = int(os.getenv("JURY_LLM_MAX_TOKENS", "8000"))

# The jurors aren't the only stage a reasoning model can starve: every
# pipeline classifier that asks for structured JSON spends hidden reasoning
# tokens first, and on an empty budget the API returns
# finish_reason="length" with EMPTY content -- which each stage then reports
# as a degraded field rather than an error, so it fails silently. Measured on
# deepseek-v4-pro, the key-phrase prompt alone spends ~7.4k reasoning tokens
# on a single conversation, so the default LLM_MAX_TOKENS (2048) never
# produced any output at all. Give these stages the same headroom as jurors.
CLASSIFIER_LLM_MAX_TOKENS: int = int(
    os.getenv("CLASSIFIER_LLM_MAX_TOKENS", "8000")
)

# Classification is not generation: picking a label or a sentiment is a task
# with one right answer, so sampling only adds variance. At the default
# LLM_TEMPERATURE (0.7) repeated runs over the same inputs disagree with
# themselves, which makes a measured accuracy change impossible to attribute
# to the prompt rather than to the sampler. Scored on the Bitext goldenset
# via scripts/eval_intent_accuracy.py, two runs of the same prompt differed
# by ~1pp from sampling alone.
CLASSIFIER_LLM_TEMPERATURE: float = float(
    os.getenv("CLASSIFIER_LLM_TEMPERATURE", "0.0")
)

# Database Configuration
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_USER: str = os.getenv("DB_USER", "admin")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
DB_NAME: str = os.getenv("DB_NAME", "adapteval_agent")

# Set to "true" to persist evaluation/metrics history in PostgreSQL
# (see src/memory_manager/postgres_store.py) instead of keeping it
# only in memory.
USE_POSTGRES_MEMORY: bool = os.getenv("USE_POSTGRES_MEMORY", "False").lower() == "true"

# The rubric catalog (src/rule_generator/criteria_store.py) is read from
# PostgreSQL on every evaluation, so this defaults to on -- unlike the
# history above, which is an optional archive. Setting it to "false" (or
# simply having no reachable database) makes the catalog in-process: the
# same seeded criteria are served, but criteria created during a run are
# forgotten when it ends.
USE_POSTGRES_CRITERIA: bool = (
    os.getenv("USE_POSTGRES_CRITERIA", "True").lower() == "true"
)

# Vector DB Configuration (for memory)
VECTOR_DB_HOST: str = os.getenv("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT: int = int(os.getenv("VECTOR_DB_PORT", "6379"))
VECTOR_DB_NAME: str = os.getenv("VECTOR_DB_NAME", "adapteval_vectors")

# Logging Configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", "logs/adapteval.log")

# Processing Configuration
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4"))
ASYNC_PROCESSING: bool = os.getenv("ASYNC_PROCESSING", "False").lower() == "true"

# Evaluation Configuration
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MIN_EVALUATORS: int = int(os.getenv("MIN_EVALUATORS", "3"))
USE_HUMAN_IN_THE_LOOP: bool = os.getenv("USE_HUMAN_IN_THE_LOOP", "True").lower() == "true"

# Memory Manager Configuration
CACHE_SIZE_MB: int = int(os.getenv("CACHE_SIZE_MB", "1024"))
CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "24"))
ENABLE_CALIBRATION: bool = os.getenv("ENABLE_CALIBRATION", "True").lower() == "true"

# Langfuse Configuration (prompt management)
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")


def validate_config() -> bool:
    """Validate required configuration parameters"""
    required = ["LLM_API_KEY", "LLM_MODEL"]
    missing = [param for param in required if not globals().get(param)]

    # A fallback that was asked for but dropped (unknown name, no key, not
    # OpenAI-compatible) would otherwise only show up as an outage, so say so
    # here instead. It is a warning, not a failure: the primary still works.
    usable = {provider["name"] for provider in LLM_FALLBACK_PROVIDERS}
    unusable = [
        name for name in LLM_FALLBACK_PROVIDER_NAMES if name not in usable
    ]
    if unusable:
        print(
            f"Warning: LLM fallback provider(s) not usable and ignored: "
            f"{unusable} (unknown name, missing api_key/model, or not "
            f"OpenAI-compatible)"
        )

    if missing:
        print(f"Warning: Missing required config parameters: {missing}")
        return False
    return True
