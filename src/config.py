"""Configuration and environment variables management"""

import os
from pathlib import Path
from typing import Optional

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
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-v4-pro")
LLM_API_KEY: str = os.getenv("DEEP_SEEK_API_KEY", "")
LLM_BASE_URL: str = os.getenv("DEEP_SEEK_BASE_URL", "https://api.deepseek.com")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))

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


def validate_config() -> bool:
    """Validate required configuration parameters"""
    required = ["LLM_API_KEY", "LLM_MODEL"]
    missing = [param for param in required if not globals().get(param)]

    if missing:
        print(f"Warning: Missing required config parameters: {missing}")
        return False
    return True
