"""Cache management for performance optimization"""

from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import hashlib

from src.config import CACHE_SIZE_MB, CACHE_TTL_HOURS
from src.shared.logger import get_logger
from src.shared.utils import safe_json_dumps

logger = get_logger(__name__)


class CacheManager:
    """Manage caching of evaluations and analyses"""

    def __init__(
        self,
        max_size_mb: int = CACHE_SIZE_MB,
        ttl_hours: int = CACHE_TTL_HOURS,
    ):
        self.max_size_mb = max_size_mb
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: Dict[str, Dict] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_bytes": 0,
        }

    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        data_str = safe_json_dumps(data)
        return hashlib.md5(data_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key not in self.cache:
            self.cache_stats["misses"] += 1
            logger.debug(f"Cache miss for key: {key}")
            return None

        entry = self.cache[key]

        # Check expiration
        if datetime.utcnow() > entry["expires_at"]:
            del self.cache[key]
            logger.debug(f"Cache entry expired: {key}")
            return None

        self.cache_stats["hits"] += 1
        logger.debug(f"Cache hit for key: {key}")
        return entry["value"]

    def set(self, key: str, value: Any) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            Success status
        """
        entry = {
            "value": value,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": datetime.utcnow() + self.ttl,
            "size_bytes": len(safe_json_dumps(value)),
        }

        self.cache[key] = entry
        self.cache_stats["total_bytes"] += entry["size_bytes"]

        # Check size limit
        if self.cache_stats["total_bytes"] > (self.max_size_mb * 1024 * 1024):
            self._evict_oldest()

        logger.debug(f"Cached value for key: {key}")
        return True

    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key in self.cache:
            self.cache_stats["total_bytes"] -= (
                self.cache[key]["size_bytes"]
            )
            del self.cache[key]
            logger.debug(f"Deleted cache entry: {key}")
            return True
        return False

    def clear(self) -> None:
        """Clear entire cache"""
        self.cache.clear()
        self.cache_stats["total_bytes"] = 0
        logger.info("Cache cleared")

    def _evict_oldest(self) -> None:
        """Evict oldest entries when cache is full"""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]["created_at"],
        )

        # Remove 20% of entries
        evict_count = max(1, len(sorted_entries) // 5)
        for key, entry in sorted_entries[:evict_count]:
            self.cache_stats["total_bytes"] -= entry["size_bytes"]
            del self.cache[key]

        logger.debug(f"Evicted {evict_count} oldest entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (
            self.cache_stats["hits"]
            / (
                self.cache_stats["hits"]
                + self.cache_stats["misses"]
            )
            if (
                self.cache_stats["hits"]
                + self.cache_stats["misses"]
            ) > 0
            else 0
        )

        return {
            "entries": len(self.cache),
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": round(hit_rate, 2),
            "total_size_mb": round(
                self.cache_stats["total_bytes"] / (1024 * 1024), 2
            ),
            "max_size_mb": self.max_size_mb,
        }

    def cache_agent_memory(
        self, agent_id: str, memory: Dict
    ) -> bool:
        """Cache agent memory mapping"""
        key = f"agent_memory_{agent_id}"
        return self.set(key, memory)

    def get_agent_memory(self, agent_id: str) -> Optional[Dict]:
        """Retrieve cached agent memory"""
        key = f"agent_memory_{agent_id}"
        return self.get(key)
