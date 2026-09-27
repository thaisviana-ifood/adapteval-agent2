"""PostgreSQL-backed persistence for evaluation and metrics history"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.shared.infra.database import PostgresConnector
from src.shared.logger import get_logger

logger = get_logger(__name__)

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "docker"
    / "postgres"
    / "init.sql"
)


class PostgresMemoryStore:
    """Persists jury memory (evaluation & metrics history) in PostgreSQL"""

    def __init__(self, connector: Optional[PostgresConnector] = None):
        self.connector = connector or PostgresConnector()
        self._connected = False

    def connect(self) -> bool:
        """Establish the underlying PostgreSQL connection"""
        self._connected = self.connector.connect()
        return self._connected

    def disconnect(self) -> None:
        """Close the underlying PostgreSQL connection"""
        if self._connected:
            self.connector.disconnect()
            self._connected = False

    def ensure_schema(self) -> bool:
        """Create the memory tables if they don't exist yet"""
        if not self._connected:
            return False

        try:
            schema_sql = SCHEMA_PATH.read_text()
            cursor = self.connector.connection.cursor()
            cursor.execute(schema_sql)
            self.connector.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to apply memory schema: {e}")
            return False

    def save_evaluation(self, record: Dict[str, Any]) -> bool:
        """Persist a single evaluation history record"""
        if not self._connected:
            return False

        try:
            cursor = self.connector.connection.cursor()
            cursor.execute(
                """
                INSERT INTO evaluation_history
                    (evaluation_id, task_type, final_score, confidence, components)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    record["evaluation_id"],
                    record["task_type"],
                    record["final_score"],
                    record["confidence"],
                    json.dumps(record.get("components", {})),
                ),
            )
            self.connector.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to persist evaluation: {e}")
            return False

    def save_metric(self, record: Dict[str, Any]) -> bool:
        """Persist a single metric history record"""
        if not self._connected:
            return False

        try:
            cursor = self.connector.connection.cursor()
            cursor.execute(
                """
                INSERT INTO metrics_history (metric_name, value, tags)
                VALUES (%s, %s, %s)
                """,
                (
                    record["metric"],
                    record["value"],
                    json.dumps(record.get("tags", {})),
                ),
            )
            self.connector.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to persist metric: {e}")
            return False

    def get_evaluation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Load the most recent evaluation history records, oldest first"""
        if not self._connected:
            return []

        try:
            cursor = self.connector.connection.cursor()
            cursor.execute(
                """
                SELECT evaluation_id, task_type, final_score, confidence,
                       components, created_at
                FROM evaluation_history
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            cursor.close()

            records = [
                {
                    "evaluation_id": row[0],
                    "task_type": row[1],
                    "final_score": row[2],
                    "confidence": row[3],
                    "components": row[4],
                    "timestamp": row[5].isoformat(),
                }
                for row in rows
            ]
            return list(reversed(records))
        except Exception as e:
            logger.error(f"Failed to load evaluation history: {e}")
            return []

    def get_metrics_history(
        self, metric_name: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Load the most recent metrics history records, oldest first"""
        if not self._connected:
            return []

        try:
            cursor = self.connector.connection.cursor()
            if metric_name:
                cursor.execute(
                    """
                    SELECT metric_name, value, tags, created_at
                    FROM metrics_history
                    WHERE metric_name = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (metric_name, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT metric_name, value, tags, created_at
                    FROM metrics_history
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            cursor.close()

            records = [
                {
                    "metric": row[0],
                    "value": row[1],
                    "tags": row[2],
                    "timestamp": row[3].isoformat(),
                }
                for row in rows
            ]
            return list(reversed(records))
        except Exception as e:
            logger.error(f"Failed to load metrics history: {e}")
            return []
