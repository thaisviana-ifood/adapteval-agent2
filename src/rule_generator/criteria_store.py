"""Persistent catalog of rubric criteria

The rule generator doesn't own its rubric: this catalog does. Every
criterion the pipeline can apply is a row in the `evaluation_criteria`
table (see `docker/postgres/init.sql`), read through `CriteriaRepository`:

* `mandatory=True` rows go into every rubric, whatever the task.
* `mandatory=False` rows are offered when the task matches
  `most_common_tasks` -- the same task, or a similar one per
  `SIMILAR_TASKS` -- or when the rule generator's context trigger for that
  name fires (`criteria.py:_CONTEXT_TRIGGERS`).
* Criteria invented at runtime (the LLM's `dynamic_criteria`, and the
  `success_criteria` of a task's objectives) are written here on first use,
  so from the next evaluation on they are ordinary catalog rows that a
  similar task can reuse.

Using a criterion updates its statistics: `avg_weight` is the running mean
of the weights it has been applied with, and `most_common_tasks` counts the
rubrics per task type it appeared in.

PostgreSQL is the store. When it is unreachable (no psycopg2, no server,
`USE_POSTGRES_CRITERIA=False`) the repository serves the same
`DEFAULT_CRITERIA` from memory and keeps runtime additions for the life of
the process, so no evaluation ever fails over catalog storage -- it just
stops accumulating history.
"""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import USE_POSTGRES_CRITERIA
from src.shared.infra.database import PostgresConnector
from src.shared.logger import get_logger

logger = get_logger(__name__)

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "docker"
    / "postgres"
    / "init.sql"
)

BOOLEAN = "boolean"
HEURISTIC = "heuristic"

_COLUMNS = (
    "name",
    "description",
    "criterion_type",
    "mandatory",
    "avg_weight",
    "most_common_tasks",
    "weight_samples",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM evaluation_criteria"


def _objective_criteria(task_type: str, names: List[str]) -> List[Dict[str, Any]]:
    """Seed rows for the `success_criteria` of one task's objectives

    `ObjectiveDefinition.get_objectives` phrases these as bare requirements
    ("Factually correct"); the rubric needs them as yes/no questions, in the
    same wording `CriteriaGenerator` has always used for them.
    """
    return [
        {
            "name": name,
            "description": f"Does the response satisfy: {name}?",
            "criterion_type": BOOLEAN,
            "mandatory": False,
            "avg_weight": 1.0,
            "most_common_tasks": {task_type: 1},
        }
        for name in names
    ]


# The rubric this repository is seeded with: every criterion the pipeline
# used to hard-code, now a catalog row. Keep this list as the single source
# of truth for the seed -- init.sql creates the table but inserts nothing.
DEFAULT_CRITERIA: List[Dict[str, Any]] = [
    # Base rubric: applied to every evaluation, hence mandatory.
    {
        "name": "Relevance",
        "description": "Is the response relevant to the query?",
        "criterion_type": BOOLEAN,
        "mandatory": True,
        "avg_weight": 1.0,
        "most_common_tasks": {},
    },
    {
        "name": "Correctness",
        "description": "Is the information in the response factually correct?",
        "criterion_type": BOOLEAN,
        "mandatory": True,
        "avg_weight": 1.0,
        "most_common_tasks": {},
    },
    {
        "name": "Clarity",
        "description": "Is the response clear and understandable?",
        "criterion_type": BOOLEAN,
        "mandatory": True,
        "avg_weight": 0.8,
        "most_common_tasks": {},
    },
    {
        "name": "Completeness",
        "description": "Does the response address all aspects of the request?",
        "criterion_type": BOOLEAN,
        "mandatory": True,
        "avg_weight": 0.9,
        "most_common_tasks": {},
    },
    # Context-triggered: selected by _CONTEXT_TRIGGERS in criteria.py, not
    # by task type, so they carry no task counts of their own at seed time.
    {
        "name": "Technical Depth",
        "description": (
            "Does the response address the technical complexity of the "
            "request appropriately?"
        ),
        "criterion_type": BOOLEAN,
        "mandatory": False,
        "avg_weight": 1.0,
        "most_common_tasks": {},
    },
    {
        "name": "Problem Solving",
        "description": "Does the response effectively solve the stated problem?",
        "criterion_type": BOOLEAN,
        "mandatory": False,
        "avg_weight": 1.2,
        "most_common_tasks": {},
    },
    # Task-specific: the success_criteria of each task in
    # ObjectiveDefinition.objectives_map.
    *_objective_criteria(
        "question_answering",
        ["Addresses all parts of question", "Factually correct", "Well-structured"],
    ),
    *_objective_criteria(
        "code_generation",
        ["Syntax correct", "Solves the problem", "No logical errors"],
    ),
    *_objective_criteria(
        "text_summarization",
        ["Preserves main points", "Concise", "No missing critical info"],
    ),
    *_objective_criteria(
        "general",
        ["Addresses user need", "Factually sound", "Well-organized"],
    ),
    # Heuristics: not jury questions but code checks in heuristics.py,
    # looked up there by name. Catalogued so the whole rubric lives in one
    # place and a check can be retired without a code change.
    {
        "name": "minimum_length",
        "description": "Response has minimum required length",
        "criterion_type": HEURISTIC,
        "mandatory": True,
        "avg_weight": 1.0,
        "most_common_tasks": {},
    },
    {
        "name": "no_empty_response",
        "description": "Response is not empty",
        "criterion_type": HEURISTIC,
        "mandatory": True,
        "avg_weight": 1.0,
        "most_common_tasks": {},
    },
    {
        "name": "coherence",
        "description": "Response is coherent",
        "criterion_type": HEURISTIC,
        "mandatory": True,
        "avg_weight": 1.0,
        "most_common_tasks": {},
    },
    {
        "name": "relevance",
        "description": "Response mentions relevant keywords",
        "criterion_type": HEURISTIC,
        "mandatory": True,
        "avg_weight": 1.0,
        "most_common_tasks": {},
    },
]

# Task types whose rubrics tend to need the same criteria, so a criterion
# learned on one is worth offering on the others when the task itself has no
# history yet. Keys cover TaskClassifier.task_types plus "general".
SIMILAR_TASKS: Dict[str, Tuple[str, ...]] = {
    "question_answering": ("reasoning", "instruction_following", "general"),
    "reasoning": ("question_answering", "classification", "general"),
    "instruction_following": ("question_answering", "general"),
    "classification": ("reasoning", "question_answering"),
    "code_generation": ("reasoning", "instruction_following"),
    "text_summarization": ("translation", "creative_writing"),
    "translation": ("text_summarization",),
    "creative_writing": ("text_summarization",),
    "general": ("question_answering", "instruction_following"),
}


def related_tasks(task_type: str) -> List[str]:
    """The task itself first, then the tasks whose criteria it can borrow"""
    task_type = task_type or "general"
    return [task_type, *SIMILAR_TASKS.get(task_type, ())]


def to_criterion(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a catalog row into the criterion dict the pipeline passes
    around (`name`/`description`/`criterion_type`/`weight`)

    The catalog stores the mean of the weights a criterion has been applied
    with; that mean is what the next rubric applies it with.
    """
    return {
        "name": row["name"],
        "description": row["description"],
        "criterion_type": row["criterion_type"],
        "weight": round(float(row.get("avg_weight", 1.0)), 4),
    }


def _default_rows(
    criterion_type: Optional[str] = None,
    mandatory: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """DEFAULT_CRITERIA rows (copies), optionally filtered"""
    rows = []
    for row in DEFAULT_CRITERIA:
        if criterion_type is not None and row["criterion_type"] != criterion_type:
            continue
        if mandatory is not None and row["mandatory"] != mandatory:
            continue
        rows.append({**copy.deepcopy(row), "weight_samples": 1})
    return rows


class _MemoryBackend:
    """Catalog kept in this process only -- the fallback when there is no
    database, and the store used by tests"""

    def __init__(self):
        self._rows: Dict[Tuple[str, str], Dict[str, Any]] = {}

    @staticmethod
    def _key(name: str, criterion_type: str) -> Tuple[str, str]:
        return (name.strip().lower(), criterion_type)

    def seed(self, rows: List[Dict[str, Any]]) -> int:
        inserted = 0
        for row in rows:
            key = self._key(row["name"], row["criterion_type"])
            if key not in self._rows:
                self._rows[key] = copy.deepcopy(row)
                inserted += 1
        return inserted

    def get(self, name: str, criterion_type: str) -> Optional[Dict[str, Any]]:
        row = self._rows.get(self._key(name, criterion_type))
        return copy.deepcopy(row) if row else None

    def create(self, row: Dict[str, Any]) -> Dict[str, Any]:
        self.seed([row])
        return self.get(row["name"], row["criterion_type"])

    def all(
        self,
        criterion_type: Optional[str] = None,
        mandatory: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        rows = [
            copy.deepcopy(row)
            for row in self._rows.values()
            if (criterion_type is None or row["criterion_type"] == criterion_type)
            and (mandatory is None or row["mandatory"] == mandatory)
        ]
        return sorted(rows, key=lambda r: (r["criterion_type"], r["name"].lower()))

    def for_tasks(
        self, tasks: List[str], criterion_type: str
    ) -> List[Dict[str, Any]]:
        wanted = set(tasks)
        return [
            copy.deepcopy(row)
            for row in self._rows.values()
            if row["criterion_type"] == criterion_type
            and not row["mandatory"]
            and wanted & set(row.get("most_common_tasks") or {})
        ]

    def delete(self, name: str, criterion_type: str) -> bool:
        return self._rows.pop(self._key(name, criterion_type), None) is not None

    def bump(
        self,
        name: str,
        criterion_type: str,
        task_type: Optional[str],
        weight: float,
    ) -> None:
        row = self._rows.get(self._key(name, criterion_type))
        if not row:
            return
        samples = int(row.get("weight_samples", 1))
        row["avg_weight"] = (
            float(row.get("avg_weight", 1.0)) * samples + float(weight)
        ) / (samples + 1)
        row["weight_samples"] = samples + 1
        if task_type:
            tasks = row.setdefault("most_common_tasks", {})
            tasks[task_type] = int(tasks.get(task_type, 0)) + 1


class _PostgresBackend:
    """Catalog stored in the `evaluation_criteria` table"""

    def __init__(self, connector: PostgresConnector):
        self.connector = connector

    @contextmanager
    def _cursor(self):
        """A cursor that rolls back on failure

        Without the rollback, one failed statement leaves the connection in
        an aborted transaction and every later query fails too -- which
        would turn a single bad write into a catalog-wide outage.
        """
        cursor = self.connector.connection.cursor()
        try:
            yield cursor
        except Exception:
            self.connector.connection.rollback()
            raise
        finally:
            cursor.close()

    def _query(self, sql: str, params: Tuple) -> List[Dict[str, Any]]:
        with self._cursor() as cursor:
            cursor.execute(sql, params)
            return [dict(zip(_COLUMNS, row)) for row in cursor.fetchall()]

    def ensure_schema(self) -> None:
        with self._cursor() as cursor:
            cursor.execute(SCHEMA_PATH.read_text())
            self.connector.connection.commit()

    def seed(self, rows: List[Dict[str, Any]]) -> int:
        with self._cursor() as cursor:
            inserted = 0
            for row in rows:
                cursor.execute(
                    """
                    INSERT INTO evaluation_criteria
                        (name, description, criterion_type, mandatory,
                         avg_weight, most_common_tasks, weight_samples)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (lower(name), criterion_type) DO NOTHING
                    """,
                    (
                        row["name"],
                        row["description"],
                        row["criterion_type"],
                        row["mandatory"],
                        row["avg_weight"],
                        json.dumps(row.get("most_common_tasks") or {}),
                        row.get("weight_samples", 1),
                    ),
                )
                inserted += cursor.rowcount
            self.connector.connection.commit()
            return inserted

    def get(self, name: str, criterion_type: str) -> Optional[Dict[str, Any]]:
        rows = self._query(
            f"{_SELECT} WHERE lower(name) = lower(%s) AND criterion_type = %s",
            (name, criterion_type),
        )
        return rows[0] if rows else None

    def create(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.seed([row])
        return self.get(row["name"], row["criterion_type"])

    def all(
        self,
        criterion_type: Optional[str] = None,
        mandatory: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if criterion_type is not None:
            clauses.append("criterion_type = %s")
            params.append(criterion_type)
        if mandatory is not None:
            clauses.append("mandatory = %s")
            params.append(mandatory)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._query(
            f"{_SELECT}{where} ORDER BY criterion_type, lower(name)",
            tuple(params),
        )

    def for_tasks(
        self, tasks: List[str], criterion_type: str
    ) -> List[Dict[str, Any]]:
        # `?|` is "has any of these top-level keys", which is exactly
        # "was used on any of these task types".
        return self._query(
            f"""
            {_SELECT}
            WHERE criterion_type = %s
              AND NOT mandatory
              AND most_common_tasks ?| %s::text[]
            """,
            (criterion_type, list(tasks)),
        )

    def delete(self, name: str, criterion_type: str) -> bool:
        with self._cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM evaluation_criteria
                WHERE lower(name) = lower(%s) AND criterion_type = %s
                """,
                (name, criterion_type),
            )
            deleted = cursor.rowcount
            self.connector.connection.commit()
            return bool(deleted)

    def bump(
        self,
        name: str,
        criterion_type: str,
        task_type: Optional[str],
        weight: float,
    ) -> None:
        with self._cursor() as cursor:
            task_update = ""
            params: List[Any] = [weight]
            if task_type:
                # Incrementing in SQL rather than read-modify-write keeps
                # the count correct when two evaluations run at once.
                task_update = """,
                    most_common_tasks = jsonb_set(
                        most_common_tasks,
                        %s::text[],
                        to_jsonb(
                            COALESCE((most_common_tasks ->> %s)::int, 0) + 1
                        ),
                        true
                    )"""
                params += [[task_type], task_type]
            params += [name, criterion_type]

            cursor.execute(
                f"""
                UPDATE evaluation_criteria
                SET avg_weight =
                        (avg_weight * weight_samples + %s)
                        / (weight_samples + 1),
                    weight_samples = weight_samples + 1,
                    updated_at = now(){task_update}
                WHERE lower(name) = lower(%s) AND criterion_type = %s
                """,
                tuple(params),
            )
            self.connector.connection.commit()


class CriteriaRepository:
    """Read/write access to the rubric catalog

    Connects lazily on first use and, if PostgreSQL is not available, falls
    back to an in-process catalog holding the same seed rows. Either way the
    callers (`CriteriaGenerator`, `HeuristicChecker`) only ever see catalog
    rows, never hard-coded rubrics.
    """

    def __init__(
        self,
        connector: Optional[PostgresConnector] = None,
        enabled: Optional[bool] = None,
    ):
        self._connector = connector
        self._enabled = USE_POSTGRES_CRITERIA if enabled is None else enabled
        self._backend = None
        self._persistent = False

    # -- lifecycle ------------------------------------------------------

    @property
    def is_persistent(self) -> bool:
        """Whether the catalog is backed by PostgreSQL (not just memory)"""
        self._ensure_backend()
        return self._persistent

    def _ensure_backend(self):
        if self._backend is not None:
            return self._backend

        if self._enabled:
            connector = self._connector or PostgresConnector()
            try:
                if connector.connect():
                    backend = _PostgresBackend(connector)
                    backend.ensure_schema()
                    self._backend = backend
                    self._persistent = True
                    self._connector = connector
                    # Only an empty catalog is seeded: re-adding every
                    # default on each startup would silently resurrect a
                    # criterion that was deliberately deleted, which would
                    # make the table a cache rather than the source of
                    # truth. `seed_defaults()` re-applies them on demand.
                    if not backend.all():
                        seeded = backend.seed(_default_rows())
                        logger.info(
                            f"Seeded {seeded} default criteria into the "
                            f"criteria catalog"
                        )
                    return self._backend
            except Exception as e:
                logger.error(f"Criteria catalog unavailable in PostgreSQL: {e}")
            logger.warning(
                "Falling back to the in-memory criteria catalog; criteria "
                "created in this run will not be persisted"
            )

        self._backend = _MemoryBackend()
        self._backend.seed(_default_rows())
        self._persistent = False
        return self._backend

    def close(self) -> None:
        """Release the database connection, if one was opened"""
        if self._persistent and self._connector:
            self._connector.disconnect()
        self._backend = None
        self._persistent = False

    def seed_defaults(self) -> int:
        """Insert any DEFAULT_CRITERIA row missing from the catalog

        Returns the number of rows inserted. Called on connect, and by
        `scripts/dump_criteria.py`.
        """
        return self._ensure_backend().seed(_default_rows())

    # -- reads ----------------------------------------------------------

    def get_mandatory(self, criterion_type: str = BOOLEAN) -> List[Dict[str, Any]]:
        """Criteria that go into every rubric, whatever the task"""
        rows = self._ensure_backend().all(
            criterion_type=criterion_type, mandatory=True
        )
        if not rows:
            # A catalog with no mandatory criterion would yield an empty
            # rubric and a meaningless evaluation, so prefer the seed.
            logger.warning(
                "No mandatory criteria in the catalog; using the defaults"
            )
            rows = _default_rows(criterion_type=criterion_type, mandatory=True)
        return rows

    def get_by_name(
        self, name: str, criterion_type: str = BOOLEAN
    ) -> Optional[Dict[str, Any]]:
        """One catalog row by name, or None if it isn't catalogued yet"""
        return self._ensure_backend().get(name, criterion_type)

    def get_for_task(
        self,
        task_type: str,
        criterion_type: str = BOOLEAN,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Optional criteria previously used on this task, or on a similar
        one (`SIMILAR_TASKS`)

        Ordered by how often they were used on the task itself first, then
        on the related tasks, so the most established criteria come first.
        """
        tasks = related_tasks(task_type)
        rows = self._ensure_backend().for_tasks(tasks, criterion_type)

        def rank(row: Dict[str, Any]) -> Tuple:
            counts = row.get("most_common_tasks") or {}
            exact = int(counts.get(tasks[0], 0))
            related = sum(int(counts.get(t, 0)) for t in tasks[1:])
            return (-exact, -related, row["name"].lower())

        rows.sort(key=rank)
        return rows[:limit] if limit else rows

    def list_all(
        self, criterion_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """The whole catalog -- used by the dump script"""
        return self._ensure_backend().all(criterion_type=criterion_type)

    # -- writes ---------------------------------------------------------

    def get_or_create(
        self,
        name: str,
        description: str,
        criterion_type: str = BOOLEAN,
        mandatory: bool = False,
        weight: float = 1.0,
        task_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the catalogued criterion, adding it if it is new

        This is how a criterion the pipeline invents at runtime (an LLM
        proposal, or an objective's success_criteria) becomes a reusable
        catalog row.
        """
        existing = self.get_by_name(name, criterion_type)
        if existing:
            return existing

        row = {
            "name": name,
            "description": description,
            "criterion_type": criterion_type,
            "mandatory": mandatory,
            "avg_weight": weight,
            "most_common_tasks": {task_type: 1} if task_type else {},
            "weight_samples": 1,
        }
        created = self._ensure_backend().create(row)
        if created:
            logger.debug(f"Added criterion to the catalog: {name}")
        return created or row

    def delete(self, name: str, criterion_type: str = BOOLEAN) -> bool:
        """Retire a criterion: it stops being applied from the next
        evaluation on, even if it is one of the defaults

        Returns whether a row was removed. Note that a default removed from
        an empty catalog comes back the next time the catalog is seeded.
        """
        return self._ensure_backend().delete(name, criterion_type)

    def record_usage(
        self, criteria: List[Dict[str, Any]], task_type: Optional[str]
    ) -> None:
        """Update `avg_weight` and `most_common_tasks` for a rubric that was
        just applied to a `task_type`"""
        backend = self._ensure_backend()
        for criterion in criteria:
            try:
                backend.bump(
                    criterion["name"],
                    criterion.get("criterion_type", BOOLEAN),
                    task_type,
                    float(criterion.get("weight", 1.0)),
                )
            except Exception as e:
                logger.warning(
                    f"Could not record usage of criterion "
                    f"{criterion.get('name')!r}: {e}"
                )


_repository: Optional[CriteriaRepository] = None


def get_criteria_repository() -> CriteriaRepository:
    """Process-wide catalog, so one connection serves every stage"""
    global _repository
    if _repository is None:
        _repository = CriteriaRepository()
    return _repository
