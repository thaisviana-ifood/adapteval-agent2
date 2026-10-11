"""Dump the criteria catalog (the `evaluation_criteria` table)

Reads the rubric catalog the pipeline uses -- from PostgreSQL when it is
reachable, otherwise from the in-process defaults -- and writes it as JSON,
as a Markdown table, or as the SQL inserts needed to load it into another
database. `--seed` first applies any DEFAULT_CRITERIA row the catalog is
missing, which is how a fresh database gets its rubric.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DATA_DIR  # noqa: E402
from src.rule_generator.criteria_store import CriteriaRepository  # noqa: E402
from src.shared.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

EXTENSIONS = {"json": "json", "markdown": "md", "sql": "sql"}


def _tasks_cell(row: Dict[str, Any]) -> str:
    """`most_common_tasks` as "task (n)" entries, most common first"""
    counts = row.get("most_common_tasks") or {}
    if not counts:
        return "-"
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{task} ({count})" for task, count in ordered)


def as_json(rows: List[Dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2, ensure_ascii=False, default=str)


def as_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| name | description | criterion_type | mandatory | avg_weight "
        "| most_common_tasks |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {description} | {criterion_type} | {mandatory} "
            "| {avg_weight:.2f} | {tasks} |".format(
                name=row["name"],
                description=row["description"].replace("|", "\\|"),
                criterion_type=row["criterion_type"],
                mandatory="yes" if row["mandatory"] else "no",
                avg_weight=float(row["avg_weight"]),
                tasks=_tasks_cell(row),
            )
        )
    return "\n".join(lines)


def as_sql(rows: List[Dict[str, Any]]) -> str:
    """Inserts that reproduce this catalog in another database

    Idempotent, so the dump can be replayed over an existing catalog
    without clobbering it.
    """

    def literal(value: Any) -> str:
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return repr(value)
        text = value if isinstance(value, str) else json.dumps(value)
        return "'" + text.replace("'", "''") + "'"

    statements = [
        "-- Dump of evaluation_criteria "
        "(see docker/postgres/init.sql for the schema)"
    ]
    for row in rows:
        statements.append(
            "INSERT INTO evaluation_criteria (name, description, "
            "criterion_type, mandatory, avg_weight, most_common_tasks, "
            "weight_samples) VALUES ({values})\n"
            "    ON CONFLICT (lower(name), criterion_type) DO NOTHING;".format(
                values=", ".join(
                    [
                        literal(row["name"]),
                        literal(row["description"]),
                        literal(row["criterion_type"]),
                        literal(bool(row["mandatory"])),
                        literal(float(row["avg_weight"])),
                        literal(json.dumps(row.get("most_common_tasks") or {}))
                        + "::jsonb",
                        literal(int(row.get("weight_samples", 1))),
                    ]
                )
            )
        )
    return "\n".join(statements) + "\n"


FORMATTERS = {"json": as_json, "markdown": as_markdown, "sql": as_sql}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump the evaluation_criteria rubric catalog"
    )
    parser.add_argument(
        "--format",
        choices=sorted(FORMATTERS),
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--criterion-type",
        choices=["boolean", "heuristic"],
        help="Dump only criteria of this type (default: all)",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help=(
            "Insert any missing DEFAULT_CRITERIA row before dumping, so a "
            "fresh database ends up with the full rubric"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Where to write the dump (default: "
            "data/output/criteria_dump.<ext>); use - for stdout"
        ),
    )
    args = parser.parse_args()

    repository = CriteriaRepository()
    try:
        if args.seed:
            inserted = repository.seed_defaults()
            logger.info(f"Seeded {inserted} missing default criteria")

        rows = repository.list_all(criterion_type=args.criterion_type)
        if not repository.is_persistent:
            logger.warning(
                "Dumping the in-memory catalog: PostgreSQL was not "
                "reachable, so this is the seed rubric, not accumulated "
                "history"
            )
    finally:
        repository.close()

    content = FORMATTERS[args.format](rows)

    if args.output and str(args.output) == "-":
        print(content)
        return

    output = args.output or (
        OUTPUT_DATA_DIR / f"criteria_dump.{EXTENSIONS[args.format]}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        content if content.endswith("\n") else content + "\n",
        encoding="utf-8",
    )
    logger.info(f"Wrote {len(rows)} criteria to {output}")


if __name__ == "__main__":
    main()
