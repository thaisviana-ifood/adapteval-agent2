"""Dump the criteria catalog (the `evaluation_criteria` table)

Reads the rubric catalog the pipeline uses -- from PostgreSQL when it is
reachable, otherwise from the in-process seed -- and writes it as JSON, as a
Markdown table, or as the SQL inserts needed to load it into another
database.

Two ways to fill a catalog:

* `--seed` applies every missing seed row: the pipeline's own
  DEFAULT_CRITERIA *and* the rubric library in `data/rubrics/*.csv`. This is
  how a fresh database gets its rubric.
* `--import-csv [DIR]` applies only the CSV library, so a new export can be
  loaded without re-adding a pipeline criterion that was deleted on purpose.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DATA_DIR  # noqa: E402
from src.rule_generator.criteria_store import (  # noqa: E402
    BOOLEAN,
    CSV_COLUMNS,
    HEURISTIC,
    RUBRIC,
    RUBRICS_CSV_DIR,
    CriteriaRepository,
)
from src.shared.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

EXTENSIONS = {"json": "json", "markdown": "md", "sql": "sql"}

# Columns of the Markdown table. The full row (including `prompt_rubrica`,
# which is a whole judge prompt) only makes sense in the JSON and SQL
# dumps -- a Markdown cell with 3,000 characters of prompt in it is not a
# table anyone can read.
MARKDOWN_COLUMNS = (
    "name",
    "description",
    "criterion_type",
    "mandatory",
    "avg_weight",
    "most_common_tasks",
    "avaliador",
    "categoria",
    "escala_saida",
    "exige_referencia",
    "tarefas_indicadas",
)
MARKDOWN_MAX_CELL = 160


def _tasks_cell(row: Dict[str, Any]) -> str:
    """`most_common_tasks` as "task (n)" entries, most common first"""
    counts = row.get("most_common_tasks") or {}
    if not counts:
        return "-"
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{task} ({count})" for task, count in ordered)


def as_json(rows: List[Dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2, ensure_ascii=False, default=str)


def _markdown_cell(row: Dict[str, Any], column: str) -> str:
    if column == "most_common_tasks":
        value = _tasks_cell(row)
    elif column == "mandatory":
        value = "yes" if row["mandatory"] else "no"
    elif column == "avg_weight":
        value = f"{float(row['avg_weight']):.2f}"
    else:
        value = str(row.get(column) or "")

    value = " ".join(value.split()).replace("|", "\\|")
    if len(value) > MARKDOWN_MAX_CELL:
        value = value[: MARKDOWN_MAX_CELL - 1].rstrip() + "…"
    return value or "-"


def as_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(MARKDOWN_COLUMNS) + " |",
        "|" + "---|" * len(MARKDOWN_COLUMNS),
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row, column) for column in MARKDOWN_COLUMNS
            )
            + " |"
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

    columns = (
        "name",
        "description",
        "criterion_type",
        "mandatory",
        "avg_weight",
        "most_common_tasks",
        "weight_samples",
        *CSV_COLUMNS,
    )
    statements = [
        "-- Dump of evaluation_criteria "
        "(see docker/postgres/init.sql for the schema)"
    ]
    for row in rows:
        values = [
            literal(row["name"]),
            literal(row["description"]),
            literal(row["criterion_type"]),
            literal(bool(row["mandatory"])),
            literal(float(row["avg_weight"])),
            literal(json.dumps(row.get("most_common_tasks") or {})) + "::jsonb",
            literal(int(row.get("weight_samples", 1))),
            *(literal(row.get(column) or "") for column in CSV_COLUMNS),
        ]
        statements.append(
            "INSERT INTO evaluation_criteria ({columns})\n"
            "    VALUES ({values})\n"
            "    ON CONFLICT (lower(name), criterion_type, lower(avaliador))"
            " DO NOTHING;".format(
                columns=", ".join(columns), values=", ".join(values)
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
        choices=[BOOLEAN, HEURISTIC, RUBRIC],
        help=(
            f"Dump only criteria of this type (default: all); "
            f"{RUBRIC!r} is the imported CSV library"
        ),
    )
    parser.add_argument(
        "--avaliador",
        help="Dump only rubrics from this framework (e.g. DeepEval, Ragas)",
    )
    parser.add_argument(
        "--categoria",
        help="Dump only rubrics whose categoria contains this text (e.g. RAG)",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help=(
            "Insert every missing seed row (DEFAULT_CRITERIA plus the CSV "
            "rubric library) before dumping, so a fresh database ends up "
            "with the full catalog"
        ),
    )
    parser.add_argument(
        "--import-csv",
        nargs="?",
        type=Path,
        const=RUBRICS_CSV_DIR,
        metavar="DIR",
        help=(
            f"Import only the rubric library from the CSVs in DIR "
            f"(default: {RUBRICS_CSV_DIR.relative_to(RUBRICS_CSV_DIR.parents[1])})"
            f" before dumping"
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
            logger.info(f"Seeded {inserted} missing criteria")
        if args.import_csv:
            repository.import_rubrics_csv(args.import_csv)

        rows = repository.list_all(
            criterion_type=args.criterion_type,
            avaliador=args.avaliador,
            categoria=args.categoria,
        )
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
