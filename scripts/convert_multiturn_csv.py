"""Convert the Kaggle Multi-Turn CSV export into the dataset JSON schema
expected by scripts/run_deep_agent.py (list of items with
input.query.content as a JSON string of {role, content} messages).

The source CSV has one row per conversation, with columns P1,R1,P2,R2,P3,R3,
P4,R4,P5: alternating user prompts (Pn) and assistant responses (Rn). Every
conversation ends on an unanswered user turn (Pn with no matching Rn), which
becomes the last message in the resulting transcript.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR  # noqa: E402

TURN_COLUMNS = ["P1", "R1", "P2", "R2", "P3", "R3", "P4", "R4", "P5"]


def convert(input_csv: Path, output_json: Path) -> int:
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    items = []
    for row in rows:
        row_id = (row.get("]") or "").strip()

        messages = []
        for idx, col in enumerate(TURN_COLUMNS):
            value = (row.get(col) or "").strip()
            if not value:
                break
            role = "user" if col.startswith("P") else "assistant"
            messages.append({"role": role, "content": value})

        if not messages:
            continue

        items.append(
            {
                "id": row_id or str(len(items) + 1),
                "datasetName": "kaggle-multi-turn",
                "metadata": {
                    "use_case": (row.get("Use case") or "").strip(),
                    "type": (row.get("Type") or "").strip(),
                    "category": (row.get("Category") or "").strip(),
                },
                "input": {
                    "query": {
                        "source": {"name": "kaggle_multi_turn_csv", "item_id": row_id},
                        "content": json.dumps(messages, ensure_ascii=False),
                    }
                },
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    return len(items)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DATA_DIR / "Multi-Turn.xlsx - Multi-Turn.csv"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else RAW_DATA_DIR / "multi_turn_kaggle_dataset.json"
    count = convert(src, dst)
    print(f"Wrote {count} conversation(s) to {dst}")
