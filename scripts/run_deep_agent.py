"""Run the thesis evaluation deep agent over a multi-turn conversation

By default this calls the LangGraph pipeline directly (src.agent.run_pipeline),
which is the reliable, schema-guaranteed entry point. Pass --use-agent to
instead go through the deepagents conversational front-end.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR, OUTPUT_DATA_DIR  # noqa: E402
from src.shared.logger import get_logger  # noqa: E402
from src.shared.utils import build_conversation_text  # noqa: E402
from src.agent import evaluate_conversation, run_pipeline  # noqa: E402

logger = get_logger(__name__)

DEFAULT_OUTPUT = OUTPUT_DATA_DIR / "deep_agent_evaluation.json"

EXAMPLE_CONVERSATION = (
    "User: Can you explain how neural networks work?\n\n"
    "Assistant: Sure! Neural networks are computing systems inspired by "
    "biological neurons. They consist of layers of interconnected nodes.\n\n"
    "User: Can you give a concrete example of training one?\n\n"
    "Assistant: Neural networks consist of layers of interconnected nodes. "
    "Each connection has a weight that is adjusted during training via "
    "backpropagation: the network makes a prediction, compares it to the "
    "expected output, and nudges the weights to reduce the error."
)


def load_conversation(dataset_path: Path, item_id: str = None) -> str:
    """Load a single conversation from a dataset export (same format as
    run_context_analysis.py), optionally selecting a specific item id"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    if item_id:
        for item in dataset:
            if str(item.get("id")) == item_id:
                return build_conversation_text(item)
        raise ValueError(f"No item with id '{item_id}' in {dataset_path}")

    return build_conversation_text(dataset[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the thesis evaluation pipeline (Análise de Contexto -> "
            "Gerador de Regras -> LLM as a Jury -> Agregação das métricas) "
            "over a multi-turn conversation"
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=f"Dataset JSON file to load a conversation from (default: {RAW_DATA_DIR}/diana-agent-csat.json)",
    )
    parser.add_argument("--item-id", default=None, help="Specific dataset item id to evaluate")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the structured evaluation JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="Go through the deepagents conversational front-end instead of calling the pipeline directly",
    )
    args = parser.parse_args()

    if args.dataset:
        conversation = load_conversation(args.dataset, args.item_id)
    else:
        dataset_path = RAW_DATA_DIR / "diana-agent-csat.json"
        conversation = (
            load_conversation(dataset_path, args.item_id)
            if dataset_path.exists()
            else EXAMPLE_CONVERSATION
        )

    logger.info("Running thesis evaluation %s", "via deep agent" if args.use_agent else "directly")
    if args.use_agent:
        result = evaluate_conversation(conversation)
    else:
        result = run_pipeline(conversation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote evaluation result to {args.output}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
