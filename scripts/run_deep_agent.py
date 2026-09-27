"""Run the thesis evaluation deep agent over a dataset of multi-turn conversations

By default this calls the LangGraph pipeline directly (src.agent.run_pipeline)
for every item in the dataset, which is the reliable, schema-guaranteed entry
point the deep agent's own tool wraps. Pass --use-agent to instead go through
the deepagents conversational front-end for each item.
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

DEFAULT_DATASET = RAW_DATA_DIR / "diana-agent-csat.json"
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


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load a dataset export (list of items with input.query.content messages)"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _langfuse_context(item: Dict[str, Any]) -> Dict[str, Any]:
    """Derive Langfuse user_id/tags/metadata for a dataset item

    user_id is the tenant/restaurant being evaluated (metadata.tenant_hash
    in Langfuse dataset exports), so all evaluations of the same tenant
    across runs group under the same Langfuse user.
    """
    item_metadata = item.get("metadata") or {}
    dataset_name = item.get("datasetName")

    return {
        "user_id": item_metadata.get("tenant_hash"),
        "tags": [dataset_name] if dataset_name else None,
        "metadata": {
            "dataset_item_id": item.get("id"),
            "source_trace_id": item.get("sourceTraceId"),
        },
    }


def evaluate_dataset(
    dataset: List[Dict[str, Any]],
    output_path: Path,
    use_agent: bool = False,
) -> List[Dict[str, Any]]:
    """Run every item in the dataset through the pipeline, writing results
    incrementally to output_path so progress survives an interruption"""
    results: List[Dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(dataset, start=1):
        item_id = item.get("id")
        logger.info(f"[{idx}/{len(dataset)}] Evaluating item {item_id}")

        conversation = item.get("_conversation_override") or build_conversation_text(item)
        langfuse_context = _langfuse_context(item)
        try:
            if use_agent:
                evaluation = evaluate_conversation(
                    conversation, thread_id=item_id, **langfuse_context
                )
            else:
                evaluation = run_pipeline(
                    conversation, evaluation_id=item_id, **langfuse_context
                )
        except Exception as e:
            logger.error(f"Item {item_id} failed: {e}")
            evaluation = {"error": str(e)}

        results.append({"id": item_id, **evaluation})

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the thesis evaluation pipeline (Análise de Contexto -> "
            "Gerador de Regras -> LLM as a Jury -> Agregação das métricas) "
            "over every conversation in a dataset"
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset JSON file to evaluate (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--item-id",
        default=None,
        help="Evaluate only this single dataset item id instead of the whole file",
    )
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

    if args.dataset.exists():
        dataset = load_dataset(args.dataset)
        if args.item_id:
            dataset = [item for item in dataset if str(item.get("id")) == args.item_id]
            if not dataset:
                raise ValueError(f"No item with id '{args.item_id}' in {args.dataset}")
    else:
        logger.warning(f"{args.dataset} not found; using a single built-in example conversation")
        dataset = [{"id": "example", "_conversation_override": EXAMPLE_CONVERSATION}]

    logger.info(
        "Running thesis evaluation over %d item(s) %s",
        len(dataset),
        "via deep agent" if args.use_agent else "directly",
    )

    results = evaluate_dataset(dataset, args.output, use_agent=args.use_agent)

    logger.info(f"Wrote {len(results)} evaluation result(s) to {args.output}")


if __name__ == "__main__":
    main()
