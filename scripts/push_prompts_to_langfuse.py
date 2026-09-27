"""Publish the local prompt templates (src/shared/llm/prompts.py) to Langfuse
Prompt Management as new, immediately-active versions.

Each template is pushed with labels=["production"], so Langfuse moves the
"production" label to the new version right away -- no manual promotion step
in the Langfuse UI. The previous version remains in the prompt's history and
can be re-promoted from there if the new one needs to be rolled back.

Requires LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to be set (see .env).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.llm.langfuse_client import get_langfuse_client  # noqa: E402
from src.shared.llm.prompts import get_default_prompts  # noqa: E402
from src.shared.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

COMMIT_MESSAGE = "v2: explicit output contract, stricter per-criterion scoring"


def main() -> None:
    langfuse = get_langfuse_client()
    if langfuse is None:
        print(
            "Langfuse credentials not configured (LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY in .env). Nothing was pushed."
        )
        sys.exit(1)

    prompts = get_default_prompts()
    for name, template in prompts.items():
        prompt = langfuse.create_prompt(
            name=name,
            prompt=template,
            labels=["production"],
            type="text",
            commit_message=COMMIT_MESSAGE,
        )
        print(f"Pushed '{name}' -> version {prompt.version} (label: production)")

    langfuse.flush()
    print(f"Done. Pushed {len(prompts)} prompt(s) to Langfuse.")


if __name__ == "__main__":
    main()
