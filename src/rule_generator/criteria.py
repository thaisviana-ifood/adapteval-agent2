"""Context-aware criteria generation

The rubric is not hard-coded here: it is assembled from the criteria
catalog in the database (`criteria_store.py`). This module decides *which*
catalogued criteria apply to one evaluation:

1. every `mandatory` criterion in the catalog;
2. the criteria the catalog has seen on this task type or a similar one,
   plus the ones a context trigger below selects;
3. criteria the LLM proposes for this particular context -- which are
   written into the catalog on first use, so a later evaluation of the same
   or a similar task finds them in step 2.

Every criterion produced here is boolean: a strict yes/no check
(`criterion_type: "boolean"`) with a description phrased as a question that
can only be answered True or False, so downstream evaluators score
pass/fail rather than a graded scale.
"""

import json
import re
from typing import Dict, Any, List, Optional

from src.config import CLASSIFIER_LLM_MAX_TOKENS
from src.shared.logger import get_logger
from src.shared.llm.client import LLMClient
from src.shared.llm.prompts import PromptManager
from src.rule_generator.criteria_store import (
    BOOLEAN,
    CriteriaRepository,
    get_criteria_repository,
    to_criterion,
)

logger = get_logger(__name__)

_MAX_DYNAMIC_CRITERIA = 3
_JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)

# How many criteria a rubric may inherit purely from the catalog's task
# history. The objectives' own criteria are always included on top of these;
# the cap only bounds how much a well-used task type can grow the rubric,
# since every extra criterion is another verdict in every juror's prompt.
_MAX_TASK_CRITERIA = 6

# Catalogued criteria that are selected by a signal in the context analysis
# rather than by task type. The condition lives here (it reads the context);
# the criterion's wording, weight and history live in the catalog, and a
# name missing from the catalog is simply not applied.
_CONTEXT_TRIGGERS = {
    "Technical Depth": lambda ctx: (
        (ctx.get("complexity", {}).get("complexity_score") or 0) > 7
    ),
    "Problem Solving": lambda ctx: (
        "fix" in (ctx.get("intent", {}).get("user_intents") or [])
    ),
}


class CriteriaGenerator:
    """Generate evaluation criteria from context and objectives"""

    def __init__(self, repository: Optional[CriteriaRepository] = None):
        self.llm_client = LLMClient(max_tokens=CLASSIFIER_LLM_MAX_TOKENS)
        self.prompt_manager = PromptManager()
        self.repository = repository or get_criteria_repository()

    def generate(
        self,
        context: Dict[str, Any],
        objectives: Dict[str, Any],
        task_type: str = "general",
    ) -> Dict[str, Any]:
        """
        Generate context-aware evaluation criteria

        Args:
            context: Context analysis results
            objectives: Success objectives
            task_type: Task type from the classifier, used to look up the
                criteria the catalog has seen on this task and on similar
                ones, and to record this rubric against

        Returns:
            Evaluation criteria
        """
        base_criteria = self._mandatory_criteria()
        contextual_criteria = self._contextual_criteria(
            context, objectives, task_type, base_criteria
        )
        dynamic_criteria = self._generate_dynamic_criteria(
            context, objectives, task_type, base_criteria + contextual_criteria
        )
        applied = base_criteria + contextual_criteria + dynamic_criteria
        weighted_criteria = self._assign_weights(applied)

        # Applying a rubric is what makes a criterion "common" for a task
        # and what keeps avg_weight an average -- see criteria_store.
        self.repository.record_usage(applied, task_type)

        criteria = {
            "base_criteria": base_criteria,
            "contextual_criteria": contextual_criteria,
            "dynamic_criteria": dynamic_criteria,
            "weighted_criteria": weighted_criteria,
            "total_criteria_count": len(weighted_criteria),
        }

        logger.debug(
            f"Generated {len(weighted_criteria)} criteria for task "
            f"'{task_type}' (catalog persistent: "
            f"{self.repository.is_persistent})"
        )
        return criteria

    def _mandatory_criteria(self) -> List[Dict]:
        """The criteria the catalog marks as mandatory -- applied to every
        evaluation, whatever the task"""
        return [
            to_criterion(row)
            for row in self.repository.get_mandatory(BOOLEAN)
        ]

    def _contextual_criteria(
        self,
        context: Dict[str, Any],
        objectives: Dict[str, Any],
        task_type: str,
        already_applied: List[Dict],
    ) -> List[Dict]:
        """Criteria specific to this context: triggered by a context signal,
        required by the task's objectives, or inherited from the catalog's
        history for this task type and similar ones"""
        seen = {c["name"].strip().lower() for c in already_applied}
        contextual: List[Dict] = []

        def add(row: Optional[Dict]) -> bool:
            if not row:
                return False
            name = row["name"].strip().lower()
            if name in seen:
                return False
            seen.add(name)
            contextual.append(to_criterion(row))
            return True

        # 1. Context triggers (complexity, user intents).
        for name, triggered in _CONTEXT_TRIGGERS.items():
            if not triggered(context):
                continue
            row = self.repository.get_by_name(name, BOOLEAN)
            if row is None:
                logger.debug(
                    f"Context trigger '{name}' fired but the criterion is "
                    f"not in the catalog; skipping it"
                )
                continue
            add(row)

        # 2. The task's own must-haves (from ObjectiveDefinition
        # .get_objectives's "success_criteria", or define_success's
        # "must_haves"). New ones join the catalog here.
        must_haves = objectives.get(
            "success_criteria", objectives.get("must_haves", [])
        )
        for criterion in must_haves:
            add(
                self.repository.get_or_create(
                    name=criterion,
                    description=f"Does the response satisfy: {criterion}?",
                    criterion_type=BOOLEAN,
                    task_type=task_type,
                )
            )

        # 3. What the catalog has learned about this task type, and about
        # tasks similar to it.
        inherited = 0
        for row in self.repository.get_for_task(task_type, BOOLEAN):
            if inherited >= _MAX_TASK_CRITERIA:
                break
            if add(row):
                inherited += 1

        return contextual

    def _generate_dynamic_criteria(
        self,
        context: Dict[str, Any],
        objectives: Dict[str, Any],
        task_type: str,
        existing_criteria: List[Dict],
    ) -> List[Dict]:
        """Use the LLM to propose complementary, context-specific criteria
        not already covered by the mandatory/contextual ones. Always boolean,
        and always added to the catalog so the next run can reuse them.
        """
        prompt = self.prompt_manager.format_prompt(
            "dynamic_criteria_generation",
            context_summary=self._summarize_context(context),
            objectives_summary=self._summarize_objectives(objectives),
            existing_criteria=self._summarize_existing(existing_criteria),
        )
        if not prompt:
            return []

        try:
            response_text = self.llm_client.call(
                prompt,
                name="rule_generator.dynamic_criteria",
                langfuse_prompt=self.prompt_manager.get_prompt_client(
                    "dynamic_criteria_generation"
                ),
            )
        except Exception as e:
            logger.warning(f"Dynamic criteria generation failed: {e}")
            return []

        existing_names = {c["name"].strip().lower() for c in existing_criteria}
        proposals = self._parse_dynamic_criteria(response_text, existing_names)
        return [
            to_criterion(
                self.repository.get_or_create(
                    name=proposal["name"],
                    description=proposal["description"],
                    criterion_type=BOOLEAN,
                    task_type=task_type,
                )
            )
            for proposal in proposals
        ]

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Compact summary of the signals relevant to criteria generation"""
        summary = {
            "topics": context.get("semantic", {}).get("topics", []),
            "user_intents": context.get("intent", {}).get("user_intents", []),
            "complexity_score": context.get("complexity", {}).get(
                "complexity_score"
            ),
        }
        return json.dumps(summary, default=str)

    def _summarize_objectives(self, objectives: Dict[str, Any]) -> str:
        """Compact summary of the objectives relevant to criteria generation"""
        summary = {
            "primary_objective": objectives.get("primary_objective")
            or objectives.get("primary"),
            "secondary_objectives": objectives.get("secondary_objectives")
            or objectives.get("secondary", []),
            "success_criteria": objectives.get("success_criteria")
            or objectives.get("must_haves", []),
        }
        return json.dumps(summary, default=str)

    def _summarize_existing(self, existing_criteria: List[Dict]) -> str:
        if not existing_criteria:
            return "None"
        return "\n".join(f"- {c['name']}" for c in existing_criteria)

    def _parse_dynamic_criteria(
        self, response_text: Optional[str], existing_names: set
    ) -> List[Dict]:
        """Parse the LLM's JSON array response into boolean criteria dicts"""
        if not response_text:
            return []

        match = _JSON_ARRAY_PATTERN.search(response_text)
        if not match:
            logger.warning(
                "Dynamic criteria generation returned no JSON array"
            )
            return []

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse dynamic criteria JSON: {e}")
            return []

        dynamic = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            if not name or not description:
                continue
            if name.lower() in existing_names:
                continue

            existing_names.add(name.lower())
            dynamic.append(
                {
                    "name": name,
                    "description": description,
                    "criterion_type": BOOLEAN,
                    "weight": 1.0,
                }
            )
            if len(dynamic) >= _MAX_DYNAMIC_CRITERIA:
                break

        return dynamic

    def _assign_weights(self, criteria: List[Dict]) -> List[Dict]:
        """Assign normalized weights to criteria"""
        total_weight = sum(c.get("weight", 1.0) for c in criteria)

        weighted = []
        for criterion in criteria:
            weight = criterion.get("weight", 1.0)
            normalized_weight = weight / total_weight if total_weight > 0 else 1.0 / len(criteria)

            weighted.append(
                {
                    **criterion,
                    "weight": weight,
                    "normalized_weight": round(
                        normalized_weight, 4
                    ),
                }
            )

        return weighted
