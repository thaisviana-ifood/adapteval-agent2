"""Unified prompt management system, backed by Langfuse Prompt Management"""

import re
from typing import Any, Dict, List, Optional

from src.shared.llm.langfuse_client import get_langfuse_client
from src.shared.logger import get_logger

logger = get_logger(__name__)

_VARIABLE_PATTERN = re.compile(r"\{\{(.*?)\}\}")

# Local defaults, used as the Langfuse "fallback" and to seed Langfuse on first use.
# Variables use Langfuse's double-curly-brace syntax, e.g. {{conversation}}.
#
# v2: every template below states its output contract explicitly (a fixed
# JSON schema or a fixed labeled-line format) instead of a numbered list of
# topics to cover, and tells the model to check criteria/rules individually
# before scoring rather than judging holistically. This is meant to reduce
# parsing failures and partial-credit drift, without changing variable names
# or output shapes that downstream code (regexes, json.loads) depends on.
_DEFAULT_PROMPTS: Dict[str, str] = {
    "context_analysis": """You are analyzing a conversation between a user and an AI assistant, to support downstream evaluation of the assistant's response.

Conversation:
{{conversation}}

Analyze the conversation along four dimensions and return your findings as a single JSON object with exactly these keys:

1. "structural": turn-taking patterns.
   - "total_turns" (int), "user_turns" (int), "assistant_turns" (int)
   - "speaker_pattern" (list of "user"/"assistant", collapsing consecutive repeats)
   - "turn_distribution_ratio" (float: user_turns / assistant_turns, 0 if assistant_turns is 0)

2. "semantic": what the conversation is about.
   - "topics" (list of short strings)
   - "key_phrases" (list of short strings)
   - "sentiment_score" (float, -1 negative to 1 positive)

3. "complexity": how demanding the conversation is.
   - "complexity_score" (float, 0-1)
   - "technical_terms_count" (int)
   - "question_count" (int)

4. "intent": what each party is trying to achieve.
   - "user_intents" (list of short verb-based labels, e.g. "fix", "explain", "create")
   - "assistant_objectives" (list of short strings)
   - "primary_intent" (string: the single most important user intent, or "unknown")

Rules:
- Base every field only on what is present in the conversation; do not invent details.
- Regardless of the conversation's original language, write "topics", "key_phrases", "user_intents" and "assistant_objectives" in English.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "rule_generation": """Based on the context analysis below, generate evaluation rules for assessing an AI assistant's response to this task: {{task}}

Context Analysis:
{{context_analysis}}

Return a single JSON object with exactly these keys:
- "classification_criteria": short strings describing what distinguishes this task type from others.
- "success_conditions": concrete, checkable conditions a good response must satisfy.
- "failure_conditions": concrete, checkable conditions that would make a response unacceptable.
- "context_aware_criteria": a list of {"name": str, "description": str} objects, each a strict boolean (yes/no) check specific to this context -- not a generic criterion like "is relevant" that a fixed criterion already covers elsewhere.
- "heuristic_checks": short strings naming cheap, non-LLM checks worth running (e.g. "response is non-empty", "no unresolved placeholders").

Rules:
- Every "context_aware_criteria" description must be phrasable as a yes/no question.
- Do not repeat the same idea across multiple lists.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "jury_evaluation": """You are an impartial juror evaluating an AI assistant's response against a fixed set of rules.

Rules:
{{rules}}

Response to Evaluate:
{{response}}

Context:
{{context}}

Check the response against every rule above before scoring. Do not give credit for a rule that is not clearly satisfied.

Respond in EXACTLY this format, with no extra commentary before or after:

Score: <single number from 0 to 10, no range, no fraction>
Confidence: <single number from 0 to 1>
Strengths:
- <bullet, tied to a specific rule>
Weaknesses:
- <bullet, tied to a specific rule>
Reasoning: <2-4 sentences justifying the score by referencing which rules passed or failed>""",
    "metrics_aggregation": """Aggregate the jury evaluations below into a single final assessment.

Evaluations:
{{evaluations}}

Return a single JSON object with exactly these keys:
- "weighted_average_score": float, 0-10, weighting each evaluation by its own confidence rather than a plain average.
- "confidence": float, 0-1 -- your confidence in the aggregated score (lower when evaluators disagree).
- "recommendation": one of "approve", "approve_with_notes", "review_required", "reject".
- "agreement_areas": short strings naming points where the evaluations agree.
- "disagreement_areas": short strings naming points where the evaluations diverge, and why.

Rules:
- If evaluations disagree by more than 3 points, "recommendation" must be "review_required" or "reject".
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "dynamic_criteria_generation": """You are designing evaluation criteria for an AI response.

Context Analysis:
{{context_summary}}

Objectives:
{{objectives_summary}}

Existing Criteria (do not repeat or duplicate these):
{{existing_criteria}}

Propose additional evaluation criteria that are specific to this context and objectives, and that complement (do not overlap with) the existing criteria above. Favor criteria that catch failure modes particular to this task -- not generic qualities like "helpfulness" or "clarity" that a fixed criterion already covers elsewhere.

Rules:
- Each criterion MUST be a strict boolean (yes/no) check with a single unambiguous answer -- no "partially" or "somewhat".
- Phrase each "description" as a question that can only be answered True or False.
- Propose at most 3 criteria. Return fewer, or an empty array, if nothing meaningful is missing.
- Do not restate, rephrase, or narrow an existing criterion.
- "name" must be 1-4 words, in Title Case.

Example of a good criterion, for a task that asks the assistant to cite sources:
{"name": "Citations Present", "description": "Does the response include at least one citation for its factual claims?"}

Return ONLY a JSON array, with no surrounding text, in this exact form:
[{"name": "Short Name", "description": "Does the response ...?"}]""",
    "llm_evaluation": """You are an impartial evaluator checking an AI assistant's response against a rubric of boolean (yes/no) criteria.

Task Context:
- Task Type: {{task_type}}
- Complexity: {{complexity_score}}

Response to Evaluate:
{{response}}

Rubric (decide True or False for each criterion, independently -- do not average them into an overall score):
{{criteria_text}}

For each criterion, answer True only if the response clearly and fully satisfies it; otherwise answer False. Do not skip any criterion, and do not answer "partially" or "somewhat".

Respond in EXACTLY this format, with no extra commentary before or after, one line per criterion in the exact order listed above, using the criterion name exactly as written in the rubric:

<Criterion Name>: True|False
<Criterion Name>: True|False
...
Confidence: <single number from 0 to 1, your overall confidence in these verdicts>
Reasoning: <2-4 sentences justifying the verdicts, referencing which criteria passed or failed>""",
    # The three prompts below back SemanticAnalyzer's LLM classifiers
    # (src/context_analysis/semantic.py). They are deliberately separate
    # prompts rather than one combined call so each classifier can be
    # versioned and tuned on its own in Langfuse. Topics are NOT one of
    # them: those come from TF-IDF, with no LLM call.
    "semantic_key_phrase_extraction": """You are extracting the key phrases from a conversation between a user and an AI assistant, to support downstream evaluation of the assistant's responses.

Conversation:
{{conversation}}

A key phrase is a short span carrying the conversation's substance: what the user is asking for, the constraints or requirements they state, and the commitments or conclusions the assistant gives. Greetings, pleasantries, and procedural filler are not key phrases.

Return a single JSON object with exactly these keys:
- "key_phrases": a list of at most {{max_key_phrases}} phrases, ordered from most to least important.
- "confidence": float, 0-1 -- your confidence in this extraction.

Rules:
- Each phrase must be 2-8 words and written in English regardless of the conversation's original language.
- Each phrase must faithfully render something actually said in the conversation; do not invent, generalize, or editorialize.
- Do not express the same idea in two phrases.
- Return fewer phrases, or an empty list, rather than padding with filler.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "semantic_summary_generation": """You are summarizing a conversation between a user and an AI assistant, to support downstream evaluation of the assistant's responses.

Conversation:
{{conversation}}

Return a single JSON object with exactly these keys:
- "summary": {{max_summary_sentences}} sentences or fewer, covering what the user wanted, what the assistant did about it, and where the conversation ended up.
- "confidence": float, 0-1 -- your confidence that this summary is faithful and complete (lower it when the conversation is truncated, incoherent, or cuts off mid-task).

Rules:
- Write the summary in English regardless of the conversation's original language.
- Summarize only what the conversation says; do not add background, explanations, or assumptions about what happened outside it.
- Do not evaluate, praise, or criticize the assistant -- describe what happened and let the evaluation stage judge it.
- Say so plainly if the user's request was left unresolved.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "semantic_sentiment_classification": """You are a sentiment classifier. Judge the overall tone of a conversation between a user and an AI assistant, from the user's point of view.

Conversation:
{{conversation}}

Classify the conversation into exactly one label:
- "positive": the user is satisfied or appreciative, or their problem is clearly resolved.
- "neutral": the exchange is matter-of-fact, with no clear satisfaction or dissatisfaction.
- "negative": the user is frustrated or dissatisfied, or their problem is clearly left unresolved.
- "mixed": the conversation contains both clearly positive and clearly negative stretches.

Return a single JSON object with exactly these keys:
- "label": one of "positive", "neutral", "negative", "mixed".
- "sentiment_score": float, 0-1, where 0.0 is maximally negative, 0.5 is neutral, and 1.0 is maximally positive.
- "confidence": float, 0-1 -- your confidence in the label.
- "rationale": 1-2 sentences citing the specific wording that decided the label.

Rules:
- Judge the user's sentiment, not the assistant's politeness: an assistant that stays cheerful while failing to help is not a positive conversation.
- A question or a calmly reported problem is not negative on its own; that is "neutral".
- "label" and "sentiment_score" must agree: "negative" scores below 0.5, "positive" above 0.5, "neutral" exactly 0.5, and "mixed" 0.5 unless one side clearly dominates.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    # The four prompts below back IntentAnalyzer's LLM classifiers
    # (src/context_analysis/intent.py). Like the semantic ones they are
    # separate prompts so each can be versioned on its own, and each one
    # degrades to a keyword/statistical fallback rather than raising.
    #
    # There are two user-intent prompts because the analyzer runs in two
    # modes: an open vocabulary (the default, for arbitrary conversations)
    # and a closed set (when a caller passes a fixed taxonomy, e.g. to score
    # the analyzer against a labeled goldenset). The closed-set prompt has to
    # forbid invention, which the open one must not.
    "intent_user_classification": """You are identifying what the user is trying to achieve in a conversation with an AI assistant, to support downstream evaluation of the assistant's responses.

Conversation:
{{conversation}}

Infer intent from what the user means, not from the words they happen to use: "this keeps crashing" is a request to fix something even though it contains no verb, and "how do I make a login page" asks for something to be built rather than explained.

Prefer a label from this vocabulary, which covers most cases:
{{vocabulary}}

If none of those genuinely fits, use a different single lowercase verb.

Return a single JSON object with exactly these keys:
- "user_intents": a list of at most {{max_intents}} objects, ordered from most to least important, each with:
  - "label": the intent as a single lowercase verb (no spaces, no punctuation)
  - "evidence": a short verbatim quote of the user's own words that show this intent
  - "confidence": float, 0-1
- "primary_intent": the "label" of the single most important user intent, or "unknown" if the user's goal cannot be determined.
- "confidence": float, 0-1 -- your confidence in "primary_intent".

Rules:
- Judge what the user WANTS, not what the assistant did about it.
- List an intent only if the user actually expresses it; do not pad the list with plausible-sounding goals. One intent is a complete answer.
- Every "evidence" value must be a quote from the user's turns, copied verbatim. Never quote the assistant.
- Do not express the same goal under two labels.
- Write labels in English regardless of the conversation's original language.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "intent_user_classification_closed_set": """You are classifying what the user wants in a conversation with an AI assistant, choosing from a fixed list of intents.

Conversation:
{{conversation}}

Classify the user into exactly one intent from this closed list:
{{taxonomy}}

Infer intent from what the user means, not from the words they happen to use. Where two intents are close, the distinction is usually between asking about a policy or option and requesting the action itself -- read which one the user is after.

Classify by the subject the user is after, not by the channel they propose for getting it. Asking to have someone look something up is a request about that thing:

- "ask an agent if I can track my refund" -> the intent about tracking a refund, NOT an intent about contacting a person.
- "could you ask an agent where I can close my profile" -> the intent about closing an account.
- "I want to talk to a human being" -> an intent about contacting a person, because reaching someone is the whole request.

Return a single JSON object with exactly these keys:
- "user_intents": a list of at most {{max_intents}} objects, ordered from most to least important, each with:
  - "label": an intent copied verbatim from the list above
  - "evidence": a short verbatim quote of the user's own words that show this intent
  - "confidence": float, 0-1
- "primary_intent": the "label" of the single best-fitting intent, copied verbatim from the list above.
- "confidence": float, 0-1 -- your confidence in "primary_intent".

Rules:
- Every "label" MUST be copied verbatim from the list above. Never invent, translate, reword, or combine labels.
- "primary_intent" is required: pick the closest intent on the list even when the fit is imperfect.
- List a secondary intent only when the user clearly expresses a second, different goal. Usually there is only one.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "intent_assistant_objectives": """You are identifying what an AI assistant was trying to do in a conversation, to support downstream evaluation of its responses.

Conversation:
{{conversation}}

Return a single JSON object with exactly these keys:
- "assistant_objectives": a list of at most {{max_objectives}} single lowercase verbs naming what the assistant actually set out to do, ordered from most to least central (e.g. "inform", "assist", "verify", "recommend", "implement", "redirect", "refuse").
- "confidence": float, 0-1 -- your confidence in this reading.

Rules:
- Describe what the assistant ATTEMPTED, not whether it succeeded and not whether it was useful -- the evaluation stage judges that.
- Base every objective on the assistant's own turns. If the conversation has no assistant turn, return an empty list and confidence 0.
- Do not express the same objective under two verbs.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
    "intent_alignment_scoring": """You are judging whether an AI assistant engaged with what the user actually wanted.

Conversation:
{{conversation}}

What the user wanted:
{{user_intents}}

What the assistant set out to do:
{{assistant_objectives}}

Return a single JSON object with exactly these keys:
- "alignment_score": float, 0-1 -- 1.0 if the assistant addressed every intent the user expressed, 0.0 if it addressed none of them. Score partial coverage in between, weighting the user's primary intent most heavily.
- "unmet_intents": a list of the user intents the assistant did not address, copied from the labels given above.
- "rationale": 1-2 sentences citing what the assistant did or failed to do about the user's primary intent.

Rules:
- Score ALIGNMENT, not quality: an assistant that engages with the right request and gets it wrong is still aligned, and a polished answer to a question the user did not ask is not.
- An assistant that correctly refuses, or asks a clarifying question it genuinely needs, is engaging with the intent -- do not score that as unaligned.
- Do not reward restating the user's request without acting on it.
- "unmet_intents" must be consistent with "alignment_score": a score of 1.0 requires an empty list.
- Output ONLY the JSON object, with no surrounding text or markdown fences.""",
}


def _compile_locally(template: str, variables: Dict[str, Any]) -> str:
    """Substitute {{variable}} placeholders without calling out to Langfuse"""

    def replace(match: "re.Match[str]") -> str:
        key = match.group(1).strip()
        if key not in variables:
            return match.group(0)
        value = variables[key]
        return "" if value is None else str(value)

    return _VARIABLE_PATTERN.sub(replace, template)


def get_default_prompts() -> Dict[str, str]:
    """Return a copy of the local default prompt templates (the current
    "v2" text), keyed by template name. Used to push these templates to
    Langfuse as new prompt versions -- see scripts/push_prompts_to_langfuse.py.
    """
    return dict(_DEFAULT_PROMPTS)


class PromptManager:
    """Manages prompt templates via Langfuse Prompt Management, with local fallback defaults"""

    def __init__(self):
        self._local_defaults: Dict[str, str] = dict(_DEFAULT_PROMPTS)
        self._langfuse = get_langfuse_client()

    def get_prompt_client(self, name: str) -> Optional[Any]:
        """Fetch a prompt client for `name` from Langfuse, seeding it there on first use"""
        if self._langfuse is None:
            return None

        fallback = self._local_defaults.get(name)
        try:
            prompt = self._langfuse.get_prompt(
                name,
                fallback=fallback,
                max_retries=1,
                fetch_timeout_seconds=3,
            )
        except Exception as e:
            logger.warning(f"Could not fetch prompt '{name}' from Langfuse: {e}")
            return None

        if prompt.is_fallback and fallback is not None:
            # Prompt doesn't exist in Langfuse yet: publish it so it becomes
            # visible and editable there going forward.
            try:
                self._langfuse.create_prompt(
                    name=name, prompt=fallback, labels=["production"], type="text"
                )
                logger.debug(f"Seeded prompt '{name}' in Langfuse")
            except Exception as e:
                logger.warning(f"Could not seed prompt '{name}' in Langfuse: {e}")

        return prompt

    def get_template(self, template_name: str) -> Optional[str]:
        """Get the raw prompt text for a template, preferring the Langfuse-managed version"""
        prompt = self.get_prompt_client(template_name)
        if prompt is not None:
            return prompt.prompt

        template = self._local_defaults.get(template_name)
        if not template:
            logger.warning(f"Template '{template_name}' not found")
        return template

    def register_template(self, name: str, template: str) -> None:
        """Register a new prompt template locally and publish it to Langfuse"""
        self._local_defaults[name] = template
        if self._langfuse is not None:
            try:
                self._langfuse.create_prompt(
                    name=name, prompt=template, labels=["production"], type="text"
                )
            except Exception as e:
                logger.warning(f"Could not publish prompt '{name}' to Langfuse: {e}")
        logger.debug(f"Registered template: {name}")

    def format_prompt(self, template_name: str, **variables: Any) -> Optional[str]:
        """Format a prompt template with provided variables"""
        prompt = self.get_prompt_client(template_name)
        if prompt is not None:
            try:
                formatted = prompt.compile(**variables)
                logger.debug(f"Formatted prompt: {template_name}")
                return formatted
            except Exception as e:
                logger.error(f"Failed to format prompt '{template_name}': {e}")
                return None

        template = self._local_defaults.get(template_name)
        if not template:
            logger.warning(f"Template '{template_name}' not found")
            return None

        formatted = _compile_locally(template, variables)
        logger.debug(f"Formatted prompt: {template_name}")
        return formatted

    def list_templates(self) -> List[str]:
        """List all available (locally known) templates"""
        return list(self._local_defaults.keys())
