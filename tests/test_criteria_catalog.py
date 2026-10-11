"""Tests for the rubric catalog and the criteria generation that reads it

Every test runs the catalog with `enabled=False`, i.e. on its in-process
backend seeded with the same rows a fresh database gets (DEFAULT_CRITERIA
plus the imported `data/rubrics/*.csv` library): the behaviour under test is
the selection logic, which is identical on both backends, and no test should
need a PostgreSQL server.
"""

import pytest

from src.rule_generator.criteria import CriteriaGenerator
from src.rule_generator.criteria_store import (
    BOOLEAN,
    CSV_COLUMNS,
    DEFAULT_CRITERIA,
    HEURISTIC,
    RUBRIC,
    CriteriaRepository,
    load_csv_rubrics,
    related_tasks,
)
from src.rule_generator.heuristics import HeuristicChecker
from src.shared.llm.client import LLMClient


@pytest.fixture
def repository():
    """A catalog backed by memory only, seeded with the defaults"""
    return CriteriaRepository(enabled=False)


class TestCriteriaRepository:
    """The catalog itself"""

    def test_seeds_every_default_criterion(self, repository):
        # The pipeline's own criteria are the rows with no avaliador; the
        # rest of the catalog is the imported library (TestRubricLibrary).
        catalogued = {
            (row["name"], row["criterion_type"])
            for row in repository.list_all(avaliador="")
        }
        assert catalogued == {
            (row["name"], row["criterion_type"]) for row in DEFAULT_CRITERIA
        }

    def test_mandatory_criteria_are_the_base_rubric(self, repository):
        names = [row["name"] for row in repository.get_mandatory(BOOLEAN)]
        assert sorted(names) == [
            "Clarity",
            "Completeness",
            "Correctness",
            "Relevance",
        ]

    def test_mandatory_is_scoped_by_criterion_type(self, repository):
        # "relevance" (a string check) and "Relevance" (a jury question)
        # are different criteria that happen to share a name.
        heuristics = [row["name"] for row in repository.get_mandatory(HEURISTIC)]
        assert "relevance" in heuristics
        assert "Relevance" not in heuristics

    def test_task_lookup_prefers_the_task_itself(self, repository):
        names = [
            row["name"] for row in repository.get_for_task("code_generation")
        ]
        assert names == [
            "No logical errors",
            "Solves the problem",
            "Syntax correct",
        ]

    def test_task_lookup_borrows_from_similar_tasks(self, repository):
        # "reasoning" has no criteria of its own; question_answering and
        # general are similar enough to lend theirs.
        assert "reasoning" not in {
            task
            for row in repository.list_all()
            for task in row["most_common_tasks"]
        }
        names = [row["name"] for row in repository.get_for_task("reasoning")]
        assert "Addresses all parts of question" in names
        assert "Addresses user need" in names
        # ... but not the ones belonging to an unrelated task.
        assert "Syntax correct" not in names

    def test_task_lookup_never_returns_mandatory_criteria(self, repository):
        # They are applied unconditionally; offering them again would only
        # duplicate them in the rubric.
        names = [row["name"] for row in repository.get_for_task("general")]
        assert "Relevance" not in names

    def test_related_tasks_starts_with_the_task_itself(self):
        assert related_tasks("translation")[0] == "translation"
        assert "text_summarization" in related_tasks("translation")

    def test_unknown_criteria_are_added_to_the_catalog(self, repository):
        assert repository.get_by_name("Cites Sources") is None

        created = repository.get_or_create(
            name="Cites Sources",
            description="Does the response cite its sources?",
            task_type="question_answering",
        )

        assert created["most_common_tasks"] == {"question_answering": 1}
        assert repository.get_by_name("Cites Sources") is not None
        # And it is reusable: a later evaluation of the same task finds it.
        assert "Cites Sources" in [
            row["name"]
            for row in repository.get_for_task("question_answering")
        ]

    def test_known_criteria_are_not_duplicated(self, repository):
        before = len(repository.list_all())

        repository.get_or_create(
            name="relevance",  # already catalogued as "Relevance"
            description="A different wording of the same check",
        )

        assert len(repository.list_all()) == before
        assert (
            repository.get_by_name("RELEVANCE")["description"]
            == "Is the response relevant to the query?"
        )

    def test_usage_updates_task_counts_and_average_weight(self, repository):
        repository.record_usage(
            [{"name": "Clarity", "criterion_type": BOOLEAN, "weight": 1.0}],
            "code_generation",
        )
        row = repository.get_by_name("Clarity")

        assert row["most_common_tasks"] == {"code_generation": 1}
        # Seeded at 0.8 with one sample, now averaged with a 1.0 sample.
        assert row["avg_weight"] == pytest.approx(0.9)
        assert row["weight_samples"] == 2

    def test_usage_of_an_uncatalogued_criterion_is_a_no_op(self, repository):
        repository.record_usage(
            [{"name": "Not In Catalog", "weight": 1.0}], "general"
        )
        assert repository.get_by_name("Not In Catalog") is None


class TestRubricLibrary:
    """The rubrics imported from data/rubrics/*.csv"""

    def test_every_csv_rubric_is_catalogued(self, repository):
        rubrics = repository.list_all(criterion_type=RUBRIC)

        assert len(rubrics) == len(load_csv_rubrics())
        assert all(row["avaliador"] for row in rubrics)
        assert all(row["prompt_rubrica"] for row in rubrics)

    def test_every_csv_column_is_kept(self, repository):
        g_eval = repository.get_by_name("G-Eval", RUBRIC, "DeepEval")

        # metrica/o_que_avalia are this table's name/description; the other
        # nine CSV columns are stored verbatim.
        assert g_eval["description"]
        for column in CSV_COLUMNS:
            assert g_eval[column], f"{column} was dropped on import"
        assert g_eval["escala_saida"].startswith("0–1")
        assert g_eval["fonte"].startswith("https://")

    def test_a_metric_name_is_scoped_by_its_framework(self, repository):
        # Three frameworks publish a "Relevance", and none of them is the
        # jury's own Relevance question.
        azure = repository.get_by_name("Relevance", RUBRIC, "Azure AI Foundry")
        bedrock = repository.get_by_name(
            "Relevance", RUBRIC, "Amazon Bedrock Evaluations"
        )
        own = repository.get_by_name("Relevance", BOOLEAN)

        assert azure["escala_saida"] != bedrock["escala_saida"]
        assert own["mandatory"] is True
        assert own["avaliador"] == ""

    def test_the_library_is_not_applied_to_evaluations(self, repository):
        # It is a reference library: catalogued and queryable, but the
        # jury's rubric is the boolean criteria only.
        assert not any(
            row["criterion_type"] == RUBRIC
            for row in repository.get_mandatory(BOOLEAN)
        )
        assert not any(
            row["criterion_type"] == RUBRIC
            for row in repository.get_for_task("question_answering", BOOLEAN)
        )

    def test_filters_narrow_the_library(self, repository):
        deepeval = repository.list_all(
            criterion_type=RUBRIC, avaliador="DeepEval"
        )
        assert deepeval
        assert {row["avaliador"] for row in deepeval} == {"DeepEval"}

        # categoria is a compound label ("Qualidade / RAG"), so it matches
        # as a substring.
        rag = repository.list_all(criterion_type=RUBRIC, categoria="RAG")
        assert rag
        assert all("rag" in row["categoria"].lower() for row in rag)

    def test_import_is_idempotent(self, repository):
        before = len(repository.list_all())

        assert repository.import_rubrics_csv() == 0
        assert len(repository.list_all()) == before

    def test_import_adds_rubrics_missing_from_the_catalog(self, repository):
        assert repository.delete("G-Eval", RUBRIC, "DeepEval")

        assert repository.import_rubrics_csv() == 1
        assert repository.get_by_name("G-Eval", RUBRIC, "DeepEval")

    def test_a_csv_without_the_expected_columns_is_skipped(self, tmp_path):
        (tmp_path / "not_rubrics.csv").write_text("a,b\n1,2\n", encoding="utf-8")

        assert load_csv_rubrics(tmp_path, refresh=True) == []


class TestCriteriaGenerator:
    """The rubric assembled for one evaluation"""

    @pytest.fixture(autouse=True)
    def no_llm_proposals(self, monkeypatch):
        """Dynamic criteria are tested on their own; keep the LLM out of
        the selection tests"""
        monkeypatch.setattr(
            LLMClient, "call", lambda self, *args, **kwargs: "[]"
        )

    @pytest.fixture
    def generator(self, repository):
        return CriteriaGenerator(repository=repository)

    def test_every_rubric_contains_the_mandatory_criteria(self, generator):
        criteria = generator.generate({}, {}, "general")

        assert [c["name"] for c in criteria["base_criteria"]] == [
            "Clarity",
            "Completeness",
            "Correctness",
            "Relevance",
        ]
        assert all(
            c["criterion_type"] == BOOLEAN
            for c in criteria["weighted_criteria"]
        )

    def test_weights_come_from_the_catalog(self, generator):
        criteria = generator.generate({}, {}, "general")
        by_name = {c["name"]: c for c in criteria["weighted_criteria"]}

        assert by_name["Clarity"]["weight"] == 0.8
        assert by_name["Completeness"]["weight"] == 0.9
        total = sum(c["weight"] for c in criteria["weighted_criteria"])
        assert by_name["Clarity"]["normalized_weight"] == pytest.approx(
            0.8 / total, abs=1e-4
        )

    def test_complexity_triggers_technical_depth(self, generator):
        plain = generator.generate(
            {"complexity": {"complexity_score": 3}}, {}, "general"
        )
        complex_ = generator.generate(
            {"complexity": {"complexity_score": 9}}, {}, "general"
        )

        names = lambda c: [x["name"] for x in c["contextual_criteria"]]  # noqa: E731
        assert "Technical Depth" not in names(plain)
        assert "Technical Depth" in names(complex_)

    def test_fix_intent_triggers_problem_solving(self, generator):
        criteria = generator.generate(
            {"intent": {"user_intents": ["fix"]}}, {}, "general"
        )

        contextual = {c["name"]: c for c in criteria["contextual_criteria"]}
        assert "Problem Solving" in contextual
        assert contextual["Problem Solving"]["weight"] == 1.2

    def test_a_trigger_without_a_catalogued_criterion_is_skipped(
        self, generator, repository
    ):
        # Deleting a row is how a criterion is retired, so a context
        # trigger must not resurrect it from code.
        assert repository.delete("Technical Depth", BOOLEAN)

        criteria = generator.generate(
            {"complexity": {"complexity_score": 9}}, {}, "general"
        )

        assert "Technical Depth" not in [
            c["name"] for c in criteria["weighted_criteria"]
        ]

    def test_objective_must_haves_join_the_catalog(self, generator, repository):
        criteria = generator.generate(
            {}, {"success_criteria": ["Handles edge cases"]}, "code_generation"
        )

        assert "Handles edge cases" in [
            c["name"] for c in criteria["contextual_criteria"]
        ]
        row = repository.get_by_name("Handles edge cases")
        assert row is not None
        assert row["most_common_tasks"]["code_generation"] >= 1

    def test_the_rubric_inherits_the_criteria_of_a_similar_task(
        self, generator
    ):
        criteria = generator.generate({}, {}, "translation")

        names = [c["name"] for c in criteria["contextual_criteria"]]
        # Seeded on text_summarization, which is similar to translation.
        assert "Preserves main points" in names

    def test_no_criterion_appears_twice(self, generator):
        criteria = generator.generate(
            {"complexity": {"complexity_score": 9}},
            {"success_criteria": ["Addresses user need"]},  # also in catalog
            "general",
        )

        names = [c["name"] for c in criteria["weighted_criteria"]]
        assert len(names) == len(set(names))

    def test_applying_a_rubric_records_its_usage(self, generator, repository):
        generator.generate({}, {}, "creative_writing")

        assert repository.get_by_name("Relevance")["most_common_tasks"] == {
            "creative_writing": 1
        }

    def test_llm_proposals_are_catalogued(self, repository, monkeypatch):
        monkeypatch.setattr(
            LLMClient,
            "call",
            lambda self, *args, **kwargs: (
                '[{"name": "Respects Tone", '
                '"description": "Does the response keep the asked-for tone?"}]'
            ),
        )
        generator = CriteriaGenerator(repository=repository)

        criteria = generator.generate({}, {}, "creative_writing")

        assert [c["name"] for c in criteria["dynamic_criteria"]] == [
            "Respects Tone"
        ]
        # The point of cataloguing them: the next run finds it by task.
        assert "Respects Tone" in [
            row["name"] for row in repository.get_for_task("creative_writing")
        ]

    def test_an_llm_failure_leaves_the_rest_of_the_rubric_intact(
        self, repository, monkeypatch
    ):
        def boom(self, *args, **kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(LLMClient, "call", boom)
        generator = CriteriaGenerator(repository=repository)

        criteria = generator.generate({}, {}, "general")

        assert criteria["dynamic_criteria"] == []
        assert criteria["total_criteria_count"] >= 4


class TestHeuristicChecker:
    """The heuristic rows of the catalog drive which checks run"""

    def test_runs_the_catalogued_heuristics(self, repository):
        checker = HeuristicChecker(repository=repository)

        assert sorted(checker.list_heuristics()) == [
            "coherence",
            "minimum_length",
            "no_empty_response",
            "relevance",
        ]

    def test_a_retired_heuristic_stops_running(self, repository):
        assert repository.delete("coherence", HEURISTIC)
        checker = HeuristicChecker(repository=repository)

        results = checker.check_all("Short.", "Short question?")

        assert "coherence" not in checker.list_heuristics()
        assert "coherence" not in (
            results["passed_checks"] + results["failed_checks"]
        )
        assert results["check_count"] == 3

    def test_a_catalogued_heuristic_without_an_implementation_is_skipped(
        self, repository
    ):
        repository.get_or_create(
            name="mentions_sources",
            description="Response names its sources",
            criterion_type=HEURISTIC,
            mandatory=True,
        )
        checker = HeuristicChecker(repository=repository)

        # Counting it would mean failing a check nothing can run.
        assert "mentions_sources" not in checker.list_heuristics()
