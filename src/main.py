"""Main entry point for Adaptive LLM Jury Agent"""

import asyncio
from typing import Dict, Any, Optional

from src.config import validate_config, USE_POSTGRES_MEMORY
from src.shared.logger import get_logger
from src.context_analysis import ContextAnalyzer
from src.rule_generator import (
    TaskClassifier,
    ObjectiveDefinition,
    CriteriaGenerator,
    HeuristicChecker,
)
from src.jury import EvaluatorPanel, AccuracyCalculator, HumanInTheLoop
from src.metrics import (
    JuryComposition,
    ThresholdCalculator,
    FinalScoreCalculator,
)
from src.memory_manager import (
    CacheManager,
    HistoryTracker,
    ErrorDetector,
    CalibrationManager,
    PostgresMemoryStore,
)

logger = get_logger(__name__)


class AdaptiveJuryAgent:
    """Main agent orchestrating the adaptive evaluation pipeline"""

    def __init__(self):
        """Initialize all components"""
        logger.info("Initializing Adaptive Jury Agent...")

        # Context analysis
        self.context_analyzer = ContextAnalyzer()

        # Rule generation
        self.task_classifier = TaskClassifier()
        self.objective_def = ObjectiveDefinition()
        self.criteria_gen = CriteriaGenerator()
        self.heuristic_checker = HeuristicChecker()

        # Jury evaluation
        self.evaluator_panel = EvaluatorPanel()
        self.accuracy_calc = AccuracyCalculator()
        self.hitl = HumanInTheLoop()

        # Metrics aggregation
        self.jury_composition = JuryComposition()
        self.threshold_calc = ThresholdCalculator()
        self.final_score_calc = FinalScoreCalculator()

        # Memory management
        self.cache = CacheManager()
        self.memory_store = self._init_memory_store()
        self.history = HistoryTracker(store=self.memory_store)
        self.error_detector = ErrorDetector()
        self.calibration = CalibrationManager()

        logger.info("Adaptive Jury Agent initialized successfully")

    def _init_memory_store(self) -> Optional[PostgresMemoryStore]:
        """Initialize the PostgreSQL-backed memory store, if enabled"""
        if not USE_POSTGRES_MEMORY:
            return None

        store = PostgresMemoryStore()
        if store.connect():
            store.ensure_schema()
            logger.info("Connected to PostgreSQL memory store")
            return store

        logger.warning(
            "USE_POSTGRES_MEMORY is enabled but the connection failed; "
            "falling back to in-memory history only"
        )
        return None

    async def evaluate(
        self,
        conversation: str,
        response: str,
        query: str = "",
    ) -> Dict[str, Any]:
        """
        Main evaluation pipeline

        Args:
            conversation: Multi-turn conversation context
            response: Response to evaluate
            query: Original user query

        Returns:
            Comprehensive evaluation result
        """
        logger.info("Starting evaluation pipeline...")

        # Check cache
        cache_key = f"eval_{hash(conversation + response)}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.info("Returning cached evaluation")
            return cached

        try:
            # 1. Context Analysis
            logger.debug("Phase 1: Context Analysis")
            context = self._analyze_context(conversation)

            # 2. Rule Generation
            logger.debug("Phase 2: Rule Generation")
            rules = self._generate_rules(context, response, query)

            # 3. Jury Evaluation
            logger.debug("Phase 3: Jury Evaluation")
            jury_result = self._evaluate_with_jury(
                response, rules, context
            )

            # 4. Metrics Aggregation
            logger.debug("Phase 4: Metrics Aggregation")
            final_result = self._aggregate_metrics(
                jury_result, rules, context
            )

            # 5. Memory Management
            logger.debug("Phase 5: Memory Management")
            self._update_memory(final_result)

            # Cache result
            self.cache.set(cache_key, final_result)

            logger.info("Evaluation complete")
            return final_result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def _analyze_context(
        self, conversation: str
    ) -> Dict[str, Any]:
        """Analyze conversation context"""
        return self.context_analyzer.analyze(conversation)

    def _generate_rules(
        self,
        context: Dict[str, Any],
        response: str,
        query: str,
    ) -> Dict[str, Any]:
        """Generate evaluation rules"""
        # Classify task
        classification = self.task_classifier.classify(context)
        task_type = classification["task_type"]

        # Define objectives
        objectives = self.objective_def.get_objectives(task_type)

        # Generate criteria
        criteria = self.criteria_gen.generate(context, objectives)

        # Run heuristics
        heuristics = self.heuristic_checker.check_all(response, query)

        return {
            "task_type": task_type,
            "objectives": objectives,
            "criteria": criteria,
            "heuristics": heuristics,
            "classification": classification,
        }

    def _evaluate_with_jury(
        self,
        response: str,
        rules: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get evaluations from jury panel"""
        jury_result = self.evaluator_panel.evaluate(
            response, rules["criteria"], context
        )

        # Check for human review need
        avg_score = jury_result.get("average_score", 5.0)
        confidence = jury_result.get("confidence", 0.5)

        if self.hitl.should_request_review(confidence):
            logger.info("Requesting human-in-the-loop review")
            self.hitl.request_review(
                "eval_001",
                response,
                rules["criteria"],
                avg_score,
            )

        return jury_result

    def _aggregate_metrics(
        self,
        jury_result: Dict[str, Any],
        rules: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate all metrics into final score"""
        # Heuristic check
        heuristics = rules.get("heuristics", {})

        # Accuracy calculation (if annotated data available)
        accuracy = self.accuracy_calc.calculate_accuracy(
            jury_result.get("average_score", 5.0),
            rules.get("criteria", {}),
        )

        # Final score calculation
        final_score = self.final_score_calc.calculate_final_score(
            jury_result,
            heuristics,
            accuracy,
            context,
        )

        # Generate report
        report = self.final_score_calc.generate_report(
            final_score
        )

        return {
            "jury_evaluation": jury_result,
            "accuracy_metrics": accuracy,
            "final_score_breakdown": final_score,
            "evaluation_report": report,
        }

    def _update_memory(self, result: Dict[str, Any]) -> None:
        """Update memory systems"""
        # Record in history
        self.history.record_evaluation(
            "eval_001",
            "general",
            result["final_score_breakdown"]["final_score"],
            result["final_score_breakdown"]["overall_confidence"],
            result["final_score_breakdown"]["component_breakdown"],
        )

        # Check for errors
        errors = self.error_detector.detect_errors(
            result["final_score_breakdown"], {}
        )
        if errors:
            logger.warning(f"Found {len(errors)} evaluation errors")

        logger.debug("Memory systems updated")


def main():
    """Main entry point"""
    if not validate_config():
        logger.warning("Some configuration values may be missing")

    agent = AdaptiveJuryAgent()

    # Example usage
    conversation = (
        "User: Explain how neural networks work\n\n"
        "Assistant: Neural networks are computing systems "
        "inspired by biological neurons..."
    )

    response = (
        "Neural networks consist of layers of interconnected nodes. "
        "Each connection has a weight that is adjusted during training..."
    )

    try:
        result = asyncio.run(
            agent.evaluate(
                conversation, response, "Explain neural networks"
            )
        )
        logger.info(f"Evaluation result: {result}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
