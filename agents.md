# Agents Architecture & Extension Guide

This document describes the agent system in the Adaptive LLM Jury Agent framework and how to extend it with custom evaluators.

## Overview

The Adaptive Jury Agent uses a **multi-evaluator panel** system where independent LLM agents assess responses according to context-aware criteria. The framework is designed for extensibility - you can add new evaluator types, customize evaluation logic, and integrate specialized assessment strategies.

## Core Agent Components

### 1. Evaluator Panel (`src/jury/panel.py`)

The `EvaluatorPanel` orchestrates multiple independent evaluators and aggregates their assessments.

```python
from src.jury import EvaluatorPanel

panel = EvaluatorPanel()
result = panel.evaluate(
    response="...",
    criteria={...},
    context={...}
)
```

**Responsibilities:**
- Coordinates evaluation across multiple evaluators
- Handles concurrent evaluations
- Aggregates scores and confidence metrics
- Produces consensus results

### 2. Evaluator Base Class (`src/jury/evaluator.py`)

All evaluators inherit from the abstract `Evaluator` base class:

```python
from src.jury import Evaluator

class MyEvaluator(Evaluator):
    def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
        """Perform evaluation and return structured result"""
        # Implementation
        pass
    
    def get_name(self) -> str:
        """Return evaluator name"""
        return "MyEvaluator"
```

**Required Methods:**
- `evaluate()` - Core evaluation logic
- `get_name()` - Evaluator identifier

**Return Format:**
```python
{
    "score": 7.5,              # 0-10 numeric score
    "confidence": 0.85,        # 0-1 confidence level
    "reasoning": "...",        # Explanation of assessment
    "component_scores": {      # Optional: break down by criteria
        "accuracy": 8.0,
        "completeness": 7.0,
        "clarity": 7.5
    },
    "flags": []                # Optional: important findings
}
```

## Built-in Evaluators

### LLMEvaluator

Uses Claude to assess responses against criteria.

```python
from src.jury.llm_evaluator import LLMEvaluator

evaluator = LLMEvaluator(model="claude-opus-4")
result = evaluator.evaluate(
    response="Neural networks are...",
    criteria={
        "accuracy": {
            "description": "Is the explanation correct?",
            "weight": 1.0
        },
        "completeness": {
            "description": "Does it cover key concepts?",
            "weight": 0.8
        }
    },
    context={"task_type": "explanation", "domain": "AI"}
)
```

**Configuration:**
- `model`: Claude model to use (default: "claude-opus-4")
- `temperature`: Sampling temperature (default: 0.7)
- `max_tokens`: Max response tokens (default: 2048)

### RulesBasedEvaluator

Fast heuristic-based evaluation using predefined rules.

```python
from src.jury.rules_based import RulesBasedEvaluator

evaluator = RulesBasedEvaluator()
result = evaluator.evaluate(
    response="...",
    criteria={...},
    context={...}
)
```

**Use Cases:**
- Quick validation checks
- Consistency verification
- Format compliance
- Baseline assessments

## Creating Custom Evaluators

### Basic Custom Evaluator

```python
from src.jury import Evaluator
from typing import Dict, Any

class DomainExpertEvaluator(Evaluator):
    """Evaluates responses from domain-specific perspective"""
    
    def __init__(self, domain: str):
        self.domain = domain
    
    def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
        """Assess response within domain context"""
        # Your evaluation logic
        score = self._calculate_domain_score(response, criteria)
        confidence = self._estimate_confidence(score)
        
        return {
            "score": score,
            "confidence": confidence,
            "reasoning": f"Domain: {self.domain}. Score rationale...",
            "component_scores": {
                "domain_accuracy": 8.0,
                "terminology": 7.5,
                "methodology": 7.0
            }
        }
    
    def get_name(self) -> str:
        return f"DomainExpertEvaluator_{self.domain}"
    
    def _calculate_domain_score(self, response: str, criteria: Dict) -> float:
        # Domain-specific scoring logic
        pass
    
    def _estimate_confidence(self, score: float) -> float:
        # Confidence estimation based on score
        pass
```

### LLM-Based Custom Evaluator

```python
from src.jury.llm_evaluator import LLMEvaluator
from src.shared.llm import LLMClient
from typing import Dict, Any

class LanguageQualityEvaluator(LLMEvaluator):
    """Specialized evaluator for linguistic quality"""
    
    def __init__(self):
        super().__init__(model="claude-opus-4")
        self.language_criteria = [
            "clarity", "conciseness", "grammar",
            "vocabulary_appropriateness", "tone"
        ]
    
    def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
        # Customize evaluation prompt for language quality
        prompt = self._build_language_prompt(response, context)
        
        llm_result = self.llm_client.call(prompt)
        
        # Parse and structure result
        return self._parse_language_result(llm_result)
    
    def get_name(self) -> str:
        return "LanguageQualityEvaluator"
    
    def _build_language_prompt(self, response: str, context: Dict) -> str:
        return f"""Evaluate the linguistic quality of this response:

Response: {response}

Assess: {', '.join(self.language_criteria)}

Provide scores 0-10 for each dimension and overall assessment."""
    
    def _parse_language_result(self, llm_result: str) -> Dict:
        # Parse LLM output into structured format
        pass
```

## Integrating Custom Evaluators

### Option 1: Add to EvaluatorPanel

```python
from src.jury import EvaluatorPanel
from custom_evaluators import MyEvaluator

# Initialize panel with custom evaluator
panel = EvaluatorPanel()
panel.add_evaluator(MyEvaluator())
panel.add_evaluator(DomainExpertEvaluator(domain="biology"))

# Use in agent
agent = AdaptiveJuryAgent()
agent.evaluator_panel = panel

# Evaluate
result = agent.evaluate(conversation, response)
```

### Option 2: Configuration-Based Loading

```python
# In src/config.py
EVALUATORS = [
    {
        "name": "llm_evaluator",
        "class": "src.jury.LLMEvaluator",
        "config": {"model": "claude-opus-4"}
    },
    {
        "name": "domain_expert",
        "class": "custom_evaluators.DomainExpertEvaluator",
        "config": {"domain": "biology"}
    },
    {
        "name": "rules_based",
        "class": "src.jury.RulesBasedEvaluator",
        "config": {}
    }
]

# Load dynamically
from src.jury import EvaluatorPanel

def load_evaluators(config):
    panel = EvaluatorPanel()
    for eval_config in config.EVALUATORS:
        evaluator = load_class(eval_config["class"])(**eval_config["config"])
        panel.add_evaluator(evaluator)
    return panel
```

## Advanced Patterns

### Evaluator with Caching

```python
from src.jury import Evaluator
from src.memory_manager import CacheManager
import hashlib

class CachedEvaluator(Evaluator):
    """Evaluator with result caching to reduce redundant assessments"""
    
    def __init__(self):
        self.cache = CacheManager(ttl=3600)  # 1 hour TTL
    
    def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
        # Create cache key from inputs
        cache_key = self._make_cache_key(response, criteria)
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Perform evaluation
        result = self._do_evaluation(response, criteria, context)
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
    
    def _make_cache_key(self, response: str, criteria: Dict) -> str:
        key_str = f"{response}:{str(criteria)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _do_evaluation(self, response: str, criteria: Dict, context: Dict) -> Dict:
        pass
    
    def get_name(self) -> str:
        return "CachedEvaluator"
```

### Weighted Evaluator Composition

```python
from src.jury import Evaluator

class CompositeEvaluator(Evaluator):
    """Combines multiple evaluators with weighted voting"""
    
    def __init__(self, evaluators: list, weights: list):
        self.evaluators = evaluators
        self.weights = weights
        
        if len(evaluators) != len(weights):
            raise ValueError("Evaluators and weights must have same length")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
    
    def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
        results = []
        
        # Get evaluations from all sub-evaluators
        for evaluator in self.evaluators:
            result = evaluator.evaluate(response, criteria, context)
            results.append(result)
        
        # Weighted aggregate
        weighted_score = sum(
            r["score"] * w for r, w in zip(results, self.weights)
        )
        avg_confidence = sum(
            r["confidence"] * w for r, w in zip(results, self.weights)
        )
        
        # Combine reasoning
        reasoning = " + ".join([
            f"{e.get_name()}: {r['reasoning']}"
            for e, r in zip(self.evaluators, results)
        ])
        
        return {
            "score": weighted_score,
            "confidence": avg_confidence,
            "reasoning": reasoning,
            "component_evaluations": results
        }
    
    def get_name(self) -> str:
        return "CompositeEvaluator"
```

### Adaptive Evaluator Selection

```python
from src.jury import Evaluator

class AdaptiveEvaluator(Evaluator):
    """Selects evaluator strategy based on context"""
    
    def __init__(self):
        self.evaluators_by_task = {
            "code_generation": CodeEvaluator(),
            "explanation": ExplanationEvaluator(),
            "summarization": SummarizationEvaluator(),
        }
        self.default = LLMEvaluator()
    
    def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
        task_type = context.get("task_type", "general")
        
        # Select evaluator based on task
        evaluator = self.evaluators_by_task.get(
            task_type, self.default
        )
        
        return evaluator.evaluate(response, criteria, context)
    
    def get_name(self) -> str:
        return "AdaptiveEvaluator"
```

## Best Practices

### 1. Score Consistency
- Always return scores in 0-10 range
- Document what each score level means for your evaluator
- Ensure consistency across different response types

```python
def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
    score = self._calculate_score(response, criteria)
    
    # Validate score range
    if not (0 <= score <= 10):
        raise ValueError(f"Score must be 0-10, got {score}")
    
    return {
        "score": score,
        "confidence": self._estimate_confidence(response, score),
        "reasoning": "..."
    }
```

### 2. Meaningful Confidence
- Confidence should reflect actual uncertainty in assessment
- Higher confidence when evaluation is clear-cut
- Lower confidence with ambiguous or edge-case responses

```python
def _estimate_confidence(self, response: str, score: float) -> float:
    # High confidence for extreme scores (very good/very bad)
    if score >= 8.5 or score <= 1.5:
        return 0.9
    
    # Medium-high for clear assessments
    if score >= 7.0 or score <= 3.0:
        return 0.75
    
    # Lower confidence for ambiguous cases
    return 0.6
```

### 3. Component Breakdown
- Always provide component scores when criteria are specific
- Helps understand where improvements are needed
- Useful for human review and feedback

```python
return {
    "score": 7.2,
    "confidence": 0.8,
    "reasoning": "Good overall, but some gaps in explanation",
    "component_scores": {
        "accuracy": 8.0,
        "clarity": 7.0,
        "completeness": 6.5
    }
}
```

### 4. Error Handling
- Catch and log evaluation failures gracefully
- Return degraded assessment rather than crashing
- Include error info in reasoning

```python
def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
    try:
        return self._do_evaluation(response, criteria, context)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        
        # Return minimal valid result
        return {
            "score": 5.0,  # Neutral score
            "confidence": 0.1,  # Low confidence
            "reasoning": f"Evaluation error: {str(e)}",
            "error": True
        }
```

### 5. Context Awareness
- Use context information to tailor evaluation
- Different tasks require different assessment strategies
- Leverage semantic and structural analysis

```python
def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
    task_type = context.get("task_type")
    complexity = context.get("complexity", {}).get("level")
    
    # Adjust evaluation based on task and complexity
    if task_type == "code_generation" and complexity == "high":
        return self._evaluate_complex_code(response, criteria)
    else:
        return self._evaluate_simple(response, criteria)
```

## Testing Evaluators

```python
import pytest
from src.jury.evaluators import MyEvaluator

class TestMyEvaluator:
    
    @pytest.fixture
    def evaluator(self):
        return MyEvaluator()
    
    def test_evaluate_returns_valid_structure(self, evaluator):
        result = evaluator.evaluate(
            response="Test response",
            criteria={"accuracy": {"weight": 1.0}},
            context={"task_type": "test"}
        )
        
        assert "score" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert 0 <= result["score"] <= 10
        assert 0 <= result["confidence"] <= 1
    
    def test_score_consistency(self, evaluator):
        responses = ["good response", "better response", "best response"]
        scores = []
        
        for resp in responses:
            result = evaluator.evaluate(resp, {}, {})
            scores.append(result["score"])
        
        # Verify increasing quality corresponds to increasing scores
        assert scores[0] <= scores[1] <= scores[2]
    
    def test_handles_edge_cases(self, evaluator):
        # Empty response
        result = evaluator.evaluate("", {}, {})
        assert "error" not in result or result.get("error") is False
        
        # Very long response
        long_resp = "word " * 10000
        result = evaluator.evaluate(long_resp, {}, {})
        assert "error" not in result or result.get("error") is False
```

## Performance Considerations

### Concurrent Evaluation
The `EvaluatorPanel` runs evaluations concurrently when possible:

```python
# This runs all evaluators in parallel
result = panel.evaluate(response, criteria, context)
```

### Caching Strategy
```python
from src.memory_manager import CacheManager

cache = CacheManager(
    max_size=1000,      # Max items
    ttl=3600            # Time-to-live in seconds
)
```

### Batch Processing
```python
from src.jury import EvaluatorPanel

panel = EvaluatorPanel()

# Evaluate multiple responses efficiently
responses = ["response1", "response2", "response3"]
results = [
    panel.evaluate(resp, criteria, context)
    for resp in responses
]
```

## Extension Examples

### Sentiment-Based Evaluator
```python
from src.jury import Evaluator
from transformers import pipeline

class SentimentEvaluator(Evaluator):
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
    
    def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
        sentiment = self.sentiment_pipeline(response)[0]
        
        # Map sentiment to score
        score_map = {
            "POSITIVE": 8.0,
            "NEGATIVE": 3.0,
            "NEUTRAL": 5.0
        }
        score = score_map[sentiment["label"]]
        
        return {
            "score": score,
            "confidence": sentiment["score"],
            "reasoning": f"Sentiment: {sentiment['label']}"
        }
    
    def get_name(self) -> str:
        return "SentimentEvaluator"
```

### Code Quality Evaluator
```python
from src.jury import Evaluator
import ast

class CodeQualityEvaluator(Evaluator):
    def evaluate(self, response: str, criteria: Dict, context: Dict) -> Dict:
        try:
            tree = ast.parse(response)
        except SyntaxError:
            return {
                "score": 0.0,
                "confidence": 1.0,
                "reasoning": "Code contains syntax errors"
            }
        
        metrics = {
            "has_docstrings": self._check_docstrings(tree),
            "function_complexity": self._check_complexity(tree),
            "follows_pep8": self._check_pep8(response)
        }
        
        score = sum(metrics.values()) / len(metrics) * 10
        
        return {
            "score": score,
            "confidence": 0.9,
            "reasoning": f"Code quality metrics: {metrics}",
            "component_scores": {k: v * 10 for k, v in metrics.items()}
        }
    
    def get_name(self) -> str:
        return "CodeQualityEvaluator"
    
    def _check_docstrings(self, tree: ast.AST) -> float:
        pass
    
    def _check_complexity(self, tree: ast.AST) -> float:
        pass
    
    def _check_pep8(self, code: str) -> float:
        pass
```

## Related Documentation

- [Architecture Overview](README.md#architecture-overview)
- [Rule Generation](./rule_generation.md)
- [Metrics Aggregation](./metrics.md)
- [Memory Management](./memory_management.md)
- [Human-in-the-Loop](./human_feedback.md)
