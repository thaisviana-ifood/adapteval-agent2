# Getting Started with Adaptive Jury Agent

This guide will help you quickly get up and running with the Adaptive Jury Agent system.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your Anthropic API key
```

### 2. Basic Usage

```python
import asyncio
from src.main import AdaptiveJuryAgent

async def main():
    agent = AdaptiveJuryAgent()
    
    conversation = """
User: Explain neural networks
Assistant: Neural networks are computational models...
User: Can you give an example?
Assistant: Here's a practical example...
"""
    
    response = "Neural networks are computing systems inspired by biological neurons..."
    
    result = await agent.evaluate(conversation, response)
    
    print(f"Final Score: {result['final_score_breakdown']['final_score']}")
    print(f"Confidence: {result['final_score_breakdown']['overall_confidence']}")

asyncio.run(main())
```

## Project Components

### 1. Context Analysis
Analyzes the conversation to understand:
- Turn patterns and speaker distribution
- Key topics and semantic content
- Text complexity and readability
- User and assistant intents

### 2. Rule Generation
Generates evaluation rules based on:
- Task type classification
- Success criteria definition
- Context-aware evaluation criteria
- Quick validation heuristics

### 3. Jury Evaluation
Multi-evaluator assessment using:
- LLM-based evaluators (Claude)
- Rule-based evaluators
- Accuracy calculation from annotated data
- Human-in-the-loop for low confidence cases

### 4. Metrics Aggregation
Combines scores using:
- Weighted jury voting
- Dynamic threshold calculation
- Component-aware final scoring

### 5. Memory Management
Optimizes performance with:
- Response caching
- Evaluation history tracking
- Error detection and validation
- Calibration against reference data

## Configuration

Key settings in `.env`:

```env
LLM_API_KEY=your_key_here
CONFIDENCE_THRESHOLD=0.7
MIN_EVALUATORS=3
LOG_LEVEL=INFO
```

## Running Examples

```bash
# Run the main agent
python -m src.main

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_context.py -v
```

## Understanding Output

The evaluation returns a comprehensive result:

```python
{
    'final_score_breakdown': {
        'final_score': 8.5,
        'overall_confidence': 0.85,
        'component_breakdown': {
            'jury': {'score': 8.5, 'weight': 0.5},
            'heuristic': {'score': 8.0, 'weight': 0.2},
            'accuracy': {'score': 8.2, 'weight': 0.2},
            'context': {'score': 8.7, 'weight': 0.1}
        }
    },
    'evaluation_report': {
        'recommendation': 'Excellent - Approved',
        'summary': {
            'strength': ['Strong jury evaluation'],
            'weakness': []
        }
    }
}
```

## Next Steps

1. **Customize Evaluators**: Create custom evaluator classes for domain-specific evaluation
2. **Load Annotated Data**: Load reference data for calibration
3. **Set Up Database**: Configure PostgreSQL for persistent storage
4. **Scale Evaluation**: Use batch processing for multiple evaluations
5. **Monitor Performance**: Use history tracking and caching stats

## Common Tasks

### Add Custom Criteria

```python
from src.rule_generator import CriteriaGenerator

cg = CriteriaGenerator()
criteria = cg.generate(context, objectives)
criteria['weighted_criteria'].append({
    'name': 'CustomMetric',
    'description': 'Your metric',
    'weight': 1.0
})
```

### Enable Human Review

```python
agent.hitl.request_review(
    eval_id="eval_001",
    response=response,
    criteria=criteria,
    automated_score=score,
    reason="Low confidence"
)
```

### Load Reference Data

```python
from src.jury import AccuracyCalculator

acc = AccuracyCalculator()
acc.load_annotated_data(your_annotated_data)
accuracy = acc.calculate_accuracy(predicted_score, criteria)
```

## Troubleshooting

### API Key Issues
- Ensure `LLM_API_KEY` is set in `.env`
- Check Anthropic API credentials

### Database Connection Errors
- Verify PostgreSQL is running
- Check `DB_HOST` and `DB_PORT` settings

### Performance Issues
- Increase `CACHE_SIZE_MB` for larger cache
- Adjust `NUM_WORKERS` for parallel processing
- Check `LOG_LEVEL` (set to WARNING to reduce I/O)

## Resources

- [Full Documentation](README.md)
- [API Reference](README.md#api-reference)
- [Test Examples](tests/)
- [Configuration Guide](README.md#configuration)

## Support

For issues or questions:
1. Check the README.md
2. Review test cases for examples
3. Check logs in `logs/` directory
4. Open an issue on GitHub
