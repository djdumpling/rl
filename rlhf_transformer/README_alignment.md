# RLHF Alignment Analysis Tools

This toolkit helps you analyze alignment issues during RLHF training by providing checkpointing, response analysis, and anomaly detection capabilities.

## Quick Start

### 1. Run Training with Checkpointing

```python
from alignment_analysis import enhanced_training_with_checkpoints
from rlhf.ipynb import RLHFArgs, reward_fn_sentiment_imdb

args = RLHFArgs(reward_fn=reward_fn_sentiment_imdb, total_phases=200)
trainer, analyzer = enhanced_training_with_checkpoints(args, checkpoint_every=50)
```

### 2. Analyze Alignment Degradation

```python
# Analyze how responses change across phases
results = analyzer.analyze_alignment_degradation([50, 100, 150, 200])

# Detect anomalies
anomalies = analyzer.detect_anomalies([50, 100, 150, 200])

# Compare specific phases
comparison = analyzer.compare_phases(50, 200)
```

### 3. Generate Responses from Saved Weights

```python
# Generate fresh responses from any checkpoint
result = analyzer.generate_responses_from_checkpoint(phase=150, num_samples=10)
print(f"Mean reward: {result['mean_reward']:.4f}")
print(f"Sample: {result['samples'][0]}")
```

## Key Features

- **Automatic Checkpointing**: Save model weights every N phases
- **Response Analysis**: Track how responses change during training
- **Anomaly Detection**: Find low-reward or unusual responses
- **Phase Comparison**: Compare any two training phases
- **Fresh Generation**: Generate new responses from saved weights

## What to Look For

1. **Reward Degradation**: Mean reward decreasing over time
2. **High Variance**: Inconsistent reward quality
3. **Low Reward Samples**: Responses that score poorly
4. **Unusual Responses**: Very short/long or nonsensical text

## Files Created

- `checkpoints/run_name/phase_XXX.pt` - Model weights
- `checkpoints/run_name/responses_phase_XXX.json` - Response data
- `checkpoints/run_name/full_responses_history.json` - Complete history

## Example Analysis Workflow

```python
# 1. Run training
trainer, analyzer = enhanced_training_with_checkpoints(args)

# 2. Quick alignment check
results = analyzer.analyze_alignment_degradation([50, 100, 150, 200])

# 3. If issues detected, investigate further
anomalies = analyzer.detect_anomalies([50, 100, 150, 200])

# 4. Compare problematic phases
comparison = analyzer.compare_phases(50, 200)

# 5. Generate fresh samples for manual inspection
samples = analyzer.generate_responses_from_checkpoint(200, num_samples=20)
```

This toolkit helps you identify when your model starts producing misaligned responses and provides the data needed to understand what went wrong. 