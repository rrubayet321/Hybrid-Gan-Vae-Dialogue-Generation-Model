# 📊 Evaluation Metrics Guide

## Overview

This guide explains how to evaluate the Hybrid GAN + VAE model using comprehensive metrics that assess text generation quality, fairness, and explainability.

## Module Structure

```
evaluation_metrics.py
├── TextGenerationEvaluator
│   ├── compute_bleu()        # N-gram precision metrics
│   ├── compute_rouge()       # N-gram recall metrics
│   ├── compute_perplexity()  # Language model quality
│   └── compute_diversity()   # Vocabulary richness
└── ModelEvaluator
    ├── evaluate_generation_quality()  # Complete text generation assessment
    ├── evaluate_fairness()            # Demographic group comparison
    ├── evaluate_with_xai()            # Explainability metrics
    └── create_evaluation_report()     # Comprehensive reporting
```

## Metrics Explained

### 1. Text Generation Metrics

#### BLEU (Bilingual Evaluation Understudy)
- **What it measures**: N-gram precision (how many n-grams in generated text appear in reference)
- **Range**: 0.0 to 1.0 (higher is better)
- **Variants**:
  - BLEU-1: Unigram precision (individual words)
  - BLEU-2: Bigram precision (2-word phrases)
  - BLEU-3: Trigram precision (3-word phrases)
  - BLEU-4: 4-gram precision (4-word phrases)

**Interpretation**:
- `> 0.40`: Excellent overlap with reference
- `0.20 - 0.40`: Good quality
- `< 0.20`: Room for improvement

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **What it measures**: N-gram recall (how many n-grams from reference appear in generated text)
- **Range**: 0.0 to 1.0 (higher is better)
- **Variants**:
  - ROUGE-1: Unigram recall
  - ROUGE-2: Bigram recall
  - ROUGE-L: Longest common subsequence

**Interpretation**:
- `> 0.40`: High content coverage
- `0.20 - 0.40`: Moderate coverage
- `< 0.20`: Low coverage

#### Perplexity
- **What it measures**: How "surprised" the model is by the generated text
- **Range**: 1.0 to ∞ (lower is better)
- **Formula**: `exp(-average log probability)`

**Interpretation**:
- `< 50`: Excellent language model quality
- `50 - 200`: Good quality
- `> 200`: Model uncertainty

#### Diversity Metrics
- **Distinct-1**: Ratio of unique unigrams to total unigrams
- **Distinct-2**: Ratio of unique bigrams to total bigrams
- **Vocab Size**: Total unique words generated
- **Avg Length**: Average text length

**Interpretation**:
- `Distinct-1/2 > 0.50`: High diversity
- `Distinct-1/2 < 0.30`: Repetitive generation

### 2. Quality Metrics

#### Discriminator Quality Score
- **What it measures**: How "real" the generated text appears to the discriminator
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - `> 0.70`: Very realistic text
  - `0.50 - 0.70`: Moderate quality
  - `< 0.50`: Easy to distinguish from real text

### 3. Fairness Metrics

#### Quality Disparity
- **What it measures**: Difference between highest and lowest quality across demographic groups
- **Range**: 0.0 to 1.0 (lower is better)
- **Formula**: `max(group_qualities) - min(group_qualities)`

**Interpretation**:
- `< 0.05`: Excellent fairness (minimal disparity)
- `0.05 - 0.10`: Acceptable disparity
- `> 0.10`: Significant unfairness

#### Per-Group Metrics
For each demographic group:
- **Count**: Number of samples
- **Quality Mean**: Average discriminator score
- **Quality Std**: Standard deviation (consistency)
- **Diversity**: Distinct-1 score

### 4. XAI Metrics

#### Latent Space Analysis
- **z_mean_avg**: Average latent representation mean (should be near 0)
- **z_mean_std**: Std of latent means (diversity)
- **z_logvar_avg**: Average latent variance (should be near 0)
- **z_logvar_std**: Std of latent variances
- **latent_diversity**: How diverse the latent representations are

**Interpretation**:
- `z_mean_avg ≈ 0`: Properly regularized VAE
- `latent_diversity > 0.002`: Good separation in latent space

## Usage Examples

### Example 1: Basic Quality Evaluation

```python
import numpy as np
import pandas as pd
import pickle
from evaluation_metrics import ModelEvaluator
from hybrid_model import HybridGANVAE
from simple_tokenizer import SimpleTokenizer

# Load model and tokenizer
with open('processed_data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

hybrid_model = HybridGANVAE(vocab_size=154, max_length=100)
hybrid_model.vae.load_weights('models/vae_pretrained.weights.h5')

# Load validation data
val_customer = np.load('processed_data/val_customer.npy')
val_agent = np.load('processed_data/val_agent.npy')

# Create evaluator
evaluator = ModelEvaluator(hybrid_model, tokenizer)

# Evaluate generation quality
results = evaluator.evaluate_generation_quality(
    val_customer[:500],
    val_agent[:500],
    sample_size=500
)

print(f"BLEU-4: {results['bleu']['bleu-4']:.4f}")
print(f"ROUGE-L: {results['rouge']['rouge-l']:.4f}")
print(f"Quality: {results['quality']['mean']:.4f}")
```

### Example 2: Fairness Evaluation

```python
# Load metadata
val_metadata = pd.read_csv('processed_data/val_metadata.csv')

# Evaluate fairness across demographics
fairness_results = evaluator.evaluate_fairness(
    val_customer,
    val_metadata,
    sensitive_attributes=['customer_segment', 'region']
)

# Check disparities
for attr, results in fairness_results.items():
    print(f"\n{attr}:")
    print(f"  Disparity: {results['quality_disparity']:.4f}")
    
    # Show per-group results
    for group, group_results in results['groups'].items():
        print(f"  {group}: {group_results['quality_mean']:.4f}")
```

### Example 3: XAI-Enhanced Evaluation

```python
from explainability import ModelExplainer

# Add explainer
explainer = ModelExplainer(hybrid_model, tokenizer)
evaluator = ModelEvaluator(hybrid_model, tokenizer, explainer)

# XAI evaluation
xai_results = evaluator.evaluate_with_xai(
    val_customer[:100],
    sample_size=100
)

print("Latent Space Analysis:")
for key, value in xai_results['latent_space'].items():
    print(f"  {key}: {value:.4f}")
```

### Example 4: Comprehensive Report

```python
# Run all evaluations
gen_results = evaluator.evaluate_generation_quality(
    val_customer[:1000],
    val_agent[:1000]
)

fairness_results = evaluator.evaluate_fairness(
    val_customer,
    val_metadata,
    ['customer_segment', 'region', 'priority']
)

xai_results = evaluator.evaluate_with_xai(val_customer[:200])

# Combine results
all_results = {
    'bleu': gen_results['bleu'],
    'rouge': gen_results['rouge'],
    'diversity': gen_results['diversity'],
    'quality': gen_results['quality'],
    'fairness': fairness_results,
    'xai': xai_results
}

# Generate report
evaluator.create_evaluation_report(
    all_results,
    save_path='results/evaluation_report.txt'
)
```

## Interpreting Results

### Good Model Indicators
✅ **Text Quality**:
- BLEU-4 > 0.15
- ROUGE-L > 0.25
- Quality > 0.60
- Distinct-1 > 0.30

✅ **Fairness**:
- Quality disparity < 0.10 across all attributes
- Similar diversity metrics across groups

✅ **XAI**:
- z_mean_avg ≈ 0 (proper regularization)
- latent_diversity > 0.001 (good separation)

### Areas for Improvement
⚠️ **Low BLEU/ROUGE**:
- May indicate generic responses
- Consider more diverse training data
- Adjust temperature sampling

⚠️ **High Quality Disparity**:
- Check for data imbalance across groups
- Apply fairness regularization
- Use bias mitigation techniques

⚠️ **Poor Latent Space**:
- z_mean_avg far from 0: Increase KL weight
- Low latent_diversity: Increase β in β-VAE

## Benchmark Comparisons

### Untrained Model
- BLEU-4: ~0.001 (random)
- ROUGE-L: ~0.003 (random)
- Quality: ~0.50 (random discriminator)
- Disparity: ~0.05 (random)

### After VAE Pretraining
- BLEU-4: ~0.05 - 0.10
- ROUGE-L: ~0.10 - 0.20
- Quality: ~0.55 - 0.65
- Disparity: ~0.03 - 0.07

### After Full Training
- BLEU-4: ~0.15 - 0.25
- ROUGE-L: ~0.25 - 0.35
- Quality: ~0.65 - 0.75
- Disparity: ~0.02 - 0.05

## Evaluation Workflow

### 1. During Training
```python
# Checkpoint evaluation every N epochs
for epoch in range(num_epochs):
    # Training code...
    
    if epoch % 5 == 0:
        results = evaluator.evaluate_generation_quality(
            val_customer[:200],
            val_agent[:200]
        )
        print(f"Epoch {epoch} - BLEU: {results['bleu']['bleu-4']:.4f}")
```

### 2. Model Comparison
```python
models = {
    'baseline': load_model('baseline_weights.h5'),
    'with_fairness': load_model('fair_weights.h5'),
    'fully_trained': load_model('final_weights.h5')
}

for name, model in models.items():
    evaluator = ModelEvaluator(model, tokenizer)
    results = evaluator.evaluate_generation_quality(
        val_customer, val_agent, sample_size=1000
    )
    print(f"{name}: BLEU-4={results['bleu']['bleu-4']:.4f}")
```

### 3. Production Monitoring
```python
# Periodic evaluation on production data
def monitor_model_quality():
    recent_customer = load_recent_messages(limit=500)
    recent_agent = load_recent_responses(limit=500)
    
    results = evaluator.evaluate_generation_quality(
        recent_customer, recent_agent
    )
    
    # Alert if quality drops
    if results['quality']['mean'] < 0.60:
        send_alert("Model quality degradation detected!")
    
    return results
```

## Troubleshooting

### Low BLEU/ROUGE Scores
**Problem**: BLEU-4 < 0.05
**Solutions**:
1. Check data quality (are references meaningful?)
2. Increase training epochs
3. Adjust learning rate
4. Use beam search instead of greedy decoding

### High Fairness Disparity
**Problem**: Disparity > 0.10
**Solutions**:
1. Balance training data across groups
2. Apply fairness regularization (see bias_mitigation.py)
3. Use adversarial debiasing
4. Evaluate if disparity comes from legitimate differences

### Poor Latent Space
**Problem**: latent_diversity < 0.001
**Solutions**:
1. Increase β in β-VAE
2. Add more diverse training examples
3. Reduce KL weight slightly (avoid posterior collapse)

## Best Practices

1. **Evaluate on multiple metrics**: Don't rely on a single metric
2. **Use consistent sample sizes**: Compare apples to apples
3. **Track over time**: Monitor trends, not just absolute values
4. **Consider human evaluation**: Metrics don't capture everything
5. **Test edge cases**: Evaluate on specific difficult scenarios
6. **Document baselines**: Know what "good" looks like for your use case

## References

- BLEU: Papineni et al. (2002)
- ROUGE: Lin (2004)
- Perplexity: Standard language modeling metric
- Fairness: Demographic parity and equalized odds
- XAI: Integrated from explainability.py module

## Quick Reference

| Metric | Good Range | Warning Range | Concerning Range |
|--------|-----------|---------------|------------------|
| BLEU-4 | > 0.15 | 0.05 - 0.15 | < 0.05 |
| ROUGE-L | > 0.25 | 0.10 - 0.25 | < 0.10 |
| Quality | > 0.65 | 0.55 - 0.65 | < 0.55 |
| Distinct-1 | > 0.30 | 0.15 - 0.30 | < 0.15 |
| Disparity | < 0.05 | 0.05 - 0.10 | > 0.10 |
| Latent Div | > 0.002 | 0.001 - 0.002 | < 0.001 |

---

**Next Steps**: See `inference.py` for deployment-ready generation pipeline!
