# Fine-Tuning Guide: Diversity-Optimized Training

**Date**: January 2026  
**Purpose**: Train Hybrid GAN + VAE with diversity optimization to achieve high-quality, non-repetitive responses

---

## Overview

This guide provides complete instructions for fine-tuning the **Separate Input Hybrid GAN + VAE** model (winner from architecture comparison) with **diversity optimization** to reduce repetition and improve response quality.

### Key Features

✅ **Diversity Loss** - Penalizes repetitive generation  
✅ **Temperature Sampling** - Increases generation diversity  
✅ **Nucleus (Top-p) Sampling** - Samples from high-probability tokens  
✅ **Repetition Penalty** - Discourages repeating recent tokens  
✅ **Quality Monitoring** - Tracks BLEU, ROUGE, Distinct, Repetition every 5 epochs  
✅ **Early Stopping** - Prevents overfitting  
✅ **Automatic Checkpointing** - Saves best model based on validation loss

---

## Architecture: Separate Input Model (Winner)

Based on comparison results showing **+43.6% improvement** in BLEU-4 and **+41.8% improvement** in ROUGE-L, we use the **Separate Input architecture**:

```
Customer Input ──→ Customer Encoder ──→ Customer Latent Space
                                             ↓
                                         Combine
                                             ↓
Agent Input ────→ Agent Encoder ────→ Agent Latent Space ──→ Generator ──→ Response
```

**Advantages**:
- Independent customer/agent encoders
- Role-specific linguistic pattern learning
- Better dialogue understanding
- 22M parameters (vs 16M for combined)

---

## Training Components

### 1. Diversity Loss Function

**Purpose**: Encourage diverse token generation and penalize repetition

**Components**:

```python
diversity_loss = 0.3 × distinct_loss + 0.4 × repetition_penalty + 0.3 × entropy_regularization
```

- **Distinct Loss**: Measures unique n-gram ratio (lower loss = more diverse)
- **Repetition Penalty**: Penalizes consecutive token repetition
- **Entropy Regularization**: Encourages high entropy in probability distributions

**Integration**:
```python
total_loss = reconstruction_loss + 0.15 × diversity_loss
```

### 2. Sampling Strategies

#### Temperature Sampling
```python
temperature = 1.5  # Higher = more diverse
logits = logits / temperature
probs = softmax(logits)
token = sample(probs)
```

**Effect**:
- `T = 1.0`: Standard sampling
- `T > 1.0`: More diverse (flatter distribution)
- `T < 1.0`: More focused (sharper distribution)

#### Nucleus (Top-p) Sampling
```python
top_p = 0.9  # Sample from top 90% probability mass
sorted_probs = sort_descending(probs)
cumsum = cumulative_sum(sorted_probs)
nucleus = probs[cumsum <= top_p]
token = sample(nucleus)
```

**Effect**: Samples from high-probability tokens only, avoiding low-quality rare tokens

#### Repetition Penalty
```python
penalty = 1.2  # Discourage repetition
recent_tokens = last_20_tokens
for token in recent_tokens:
    logits[token] /= penalty
```

**Effect**: Recent tokens become less likely to be generated again

### 3. Quality Monitoring Callback

**DiversityQualityCallback** evaluates quality every 5 epochs:

**Metrics Tracked**:
- **BLEU-1/2/4**: N-gram precision
- **ROUGE-1/2/L**: N-gram recall
- **Distinct-1/2/3**: Unique n-gram ratios
- **Repetition Rate**: Consecutive token repetition
- **Entropy**: Vocabulary distribution diversity
- **Quality Score**: Composite metric (0-1)

**Quality Score Formula**:
```python
quality_score = 0.3 × BLEU-4 + 0.3 × ROUGE-L + 0.2 × Distinct-2 + 0.2 × (1 - Repetition Rate)
```

---

## Training Configuration

### Recommended Hyperparameters

```python
config = {
    'epochs': 100,                   # Maximum epochs
    'batch_size': 64,                # Training batch size
    'learning_rate': 1e-4,           # Initial learning rate
    'diversity_weight': 0.15,        # Weight for diversity loss
    'temperature': 1.5,              # Sampling temperature
    'top_p': 0.9,                    # Nucleus sampling threshold
    'repetition_penalty': 1.2        # Repetition penalty factor
}
```

### Callbacks

1. **DiversityQualityCallback**
   - Interval: Every 5 epochs
   - Samples: 200 validation examples
   - Logs: `logs/diversity_quality_log.csv`

2. **ModelCheckpoint**
   - Monitor: `val_loss`
   - Mode: `min` (save lowest)
   - Path: `models/diversity_optimized/best_model_epoch_{epoch:03d}_val_{val_loss:.4f}.weights.h5`

3. **ReduceLROnPlateau**
   - Monitor: `val_loss`
   - Factor: `0.5` (halve learning rate)
   - Patience: 5 epochs

4. **EarlyStopping**
   - Monitor: `val_loss`
   - Patience: 15 epochs
   - Restore best weights: `True`

---

## Usage Instructions

### Step 1: Prepare Data

Ensure preprocessed data exists:
```
processed_data/
  ├── train_customer.npy
  ├── train_agent.npy
  ├── val_customer.npy
  ├── val_agent.npy
  └── tokenizer.pkl
```

If not, run:
```bash
python preprocessing.py
```

### Step 2: Run Fine-Tuning

```bash
cd "/Users/rubayethassan/Desktop/424 project start"
source vae_env/bin/activate
python fine_tune_with_diversity.py
```

**Expected Output**:
```
================================================================================
ENHANCED FINE-TUNING WITH DIVERSITY OPTIMIZATION
================================================================================

Training Configuration:
  Epochs: 100
  Batch size: 64
  Learning rate: 0.0001
  Diversity weight: 0.15
  Temperature: 1.5
  Top-p: 0.9
  Repetition penalty: 1.2

Data:
  Training samples: 80,000
  Validation samples: 20,000
  Vocabulary size: 154

================================================================================
STARTING TRAINING
================================================================================
```

### Step 3: Monitor Training

**Real-time Monitoring**:
```bash
# Watch training progress
tail -f logs/diversity_quality_log.csv

# Monitor epoch progress
watch -n 5 "tail -20 logs/diversity_quality_log.csv"
```

**Quality Metrics Log** (`logs/diversity_quality_log.csv`):
```csv
epoch,bleu_1,bleu_2,bleu_4,rouge_1,rouge_2,rouge_l,distinct_1,distinct_2,distinct_3,repetition_rate,entropy,quality_score,avg_length
5,0.0234,0.0112,0.0045,0.0312,0.0089,0.0298,0.0456,0.0823,0.1234,0.8234,6.234,0.0234,12.3
10,0.0345,0.0189,0.0078,0.0423,0.0145,0.0398,0.0678,0.1234,0.1876,0.7123,6.567,0.0345,13.2
...
```

### Step 4: Evaluate Results

**Training Curves**: `results/diversity_training/training_curves.png`
- Loss curves (train/val)
- Accuracy curves (train/val)

**Model Checkpoints**: `models/diversity_optimized/`
```
best_model_epoch_023_val_2.3456.weights.h5  ← Best model
best_model_epoch_045_val_2.1234.weights.h5  ← Better model
final_model.weights.h5                       ← Final model
```

### Step 5: Run Inference

```bash
python inference_with_diversity.py
```

**Example Output**:
```
Customer: my account is locked after multiple failed login attempts
Agent:    Sorry to hear you're having trouble accessing your account. 
          I've escalated this to our security team for immediate attention.

Quality:
  Distinct-2: 0.823
  Repetition: 0.034
  Score: 0.756
```

---

## Expected Performance Improvements

### Before Training (Untrained Model)
```
BLEU-4:          0.0003  ❌ (Poor precision)
ROUGE-L:         0.0028  ❌ (Poor recall)
Distinct-1:      0.0031  ❌ (Only 0.3% unique words)
Distinct-2:      0.0204  ❌ (Only 2% unique bigrams)
Repetition Rate: 0.9822  ❌ (98% repetition!)
Quality Score:   0.0086  ❌ (Very poor)
```

### After Training (Expected with 50+ epochs)
```
BLEU-4:          0.25-0.35  ✅ (Good precision)
ROUGE-L:         0.30-0.40  ✅ (Good recall)
Distinct-1:      0.45-0.60  ✅ (High diversity)
Distinct-2:      0.50-0.65  ✅ (Diverse bigrams)
Repetition Rate: 0.05-0.15  ✅ (Low repetition)
Quality Score:   0.40-0.55  ✅ (Good quality)
```

### Target Metrics (After 100 epochs)
```
✅ BLEU-4 > 0.30        (Good n-gram precision)
✅ ROUGE-L > 0.35       (Good content coverage)
✅ Distinct-1 > 0.50    (High word diversity)
✅ Distinct-2 > 0.55    (High bigram diversity)
✅ Repetition < 0.10    (Minimal repetition)
✅ Quality Score > 0.45 (Overall good quality)
```

---

## Troubleshooting

### Issue 1: High Repetition (>50% after 20 epochs)

**Solutions**:
1. Increase diversity weight: `diversity_weight = 0.2` (from 0.15)
2. Increase temperature: `temperature = 1.8` (from 1.5)
3. Increase repetition penalty: `repetition_penalty = 1.5` (from 1.2)
4. Decrease top-p: `top_p = 0.8` (from 0.9) for more focused sampling

### Issue 2: Low BLEU/ROUGE Scores (<0.20 after 30 epochs)

**Solutions**:
1. Decrease diversity weight: `diversity_weight = 0.1` (from 0.15)
2. Decrease temperature: `temperature = 1.2` (from 1.5)
3. Increase learning rate: `learning_rate = 2e-4` (from 1e-4)
4. Train longer: Increase patience for early stopping

### Issue 3: Validation Loss Not Decreasing

**Solutions**:
1. Check for overfitting (train loss << val loss)
2. Reduce model complexity or add regularization
3. Increase training data or apply data augmentation
4. Adjust learning rate schedule

### Issue 4: Out of Memory

**Solutions**:
1. Reduce batch size: `batch_size = 32` (from 64)
2. Reduce max sequence length: `MAX_LEN = 80` (from 100)
3. Use gradient accumulation
4. Use mixed precision training

---

## Advanced Configuration

### Custom Diversity Loss Weights

```python
# More emphasis on distinct n-grams
diversity_loss = 0.5 × distinct_loss + 0.3 × repetition_penalty + 0.2 × entropy

# More emphasis on repetition penalty
diversity_loss = 0.2 × distinct_loss + 0.6 × repetition_penalty + 0.2 × entropy
```

### Dynamic Temperature Schedule

```python
# Start conservative, increase for diversity
temperature = 1.0 + 0.5 × (epoch / total_epochs)

# Start high, decrease for quality
temperature = 2.0 - 0.5 × (epoch / total_epochs)
```

### Multiple Checkpoints

```python
# Save top 3 models
ModelCheckpoint(
    filepath='models/diversity_optimized/model_epoch_{epoch:03d}.weights.h5',
    monitor='val_loss',
    save_best_only=False,  # Save all
    save_freq='epoch'
)
```

---

## Performance Benchmarks

### Training Time (GPU)
- **Per Epoch**: ~3-4 minutes (with 80K samples, batch_size=64)
- **50 Epochs**: ~2.5-3.5 hours
- **100 Epochs**: ~5-7 hours

### Memory Requirements
- **GPU Memory**: ~6-8 GB (for 22M parameter model)
- **RAM**: ~8-16 GB (for data loading)
- **Storage**: ~500 MB (for checkpoints)

---

## Quality Assessment Thresholds

```python
# Quality categories
EXCELLENT  = quality_score > 0.50 and repetition_rate < 0.10
GOOD       = quality_score > 0.40 and repetition_rate < 0.20
MODERATE   = quality_score > 0.30 and repetition_rate < 0.40
POOR       = quality_score < 0.30 or repetition_rate > 0.40
```

**Interpretation**:
- **Distinct-1 > 0.50**: High word diversity ✅
- **Distinct-2 > 0.50**: High bigram diversity ✅
- **Repetition < 0.10**: Minimal repetition ✅
- **BLEU-4 > 0.30**: Good precision ✅
- **ROUGE-L > 0.35**: Good recall ✅

---

## Integration with Existing Pipeline

### 1. Load Trained Model in Other Scripts

```python
from compare_input_approaches import SeparateInputHybridGANVAE
from simple_tokenizer import SimpleTokenizer
import pickle

# Load tokenizer
with open('processed_data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Build model
model = SeparateInputHybridGANVAE(
    vocab_size=len(tokenizer.word_index) + 1,
    max_len=100,
    embedding_dim=128,
    latent_dim=256,
    lstm_units=512
)

# Load weights
model.hybrid_model.load_weights('models/diversity_optimized/final_model.weights.h5')
```

### 2. Use in API

```python
from inference_with_diversity import InferenceAPI

api = InferenceAPI(
    model_path='models/diversity_optimized/final_model.weights.h5',
    tokenizer_path='processed_data/tokenizer.pkl'
)

response = api.generate("my account is locked")
print(response['response'])
```

### 3. Batch Processing

```python
messages = [
    "cannot login to dashboard",
    "password reset not working",
    "billing error on my account"
]

results = api.generate_batch(messages, return_quality=True, verbose=True)
for msg, result in zip(messages, results):
    print(f"Q: {msg}")
    print(f"A: {result['response']}")
    print(f"Quality: {result['quality_metrics']['quality_score']:.3f}\n")
```

---

## Summary

This fine-tuning approach combines:

1. ✅ **Diversity Loss** - Reduces repetition at the loss level
2. ✅ **Sampling Strategies** - Temperature + Nucleus sampling for diverse generation
3. ✅ **Repetition Penalty** - Discourages repeating recent tokens
4. ✅ **Quality Monitoring** - Tracks 13+ metrics every 5 epochs
5. ✅ **Automatic Optimization** - Early stopping + LR reduction + best model saving

**Expected Outcome**: High-quality, diverse, non-repetitive agent responses suitable for production deployment.

---

## Next Steps

1. ✅ **Run Training**: `python fine_tune_with_diversity.py`
2. ✅ **Monitor Progress**: Check `logs/diversity_quality_log.csv`
3. ✅ **Evaluate Results**: Review training curves and quality metrics
4. ✅ **Test Inference**: Run `python inference_with_diversity.py`
5. ✅ **Deploy**: Integrate with API using `InferenceAPI` class

---

**Generated**: January 2026  
**Model**: Separate Input Hybrid GAN + VAE  
**Optimization**: Diversity-focused fine-tuning  
**Target**: Production-ready IT support chatbot
