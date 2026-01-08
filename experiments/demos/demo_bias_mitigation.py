"""
Quick demo of bias mitigation
Shows fairness evaluation before and after training
"""

import numpy as np
import pandas as pd
import pickle
import config
from hybrid_model import HybridGANVAE
from bias_mitigation import evaluate_fairness

print("=" * 70)
print("BIAS MITIGATION DEMO")
print("=" * 70)

# Load preprocessing config
with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
    preprocess_config = pickle.load(f)

vocab_size = preprocess_config['vocab_size']
max_length = preprocess_config['max_length']

print(f"\nVocabulary size: {vocab_size}")
print(f"Max sequence length: {max_length}")

# Load validation data (subset for quick demo)
print("\nLoading validation data...")
val_customer = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')[:2000]
val_metadata = pd.read_csv(f'{config.PROCESSED_DATA_DIR}/val_metadata.csv').iloc[:2000]

print(f"Validation samples: {len(val_customer):,}")

# Build hybrid model
print("\nBuilding hybrid model...")
hybrid = HybridGANVAE(vocab_size, max_length)

print("\n" + "=" * 70)
print("EVALUATING FAIRNESS (UNTRAINED MODEL)")
print("=" * 70)
print("\nNote: Untrained models typically show random predictions")
print("      Detector accuracy ≈ 1/num_classes indicates no bias")

# Evaluate fairness
metrics, bias_aware = evaluate_fairness(
    hybrid,
    val_customer,
    val_metadata,
    sensitive_attributes=['customer_segment', 'region']
)

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

print("\n1. BIAS DETECTOR PERFORMANCE:")
print("   - Measures how well demographics can be predicted from latent space")
print("   - Random guessing = no bias (good!)")
print("   - High accuracy = bias detected (bad!)")

print("\n2. DEMOGRAPHIC PARITY:")
print("   - Measures similarity of representations across groups")
print("   - Low values = groups treated equally (good!)")
print("   - High values = systematic differences (bad!)")

print("\n3. WHAT TO EXPECT AFTER TRAINING:")
print("   - Without fairness: Detector accuracy may increase (bias learned)")
print("   - With fairness: Detector accuracy stays low (bias prevented)")

print("\n" + "=" * 70)
print("✅ DEMO COMPLETE!")
print("=" * 70)

print("\nNext steps:")
print("  1. Run: python train_with_fairness.py")
print("     - Trains model with fairness constraints")
print("     - Monitors bias metrics throughout training")
print("  2. Compare fairness before vs. after training")
print("  3. Verify bias detector accuracy remains ≈ random")
print("  4. Check demographic parity stays low")

print("\nFor full guide, see: BIAS_MITIGATION_GUIDE.md")
