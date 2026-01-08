"""
Quick demo of XAI/Explainability features
Shows how to interpret model decisions
"""

import numpy as np
import pandas as pd
import pickle
import config
from explainability import ModelExplainer

print("=" * 70)
print("XAI (EXPLAINABILITY) DEMO")
print("=" * 70)

# Load preprocessing config
with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
    preprocess_config = pickle.load(f)

vocab_size = preprocess_config['vocab_size']
max_length = preprocess_config['max_length']

# Load tokenizer
with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print(f"\nVocabulary size: {vocab_size}")
print(f"Max sequence length: {max_length}")

# Load validation data
val_customer = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')[:100]

print(f"Validation samples loaded: {len(val_customer)}")

# Build hybrid model
print("\nBuilding hybrid model...")
from hybrid_model import HybridGANVAE
hybrid = HybridGANVAE(vocab_size, max_length)

# Initialize explainer
print("\n" + "=" * 70)
print("INITIALIZING EXPLAINER")
print("=" * 70)

explainer = ModelExplainer(hybrid, tokenizer)

# Select interesting examples
test_message = val_customer[0:1]

print("\n" + "=" * 70)
print("EXAMPLE 1: TOKEN IMPORTANCE ANALYSIS")
print("=" * 70)

token_importance = explainer.analyze_token_importance(test_message)

print(f"\nAnalyzed {len(token_importance)} tokens in customer message")
print("\nTop 10 Most Important Tokens (Combined Attention + Saliency):")
for i, (token, score) in enumerate(list(token_importance.items())[:10]):
    bar = "█" * int(score * 30)
    print(f"  {i+1:2d}. {token:15s} {score:.4f} {bar}")

print("\n" + "=" * 70)
print("EXAMPLE 2: COMPREHENSIVE EXPLANATION")
print("=" * 70)

explanation = explainer.explain_generation(test_message, top_k=8)

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

print("\n1. ATTENTION MECHANISM:")
print("   - Shows which tokens the encoder focuses on")
print("   - Higher attention = more important for encoding")
print("   - Bidirectional LSTM uses context from both directions")

print("\n2. SALIENCY MAP:")
print("   - Measures gradient of output w.r.t. input")
print("   - Shows which tokens most influence the latent space")
print("   - High saliency = changing this token changes output significantly")

print("\n3. TOKEN IMPORTANCE:")
print("   - Combines attention and saliency")
print("   - Gives holistic view of token contributions")
print("   - Top tokens are most influential for generation")

print("\n4. LATENT SPACE:")
print("   - 256-dimensional compressed representation")
print(f"   - Mean magnitude: {explanation['latent_representation']['mean_magnitude']:.4f}")
print(f"   - Variance: {explanation['latent_representation']['variance']:.4f}")
print("   - Captures semantic meaning of customer message")

print("\n5. RESPONSE GENERATION:")
print("   - Latent space → GAN Generator → Agent Response")
print(f"   - Quality Score: {explanation['quality_score']:.4f}")
print("   - Discriminator evaluates realism")

print("\n" + "=" * 70)
print("EXAMPLE 3: VISUALIZATIONS")
print("=" * 70)

print("\nGenerating attention visualization...")
try:
    explainer.visualize_attention(
        test_message,
        save_path=f'{config.RESULTS_DIR}/demo_attention.png'
    )
    print("✓ Saved: results/demo_attention.png")
except Exception as e:
    print(f"  Note: Visualization skipped ({str(e)[:50]}...)")

print("\nGenerating saliency map...")
try:
    explainer.visualize_saliency_map(
        test_message,
        save_path=f'{config.RESULTS_DIR}/demo_saliency.png'
    )
    print("✓ Saved: results/demo_saliency.png")
except Exception as e:
    print(f"  Note: Visualization skipped ({str(e)[:50]}...)")

print("\n" + "=" * 70)
print("EXAMPLE 4: COMPARING MULTIPLE INPUTS")
print("=" * 70)

print("\nAnalyzing 3 different customer messages...\n")

for idx in [0, 5, 10]:
    msg = val_customer[idx:idx+1]
    importance = explainer.analyze_token_importance(msg)
    
    input_text = tokenizer.sequences_to_texts([msg[0]])[0]
    print(f"Message {idx+1}: \"{input_text[:60]}...\"")
    
    top_3 = list(importance.items())[:3]
    print(f"  Top 3 tokens: {', '.join([f'{t}({s:.3f})' for t, s in top_3])}")
    print()

print("=" * 70)
print("✅ EXPLAINABILITY DEMO COMPLETE!")
print("=" * 70)

print("\nWhat You Learned:")
print("  ✓ How to analyze token importance")
print("  ✓ How to visualize attention weights")
print("  ✓ How to compute saliency maps")
print("  ✓ How to explain model generations")

print("\nNext Steps:")
print("  1. Train model for better explanations")
print("  2. Use create_comprehensive_report() for full analysis")
print("  3. Compare explanations before/after training")
print("  4. Analyze fairness through explanations")

print("\nKey Methods:")
print("  • explainer.analyze_token_importance(msg)")
print("  • explainer.explain_generation(msg, top_k=10)")
print("  • explainer.visualize_attention(msg, save_path='...')")
print("  • explainer.visualize_saliency_map(msg, save_path='...')")
print("  • explainer.visualize_latent_space(data, labels, save_path='...')")
print("  • explainer.create_comprehensive_report(msg, output_dir='...')")
