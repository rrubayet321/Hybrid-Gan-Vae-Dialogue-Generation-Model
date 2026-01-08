"""
Quick test script for VAE architecture without full TensorFlow initialization
"""

import pickle
import config

# Test that we can load the preprocessing config
print("Testing VAE setup...")

# Load preprocessing config
with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
    preprocess_config = pickle.load(f)

vocab_size = preprocess_config['vocab_size']
max_length = preprocess_config['max_length']

print(f"✅ Preprocessing config loaded")
print(f"   Vocabulary size: {vocab_size}")
print(f"   Max sequence length: {max_length}")

print(f"\n✅ VAE configuration:")
print(f"   Embedding dimension: {config.VAE_EMBEDDING_DIM}")
print(f"   Latent dimension: {config.VAE_LATENT_DIM}")
print(f"   Hidden dimension: {config.VAE_HIDDEN_DIM}")
print(f"   Dropout: {config.VAE_DROPOUT}")
print(f"   Learning rate: {config.VAE_LEARNING_RATE}")
print(f"   Batch size: {config.VAE_BATCH_SIZE}")
print(f"   Epochs: {config.VAE_EPOCHS}")

print("\n✅ Ready to build VAE model")
print("   Run 'python3 train_vae.py' to start training")
