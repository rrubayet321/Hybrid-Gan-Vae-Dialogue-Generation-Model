"""
Quick test of VAE model architecture
Verifies the model builds correctly without full training
"""

import numpy as np
import pickle
import config
from vae_model import VAEModel, build_vae

def test_vae_architecture():
    """Test VAE model can be built"""
    print("\n" + "=" * 60)
    print("TESTING VAE ARCHITECTURE")
    print("=" * 60)
    
    # Load preprocessing config
    print("\nLoading preprocessing config...")
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    print(f"✓ Vocab size: {vocab_size}")
    print(f"✓ Max length: {max_length}")
    
    # Build VAE
    print("\nBuilding VAE model...")
    vae_model = build_vae(vocab_size, max_length)
    
    print(f"\n✓ VAE model built successfully!")
    print(f"  Embedding dim: {vae_model.embedding_dim}")
    print(f"  Latent dim: {vae_model.latent_dim}")
    print(f"  Hidden dim: {vae_model.hidden_dim}")
    print(f"  Dropout: {vae_model.dropout}")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    dummy_input = np.random.randint(0, vocab_size, size=(10, max_length))
    
    # Test encoder
    print("  Testing encoder...")
    z_mean, z_log_var, z = vae_model.encode(dummy_input)
    print(f"    ✓ Latent vectors shape: {z.shape}")
    
    # Test decoder
    print("  Testing decoder...")
    decoded = vae_model.decode(z)
    print(f"    ✓ Decoded shape: {decoded.shape}")
    
    # Test full VAE
    print("  Testing full VAE...")
    reconstructed = vae_model.reconstruct(dummy_input)
    print(f"    ✓ Reconstructed shape: {reconstructed.shape}")
    
    # Verify shapes
    assert z.shape == (10, vae_model.latent_dim), "Latent space shape mismatch"
    assert decoded.shape == (10, max_length, vocab_size), "Decoder output shape mismatch"
    assert reconstructed.shape == (10, max_length, vocab_size), "VAE output shape mismatch"
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nVAE architecture is ready for training.")
    print(f"Run: python train_vae_complete.py")
    
    return vae_model

if __name__ == "__main__":
    test_vae_architecture()
