"""
Quick Demo of Hybrid GAN + VAE Training
Tests the training pipeline with a small subset and few epochs
"""

import numpy as np
import pickle
import config
from train_hybrid import HybridTrainer

def quick_demo():
    """Quick training demonstration with subset"""
    print("\n" + "=" * 70)
    print("HYBRID GAN + VAE TRAINING DEMO")
    print("=" * 70)
    print("Training with small subset for demonstration...")
    
    # Load preprocessing config
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    # Load tokenizer
    with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Max length: {max_length}")
    
    # Create trainer
    trainer = HybridTrainer(vocab_size, max_length)
    
    # Load data
    full_data = trainer.load_data()
    
    # Use subset for demo (5000 training, 1000 validation)
    print("\nUsing subset for demo:")
    data = {
        'train_customer': full_data['train_customer'][:5000],
        'train_agent': full_data['train_agent'][:5000],
        'val_customer': full_data['val_customer'][:1000],
        'val_agent': full_data['val_agent'][:1000],
        'train_metadata': full_data['train_metadata'][:5000],
        'val_metadata': full_data['val_metadata'][:1000]
    }
    print(f"  Training: {len(data['train_customer']):,} samples")
    print(f"  Validation: {len(data['val_customer']):,} samples")
    
    # Quick training (few epochs)
    print("\nTraining with reduced epochs for demo...")
    trainer.train_complete(
        data,
        vae_pretrain_epochs=3,  # Just 3 epochs
        gan_epochs=5,           # Just 5 epochs
        hybrid_epochs=3,        # Just 3 epochs
        batch_size=64
    )
    
    # Plot results
    trainer.plot_training_history(f'{config.RESULTS_DIR}/hybrid_demo_history.png')
    
    # Evaluate generation
    print("\nGenerating sample responses...")
    trainer.evaluate_generation(data, tokenizer, num_samples=3)
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETED!")
    print("=" * 70)
    print("\nThe hybrid model successfully trained on subset!")
    print("For full training, run: python train_hybrid.py")

if __name__ == "__main__":
    quick_demo()
