"""
Quick VAE Training Demo
Trains for just a few epochs to demonstrate the training process
"""

import numpy as np
import pickle
import tensorflow as tf

import config
from train_vae_complete import VAETrainer

def quick_train_demo():
    """Quick training demonstration"""
    print("\n" + "=" * 70)
    print("VAE QUICK TRAINING DEMO")
    print("=" * 70)
    print("Training for 5 epochs to demonstrate the process...")
    
    # Load preprocessing config
    print("\nLoading preprocessing config...")
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    print(f"✓ Vocab size: {vocab_size}")
    print(f"✓ Max length: {max_length}")
    
    # Load tokenizer
    with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"✓ Tokenizer loaded")
    
    # Create trainer
    print("\nCreating VAE trainer...")
    trainer = VAETrainer(vocab_size, max_length)
    
    # Load data (use subset for quick demo)
    print("\nLoading data subset...")
    train_data = np.load(f'{config.PROCESSED_DATA_DIR}/train_customer.npy')
    val_data = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')
    
    # Use subset for quick demo
    train_subset = train_data[:10000]  # 10K samples
    val_subset = val_data[:2000]  # 2K samples
    
    print(f"✓ Training subset: {len(train_subset):,} samples")
    print(f"✓ Validation subset: {len(val_subset):,} samples")
    
    # Train for 5 epochs
    print("\n" + "=" * 70)
    print("STARTING TRAINING (5 epochs on subset)")
    print("=" * 70)
    
    history = trainer.train(
        train_subset, 
        val_subset,
        epochs=5,
        batch_size=64,
        run_name='quick_demo'
    )
    
    # Show reconstruction examples
    print("\n" + "=" * 70)
    print("RECONSTRUCTION EXAMPLES")
    print("=" * 70)
    
    trainer.evaluate_reconstruction(val_subset, tokenizer, num_samples=3)
    
    # Plot training history
    trainer.plot_training_history(f'{config.RESULTS_DIR}/vae_demo_history.png')
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETED!")
    print("=" * 70)
    print("\nThe VAE learned to encode and reconstruct customer messages!")
    print(f"Training plot saved to: {config.RESULTS_DIR}/vae_demo_history.png")
    print("\nFor full training, run: python train_vae_complete.py")

if __name__ == "__main__":
    quick_train_demo()
