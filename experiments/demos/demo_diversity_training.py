"""
Quick Demo: Fine-tuning with Diversity Optimization (5 epochs)
"""

# Use the main fine-tuning script but with reduced epochs for demonstration
import sys
sys.path.insert(0, '.')

from fine_tune_with_diversity import *

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("QUICK DEMO: DIVERSITY-OPTIMIZED FINE-TUNING (5 EPOCHS)")
    print(f"{'='*80}\n")
    
    # Load data and tokenizer
    print("📦 Loading data and tokenizer...")
    
    with open('processed_data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    customer_train = np.load('processed_data/train_customer.npy')
    agent_train = np.load('processed_data/train_agent.npy')
    
    customer_val = np.load('processed_data/val_customer.npy')
    agent_val = np.load('processed_data/val_agent.npy')
    
    # Targets
    targets_train = agent_train[:, 1:]
    targets_val = agent_val[:, 1:]
    
    # Adjust inputs
    customer_train = customer_train[:, :-1]
    agent_train = agent_train[:, :-1]
    customer_val = customer_val[:, :-1]
    agent_val = agent_val[:, :-1]
    
    print(f"✓ Loaded data:")
    print(f"  Training: {len(customer_train):,} samples")
    print(f"  Validation: {len(customer_val):,} samples")
    print(f"  Vocabulary: {len(tokenizer.word_index)} tokens")
    
    # Build model
    print(f"\n{'='*80}")
    print("BUILDING MODEL")
    print(f"{'='*80}\n")
    
    model = SeparateInputHybridGANVAE(
        vocab_size=len(tokenizer.word_index) + 1,
        max_length=customer_train.shape[1],
        latent_dim=VAE_LATENT_DIM
    )
    
    print("✓ Model built")
    
    # Training configuration (5 epochs for demo)
    config = {
        'epochs': 5,  # Short demo
        'batch_size': 64,
        'learning_rate': 1e-4,
        'diversity_weight': 0.15,
        'temperature': 1.5,
        'top_p': 0.9,
        'repetition_penalty': 1.2,
        'checkpoint_dir': 'models/demo_diversity',
        'log_dir': 'logs/demo'
    }
    
    print(f"\n📋 Demo Configuration (5 epochs):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train
    print(f"\n{'='*80}")
    print("STARTING 5-EPOCH DEMO TRAINING")
    print(f"{'='*80}\n")
    
    history = train_with_diversity_optimization(
        model=model,
        train_data=(customer_train, agent_train, targets_train),
        val_data=(customer_val, agent_val, targets_val),
        tokenizer=tokenizer,
        **config
    )
    
    print(f"\n{'='*80}")
    print("✅ DEMO COMPLETED!")
    print(f"{'='*80}\n")
    print("Results:")
    print(f"  Final train loss: {history['loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Training curves: results/diversity_training/training_curves.png")
    print(f"  Quality log: logs/demo/diversity_quality_log.csv")
    print("\nNote: This was a 5-epoch demo. For production, run full 100-epoch training:")
    print("  python fine_tune_with_diversity.py")
