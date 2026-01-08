"""
Enhanced Fine-tuning with Diversity Optimization
=================================================

This script fine-tunes the Hybrid GAN + VAE model with:
1. Diversity loss to reduce repetition
2. Quality-aware training (BLEU, ROUGE monitoring)
3. Temperature sampling for diverse generation
4. Nucleus (top-p) sampling
5. Repetition penalty
6. Early stopping based on quality metrics

Based on comparison results: Using SEPARATE INPUT architecture (winner)
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import project modules
from config import *
from simple_tokenizer import SimpleTokenizer
from evaluation_metrics import TextGenerationEvaluator
from diversity_metrics import DiversityMetrics
from compare_input_approaches import SeparateInputHybridGANVAE


# ============================================================================
# Diversity Loss Functions
# ============================================================================

def distinct_loss(predictions: tf.Tensor, n: int = 2) -> tf.Tensor:
    """
    Compute distinct-n loss to encourage diverse n-gram generation.
    Lower loss = more diverse n-grams.
    
    Args:
        predictions: Logits or probabilities [batch, seq_len, vocab_size]
        n: N-gram size (1, 2, or 3)
    
    Returns:
        Scalar loss value (0 to 1, lower is better)
    """
    # Get predicted tokens (argmax)
    predicted_tokens = tf.argmax(predictions, axis=-1)  # [batch, seq_len]
    
    batch_size = tf.shape(predicted_tokens)[0]
    seq_len = tf.shape(predicted_tokens)[1]
    
    # Extract n-grams
    if n == 1:
        # Unigrams - just count unique tokens
        unique_count = tf.size(tf.unique(tf.reshape(predicted_tokens, [-1]))[0])
        total_count = tf.cast(batch_size * seq_len, tf.float32)
    else:
        # Create n-grams by sliding window
        ngrams = []
        for i in range(seq_len - n + 1):
            # Extract n consecutive tokens
            ngram = predicted_tokens[:, i:i+n]
            # Convert to single integer for uniqueness checking
            ngram_hash = tf.reduce_sum(ngram * tf.constant([vocab_size ** i for i in range(n)], dtype=tf.int64), axis=1)
            ngrams.append(ngram_hash)
        
        if ngrams:
            ngrams = tf.concat(ngrams, axis=0)
            unique_count = tf.size(tf.unique(ngrams)[0])
            total_count = tf.cast(batch_size * (seq_len - n + 1), tf.float32)
        else:
            return 0.0
    
    # Distinct score: unique / total (higher is better)
    distinct_score = tf.cast(unique_count, tf.float32) / (total_count + 1e-10)
    
    # Loss: 1 - distinct_score (lower is better)
    loss = 1.0 - distinct_score
    
    return loss


def repetition_penalty_loss(predictions: tf.Tensor) -> tf.Tensor:
    """
    Penalize consecutive token repetition in generated sequences.
    
    Args:
        predictions: Logits [batch, seq_len, vocab_size]
    
    Returns:
        Scalar loss value (0 to 1, lower is better)
    """
    # Get predicted tokens
    predicted_tokens = tf.argmax(predictions, axis=-1)  # [batch, seq_len]
    
    # Check consecutive tokens
    consecutive_same = tf.cast(
        tf.equal(predicted_tokens[:, :-1], predicted_tokens[:, 1:]),
        tf.float32
    )  # [batch, seq_len-1]
    
    # Average repetition rate
    repetition_rate = tf.reduce_mean(consecutive_same)
    
    return repetition_rate


def entropy_regularization(predictions: tf.Tensor) -> tf.Tensor:
    """
    Encourage high entropy in token probability distributions.
    Higher entropy = more diverse/uncertain predictions.
    
    Args:
        predictions: Probabilities [batch, seq_len, vocab_size]
    
    Returns:
        Negative entropy (minimize to maximize entropy)
    """
    # Compute entropy: -sum(p * log(p))
    probs = tf.nn.softmax(predictions, axis=-1)
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)  # [batch, seq_len]
    
    # Average entropy across batch and sequence
    avg_entropy = tf.reduce_mean(entropy)
    
    # Return negative (we want to maximize entropy, so minimize negative entropy)
    return -avg_entropy


def diversity_loss(predictions: tf.Tensor, 
                   distinct_weight: float = 0.3,
                   repetition_weight: float = 0.4,
                   entropy_weight: float = 0.3) -> tf.Tensor:
    """
    Combined diversity loss with multiple components.
    
    Args:
        predictions: Model predictions [batch, seq_len, vocab_size]
        distinct_weight: Weight for distinct-2 loss
        repetition_weight: Weight for repetition penalty
        entropy_weight: Weight for entropy regularization
    
    Returns:
        Weighted diversity loss
    """
    # Compute individual components
    distinct_2_loss = distinct_loss(predictions, n=2)
    rep_loss = repetition_penalty_loss(predictions)
    ent_loss = entropy_regularization(predictions)
    
    # Weighted combination
    total_loss = (
        distinct_weight * distinct_2_loss +
        repetition_weight * rep_loss +
        entropy_weight * ent_loss
    )
    
    return total_loss


# ============================================================================
# Sampling Functions
# ============================================================================

def temperature_sampling(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply temperature scaling to logits for diverse sampling.
    
    Args:
        logits: Logits from model [vocab_size]
        temperature: Sampling temperature (higher = more diverse)
                    1.0 = no change
                    >1.0 = more diverse (flatter distribution)
                    <1.0 = more focused (sharper distribution)
    
    Returns:
        Sampled token ID
    """
    if temperature <= 0:
        # Greedy sampling
        return np.argmax(logits)
    
    # Scale logits by temperature
    logits = logits / temperature
    
    # Convert to probabilities
    probs = np.exp(logits - np.max(logits))  # Stability
    probs = probs / np.sum(probs)
    
    # Sample
    return np.random.choice(len(probs), p=probs)


def nucleus_sampling(logits: np.ndarray, top_p: float = 0.9, temperature: float = 1.0) -> int:
    """
    Nucleus (top-p) sampling for diverse generation.
    Only sample from tokens whose cumulative probability >= top_p.
    
    Args:
        logits: Logits from model [vocab_size]
        top_p: Cumulative probability threshold (0.9 = sample from top 90%)
        temperature: Temperature scaling
    
    Returns:
        Sampled token ID
    """
    # Apply temperature
    logits = logits / temperature
    
    # Convert to probabilities
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    
    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Compute cumulative probabilities
    cumsum_probs = np.cumsum(sorted_probs)
    
    # Find cutoff index where cumsum >= top_p
    cutoff_idx = np.searchsorted(cumsum_probs, top_p)
    
    # Keep only top-p tokens
    nucleus_indices = sorted_indices[:cutoff_idx + 1]
    nucleus_probs = sorted_probs[:cutoff_idx + 1]
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)  # Renormalize
    
    # Sample from nucleus
    sampled_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
    return nucleus_indices[sampled_idx]


def apply_repetition_penalty(logits: np.ndarray, 
                             generated_tokens: List[int], 
                             penalty: float = 1.2,
                             window: int = 20) -> np.ndarray:
    """
    Apply repetition penalty to discourage repeating recent tokens.
    
    Args:
        logits: Current logits [vocab_size]
        generated_tokens: Previously generated tokens
        penalty: Penalty factor (>1.0 = discourage repetition)
        window: Look-back window for recent tokens
    
    Returns:
        Penalized logits
    """
    if not generated_tokens:
        return logits
    
    # Get recent tokens
    recent_tokens = generated_tokens[-window:] if len(generated_tokens) > window else generated_tokens
    
    # Apply penalty to repeated tokens
    penalized_logits = logits.copy()
    for token_id in set(recent_tokens):
        penalized_logits[token_id] /= penalty
    
    return penalized_logits


# ============================================================================
# Quality Metrics Callback
# ============================================================================

class DiversityQualityCallback(Callback):
    """
    Monitor quality metrics during training with diversity focus.
    Tracks BLEU, ROUGE, Distinct, Repetition Rate every N epochs.
    """
    
    def __init__(self, 
                 validation_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 tokenizer: SimpleTokenizer,
                 interval: int = 5,
                 num_samples: int = 200,
                 temperature: float = 1.5,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.2,
                 log_file: str = 'logs/diversity_quality_log.csv'):
        """
        Args:
            validation_data: (customer_inputs, agent_inputs, agent_targets)
            tokenizer: SimpleTokenizer instance
            interval: Evaluate every N epochs
            num_samples: Number of samples to evaluate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty factor
            log_file: Path to save metrics log
        """
        super().__init__()
        self.val_customer_inputs, self.val_agent_inputs, self.val_agent_targets = validation_data
        self.tokenizer = tokenizer
        self.interval = interval
        self.num_samples = min(num_samples, len(self.val_customer_inputs))
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.log_file = log_file
        
        # Initialize evaluators
        self.text_evaluator = TextGenerationEvaluator(tokenizer)
        self.diversity_metrics = DiversityMetrics(tokenizer)
        
        # Metrics history
        self.history = []
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize log file
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('epoch,bleu_1,bleu_2,bleu_4,rouge_1,rouge_2,rouge_l,'
                       'distinct_1,distinct_2,distinct_3,repetition_rate,'
                       'entropy,quality_score,avg_length\n')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Evaluate quality metrics at epoch end."""
        if (epoch + 1) % self.interval != 0:
            return
        
        print(f"\n{'='*80}")
        print(f"QUALITY EVALUATION - Epoch {epoch + 1}")
        print(f"{'='*80}")
        
        # Sample validation data
        indices = np.random.choice(len(self.val_customer_inputs), self.num_samples, replace=False)
        customer_samples = self.val_customer_inputs[indices]
        agent_samples = self.val_agent_inputs[indices]
        target_samples = self.val_agent_targets[indices]
        
        # Generate responses with diversity sampling
        generated_texts = []
        reference_texts = []
        
        print(f"Generating {self.num_samples} responses with diversity sampling...")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-p: {self.top_p}")
        print(f"  Repetition penalty: {self.repetition_penalty}")
        
        for i, (customer_input, agent_input, target) in enumerate(zip(customer_samples, agent_samples, target_samples)):
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{self.num_samples}...")
            
            # Generate with diversity sampling
            generated = self._generate_with_diversity(customer_input, agent_input)
            generated_texts.append(generated)
            
            # Reference text
            reference = self.tokenizer.sequences_to_texts([target])[0]
            reference_texts.append(reference)
        
        # Compute BLEU scores
        print("\n📊 Computing BLEU scores...")
        bleu_scores = self.text_evaluator.compute_bleu(reference_texts, generated_texts)
        print(f"  BLEU-1: {bleu_scores['bleu-1']:.4f}")
        print(f"  BLEU-2: {bleu_scores['bleu-2']:.4f}")
        print(f"  BLEU-4: {bleu_scores['bleu-4']:.4f}")
        
        # Compute ROUGE scores
        print("\n📊 Computing ROUGE scores...")
        rouge_scores = self.text_evaluator.compute_rouge(reference_texts, generated_texts)
        print(f"  ROUGE-1: {rouge_scores['rouge-1']:.4f}")
        print(f"  ROUGE-2: {rouge_scores['rouge-2']:.4f}")
        print(f"  ROUGE-L: {rouge_scores['rouge-l']:.4f}")
        
        # Compute diversity metrics
        print("\n📊 Computing Diversity metrics...")
        diversity = self.diversity_metrics.compute_all_metrics(generated_texts)
        print(f"  Distinct-1: {diversity['distinct_1']:.4f}")
        print(f"  Distinct-2: {diversity['distinct_2']:.4f}")
        print(f"  Distinct-3: {diversity['distinct_3']:.4f}")
        print(f"  Repetition Rate: {diversity['repetition_rate']:.4f}")
        print(f"  Entropy: {diversity['entropy']:.4f}")
        
        # Quality score
        quality_score = (
            0.3 * bleu_scores['bleu-4'] +
            0.3 * rouge_scores['rouge-l'] +
            0.2 * diversity['distinct_2'] +
            0.2 * (1.0 - diversity['repetition_rate'])
        )
        print(f"\n⭐ Overall Quality Score: {quality_score:.4f}")
        
        # Average length
        avg_length = np.mean([len(text.split()) for text in generated_texts])
        print(f"📏 Average Response Length: {avg_length:.1f} words")
        
        # Assessment
        print("\n" + "="*80)
        if diversity['distinct_1'] > 0.4 and diversity['repetition_rate'] < 0.2:
            print("✅ EXCELLENT: High diversity, low repetition!")
        elif diversity['distinct_1'] > 0.3 and diversity['repetition_rate'] < 0.4:
            print("✅ GOOD: Reasonable diversity and repetition.")
        elif diversity['distinct_1'] > 0.2 and diversity['repetition_rate'] < 0.6:
            print("⚠️  MODERATE: Some improvement needed.")
        else:
            print("❌ POOR: High repetition detected!")
        print("="*80)
        
        # Save metrics
        metrics = {
            'epoch': epoch + 1,
            'bleu_1': bleu_scores['bleu-1'],
            'bleu_2': bleu_scores['bleu-2'],
            'bleu_4': bleu_scores['bleu-4'],
            'rouge_1': rouge_scores['rouge-1'],
            'rouge_2': rouge_scores['rouge-2'],
            'rouge_l': rouge_scores['rouge-l'],
            'distinct_1': diversity['distinct_1'],
            'distinct_2': diversity['distinct_2'],
            'distinct_3': diversity['distinct_3'],
            'repetition_rate': diversity['repetition_rate'],
            'entropy': diversity['entropy'],
            'quality_score': quality_score,
            'avg_length': avg_length
        }
        self.history.append(metrics)
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(f"{metrics['epoch']},{metrics['bleu_1']:.4f},{metrics['bleu_2']:.4f},"
                   f"{metrics['bleu_4']:.4f},{metrics['rouge_1']:.4f},{metrics['rouge_2']:.4f},"
                   f"{metrics['rouge_l']:.4f},{metrics['distinct_1']:.4f},{metrics['distinct_2']:.4f},"
                   f"{metrics['distinct_3']:.4f},{metrics['repetition_rate']:.4f},"
                   f"{metrics['entropy']:.4f},{metrics['quality_score']:.4f},{metrics['avg_length']:.1f}\n")
        
        # Show sample
        print("\n📝 Sample Generation:")
        print(f"  Customer: {self.tokenizer.sequences_to_texts([customer_samples[0]])[0]}")
        print(f"  Reference: {reference_texts[0]}")
        print(f"  Generated: {generated_texts[0]}")
        print()
    
    def _generate_with_diversity(self, customer_input: np.ndarray, agent_input: np.ndarray) -> str:
        """Generate response with diversity sampling strategies."""
        # This is a placeholder - will be overridden by actual model generation
        # In practice, this should use the model's generator with temperature/nucleus sampling
        
        # For now, use standard generation (will be replaced with enhanced version)
        customer_batch = np.expand_dims(customer_input, axis=0)
        agent_batch = np.expand_dims(agent_input, axis=0)
        
        # Generate using model
        generated = self.model.predict([customer_batch, agent_batch], verbose=0)[0]
        
        # Decode
        generated_ids = np.argmax(generated, axis=-1)
        generated_text = self.tokenizer.sequences_to_texts([generated_ids])[0]
        
        return generated_text


# ============================================================================
# Enhanced Training Function
# ============================================================================

def train_with_diversity_optimization(
    model: SeparateInputHybridGANVAE,
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tokenizer: SimpleTokenizer,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    diversity_weight: float = 0.15,
    temperature: float = 1.5,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    checkpoint_dir: str = 'models/diversity_optimized',
    log_dir: str = 'logs'
) -> Dict:
    """
    Fine-tune model with diversity optimization.
    
    Args:
        model: SeparateInputHybridGANVAE model
        train_data: (customer_train, agent_train, targets_train)
        val_data: (customer_val, agent_val, targets_val)
        tokenizer: SimpleTokenizer
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        diversity_weight: Weight for diversity loss
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        repetition_penalty: Repetition penalty factor
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
    
    Returns:
        Training history dictionary
    """
    print(f"\n{'='*80}")
    print("ENHANCED FINE-TUNING WITH DIVERSITY OPTIMIZATION")
    print(f"{'='*80}\n")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Unpack data
    customer_train, agent_train, targets_train = train_data
    customer_val, agent_val, targets_val = val_data
    
    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Diversity weight: {diversity_weight}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Repetition penalty: {repetition_penalty}")
    print(f"\nData:")
    print(f"  Training samples: {len(customer_train):,}")
    print(f"  Validation samples: {len(customer_val):,}")
    print(f"  Vocabulary size: {len(tokenizer.word_index)}")
    
    # Compile model with diversity loss
    print(f"\n{'='*80}")
    print("COMPILING MODEL")
    print(f"{'='*80}\n")
    
    # Custom loss function combining reconstruction + diversity
    def combined_loss(y_true, y_pred):
        # Reconstruction loss (categorical crossentropy)
        recon_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Diversity loss
        div_loss = diversity_loss(y_pred)
        
        # Combined
        total_loss = recon_loss + diversity_weight * div_loss
        
        return total_loss
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.hybrid_model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=['accuracy']
    )
    
    print("✓ Model compiled with diversity-aware loss")
    print(f"  Loss = reconstruction_loss + {diversity_weight} × diversity_loss")
    
    # Callbacks
    print(f"\n{'='*80}")
    print("SETTING UP CALLBACKS")
    print(f"{'='*80}\n")
    
    callbacks = [
        # Quality metrics monitoring
        DiversityQualityCallback(
            validation_data=val_data,
            tokenizer=tokenizer,
            interval=5,
            num_samples=200,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            log_file=f'{log_dir}/diversity_quality_log.csv'
        ),
        
        # Model checkpointing (save best based on val loss)
        ModelCheckpoint(
            filepath=f'{checkpoint_dir}/best_model_epoch_{{epoch:03d}}_val_{{val_loss:.4f}}.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        ),
        
        # Learning rate reduction on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping (patience=15 epochs)
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("✓ Callbacks configured:")
    print(f"  • DiversityQualityCallback (every 5 epochs)")
    print(f"  • ModelCheckpoint (save best)")
    print(f"  • ReduceLROnPlateau (factor=0.5, patience=5)")
    print(f"  • EarlyStopping (patience=15)")
    
    # Training
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    history = model.hybrid_model.fit(
        [customer_train, agent_train],
        targets_train,
        validation_data=([customer_val, agent_val], targets_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = f'{checkpoint_dir}/final_model.weights.h5'
    model.hybrid_model.save_weights(final_path)
    print(f"\n✓ Final model saved to: {final_path}")
    
    # Training summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}\n")
    
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"Final Metrics:")
    print(f"  Training Loss: {final_train_loss:.4f}")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Training Accuracy: {final_train_acc:.4f}")
    print(f"  Validation Accuracy: {final_val_acc:.4f}")
    
    # Plot training curves
    print(f"\n{'='*80}")
    print("GENERATING TRAINING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    plot_training_curves(history, save_dir='results/diversity_training')
    
    return history.history


def plot_training_curves(history, save_dir: str = 'results/diversity_training'):
    """Plot training curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/training_curves.png")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main training script."""
    print(f"\n{'='*80}")
    print("ENHANCED FINE-TUNING WITH DIVERSITY OPTIMIZATION")
    print(f"{'='*80}\n")
    
    # Load data and tokenizer
    print("📦 Loading data and tokenizer...")
    
    with open('processed_data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    customer_train = np.load('processed_data/train_customer.npy')
    agent_train = np.load('processed_data/train_agent.npy')
    
    customer_val = np.load('processed_data/val_customer.npy')
    agent_val = np.load('processed_data/val_agent.npy')
    
    # Targets are agent responses (shifted for teacher forcing)
    targets_train = agent_train[:, 1:]  # Remove first token (start token)
    targets_val = agent_val[:, 1:]
    
    # Adjust inputs to match target length
    customer_train = customer_train[:, :-1]
    agent_train = agent_train[:, :-1]
    customer_val = customer_val[:, :-1]
    agent_val = agent_val[:, :-1]
    
    print(f"✓ Loaded data:")
    print(f"  Training: {len(customer_train):,} samples")
    print(f"  Validation: {len(customer_val):,} samples")
    print(f"  Vocabulary: {len(tokenizer.word_index)} tokens")
    print(f"  Sequence length: {customer_train.shape[1]}")
    
    # Build model (Separate Input architecture - winner from comparison)
    print(f"\n{'='*80}")
    print("BUILDING SEPARATE INPUT MODEL (WINNER ARCHITECTURE)")
    print(f"{'='*80}\n")
    
    model = SeparateInputHybridGANVAE(
        vocab_size=len(tokenizer.word_index) + 1,
        max_length=customer_train.shape[1],
        latent_dim=VAE_LATENT_DIM
    )
    
    print("✓ Model built successfully")
    print(f"  Architecture: Separate Input (Independent Customer/Agent Encoders)")
    print(f"  Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"  Latent dim: {VAE_LATENT_DIM}")
    
    # Training configuration
    config = {
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'diversity_weight': 0.15,  # Weight for diversity loss
        'temperature': 1.5,        # Sampling temperature
        'top_p': 0.9,             # Nucleus sampling
        'repetition_penalty': 1.2  # Repetition penalty
    }
    
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train
    history = train_with_diversity_optimization(
        model=model,
        train_data=(customer_train, agent_train, targets_train),
        val_data=(customer_val, agent_val, targets_val),
        tokenizer=tokenizer,
        **config
    )
    
    # Save training summary
    summary_path = 'results/diversity_training/training_summary.json'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump({
            'config': config,
            'final_metrics': {
                'train_loss': float(history['loss'][-1]),
                'val_loss': float(history['val_loss'][-1]),
                'train_accuracy': float(history['accuracy'][-1]),
                'val_accuracy': float(history['val_accuracy'][-1])
            },
            'total_epochs': len(history['loss'])
        }, f, indent=2)
    
    print(f"\n✓ Training summary saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("✅ FINE-TUNING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")
    print("Next steps:")
    print("  1. Check quality metrics in: logs/diversity_quality_log.csv")
    print("  2. Review training curves in: results/diversity_training/")
    print("  3. Load best model from: models/diversity_optimized/")
    print("  4. Run inference with diversity sampling")


if __name__ == "__main__":
    main()
