"""
Hybrid GAN + VAE Training Pipeline

Trains the complete hybrid model with alternating updates:
1. VAE training (encoder + decoder)
2. GAN Discriminator training (real vs fake)
3. GAN Generator training (fool discriminator)
4. Hybrid end-to-end training

Combines reconstruction, adversarial, and regularization losses.
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
import os

import config
from hybrid_model import HybridGANVAE
from simple_tokenizer import SimpleTokenizer


class HybridTrainer:
    """
    Complete training pipeline for Hybrid GAN + VAE model
    
    Manages:
    - VAE pretraining
    - GAN adversarial training
    - Hybrid end-to-end training
    - Loss tracking and visualization
    """
    
    def __init__(self, vocab_size, max_length):
        """
        Initialize Hybrid Trainer
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Build hybrid model
        print("Building Hybrid GAN + VAE model...")
        self.hybrid = HybridGANVAE(vocab_size, max_length)
        
        # Training history
        self.history = {
            'vae_loss': [],
            'vae_reconstruction_loss': [],
            'vae_kl_loss': [],
            'discriminator_loss': [],
            'discriminator_accuracy': [],
            'generator_loss': [],
            'hybrid_loss': []
        }
        
        print("✓ Hybrid trainer initialized")
    
    def load_data(self):
        """Load preprocessed data"""
        print("\nLoading data...")
        
        # Load customer and agent sequences
        train_customer = np.load(f'{config.PROCESSED_DATA_DIR}/train_customer.npy')
        train_agent = np.load(f'{config.PROCESSED_DATA_DIR}/train_agent.npy')
        val_customer = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')
        val_agent = np.load(f'{config.PROCESSED_DATA_DIR}/val_agent.npy')
        
        # Load metadata
        train_metadata = pd.read_csv(f'{config.PROCESSED_DATA_DIR}/train_metadata.csv')
        val_metadata = pd.read_csv(f'{config.PROCESSED_DATA_DIR}/val_metadata.csv')
        
        print(f"✓ Training: {len(train_customer):,} customer-agent pairs")
        print(f"✓ Validation: {len(val_customer):,} customer-agent pairs")
        
        return {
            'train_customer': train_customer,
            'train_agent': train_agent,
            'val_customer': val_customer,
            'val_agent': val_agent,
            'train_metadata': train_metadata,
            'val_metadata': val_metadata
        }
    
    def pretrain_vae(self, data, epochs=10, batch_size=64):
        """
        Pretrain VAE on customer messages
        
        Args:
            data: Dictionary with training data
            epochs: Number of pretraining epochs
            batch_size: Batch size
        """
        print("\n" + "=" * 70)
        print("PHASE 1: VAE PRETRAINING")
        print("=" * 70)
        print(f"Pretraining VAE for {epochs} epochs...")
        
        train_customer = data['train_customer']
        val_customer = data['val_customer']
        
        # Compile VAE
        self.hybrid.vae.compile_model(learning_rate=config.VAE_LEARNING_RATE)
        
        # Train VAE
        history = self.hybrid.vae.vae.fit(
            train_customer,
            validation_data=val_customer,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Store history
        for key in ['loss', 'reconstruction_loss', 'kl_loss']:
            if key in history.history:
                self.history[f'vae_{key}'].extend(history.history[key])
        
        print(f"\n✓ VAE pretraining completed")
        print(f"  Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    def train_gan(self, data, epochs=50, batch_size=64, disc_updates=1, gen_updates=1):
        """
        Train GAN (Generator + Discriminator) adversarially
        
        Args:
            data: Dictionary with training data
            epochs: Number of training epochs
            batch_size: Batch size
            disc_updates: Discriminator updates per epoch
            gen_updates: Generator updates per epoch
        """
        print("\n" + "=" * 70)
        print("PHASE 2: GAN ADVERSARIAL TRAINING")
        print("=" * 70)
        print(f"Training GAN for {epochs} epochs...")
        print(f"Discriminator updates: {disc_updates}, Generator updates: {gen_updates}")
        
        train_customer = data['train_customer']
        train_agent = data['train_agent']
        
        # Compile discriminator and generator
        from gan_model import GAN
        gan = GAN(self.hybrid.gan_generator, self.hybrid.gan_discriminator)
        gan.compile_models(
            discriminator_lr=config.GAN_LEARNING_RATE_D,
            generator_lr=config.GAN_LEARNING_RATE_G
        )
        
        num_batches = len(train_customer) // batch_size
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Shuffle data
            indices = np.random.permutation(len(train_customer))
            train_customer_shuffled = train_customer[indices]
            train_agent_shuffled = train_agent[indices]
            
            epoch_d_loss = []
            epoch_d_acc = []
            epoch_g_loss = []
            
            for batch in range(num_batches):
                # Get batch
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                customer_batch = train_customer_shuffled[start_idx:end_idx]
                agent_batch = train_agent_shuffled[start_idx:end_idx]
                
                # Encode customer messages to latent space
                _, _, latent_vectors = self.hybrid.vae.encoder.predict(customer_batch, verbose=0)
                
                # Generate fake agent responses
                generated_responses = self.hybrid.gan_generator.generate(latent_vectors, temperature=1.0)
                fake_agent_tokens = np.argmax(generated_responses, axis=-1)
                
                # Train discriminator
                for _ in range(disc_updates):
                    d_loss = gan.train_discriminator(agent_batch, fake_agent_tokens)
                    epoch_d_loss.append(d_loss[0])
                    epoch_d_acc.append(d_loss[1])
                
                # Train generator
                for _ in range(gen_updates):
                    g_loss = gan.train_generator(latent_vectors)
                    epoch_g_loss.append(g_loss[0])
                
                # Print progress
                if (batch + 1) % 50 == 0:
                    print(f"  Batch {batch+1}/{num_batches} - "
                          f"D loss: {np.mean(epoch_d_loss):.4f}, "
                          f"D acc: {np.mean(epoch_d_acc):.4f}, "
                          f"G loss: {np.mean(epoch_g_loss):.4f}")
            
            # Store epoch metrics
            self.history['discriminator_loss'].append(np.mean(epoch_d_loss))
            self.history['discriminator_accuracy'].append(np.mean(epoch_d_acc))
            self.history['generator_loss'].append(np.mean(epoch_g_loss))
            
            print(f"Epoch {epoch+1} - "
                  f"D loss: {np.mean(epoch_d_loss):.4f}, "
                  f"D acc: {np.mean(epoch_d_acc):.4f}, "
                  f"G loss: {np.mean(epoch_g_loss):.4f}")
        
        print(f"\n✓ GAN training completed")
    
    def train_hybrid_end_to_end(self, data, epochs=30, batch_size=64):
        """
        Train complete hybrid model end-to-end
        
        Args:
            data: Dictionary with training data
            epochs: Number of training epochs
            batch_size: Batch size
        """
        print("\n" + "=" * 70)
        print("PHASE 3: HYBRID END-TO-END TRAINING")
        print("=" * 70)
        print(f"Training hybrid model for {epochs} epochs...")
        
        train_customer = data['train_customer']
        train_agent = data['train_agent']
        val_customer = data['val_customer']
        val_agent = data['val_agent']
        
        # Compile hybrid model
        self.hybrid.compile_hybrid_model(learning_rate=config.HYBRID_LEARNING_RATE)
        
        # Prepare data for hybrid training
        # Outputs: generated_response, reconstructed_customer
        train_outputs = {
            'generated_response': train_agent,
            'reconstructed_customer': train_customer
        }
        
        val_outputs = {
            'generated_response': val_agent,
            'reconstructed_customer': val_customer
        }
        
        # Train hybrid model
        history = self.hybrid.hybrid_model.fit(
            train_customer,
            train_outputs,
            validation_data=(val_customer, val_outputs),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Store history
        self.history['hybrid_loss'].extend(history.history['loss'])
        
        print(f"\n✓ Hybrid end-to-end training completed")
        print(f"  Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    def train_complete(self, data, 
                      vae_pretrain_epochs=10,
                      gan_epochs=30,
                      hybrid_epochs=20,
                      batch_size=64):
        """
        Complete training pipeline: VAE → GAN → Hybrid
        
        Args:
            data: Dictionary with training data
            vae_pretrain_epochs: VAE pretraining epochs
            gan_epochs: GAN training epochs
            hybrid_epochs: Hybrid end-to-end epochs
            batch_size: Batch size
        """
        print("\n" + "=" * 70)
        print("COMPLETE HYBRID GAN + VAE TRAINING PIPELINE")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Phase 1: VAE Pretraining
        self.pretrain_vae(data, epochs=vae_pretrain_epochs, batch_size=batch_size)
        
        # Phase 2: GAN Training
        self.train_gan(data, epochs=gan_epochs, batch_size=batch_size)
        
        # Phase 3: Hybrid End-to-End Training
        self.train_hybrid_end_to_end(data, epochs=hybrid_epochs, batch_size=batch_size)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED!")
        print("=" * 70)
        print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        print("\nPlotting training history...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # VAE losses
        if len(self.history['vae_loss']) > 0:
            ax = axes[0, 0]
            ax.plot(self.history['vae_loss'], label='Total Loss', linewidth=2)
            if len(self.history['vae_reconstruction_loss']) > 0:
                ax.plot(self.history['vae_reconstruction_loss'], label='Reconstruction', linewidth=2)
            if len(self.history['vae_kl_loss']) > 0:
                ax.plot(self.history['vae_kl_loss'], label='KL Divergence', linewidth=2)
            ax.set_title('VAE Training Losses', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Discriminator
        if len(self.history['discriminator_loss']) > 0:
            ax = axes[0, 1]
            ax.plot(self.history['discriminator_loss'], label='D Loss', linewidth=2, color='red')
            ax.set_title('Discriminator Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax2 = ax.twinx()
            ax2.plot(self.history['discriminator_accuracy'], label='D Accuracy', linewidth=2, color='green')
            ax2.set_ylabel('Accuracy')
            ax2.legend(loc='lower right')
        
        # Generator
        if len(self.history['generator_loss']) > 0:
            ax = axes[1, 0]
            ax.plot(self.history['generator_loss'], label='G Loss', linewidth=2, color='blue')
            ax.set_title('Generator Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hybrid
        if len(self.history['hybrid_loss']) > 0:
            ax = axes[1, 1]
            ax.plot(self.history['hybrid_loss'], label='Hybrid Loss', linewidth=2, color='purple')
            ax.set_title('Hybrid End-to-End Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training plot saved to: {save_path}")
        else:
            plt.savefig(f'{config.RESULTS_DIR}/hybrid_training_history.png', dpi=300, bbox_inches='tight')
            print(f"✓ Training plot saved to: {config.RESULTS_DIR}/hybrid_training_history.png")
        
        plt.close()
    
    def evaluate_generation(self, data, tokenizer, num_samples=5):
        """
        Evaluate generation quality
        
        Args:
            data: Dictionary with validation data
            tokenizer: Tokenizer for decoding
            num_samples: Number of samples to evaluate
        """
        print("\n" + "=" * 70)
        print("EVALUATING GENERATION QUALITY")
        print("=" * 70)
        
        val_customer = data['val_customer']
        val_agent = data['val_agent']
        
        # Get random samples
        indices = np.random.choice(len(val_customer), num_samples, replace=False)
        customer_samples = val_customer[indices]
        agent_samples = val_agent[indices]
        
        # Generate responses
        generated_responses = self.hybrid.generate_response(customer_samples)
        generated_tokens = np.argmax(generated_responses, axis=-1)
        
        # Evaluate with discriminator
        quality_scores = self.hybrid.evaluate_response_quality(generated_tokens)
        
        # Decode sequences
        reverse_word_index = {idx: word for word, idx in tokenizer.word_index.items()}
        
        def decode_sequence(sequence):
            if len(sequence.shape) > 1:
                sequence = np.argmax(sequence, axis=-1)
            words = []
            for idx in sequence:
                if idx == 0:
                    continue
                word = reverse_word_index.get(idx, '<UNK>')
                words.append(word)
            return ' '.join(words)
        
        # Display results
        for i, idx in enumerate(indices, 1):
            print(f"\n{'─' * 70}")
            print(f"Example {i}:")
            print(f"{'─' * 70}")
            
            customer_text = decode_sequence(customer_samples[i-1])
            real_agent_text = decode_sequence(agent_samples[i-1])
            generated_agent_text = decode_sequence(generated_tokens[i-1])
            
            print(f"Customer:  {customer_text}")
            print(f"Real Agent: {real_agent_text}")
            print(f"Generated:  {generated_agent_text}")
            print(f"Quality Score: {quality_scores[i-1][0]:.4f} (0=fake, 1=real)")
    
    def save_models(self, path=None):
        """Save all trained models"""
        if path is None:
            path = config.MODELS_DIR
        
        print(f"\nSaving models to {path}/...")
        self.hybrid.save_models(path)
        
        # Save training history
        with open(f'{path}/hybrid_training_history.pkl', 'wb') as f:
            pickle.dump(self.history, f)
        
        print("✓ All models and history saved!")


def main():
    """Main training function"""
    print("\n" + "=" * 70)
    print("HYBRID GAN + VAE TRAINING PIPELINE")
    print("=" * 70)
    
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
    data = trainer.load_data()
    
    # Train complete pipeline
    trainer.train_complete(
        data,
        vae_pretrain_epochs=10,
        gan_epochs=20,
        hybrid_epochs=10,
        batch_size=64
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate generation
    trainer.evaluate_generation(data, tokenizer, num_samples=5)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
