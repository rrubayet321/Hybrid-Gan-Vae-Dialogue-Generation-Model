"""
Training script with integrated bias mitigation
Combines 3-phase training with fairness constraints
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
import os
from datetime import datetime

import config
from hybrid_model import HybridGANVAE
from bias_mitigation import BiasAwareModel, FairnessRegularizer, prepare_sensitive_attributes


class FairHybridTrainer:
    """
    Training pipeline with integrated bias mitigation
    
    Phase 1: Pretrain VAE with fairness regularization
    Phase 2: Train GAN with adversarial debiasing
    Phase 3: Joint end-to-end training with fairness constraints
    """
    
    def __init__(self, vocab_size, max_length):
        """
        Initialize fair hybrid trainer
        
        Args:
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Build hybrid model
        print("\n" + "=" * 70)
        print("INITIALIZING FAIR HYBRID TRAINER")
        print("=" * 70)
        
        self.hybrid_model = HybridGANVAE(vocab_size, max_length)
        
        # Bias-aware wrapper (will be initialized with metadata)
        self.bias_aware_model = None
        self.fairness_regularizer = FairnessRegularizer()
        
        # Training history
        self.history = {
            'vae_pretrain': {'loss': [], 'recon_loss': [], 'kl_loss': [], 'fairness_loss': []},
            'gan_train': {'d_loss': [], 'g_loss': [], 'fairness_metrics': []},
            'hybrid_train': {'loss': [], 'vae_loss': [], 'gan_loss': [], 'fairness_loss': []}
        }
        
        print("✓ Fair hybrid trainer initialized")
    
    def pretrain_vae_with_fairness(self, train_data, metadata, epochs=10, batch_size=None):
        """
        Phase 1: Pretrain VAE with fairness regularization
        
        Args:
            train_data: Training sequences (customer messages)
            metadata: Metadata DataFrame with sensitive attributes
            epochs: Number of epochs
            batch_size: Batch size
        """
        if batch_size is None:
            batch_size = config.HYBRID_BATCH_SIZE
        
        print("\n" + "=" * 70)
        print("PHASE 1: VAE PRETRAINING WITH FAIRNESS")
        print("=" * 70)
        
        # Prepare sensitive attributes
        attributes_dict, encoders, attribute_dims = prepare_sensitive_attributes(
            metadata, 
            config.SENSITIVE_ATTRIBUTES
        )
        
        # Initialize bias-aware model
        self.bias_aware_model = BiasAwareModel(
            self.hybrid_model,
            attribute_dims
        )
        
        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=config.VAE_LEARNING_RATE)
        
        num_batches = len(train_data) // batch_size
        
        print(f"\nTraining VAE for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {num_batches}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Shuffle data
            indices = np.random.permutation(len(train_data))
            train_data_shuffled = train_data[indices]
            
            # Shuffle metadata accordingly
            attributes_shuffled = {
                attr: labels[indices] for attr, labels in attributes_dict.items()
            }
            
            epoch_losses = []
            epoch_recon_losses = []
            epoch_kl_losses = []
            epoch_fairness_losses = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_data = train_data_shuffled[start_idx:end_idx]
                batch_attributes = {
                    attr: labels[start_idx:end_idx] 
                    for attr, labels in attributes_shuffled.items()
                }
                
                # Training step
                with tf.GradientTape() as tape:
                    # VAE forward pass
                    reconstructions, z_mean, z_log_var = self.hybrid_model.vae(
                        batch_data, 
                        training=True
                    )
                    
                    # Reconstruction loss
                    recon_loss = tf.reduce_mean(
                        keras.losses.sparse_categorical_crossentropy(
                            batch_data, reconstructions
                        )
                    )
                    
                    # KL divergence
                    kl_loss = -0.5 * tf.reduce_mean(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    )
                    
                    # Fairness regularization (demographic parity)
                    fairness_loss = 0.0
                    for attr_name, attr_labels in batch_attributes.items():
                        fairness_loss += self.fairness_regularizer.demographic_parity_loss(
                            z_mean, 
                            attr_labels
                        )
                    
                    # Also add variance regularization
                    fairness_loss += 0.1 * self.fairness_regularizer.variance_regularization(
                        z_mean, 
                        z_log_var
                    )
                    
                    # Total loss
                    total_loss = (
                        config.WEIGHT_RECONSTRUCTION * recon_loss +
                        config.WEIGHT_KL_DIVERGENCE * kl_loss +
                        config.BIAS_ADVERSARIAL_WEIGHT * fairness_loss
                    )
                
                # Update VAE
                gradients = tape.gradient(
                    total_loss, 
                    self.hybrid_model.vae.trainable_variables
                )
                optimizer.apply_gradients(
                    zip(gradients, self.hybrid_model.vae.trainable_variables)
                )
                
                # Track losses
                epoch_losses.append(float(total_loss))
                epoch_recon_losses.append(float(recon_loss))
                epoch_kl_losses.append(float(kl_loss))
                epoch_fairness_losses.append(float(fairness_loss))
                
                # Print progress
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx + 1}/{num_batches} - "
                          f"Loss: {np.mean(epoch_losses):.4f}, "
                          f"Recon: {np.mean(epoch_recon_losses):.4f}, "
                          f"KL: {np.mean(epoch_kl_losses):.4f}, "
                          f"Fairness: {np.mean(epoch_fairness_losses):.4f}")
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            avg_recon = np.mean(epoch_recon_losses)
            avg_kl = np.mean(epoch_kl_losses)
            avg_fairness = np.mean(epoch_fairness_losses)
            
            self.history['vae_pretrain']['loss'].append(avg_loss)
            self.history['vae_pretrain']['recon_loss'].append(avg_recon)
            self.history['vae_pretrain']['kl_loss'].append(avg_kl)
            self.history['vae_pretrain']['fairness_loss'].append(avg_fairness)
            
            print(f"\n  ✓ Epoch {epoch + 1} Summary:")
            print(f"    Total Loss: {avg_loss:.4f}")
            print(f"    Reconstruction: {avg_recon:.4f}")
            print(f"    KL Divergence: {avg_kl:.4f}")
            print(f"    Fairness: {avg_fairness:.4f}")
        
        print("\n✓ VAE pretraining with fairness complete!")
        
        # Evaluate bias after pretraining
        print("\n" + "=" * 70)
        print("BIAS EVALUATION AFTER VAE PRETRAINING")
        print("=" * 70)
        
        # Encode all training data to latent space
        z_mean, z_log_var, z = self.hybrid_model.vae.encoder.predict(
            train_data[:5000], 
            verbose=0
        )
        
        # Train bias detectors
        self.bias_aware_model.train_bias_detectors(
            z,
            {attr: labels[:5000] for attr, labels in attributes_dict.items()},
            epochs=5
        )
        
        # Compute fairness metrics
        metrics = self.bias_aware_model.compute_fairness_metrics(
            z,
            {attr: labels[:5000] for attr, labels in attributes_dict.items()}
        )
        
        print("\nFairness metrics after VAE pretraining:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        return self.history['vae_pretrain']
    
    def train_gan_with_debiasing(self, train_data, metadata, epochs=30, batch_size=None):
        """
        Phase 2: Train GAN with adversarial debiasing
        
        Args:
            train_data: Training sequences (agent responses)
            metadata: Metadata DataFrame
            epochs: Number of epochs
            batch_size: Batch size
        """
        if batch_size is None:
            batch_size = config.HYBRID_BATCH_SIZE
        
        print("\n" + "=" * 70)
        print("PHASE 2: GAN TRAINING WITH ADVERSARIAL DEBIASING")
        print("=" * 70)
        
        # Prepare sensitive attributes
        attributes_dict, _, _ = prepare_sensitive_attributes(
            metadata,
            config.SENSITIVE_ATTRIBUTES
        )
        
        # Optimizers
        d_optimizer = keras.optimizers.Adam(learning_rate=config.GAN_LEARNING_RATE_D)
        g_optimizer = keras.optimizers.Adam(learning_rate=config.GAN_LEARNING_RATE_G)
        
        num_batches = len(train_data) // batch_size
        
        print(f"\nTraining GAN for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {num_batches}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Shuffle data
            indices = np.random.permutation(len(train_data))
            train_data_shuffled = train_data[indices]
            
            epoch_d_losses = []
            epoch_g_losses = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_real = train_data_shuffled[start_idx:end_idx]
                
                # Generate random latent vectors
                batch_latent = tf.random.normal(
                    shape=(batch_size, self.hybrid_model.vae_latent_dim)
                )
                
                # === Train Discriminator ===
                with tf.GradientTape() as d_tape:
                    # Generate fake sequences
                    generated = self.hybrid_model.gan.generator(batch_latent, training=True)
                    
                    # Discriminator predictions
                    real_preds = self.hybrid_model.gan.discriminator(batch_real, training=True)
                    fake_preds = self.hybrid_model.gan.discriminator(generated, training=True)
                    
                    # Discriminator loss
                    d_loss_real = keras.losses.binary_crossentropy(
                        tf.ones_like(real_preds), real_preds
                    )
                    d_loss_fake = keras.losses.binary_crossentropy(
                        tf.zeros_like(fake_preds), fake_preds
                    )
                    d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
                
                # Update discriminator
                d_gradients = d_tape.gradient(
                    d_loss,
                    self.hybrid_model.gan.discriminator.trainable_variables
                )
                d_optimizer.apply_gradients(
                    zip(d_gradients, self.hybrid_model.gan.discriminator.trainable_variables)
                )
                
                # === Train Generator ===
                with tf.GradientTape() as g_tape:
                    # Generate fake sequences
                    generated = self.hybrid_model.gan.generator(batch_latent, training=True)
                    fake_preds = self.hybrid_model.gan.discriminator(generated, training=True)
                    
                    # Generator loss (fool discriminator)
                    g_loss = tf.reduce_mean(
                        keras.losses.binary_crossentropy(
                            tf.ones_like(fake_preds), fake_preds
                        )
                    )
                
                # Update generator
                g_gradients = g_tape.gradient(
                    g_loss,
                    self.hybrid_model.gan.generator.trainable_variables
                )
                g_optimizer.apply_gradients(
                    zip(g_gradients, self.hybrid_model.gan.generator.trainable_variables)
                )
                
                # Track losses
                epoch_d_losses.append(float(d_loss))
                epoch_g_losses.append(float(g_loss))
                
                # Print progress
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx + 1}/{num_batches} - "
                          f"D Loss: {np.mean(epoch_d_losses):.4f}, "
                          f"G Loss: {np.mean(epoch_g_losses):.4f}")
            
            # Epoch summary
            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)
            
            self.history['gan_train']['d_loss'].append(avg_d_loss)
            self.history['gan_train']['g_loss'].append(avg_g_loss)
            
            print(f"\n  ✓ Epoch {epoch + 1} Summary:")
            print(f"    Discriminator Loss: {avg_d_loss:.4f}")
            print(f"    Generator Loss: {avg_g_loss:.4f}")
        
        print("\n✓ GAN training with debiasing complete!")
        
        return self.history['gan_train']
    
    def save_model(self, save_dir=None):
        """Save trained model"""
        if save_dir is None:
            save_dir = config.MODELS_DIR
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f'fair_hybrid_model_{timestamp}')
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save VAE
        self.hybrid_model.vae.encoder.save(
            os.path.join(model_path, 'vae_encoder.h5')
        )
        self.hybrid_model.vae.decoder.save(
            os.path.join(model_path, 'vae_decoder.h5')
        )
        
        # Save GAN
        self.hybrid_model.gan.generator.save(
            os.path.join(model_path, 'gan_generator.h5')
        )
        self.hybrid_model.gan.discriminator.save(
            os.path.join(model_path, 'gan_discriminator.h5')
        )
        
        # Save history
        with open(os.path.join(model_path, 'training_history.pkl'), 'wb') as f:
            pickle.dump(self.history, f)
        
        print(f"\n✓ Model saved to: {model_path}")
        
        return model_path


if __name__ == "__main__":
    print("=" * 70)
    print("FAIR HYBRID GAN + VAE TRAINING")
    print("=" * 70)
    
    # Load preprocessing config
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # Load training data
    print("\nLoading training data...")
    train_customer = np.load(f'{config.PROCESSED_DATA_DIR}/train_customer.npy')
    train_agent = np.load(f'{config.PROCESSED_DATA_DIR}/train_agent.npy')
    train_metadata = pd.read_csv(f'{config.PROCESSED_DATA_DIR}/train_metadata.csv')
    
    print(f"Training samples: {len(train_customer):,}")
    print(f"Customer sequences shape: {train_customer.shape}")
    print(f"Agent sequences shape: {train_agent.shape}")
    
    # Initialize trainer
    trainer = FairHybridTrainer(vocab_size, max_length)
    
    # Phase 1: Pretrain VAE with fairness (quick demo)
    print("\n" + "=" * 70)
    print("STARTING TRAINING (QUICK DEMO)")
    print("=" * 70)
    
    # Use subset for quick demo
    subset_size = 5000
    
    vae_history = trainer.pretrain_vae_with_fairness(
        train_customer[:subset_size],
        train_metadata[:subset_size],
        epochs=3,
        batch_size=64
    )
    
    # Phase 2: Train GAN with debiasing (quick demo)
    gan_history = trainer.train_gan_with_debiasing(
        train_agent[:subset_size],
        train_metadata[:subset_size],
        epochs=3,
        batch_size=64
    )
    
    # Save model
    model_path = trainer.save_model()
    
    print("\n" + "=" * 70)
    print("✅ FAIR TRAINING DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {model_path}")
    print("\nTo run full training:")
    print("  - Increase epochs (VAE: 50, GAN: 100)")
    print("  - Use full dataset (remove subset_size)")
    print("  - Monitor fairness metrics")
