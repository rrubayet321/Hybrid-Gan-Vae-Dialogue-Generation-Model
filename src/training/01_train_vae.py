"""
Training script for VAE model
Trains the Variational Autoencoder on customer messages
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import config
from vae_model import build_vae
import os


class VAETrainer:
    """Handles VAE model training"""
    
    def __init__(self, vae_model):
        self.vae_model = vae_model
        self.history = None
        self.tokenizer = None
        
    def load_data(self):
        """Load preprocessed training and validation data"""
        print("Loading preprocessed data...")
        
        train_customer = np.load(f'{config.PROCESSED_DATA_DIR}/train_customer.npy')
        val_customer = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')
        
        # Load tokenizer for text generation
        with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print(f"Training samples: {len(train_customer)}")
        print(f"Validation samples: {len(val_customer)}")
        
        return train_customer, val_customer
    
    def train(self, train_data, val_data, epochs=None, batch_size=None):
        """
        Train the VAE model
        
        Args:
            train_data: Training sequences
            val_data: Validation sequences
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if epochs is None:
            epochs = config.VAE_EPOCHS
        if batch_size is None:
            batch_size = config.VAE_BATCH_SIZE
        
        print(f"\nTraining VAE for {epochs} epochs with batch size {batch_size}")
        
        # Callbacks
        callbacks = [
            # Model checkpointing
            keras.callbacks.ModelCheckpoint(
                filepath=f'{config.MODELS_DIR}/vae_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=f'{config.LOGS_DIR}/vae',
                histogram_freq=1
            )
        ]
        
        # Train model
        # Note: For VAE, input and output are the same (reconstruction)
        self.history = self.vae_model.vae.fit(
            train_data,
            train_data,  # Target is the same as input (reconstruction)
            validation_data=(val_data, val_data),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training complete!")
        
        return self.history
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('VAE Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot (if available)
        if 'lr' in self.history.history:
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def evaluate_reconstruction(self, data, num_samples=5):
        """
        Evaluate reconstruction quality on sample data
        
        Args:
            data: Input sequences
            num_samples: Number of samples to show
        """
        print(f"\nEvaluating reconstruction on {num_samples} samples...")
        
        # Get random samples
        indices = np.random.choice(len(data), num_samples, replace=False)
        samples = data[indices]
        
        # Reconstruct
        reconstructed = self.vae_model.reconstruct(samples)
        
        # Convert to text
        print("\n" + "="*60)
        print("Reconstruction Examples:")
        print("="*60)
        
        for i, (original, recon) in enumerate(zip(samples, reconstructed)):
            # Decode original
            original_text = self.tokenizer.sequences_to_texts([original.tolist()])[0]
            
            # Get most likely tokens from reconstruction
            recon_tokens = np.argmax(recon, axis=-1)
            recon_text = self.tokenizer.sequences_to_texts([recon_tokens.tolist()])[0]
            
            print(f"\nExample {i+1}:")
            print(f"Original:      {original_text}")
            print(f"Reconstructed: {recon_text}")
    
    def visualize_latent_space(self, data, num_samples=1000, save_path=None):
        """
        Visualize the learned latent space using t-SNE or PCA
        
        Args:
            data: Input sequences
            num_samples: Number of samples to visualize
        """
        print(f"\nVisualizing latent space with {num_samples} samples...")
        
        # Sample data
        if len(data) > num_samples:
            indices = np.random.choice(len(data), num_samples, replace=False)
            samples = data[indices]
        else:
            samples = data
        
        # Encode to latent space
        z_mean, _, _ = self.vae_model.encode(samples)
        
        # Use PCA for dimensionality reduction to 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_mean)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=10)
        plt.title('VAE Latent Space Visualization (PCA)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Latent space visualization saved to {save_path}")
        
        plt.show()
        
        print(f"Latent space variance explained: {pca.explained_variance_ratio_}")
    
    def generate_samples(self, num_samples=10, temperature=1.0):
        """
        Generate new samples from random latent vectors
        
        Args:
            num_samples: Number of samples to generate
            temperature: Controls randomness
        """
        print(f"\nGenerating {num_samples} samples from latent space...")
        
        # Sample from latent space
        latent_vectors = self.vae_model.sample_latent(num_samples)
        
        # Generate sequences
        generated = self.vae_model.generate_from_latent(latent_vectors, temperature)
        
        # Convert to tokens
        generated_tokens = np.argmax(generated, axis=-1)
        
        # Convert to text
        generated_texts = self.tokenizer.sequences_to_texts(generated_tokens.tolist())
        
        print("\n" + "="*60)
        print(f"Generated Samples (temperature={temperature}):")
        print("="*60)
        
        for i, text in enumerate(generated_texts):
            print(f"{i+1}. {text}")
        
        return generated_texts


def main():
    """Main training pipeline"""
    print("="*60)
    print("VAE Training Pipeline")
    print("="*60)
    
    # Set GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠️  No GPU available, using CPU")
    
    # Load preprocessing config
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # Build VAE model
    print("\n" + "="*60)
    print("Building VAE Model")
    print("="*60)
    vae_model = build_vae(vocab_size, max_length)
    
    # Create trainer
    trainer = VAETrainer(vae_model)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    train_data, val_data = trainer.load_data()
    
    # Train model
    print("\n" + "="*60)
    print("Training VAE")
    print("="*60)
    history = trainer.train(train_data, val_data)
    
    # Plot training history
    print("\n" + "="*60)
    print("Training Results")
    print("="*60)
    trainer.plot_training_history(f'{config.RESULTS_DIR}/vae_training_history.png')
    
    # Evaluate reconstruction
    print("\n" + "="*60)
    print("Evaluating Reconstruction")
    print("="*60)
    trainer.evaluate_reconstruction(val_data, num_samples=10)
    
    # Visualize latent space
    print("\n" + "="*60)
    print("Latent Space Visualization")
    print("="*60)
    trainer.visualize_latent_space(
        val_data, 
        num_samples=1000, 
        save_path=f'{config.RESULTS_DIR}/vae_latent_space.png'
    )
    
    # Generate samples
    print("\n" + "="*60)
    print("Generating Samples")
    print("="*60)
    trainer.generate_samples(num_samples=10, temperature=1.0)
    
    # Save model
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    vae_model.save_model(config.MODELS_DIR)
    
    print("\n" + "="*60)
    print("✅ VAE Training Complete!")
    print("="*60)
    print(f"Best model saved to: {config.MODELS_DIR}/vae_best.keras")
    print(f"Training history: {config.RESULTS_DIR}/vae_training_history.png")
    print(f"Latent space plot: {config.RESULTS_DIR}/vae_latent_space.png")


if __name__ == "__main__":
    main()
