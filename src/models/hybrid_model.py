"""
Hybrid GAN + VAE Model for Customer-Agent Dialogue Generation

This model combines:
1. VAE Encoder: Encodes customer messages to latent space
2. GAN Generator: Generates agent responses from latent vectors
3. GAN Discriminator: Evaluates response quality
4. VAE Decoder: Reconstructs customer messages (regularization)

The hybrid model learns to generate realistic agent responses given customer messages.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input
import numpy as np

import config
from vae_model import VAEModel
from gan_model import GANGenerator, GANDiscriminator


class HybridGANVAE:
    """
    Hybrid GAN + VAE Model
    
    Architecture:
    Customer Message → VAE Encoder → Latent Space → GAN Generator → Agent Response
                                ↓
                         VAE Decoder (reconstruction regularization)
                                ↓
                    GAN Discriminator (evaluates response quality)
    """
    
    def __init__(self, vocab_size, max_length, 
                 vae_latent_dim=None, 
                 vae_config=None,
                 gan_config=None):
        """
        Initialize Hybrid GAN + VAE model
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            vae_latent_dim: Dimension of VAE latent space
            vae_config: Optional VAE configuration dict
            gan_config: Optional GAN configuration dict
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vae_latent_dim = vae_latent_dim or config.VAE_LATENT_DIM
        
        self.vae_config = vae_config or {}
        self.gan_config = gan_config or {}
        
        # Build components
        self.vae = None
        self.gan_generator = None
        self.gan_discriminator = None
        self.hybrid_model = None
        
        self._build_components()
        self._build_hybrid_model()
    
    def _build_components(self):
        """Build VAE and GAN components"""
        print("Building Hybrid GAN + VAE components...")
        
        # Build VAE
        print("\n1. Building VAE...")
        self.vae = VAEModel(
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            embedding_dim=self.vae_config.get('embedding_dim', config.VAE_EMBEDDING_DIM),
            latent_dim=self.vae_latent_dim,
            hidden_dim=self.vae_config.get('hidden_dim', config.VAE_HIDDEN_DIM),
            dropout=self.vae_config.get('dropout', config.VAE_DROPOUT)
        )
        
        # Build GAN Generator
        print("\n2. Building GAN Generator...")
        self.gan_generator = GANGenerator(
            latent_dim=self.vae_latent_dim,
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            embedding_dim=self.gan_config.get('embedding_dim', config.GAN_EMBEDDING_DIM),
            hidden_dim=self.gan_config.get('hidden_dim', config.GAN_GENERATOR_HIDDEN_DIM),
            dropout=self.gan_config.get('dropout', config.GAN_DROPOUT)
        )
        
        # Build GAN Discriminator
        print("\n3. Building GAN Discriminator...")
        self.gan_discriminator = GANDiscriminator(
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            embedding_dim=self.gan_config.get('embedding_dim', config.GAN_EMBEDDING_DIM),
            hidden_dim=self.gan_config.get('hidden_dim', config.GAN_DISCRIMINATOR_HIDDEN_DIM),
            dropout=self.gan_config.get('dropout', config.GAN_DROPOUT)
        )
        
        print("\n✓ All components built successfully!")
    
    def _build_hybrid_model(self):
        """
        Build the complete hybrid model
        
        Flow:
        1. Customer message → VAE Encoder → Latent vector
        2. Latent vector → GAN Generator → Generated agent response
        3. Generated response → Discriminator → Authenticity score
        4. Latent vector → VAE Decoder → Reconstructed customer message
        """
        print("\n4. Building Hybrid Model...")
        
        # Input: Customer message
        customer_input = Input(shape=(self.max_length,), name='customer_message_input')
        
        # VAE Encoding
        z_mean, z_log_var, z = self.vae.encoder(customer_input)
        
        # GAN Generation (agent response)
        generated_response = self.gan_generator.generator(z)
        
        # VAE Reconstruction (regularization)
        reconstructed_customer = self.vae.decoder(z)
        
        # Build hybrid model
        self.hybrid_model = Model(
            inputs=customer_input,
            outputs={
                'generated_response': generated_response,
                'reconstructed_customer': reconstructed_customer,
                'latent_mean': z_mean,
                'latent_log_var': z_log_var,
                'latent_sample': z
            },
            name='hybrid_gan_vae'
        )
        
        print("\n✓ Hybrid model built!")
        print("\nHybrid Model Summary:")
        self.hybrid_model.summary()
        
        return self.hybrid_model
    
    def generate_response(self, customer_messages):
        """
        Generate agent responses from customer messages
        
        Args:
            customer_messages: Customer message sequences (batch, max_length)
        
        Returns:
            Generated agent responses (batch, max_length, vocab_size)
        """
        outputs = self.hybrid_model.predict(customer_messages)
        return outputs['generated_response']
    
    def encode_customer_messages(self, customer_messages):
        """
        Encode customer messages to latent space
        
        Args:
            customer_messages: Customer message sequences
        
        Returns:
            z_mean, z_log_var, z (latent representations)
        """
        z_mean, z_log_var, z = self.vae.encoder.predict(customer_messages)
        return z_mean, z_log_var, z
    
    def decode_latent_vectors(self, latent_vectors, mode='agent'):
        """
        Decode latent vectors to sequences
        
        Args:
            latent_vectors: Latent representations
            mode: 'agent' for agent responses, 'customer' for customer messages
        
        Returns:
            Decoded sequences
        """
        if mode == 'agent':
            return self.gan_generator.generator.predict(latent_vectors)
        elif mode == 'customer':
            return self.vae.decoder.predict(latent_vectors)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'agent' or 'customer'")
    
    def evaluate_response_quality(self, responses):
        """
        Evaluate quality of agent responses using discriminator
        
        Args:
            responses: Agent response sequences (batch, max_length)
        
        Returns:
            Quality scores (batch,) - higher = better quality
        """
        # Convert probabilities to tokens if needed
        if len(responses.shape) == 3:  # (batch, max_length, vocab_size)
            responses = np.argmax(responses, axis=-1)
        
        return self.gan_discriminator.discriminator.predict(responses)
    
    def compile_hybrid_model(self, learning_rate=None):
        """
        Compile the hybrid model with custom training objectives
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if learning_rate is None:
            learning_rate = config.HYBRID_LEARNING_RATE
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Define losses for each output
        losses = {
            'generated_response': 'sparse_categorical_crossentropy',
            'reconstructed_customer': 'sparse_categorical_crossentropy',
        }
        
        # Loss weights from config
        loss_weights = {
            'generated_response': config.WEIGHT_ADVERSARIAL,
            'reconstructed_customer': config.WEIGHT_RECONSTRUCTION,
        }
        
        self.hybrid_model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights
        )
        
        print(f"\n✓ Hybrid model compiled with learning rate: {learning_rate}")
        print(f"  Loss weights: {loss_weights}")
    
    def get_model_summary(self):
        """Get summary of all model components"""
        summary = {
            'vae_encoder_params': self.vae.encoder.count_params(),
            'vae_decoder_params': self.vae.decoder.count_params(),
            'gan_generator_params': self.gan_generator.generator.count_params(),
            'gan_discriminator_params': self.gan_discriminator.discriminator.count_params(),
            'total_params': (
                self.vae.encoder.count_params() +
                self.vae.decoder.count_params() +
                self.gan_generator.generator.count_params() +
                self.gan_discriminator.discriminator.count_params()
            )
        }
        return summary
    
    def save_models(self, path):
        """Save all model components"""
        print(f"\nSaving Hybrid GAN + VAE models to {path}/...")
        
        # Save VAE components
        self.vae.save_model(path)
        
        # Save GAN components
        self.gan_generator.generator.save(f'{path}/gan_generator.keras')
        self.gan_discriminator.discriminator.save(f'{path}/gan_discriminator.keras')
        
        # Save hybrid model
        self.hybrid_model.save(f'{path}/hybrid_model.keras')
        
        print("✓ All models saved successfully!")
    
    def load_models(self, path):
        """Load all model components"""
        print(f"\nLoading Hybrid GAN + VAE models from {path}/...")
        
        # Load VAE components
        self.vae.load_model(path)
        
        # Load GAN components
        self.gan_generator.generator = keras.models.load_model(f'{path}/gan_generator.keras')
        self.gan_discriminator.discriminator = keras.models.load_model(f'{path}/gan_discriminator.keras')
        
        # Load hybrid model
        self.hybrid_model = keras.models.load_model(f'{path}/hybrid_model.keras')
        
        print("✓ All models loaded successfully!")


def build_hybrid_model(vocab_size, max_length):
    """
    Convenience function to build complete Hybrid GAN + VAE model
    
    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length
    
    Returns:
        HybridGANVAE instance
    """
    print("\n" + "=" * 70)
    print("BUILDING HYBRID GAN + VAE MODEL")
    print("=" * 70)
    
    hybrid = HybridGANVAE(vocab_size=vocab_size, max_length=max_length)
    
    # Print model summary
    summary = hybrid.get_model_summary()
    
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"VAE Encoder parameters:      {summary['vae_encoder_params']:,}")
    print(f"VAE Decoder parameters:      {summary['vae_decoder_params']:,}")
    print(f"GAN Generator parameters:    {summary['gan_generator_params']:,}")
    print(f"GAN Discriminator parameters: {summary['gan_discriminator_params']:,}")
    print(f"{'─' * 70}")
    print(f"TOTAL parameters:            {summary['total_params']:,}")
    print("=" * 70)
    
    return hybrid


if __name__ == "__main__":
    # Test Hybrid model architecture
    print("Testing Hybrid GAN + VAE architecture...")
    
    # Load preprocessing config
    import pickle
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # Build Hybrid model
    hybrid = build_hybrid_model(vocab_size, max_length)
    
    # Test with dummy data
    print("\n" + "=" * 70)
    print("TESTING WITH DUMMY DATA")
    print("=" * 70)
    
    dummy_customer = np.random.randint(0, vocab_size, size=(5, max_length))
    print(f"\nDummy customer messages shape: {dummy_customer.shape}")
    
    # Test encoding
    print("\n1. Testing customer message encoding...")
    z_mean, z_log_var, z = hybrid.encode_customer_messages(dummy_customer)
    print(f"   ✓ Latent vectors shape: {z.shape}")
    
    # Test agent response generation
    print("\n2. Testing agent response generation...")
    agent_responses = hybrid.decode_latent_vectors(z, mode='agent')
    print(f"   ✓ Generated responses shape: {agent_responses.shape}")
    
    # Test customer reconstruction
    print("\n3. Testing customer message reconstruction...")
    reconstructed = hybrid.decode_latent_vectors(z, mode='customer')
    print(f"   ✓ Reconstructed messages shape: {reconstructed.shape}")
    
    # Test hybrid model forward pass
    print("\n4. Testing hybrid model forward pass...")
    outputs = hybrid.hybrid_model.predict(dummy_customer, verbose=0)
    print(f"   ✓ Generated response shape: {outputs['generated_response'].shape}")
    print(f"   ✓ Reconstructed customer shape: {outputs['reconstructed_customer'].shape}")
    print(f"   ✓ Latent mean shape: {outputs['latent_mean'].shape}")
    
    # Test discriminator evaluation
    print("\n5. Testing discriminator evaluation...")
    response_tokens = np.argmax(agent_responses, axis=-1)
    quality_scores = hybrid.evaluate_response_quality(response_tokens)
    print(f"   ✓ Quality scores shape: {quality_scores.shape}")
    print(f"   ✓ Quality scores: {quality_scores.flatten()[:5]}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nHybrid GAN + VAE model is ready for training!")
