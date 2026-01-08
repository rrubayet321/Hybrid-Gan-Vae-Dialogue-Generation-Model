"""
Generative Adversarial Network (GAN) Implementation for Text Generation
Generator creates agent responses from customer message latent vectors
Discriminator evaluates the quality and realism of generated responses
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, Dropout, 
    LeakyReLU, Embedding, Concatenate, Attention, LayerNormalization
)
import numpy as np
import config


class GANGenerator:
    """
    Generator Network: Creates agent responses from customer latent vectors
    
    Takes customer message latent representation (from VAE encoder) and generates
    realistic agent response sequences.
    """
    
    def __init__(self, latent_dim, vocab_size, max_length, 
                 embedding_dim=None, hidden_dim=None, dropout=None):
        """
        Initialize GAN Generator
        
        Args:
            latent_dim: Dimension of input latent space (from VAE)
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
        """
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim or config.GAN_EMBEDDING_DIM
        self.hidden_dim = hidden_dim or config.GAN_GENERATOR_HIDDEN_DIM
        self.dropout = dropout or config.GAN_DROPOUT
        
        self.generator = None
        self._build_generator()
    
    def _build_generator(self):
        """
        Build the Generator network with attention mechanism
        
        Architecture:
        1. Input: Latent vector (customer message encoding)
        2. Dense layers: Expand latent representation
        3. RepeatVector: Prepare for sequence generation
        4. LSTM layers: Generate sequence with temporal dependencies
        5. Attention: Focus on relevant parts of latent representation
        6. Output: Probability distribution over vocabulary
        """
        # Input: Latent vector from VAE encoder
        latent_input = Input(shape=(self.latent_dim,), name='generator_latent_input')
        
        # Expand latent representation
        x = Dense(self.hidden_dim, name='gen_dense_1')(latent_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        x = Dense(self.hidden_dim, name='gen_dense_2')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        # Repeat for sequence length
        x = layers.RepeatVector(self.max_length, name='gen_repeat')(x)
        
        # LSTM layers for sequence generation
        x = LSTM(self.hidden_dim, return_sequences=True, name='gen_lstm_1')(x)
        x = Dropout(self.dropout)(x)
        x = LayerNormalization()(x)
        
        x = LSTM(self.hidden_dim, return_sequences=True, name='gen_lstm_2')(x)
        x = Dropout(self.dropout)(x)
        x = LayerNormalization()(x)
        
        # Additional context layer
        x = LSTM(self.hidden_dim // 2, return_sequences=True, name='gen_lstm_3')(x)
        x = Dropout(self.dropout)(x)
        
        # Output layer - probability distribution over vocabulary
        output = Dense(
            self.vocab_size, 
            activation='softmax', 
            name='generator_output'
        )(x)
        
        # Create generator model
        self.generator = Model(latent_input, output, name='generator')
        
        print("Generator architecture:")
        self.generator.summary()
        
        return self.generator
    
    def generate(self, latent_vectors, temperature=1.0):
        """
        Generate agent responses from latent vectors
        
        Args:
            latent_vectors: Latent representations (batch, latent_dim)
            temperature: Controls randomness (higher = more random)
        
        Returns:
            Generated sequences (batch, max_length, vocab_size)
        """
        predictions = self.generator.predict(latent_vectors)
        
        if temperature != 1.0:
            # Apply temperature scaling
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions)
            predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)
        
        return predictions
    
    def sample_sequences(self, latent_vectors):
        """
        Sample discrete token sequences from generator output
        
        Args:
            latent_vectors: Latent representations
        
        Returns:
            Token sequences (batch, max_length)
        """
        predictions = self.generator.predict(latent_vectors)
        # Sample from categorical distribution
        sequences = np.argmax(predictions, axis=-1)
        return sequences


class GANDiscriminator:
    """
    Discriminator Network: Distinguishes real from generated agent responses
    
    Evaluates whether a given agent response is real (from dataset) or 
    fake (generated by the generator).
    """
    
    def __init__(self, vocab_size, max_length, embedding_dim=None, 
                 hidden_dim=None, dropout=None):
        """
        Initialize GAN Discriminator
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim or config.GAN_EMBEDDING_DIM
        self.hidden_dim = hidden_dim or config.GAN_DISCRIMINATOR_HIDDEN_DIM
        self.dropout = dropout or config.GAN_DROPOUT
        
        self.discriminator = None
        self._build_discriminator()
    
    def _build_discriminator(self):
        """
        Build the Discriminator network
        
        Architecture:
        1. Input: Token sequences (real or generated)
        2. Embedding: Convert tokens to dense vectors
        3. Bidirectional LSTM: Capture sequence patterns
        4. Dense layers: Extract features
        5. Output: Binary classification (real/fake)
        """
        # Input: Token sequence
        sequence_input = Input(shape=(self.max_length,), name='discriminator_input')
        
        # Embedding layer
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='disc_embedding'
        )(sequence_input)
        
        # Bidirectional LSTM for sequence processing
        x = Bidirectional(
            LSTM(self.hidden_dim, return_sequences=True, name='disc_lstm_1'),
            name='disc_bidirectional_1'
        )(x)
        x = Dropout(self.dropout)(x)
        x = LayerNormalization()(x)
        
        # Second LSTM layer
        x = Bidirectional(
            LSTM(self.hidden_dim // 2, return_sequences=False, name='disc_lstm_2'),
            name='disc_bidirectional_2'
        )(x)
        x = Dropout(self.dropout)(x)
        x = LayerNormalization()(x)
        
        # Dense layers for classification
        x = Dense(self.hidden_dim // 2, name='disc_dense_1')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.dropout)(x)
        
        x = Dense(self.hidden_dim // 4, name='disc_dense_2')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(self.dropout)(x)
        
        # Output: Binary classification (real=1, fake=0)
        output = Dense(1, activation='sigmoid', name='discriminator_output')(x)
        
        # Create discriminator model
        self.discriminator = Model(sequence_input, output, name='discriminator')
        
        print("\nDiscriminator architecture:")
        self.discriminator.summary()
        
        return self.discriminator
    
    def predict_authenticity(self, sequences):
        """
        Predict if sequences are real or fake
        
        Args:
            sequences: Token sequences (batch, max_length)
        
        Returns:
            Probabilities (batch,) - closer to 1 = more likely real
        """
        return self.discriminator.predict(sequences)


class GAN:
    """
    Complete GAN model combining Generator and Discriminator
    
    Manages the adversarial training process between generator and discriminator.
    """
    
    def __init__(self, generator, discriminator):
        """
        Initialize GAN
        
        Args:
            generator: GANGenerator instance
            discriminator: GANDiscriminator instance
        """
        self.generator_model = generator
        self.discriminator_model = discriminator
        
        self.generator = generator.generator
        self.discriminator = discriminator.discriminator
        
        self.gan = None
        self._build_gan()
    
    def _build_gan(self):
        """
        Build the combined GAN model
        
        For GAN training:
        - Generator tries to fool discriminator
        - Discriminator tries to distinguish real from fake
        """
        # Make discriminator non-trainable when training generator
        self.discriminator.trainable = False
        
        # Input: Latent vector
        latent_input = Input(shape=(self.generator_model.latent_dim,), name='gan_input')
        
        # Generate sequence
        generated_sequence = self.generator(latent_input)
        
        # Get argmax to convert probabilities to tokens for discriminator
        # Note: In practice, we use Gumbel-Softmax or similar for differentiability
        generated_tokens = layers.Lambda(
            lambda x: tf.argmax(x, axis=-1),
            name='token_sampling'
        )(generated_sequence)
        
        # Discriminator evaluation
        validity = self.discriminator(generated_tokens)
        
        # Create GAN model
        self.gan = Model(latent_input, validity, name='gan')
        
        print("\nComplete GAN architecture:")
        self.gan.summary()
        
        return self.gan
    
    def compile_models(self, 
                      discriminator_lr=None, 
                      generator_lr=None,
                      discriminator_optimizer=None,
                      generator_optimizer=None):
        """
        Compile both discriminator and GAN (generator)
        
        Args:
            discriminator_lr: Learning rate for discriminator
            generator_lr: Learning rate for generator
            discriminator_optimizer: Optional custom optimizer for discriminator
            generator_optimizer: Optional custom optimizer for generator
        """
        if discriminator_lr is None:
            discriminator_lr = config.GAN_LEARNING_RATE_D
        if generator_lr is None:
            generator_lr = config.GAN_LEARNING_RATE_G
        
        # Compile discriminator
        if discriminator_optimizer is None:
            discriminator_optimizer = keras.optimizers.Adam(
                learning_rate=discriminator_lr,
                beta_1=0.5
            )
        
        self.discriminator.trainable = True
        self.discriminator.compile(
            optimizer=discriminator_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Compile GAN (generator training)
        if generator_optimizer is None:
            generator_optimizer = keras.optimizers.Adam(
                learning_rate=generator_lr,
                beta_1=0.5
            )
        
        self.discriminator.trainable = False
        self.gan.compile(
            optimizer=generator_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n✓ Models compiled:")
        print(f"  Discriminator LR: {discriminator_lr}")
        print(f"  Generator LR: {generator_lr}")
    
    def train_discriminator(self, real_sequences, fake_sequences):
        """
        Train discriminator on real and fake sequences
        
        Args:
            real_sequences: Real agent responses
            fake_sequences: Generated agent responses
        
        Returns:
            Discriminator loss and accuracy
        """
        batch_size = len(real_sequences)
        
        # Labels for real and fake
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Train on real sequences
        d_loss_real = self.discriminator.train_on_batch(real_sequences, real_labels)
        
        # Train on fake sequences
        d_loss_fake = self.discriminator.train_on_batch(fake_sequences, fake_labels)
        
        # Average loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        return d_loss
    
    def train_generator(self, latent_vectors):
        """
        Train generator to fool discriminator
        
        Args:
            latent_vectors: Latent representations from VAE
        
        Returns:
            Generator loss
        """
        batch_size = len(latent_vectors)
        
        # Labels: Generator wants discriminator to think fake is real
        misleading_labels = np.ones((batch_size, 1))
        
        # Train generator
        g_loss = self.gan.train_on_batch(latent_vectors, misleading_labels)
        
        return g_loss
    
    def save_models(self, path):
        """Save generator and discriminator"""
        self.generator.save(f'{path}/gan_generator.keras')
        self.discriminator.save(f'{path}/gan_discriminator.keras')
        print(f"✓ GAN models saved to {path}/")
    
    def load_models(self, path):
        """Load generator and discriminator"""
        self.generator = keras.models.load_model(f'{path}/gan_generator.keras')
        self.discriminator = keras.models.load_model(f'{path}/gan_discriminator.keras')
        print(f"✓ GAN models loaded from {path}/")


def build_gan(vocab_size, max_length, latent_dim):
    """
    Convenience function to build complete GAN
    
    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length
        latent_dim: Dimension of latent space (from VAE)
    
    Returns:
        GAN instance with compiled models
    """
    print("Building GAN model...")
    
    # Build generator
    generator = GANGenerator(
        latent_dim=latent_dim,
        vocab_size=vocab_size,
        max_length=max_length
    )
    
    # Build discriminator
    discriminator = GANDiscriminator(
        vocab_size=vocab_size,
        max_length=max_length
    )
    
    # Build complete GAN
    gan = GAN(generator, discriminator)
    gan.compile_models()
    
    print("\n✓ GAN model built successfully!")
    
    return gan


if __name__ == "__main__":
    # Test GAN architecture
    print("Testing GAN architecture...")
    
    # Load preprocessing config
    import pickle
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    latent_dim = config.VAE_LATENT_DIM  # Use VAE latent dimension
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    print(f"Latent dimension: {latent_dim}")
    
    # Build GAN
    gan = build_gan(vocab_size, max_length, latent_dim)
    
    print("\n✅ GAN architecture built successfully!")
    print(f"Generator parameters: {gan.generator.count_params():,}")
    print(f"Discriminator parameters: {gan.discriminator.count_params():,}")
    print(f"Total GAN parameters: {gan.generator.count_params() + gan.discriminator.count_params():,}")
