"""
Variational Autoencoder (VAE) Implementation for Text Generation
Encodes customer messages into latent space and reconstructs them
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Lambda, Dropout
import tensorflow.keras.backend as K
import numpy as np
import config


class VAEModel:
    """Variational Autoencoder for text encoding and generation"""
    
    def __init__(self, vocab_size, max_length, embedding_dim=None, 
                 latent_dim=None, hidden_dim=None, dropout=None):
        """
        Initialize VAE model
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimension of word embeddings
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim or config.VAE_EMBEDDING_DIM
        self.latent_dim = latent_dim or config.VAE_LATENT_DIM
        self.hidden_dim = hidden_dim or config.VAE_HIDDEN_DIM
        self.dropout = dropout or config.VAE_DROPOUT
        
        # Build the model components
        self.encoder = None
        self.decoder = None
        self.vae = None
        
        self._build_encoder()
        self._build_decoder()
        self._build_vae()
    
    def _build_encoder(self):
        """Build the encoder network"""
        # Input layer
        encoder_inputs = Input(shape=(self.max_length,), name='encoder_input')
        
        # Embedding layer
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        
        # Bidirectional LSTM for better context understanding
        x = Bidirectional(
            LSTM(self.hidden_dim, return_sequences=True, name='encoder_lstm_1'),
            name='encoder_bidirectional'
        )(x)
        
        x = Dropout(self.dropout)(x)
        
        # Second LSTM to get final encoding
        encoded = LSTM(self.hidden_dim, return_sequences=False, name='encoder_lstm_2')(x)
        
        x = Dropout(self.dropout)(encoded)
        
        # Latent space parameters
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling layer using reparameterization trick
        z = Lambda(self._sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        # Create encoder model
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        print("Encoder architecture:")
        self.encoder.summary()
        
        return self.encoder
    
    def _sampling(self, args):
        """
        Reparameterization trick: sample from latent space
        z = mean + std * epsilon
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        
        # Sample epsilon from standard normal distribution
        epsilon = K.random_normal(shape=(batch, dim))
        
        # Return sampled point
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def _build_decoder(self):
        """Build the decoder network"""
        # Latent input
        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        
        # Expand to sequence
        x = Dense(self.hidden_dim, activation='relu', name='decoder_dense')(latent_inputs)
        x = Dropout(self.dropout)(x)
        
        # Repeat to match sequence length
        x = layers.RepeatVector(self.max_length, name='decoder_repeat')(x)
        
        # LSTM layers for sequence generation
        x = LSTM(self.hidden_dim, return_sequences=True, name='decoder_lstm_1')(x)
        x = Dropout(self.dropout)(x)
        
        x = LSTM(self.hidden_dim, return_sequences=True, name='decoder_lstm_2')(x)
        x = Dropout(self.dropout)(x)
        
        # Output layer - probability distribution over vocabulary
        decoder_outputs = Dense(
            self.vocab_size, 
            activation='softmax', 
            name='decoder_output'
        )(x)
        
        # Create decoder model
        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')
        
        print("\nDecoder architecture:")
        self.decoder.summary()
        
        return self.decoder
    
    def _build_vae(self):
        """Build the complete VAE model as a custom model class"""
        
        # Create a custom VAE model class
        class VAECustom(Model):
            def __init__(self, encoder, decoder, **kwargs):
                super(VAECustom, self).__init__(**kwargs)
                self.encoder_model = encoder
                self.decoder_model = decoder
                self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
                self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
                self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            
            @property
            def metrics(self):
                return [
                    self.total_loss_tracker,
                    self.reconstruction_loss_tracker,
                    self.kl_loss_tracker,
                ]
            
            def call(self, inputs):
                z_mean, z_log_var, z = self.encoder_model(inputs)
                reconstruction = self.decoder_model(z)
                return reconstruction
            
            def train_step(self, data):
                if isinstance(data, tuple):
                    data = data[0]
                
                with tf.GradientTape() as tape:
                    # Forward pass
                    z_mean, z_log_var, z = self.encoder_model(data)
                    reconstruction = self.decoder_model(z)
                    
                    # Reconstruction loss
                    reconstruction_loss = keras.losses.sparse_categorical_crossentropy(data, reconstruction)
                    reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=-1)
                    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
                    
                    # KL divergence loss
                    kl_loss = -0.5 * tf.reduce_sum(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                        axis=-1
                    )
                    kl_loss = tf.reduce_mean(kl_loss)
                    
                    # Total loss
                    total_loss = reconstruction_loss + kl_loss * config.WEIGHT_KL_DIVERGENCE
                
                # Compute gradients
                trainable_vars = self.trainable_variables
                gradients = tape.gradient(total_loss, trainable_vars)
                
                # Update weights
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                
                # Update metrics
                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.kl_loss_tracker.update_state(kl_loss)
                
                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                }
            
            def test_step(self, data):
                if isinstance(data, tuple):
                    data = data[0]
                
                # Forward pass
                z_mean, z_log_var, z = self.encoder_model(data)
                reconstruction = self.decoder_model(z)
                
                # Reconstruction loss
                reconstruction_loss = keras.losses.sparse_categorical_crossentropy(data, reconstruction)
                reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=-1)
                reconstruction_loss = tf.reduce_mean(reconstruction_loss)
                
                # KL divergence loss
                kl_loss = -0.5 * tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=-1
                )
                kl_loss = tf.reduce_mean(kl_loss)
                
                # Total loss
                total_loss = reconstruction_loss + kl_loss * config.WEIGHT_KL_DIVERGENCE
                
                # Update metrics
                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.kl_loss_tracker.update_state(kl_loss)
                
                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                }
        
        # Create the custom VAE model
        self.vae = VAECustom(self.encoder, self.decoder, name='vae')
        
        # Build the model by calling it with sample input
        sample_input = tf.zeros((1, self.max_length), dtype=tf.int32)
        _ = self.vae(sample_input)
        
        print("\nComplete VAE architecture:")
        self.vae.summary()
        
        return self.vae
    
    def compile_model(self, learning_rate=None):
        """Compile the VAE model"""
        if learning_rate is None:
            learning_rate = config.VAE_LEARNING_RATE
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.vae.compile(optimizer=optimizer)
        
        print(f"\nVAE model compiled with learning rate: {learning_rate}")
    
    def reconstruct(self, sequences):
        """Reconstruct sequences from input"""
        return self.vae.predict(sequences)
    
    def encode(self, sequences):
        """Encode sequences to latent space"""
        z_mean, z_log_var, z = self.encoder.predict(sequences)
        return z_mean, z_log_var, z
    
    def decode(self, latent_vectors):
        """Decode latent vectors to sequences"""
        return self.decoder.predict(latent_vectors)
    
    def generate_from_latent(self, latent_vectors, temperature=1.0):
        """
        Generate sequences from latent vectors with temperature control
        
        Args:
            latent_vectors: Latent space vectors
            temperature: Controls randomness (higher = more random)
        """
        predictions = self.decoder.predict(latent_vectors)
        
        if temperature != 1.0:
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions)
            predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)
        
        return predictions
    
    def sample_latent(self, num_samples):
        """Sample random points from latent space"""
        return np.random.normal(0, 1, size=(num_samples, self.latent_dim))
    
    def save_model(self, path):
        """Save the VAE model"""
        self.vae.save(f'{path}/vae_full.keras')
        self.encoder.save(f'{path}/vae_encoder.keras')
        self.decoder.save(f'{path}/vae_decoder.keras')
        print(f"VAE model saved to {path}/")
    
    def load_model(self, path):
        """Load the VAE model"""
        self.vae = keras.models.load_model(
            f'{path}/vae_full.keras',
            custom_objects={'vae_loss': self.vae_loss}
        )
        self.encoder = keras.models.load_model(f'{path}/vae_encoder.keras')
        self.decoder = keras.models.load_model(f'{path}/vae_decoder.keras')
        print(f"VAE model loaded from {path}/")


def build_vae(vocab_size, max_length):
    """
    Convenience function to build and compile a VAE model
    
    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length
    
    Returns:
        Compiled VAE model instance
    """
    vae_model = VAEModel(vocab_size, max_length)
    vae_model.compile_model()
    return vae_model


if __name__ == "__main__":
    # Test VAE architecture
    print("Testing VAE architecture...")
    
    # Load preprocessing config
    import pickle
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # Build VAE
    vae_model = build_vae(vocab_size, max_length)
    
    print("\n✅ VAE architecture built successfully!")
    print(f"Latent dimension: {vae_model.latent_dim}")
    print(f"Hidden dimension: {vae_model.hidden_dim}")
    print(f"Embedding dimension: {vae_model.embedding_dim}")
