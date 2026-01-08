"""
Configuration file for Hybrid GAN + VAE Text Generation Model
Contains all hyperparameters and settings for the project
"""

import os

# ===========================
# Data Configuration
# ===========================
DATA_PATH = 'synthetic_it_support_tickets.csv'
PROCESSED_DATA_DIR = 'processed_data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data preprocessing
TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation
MAX_VOCAB_SIZE = 10000  # Maximum vocabulary size
MAX_SEQUENCE_LENGTH = 100  # Maximum sequence length for padding
MIN_WORD_FREQUENCY = 2  # Minimum word frequency to be included in vocabulary
RANDOM_SEED = 42

# ===========================
# VAE Configuration
# ===========================
VAE_EMBEDDING_DIM = 128
VAE_LATENT_DIM = 256
VAE_HIDDEN_DIM = 512
VAE_DROPOUT = 0.3
VAE_LEARNING_RATE = 0.001
VAE_BATCH_SIZE = 64
VAE_EPOCHS = 50

# ===========================
# GAN Configuration
# ===========================
GAN_EMBEDDING_DIM = 128
GAN_GENERATOR_HIDDEN_DIM = 512
GAN_DISCRIMINATOR_HIDDEN_DIM = 256
GAN_DROPOUT = 0.3
GAN_LEARNING_RATE_G = 0.0002  # Generator learning rate
GAN_LEARNING_RATE_D = 0.0002  # Discriminator learning rate
GAN_BATCH_SIZE = 64
GAN_EPOCHS = 100
GAN_GRADIENT_PENALTY_WEIGHT = 10.0  # Weight for gradient penalty

# ===========================
# Hybrid Model Configuration
# ===========================
HYBRID_BATCH_SIZE = 64
HYBRID_EPOCHS = 100
HYBRID_LEARNING_RATE = 0.0001

# Loss weights for hybrid model
WEIGHT_RECONSTRUCTION = 1.0
WEIGHT_KL_DIVERGENCE = 0.1
WEIGHT_ADVERSARIAL = 1.0
WEIGHT_CONTENT_PRESERVATION = 0.5

# ===========================
# Bias Mitigation Configuration
# ===========================
BIAS_DETECTOR_HIDDEN_DIM = 128
BIAS_LEARNING_RATE = 0.001
BIAS_ADVERSARIAL_WEIGHT = 0.5
SENSITIVE_ATTRIBUTES = ['customer_segment', 'region']  # Attributes to debias

# Fairness thresholds
DEMOGRAPHIC_PARITY_THRESHOLD = 0.1
EQUALIZED_ODDS_THRESHOLD = 0.1
DISPARATE_IMPACT_THRESHOLD = 0.8

# ===========================
# XAI Configuration
# ===========================
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP analysis
ATTENTION_VISUALIZATION = True
FEATURE_IMPORTANCE_TOP_K = 20  # Top K important features to display

# ===========================
# Evaluation Configuration
# ===========================
BLEU_MAX_N = 4  # Maximum n-gram for BLEU score
EVALUATION_SAMPLE_SIZE = 1000  # Number of samples for evaluation

# ===========================
# Training Configuration
# ===========================
EARLY_STOPPING_PATIENCE = 10
CHECKPOINT_SAVE_FREQ = 5  # Save checkpoint every N epochs
GPU_MEMORY_LIMIT = 0.8  # Limit GPU memory usage (0.8 = 80%)
