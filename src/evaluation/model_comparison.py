"""
Compare Combined vs Separate Input Approaches for Dialogue Generation

This script implements and compares two input processing strategies:
1. Combined Input: Customer + Agent messages processed together as single sequence
2. Separate Input: Customer and Agent messages processed independently

Evaluation includes: BLEU, ROUGE, Perplexity, Diversity, Repetition metrics
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow import keras

import config
from diversity_metrics import DiversityMetrics, compare_model_diversity
from evaluation_metrics import TextGenerationEvaluator
from hybrid_model import HybridGANVAE
from vae_model import build_vae


class SeparateInputHybridGANVAE:
    """
    Hybrid GAN + VAE with SEPARATE input processing.
    Customer and Agent messages processed independently.
    """
    
    def __init__(self, vocab_size, max_length, latent_dim=256):
        """
        Initialize Separate Input Hybrid model.
        
        Args:
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            latent_dim: Latent space dimension
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.latent_dim = latent_dim
        
        print("\n" + "="*80)
        print("BUILDING SEPARATE INPUT HYBRID GAN + VAE")
        print("="*80)
        
        # Build separate encoders for customer and agent
        self._build_separate_encoders()
        
        # Build shared decoder
        self._build_decoder()
        
        # Build GAN components
        self._build_generator()
        self._build_discriminator()
        
        # Build hybrid model
        self._build_hybrid_model()
        
        print("\n✓ Separate Input Hybrid model built successfully!")
    
    def _build_separate_encoders(self):
        """Build separate encoders for customer and agent messages."""
        print("\n1. Building Separate Encoders...")
        
        # Customer encoder
        customer_input = keras.layers.Input(shape=(self.max_length,), name='customer_input')
        
        # Embedding
        customer_embed = keras.layers.Embedding(
            self.vocab_size, 128, mask_zero=True, name='customer_embedding'
        )(customer_input)
        
        # Mask for padded sequences
        customer_mask = keras.layers.Lambda(
            lambda x: tf.not_equal(x, 0), name='customer_mask'
        )(customer_input)
        
        # Bidirectional LSTM
        customer_lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(512, return_sequences=True, dropout=0.3),
            name='customer_bidirectional'
        )(customer_embed, mask=customer_mask)
        
        customer_lstm = keras.layers.Dropout(0.3)(customer_lstm)
        
        # Second LSTM
        customer_encoded = keras.layers.LSTM(
            512, return_sequences=False, dropout=0.3, name='customer_lstm_2'
        )(customer_lstm, mask=customer_mask)
        
        customer_encoded = keras.layers.Dropout(0.3)(customer_encoded)
        
        # Latent space parameters
        customer_z_mean = keras.layers.Dense(self.latent_dim, name='customer_z_mean')(customer_encoded)
        customer_z_log_var = keras.layers.Dense(self.latent_dim, name='customer_z_log_var')(customer_encoded)
        
        # Sampling layer
        customer_z = keras.layers.Lambda(
            self._sampling, name='customer_z'
        )([customer_z_mean, customer_z_log_var])
        
        self.customer_encoder = keras.Model(
            customer_input,
            [customer_z_mean, customer_z_log_var, customer_z],
            name='customer_encoder'
        )
        
        print("  ✓ Customer encoder built")
        
        # Agent encoder (optional - for training only)
        agent_input = keras.layers.Input(shape=(self.max_length,), name='agent_input')
        
        agent_embed = keras.layers.Embedding(
            self.vocab_size, 128, mask_zero=True, name='agent_embedding'
        )(agent_input)
        
        agent_mask = keras.layers.Lambda(
            lambda x: tf.not_equal(x, 0), name='agent_mask'
        )(agent_input)
        
        agent_lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(512, return_sequences=True, dropout=0.3),
            name='agent_bidirectional'
        )(agent_embed, mask=agent_mask)
        
        agent_lstm = keras.layers.Dropout(0.3)(agent_lstm)
        
        agent_encoded = keras.layers.LSTM(
            512, return_sequences=False, dropout=0.3, name='agent_lstm_2'
        )(agent_lstm, mask=agent_mask)
        
        agent_encoded = keras.layers.Dropout(0.3)(agent_encoded)
        
        agent_z_mean = keras.layers.Dense(self.latent_dim, name='agent_z_mean')(agent_encoded)
        agent_z_log_var = keras.layers.Dense(self.latent_dim, name='agent_z_log_var')(agent_encoded)
        
        agent_z = keras.layers.Lambda(
            self._sampling, name='agent_z'
        )([agent_z_mean, agent_z_log_var])
        
        self.agent_encoder = keras.Model(
            agent_input,
            [agent_z_mean, agent_z_log_var, agent_z],
            name='agent_encoder'
        )
        
        print("  ✓ Agent encoder built")
    
    def _sampling(self, args):
        """Reparameterization trick."""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def _build_decoder(self):
        """Build decoder (shared for reconstruction)."""
        print("\n2. Building Decoder...")
        
        latent_input = keras.layers.Input(shape=(self.latent_dim,), name='decoder_latent_input')
        
        # Dense layer
        decoded = keras.layers.Dense(512, activation='relu', name='decoder_dense')(latent_input)
        decoded = keras.layers.Dropout(0.3)(decoded)
        
        # Repeat for sequence
        decoded = keras.layers.RepeatVector(self.max_length, name='decoder_repeat')(decoded)
        
        # LSTM layers
        decoded = keras.layers.LSTM(512, return_sequences=True, dropout=0.3, name='decoder_lstm_1')(decoded)
        decoded = keras.layers.Dropout(0.3)(decoded)
        
        decoded = keras.layers.LSTM(512, return_sequences=True, dropout=0.3, name='decoder_lstm_2')(decoded)
        decoded = keras.layers.Dropout(0.3)(decoded)
        
        # Output
        decoder_output = keras.layers.Dense(
            self.vocab_size, activation='softmax', name='decoder_output'
        )(decoded)
        
        self.decoder = keras.Model(latent_input, decoder_output, name='decoder')
        print("  ✓ Decoder built")
    
    def _build_generator(self):
        """Build GAN generator."""
        print("\n3. Building Generator...")
        
        latent_input = keras.layers.Input(shape=(self.latent_dim,), name='generator_latent_input')
        
        # Dense layers with layer norm
        gen = keras.layers.Dense(512, name='gen_dense_1')(latent_input)
        gen = keras.layers.LeakyReLU(alpha=0.2)(gen)
        gen = keras.layers.LayerNormalization()(gen)
        gen = keras.layers.Dropout(0.3)(gen)
        
        gen = keras.layers.Dense(512, name='gen_dense_2')(gen)
        gen = keras.layers.LeakyReLU(alpha=0.2)(gen)
        gen = keras.layers.LayerNormalization()(gen)
        gen = keras.layers.Dropout(0.3)(gen)
        
        # Repeat for sequence
        gen = keras.layers.RepeatVector(self.max_length, name='gen_repeat')(gen)
        
        # LSTM layers
        gen = keras.layers.LSTM(512, return_sequences=True, dropout=0.3, name='gen_lstm_1')(gen)
        gen = keras.layers.Dropout(0.3)(gen)
        gen = keras.layers.LayerNormalization()(gen)
        
        gen = keras.layers.LSTM(512, return_sequences=True, dropout=0.3, name='gen_lstm_2')(gen)
        gen = keras.layers.Dropout(0.3)(gen)
        gen = keras.layers.LayerNormalization()(gen)
        
        gen = keras.layers.LSTM(256, return_sequences=True, dropout=0.3, name='gen_lstm_3')(gen)
        gen = keras.layers.Dropout(0.3)(gen)
        
        # Output
        generator_output = keras.layers.Dense(
            self.vocab_size, activation='softmax', name='generator_output'
        )(gen)
        
        self.generator = keras.Model(latent_input, generator_output, name='generator')
        print("  ✓ Generator built")
    
    def _build_discriminator(self):
        """Build GAN discriminator."""
        print("\n4. Building Discriminator...")
        
        sequence_input = keras.layers.Input(shape=(self.max_length,), name='discriminator_input')
        
        # Embedding
        disc = keras.layers.Embedding(
            self.vocab_size, 128, mask_zero=True, name='disc_embedding'
        )(sequence_input)
        
        # Mask
        mask = keras.layers.Lambda(lambda x: tf.not_equal(x, 0))(sequence_input)
        
        # Bidirectional LSTM
        disc = keras.layers.Bidirectional(
            keras.layers.LSTM(256, return_sequences=True, dropout=0.3),
            name='disc_bidirectional_1'
        )(disc, mask=mask)
        disc = keras.layers.Dropout(0.3)(disc)
        disc = keras.layers.LayerNormalization()(disc)
        
        disc = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=False, dropout=0.3),
            name='disc_bidirectional_2'
        )(disc, mask=mask)
        disc = keras.layers.Dropout(0.3)(disc)
        disc = keras.layers.LayerNormalization()(disc)
        
        # Dense layers
        disc = keras.layers.Dense(128, name='disc_dense_1')(disc)
        disc = keras.layers.LeakyReLU(alpha=0.2)(disc)
        disc = keras.layers.Dropout(0.3)(disc)
        
        disc = keras.layers.Dense(64, name='disc_dense_2')(disc)
        disc = keras.layers.LeakyReLU(alpha=0.2)(disc)
        disc = keras.layers.Dropout(0.3)(disc)
        
        # Output
        discriminator_output = keras.layers.Dense(1, activation='sigmoid', name='discriminator_output')(disc)
        
        self.discriminator = keras.Model(sequence_input, discriminator_output, name='discriminator')
        print("  ✓ Discriminator built")
    
    def _build_hybrid_model(self):
        """Build hybrid model with separate inputs."""
        print("\n5. Building Hybrid Model...")
        
        # Inputs
        customer_input = keras.layers.Input(shape=(self.max_length,), name='customer_message_input')
        
        # Encode customer message only
        z_mean, z_log_var, z = self.customer_encoder(customer_input)
        
        # Generate response
        generated_response = self.generator(z)
        
        # Reconstruct customer message (for VAE loss)
        reconstructed_customer = self.decoder(z)
        
        # Create model
        self.hybrid_model = keras.Model(
            inputs=customer_input,
            outputs={
                'z_mean': z_mean,
                'z_log_var': z_log_var,
                'z': z,
                'generated_response': generated_response,
                'reconstructed_customer': reconstructed_customer
            },
            name='separate_input_hybrid_gan_vae'
        )
        
        print("  ✓ Hybrid model built with separate inputs!")


class InputApproachComparator:
    """
    Compare Combined vs Separate input approaches for dialogue generation.
    """
    
    def __init__(self, tokenizer, val_customer, val_agent):
        """
        Initialize comparator.
        
        Args:
            tokenizer: SimpleTokenizer instance
            val_customer: Validation customer messages
            val_agent: Validation agent responses
        """
        self.tokenizer = tokenizer
        self.val_customer = val_customer
        self.val_agent = val_agent
        self.vocab_size = len(tokenizer.word_index) + 1
        
        print("\n" + "="*80)
        print("INPUT APPROACH COMPARATOR")
        print("="*80)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Validation samples: {len(val_customer):,}")
        
        # Initialize evaluators
        self.text_evaluator = TextGenerationEvaluator(tokenizer)
        self.diversity_calculator = DiversityMetrics(tokenizer)
        
        # Build models
        self.combined_model = None
        self.separate_model = None
        
        # Results storage
        self.results = {
            'combined': {},
            'separate': {}
        }
    
    def build_models(self):
        """Build both combined and separate input models."""
        print("\n" + "="*80)
        print("BUILDING MODELS")
        print("="*80)
        
        # Combined input model (current approach)
        print("\n📦 Building Combined Input Model...")
        self.combined_model = HybridGANVAE(
            self.vocab_size,
            config.MAX_SEQUENCE_LENGTH
        )
        print("✓ Combined input model built")
        
        # Separate input model (new approach)
        print("\n📦 Building Separate Input Model...")
        self.separate_model = SeparateInputHybridGANVAE(
            self.vocab_size,
            config.MAX_SEQUENCE_LENGTH
        )
        print("✓ Separate input model built")
    
    def generate_responses(self, num_samples=500):
        """
        Generate responses using both approaches.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Tuple of (combined_texts, separate_texts, reference_texts, inputs)
        """
        print("\n" + "="*80)
        print(f"GENERATING RESPONSES ({num_samples} samples)")
        print("="*80)
        
        sample_inputs = self.val_customer[:num_samples]
        sample_references = self.val_agent[:num_samples]
        
        # Generate with combined approach
        print("\n1. Generating with Combined Input approach...")
        combined_predictions = self.combined_model.hybrid_model.predict(
            sample_inputs, verbose=0
        )
        
        if isinstance(combined_predictions, dict):
            combined_logits = combined_predictions['generated_response']
        elif isinstance(combined_predictions, (list, tuple)):
            combined_logits = combined_predictions[1]
        else:
            combined_logits = combined_predictions
        
        combined_sequences = np.argmax(combined_logits, axis=-1)
        combined_texts = self.tokenizer.sequences_to_texts(combined_sequences)
        
        print(f"  ✓ Generated {len(combined_texts)} responses")
        
        # Generate with separate approach
        print("\n2. Generating with Separate Input approach...")
        separate_predictions = self.separate_model.hybrid_model.predict(
            sample_inputs, verbose=0
        )
        
        if isinstance(separate_predictions, dict):
            separate_logits = separate_predictions['generated_response']
        elif isinstance(separate_predictions, (list, tuple)):
            separate_logits = separate_predictions[1]
        else:
            separate_logits = separate_predictions
        
        separate_sequences = np.argmax(separate_logits, axis=-1)
        separate_texts = self.tokenizer.sequences_to_texts(separate_sequences)
        
        print(f"  ✓ Generated {len(separate_texts)} responses")
        
        # Get reference texts
        reference_texts = self.tokenizer.sequences_to_texts(sample_references)
        input_texts = self.tokenizer.sequences_to_texts(sample_inputs)
        
        return combined_texts, separate_texts, reference_texts, input_texts
    
    def evaluate_approach(self, approach_name, generated_texts, reference_texts):
        """
        Evaluate a single approach with all metrics.
        
        Args:
            approach_name: Name of approach ('combined' or 'separate')
            generated_texts: Generated texts
            reference_texts: Reference texts
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING {approach_name.upper()} APPROACH")
        print(f"{'='*80}")
        
        metrics = {}
        
        # BLEU scores
        print("\n📊 Computing BLEU scores...")
        bleu_scores = self.text_evaluator.compute_bleu(
            reference_texts, generated_texts
        )
        metrics.update(bleu_scores)
        print(f"  ✓ BLEU-1: {bleu_scores['bleu-1']:.4f}")
        print(f"  ✓ BLEU-2: {bleu_scores['bleu-2']:.4f}")
        print(f"  ✓ BLEU-4: {bleu_scores['bleu-4']:.4f}")
        
        # ROUGE scores
        print("\n📊 Computing ROUGE scores...")
        rouge_scores = self.text_evaluator.compute_rouge(
            reference_texts, generated_texts
        )
        metrics.update(rouge_scores)
        print(f"  ✓ ROUGE-1: {rouge_scores['rouge-1']:.4f}")
        print(f"  ✓ ROUGE-2: {rouge_scores['rouge-2']:.4f}")
        print(f"  ✓ ROUGE-L: {rouge_scores['rouge-l']:.4f}")
        
        # Diversity metrics
        print("\n📊 Computing Diversity metrics...")
        diversity_metrics = self.text_evaluator.compute_diversity(
            generated_texts, verbose=False
        )
        metrics.update(diversity_metrics)
        print(f"  ✓ Distinct-1: {diversity_metrics['distinct_1']:.4f}")
        print(f"  ✓ Distinct-2: {diversity_metrics['distinct_2']:.4f}")
        print(f"  ✓ Repetition Rate: {diversity_metrics['repetition_rate']:.4f}")
        
        # Perplexity (simplified calculation)
        print("\n📊 Computing Perplexity...")
        try:
            perplexity = self.text_evaluator.compute_perplexity(generated_texts)
            metrics['perplexity'] = perplexity
            print(f"  ✓ Perplexity: {perplexity:.2f}")
        except Exception as e:
            print(f"  ⚠️  Could not compute perplexity: {e}")
            metrics['perplexity'] = 0.0
        
        # Quality score (composite)
        quality_score = (
            bleu_scores['bleu-4'] * 0.3 +
            rouge_scores['rouge-l'] * 0.3 +
            diversity_metrics['distinct_2'] * 0.2 +
            (1.0 - diversity_metrics['repetition_rate']) * 0.2
        )
        metrics['quality_score'] = quality_score
        print(f"\n⭐ Overall Quality Score: {quality_score:.4f}")
        
        return metrics
    
    def compare_approaches(self, num_samples=500):
        """
        Complete comparison of both approaches.
        
        Args:
            num_samples: Number of samples to evaluate
        
        Returns:
            Dictionary with comparison results
        """
        # Generate responses
        combined_texts, separate_texts, reference_texts, input_texts = \
            self.generate_responses(num_samples)
        
        # Evaluate combined approach
        self.results['combined'] = self.evaluate_approach(
            'combined', combined_texts, reference_texts
        )
        
        # Evaluate separate approach
        self.results['separate'] = self.evaluate_approach(
            'separate', separate_texts, reference_texts
        )
        
        # Store texts for visualization
        self.results['combined']['generated_texts'] = combined_texts
        self.results['separate']['generated_texts'] = separate_texts
        self.results['reference_texts'] = reference_texts
        self.results['input_texts'] = input_texts
        
        return self.results
    
    def create_comparison_visualizations(self, save_dir='results/input_comparison'):
        """
        Create comprehensive visualizations comparing both approaches.
        
        Args:
            save_dir: Directory to save visualizations
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("CREATING COMPARISON VISUALIZATIONS")
        print("="*80)
        
        # 1. Metrics comparison bar chart
        self._plot_metrics_comparison(save_dir)
        
        # 2. Diversity comparison
        self._plot_diversity_comparison(save_dir)
        
        # 3. Example responses side-by-side
        self._create_response_examples(save_dir)
        
        # 4. Detailed metrics table
        self._create_metrics_table(save_dir)
        
        print(f"\n✓ All visualizations saved to: {save_dir}/")
    
    def _plot_metrics_comparison(self, save_dir):
        """Create bar chart comparing all metrics."""
        print("\n📊 Creating metrics comparison bar chart...")
        
        metrics_to_plot = [
            ('bleu-1', 'BLEU-1'),
            ('bleu-2', 'BLEU-2'),
            ('bleu-4', 'BLEU-4'),
            ('rouge-1', 'ROUGE-1'),
            ('rouge-2', 'ROUGE-2'),
            ('rouge-l', 'ROUGE-L'),
            ('distinct_1', 'Distinct-1'),
            ('distinct_2', 'Distinct-2'),
            ('quality_score', 'Quality Score')
        ]
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()
        
        for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            combined_val = self.results['combined'].get(metric_key, 0)
            separate_val = self.results['separate'].get(metric_key, 0)
            
            bars = ax.bar(
                ['Combined\nInput', 'Separate\nInput'],
                [combined_val, separate_val],
                color=['#3498db', '#e74c3c'],
                edgecolor='black',
                linewidth=1.5
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold'
                )
            
            # Calculate improvement
            if combined_val > 0:
                improvement = ((separate_val - combined_val) / combined_val) * 100
                ax.set_title(
                    f'{metric_name}\n({"+" if improvement >= 0 else ""}{improvement:.1f}% change)',
                    fontsize=12, fontweight='bold'
                )
            else:
                ax.set_title(metric_name, fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Score', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, max(combined_val, separate_val) * 1.2 if max(combined_val, separate_val) > 0 else 1)
        
        plt.suptitle(
            'Input Approach Comparison: Combined vs Separate\n' +
            'All Evaluation Metrics',
            fontsize=16, fontweight='bold', y=0.995
        )
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(f'{save_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_dir}/metrics_comparison.png")
    
    def _plot_diversity_comparison(self, save_dir):
        """Create visualization focusing on diversity and repetition."""
        print("\n📊 Creating diversity comparison chart...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Diversity metrics (higher is better)
        ax1 = axes[0]
        diversity_metrics = ['distinct_1', 'distinct_2', 'distinct_3']
        combined_diversity = [self.results['combined'].get(m, 0) for m in diversity_metrics]
        separate_diversity = [self.results['separate'].get(m, 0) for m in diversity_metrics]
        
        x = np.arange(len(diversity_metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, combined_diversity, width, label='Combined Input',
                       color='#3498db', edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, separate_diversity, width, label='Separate Input',
                       color='#e74c3c', edgecolor='black', linewidth=1.5)
        
        ax1.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score (Higher is Better)', fontsize=12, fontweight='bold')
        ax1.set_title('Diversity Metrics Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Distinct-1', 'Distinct-2', 'Distinct-3'])
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Repetition rate (lower is better)
        ax2 = axes[1]
        combined_rep = self.results['combined'].get('repetition_rate', 0)
        separate_rep = self.results['separate'].get('repetition_rate', 0)
        
        bars = ax2.bar(
            ['Combined\nInput', 'Separate\nInput'],
            [combined_rep, separate_rep],
            color=['#3498db', '#e74c3c'],
            edgecolor='black',
            linewidth=1.5
        )
        
        ax2.set_ylabel('Repetition Rate (Lower is Better)', fontsize=12, fontweight='bold')
        ax2.set_title('Repetition Rate Comparison', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels and quality indicators
        for bar, val in zip(bars, [combined_rep, separate_rep]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Quality indicator
            if val < 0.1:
                quality = '✅ Excellent'
                color = 'green'
            elif val < 0.2:
                quality = '✓ Good'
                color = 'blue'
            elif val < 0.3:
                quality = '⚠ Moderate'
                color = 'orange'
            else:
                quality = '❌ High'
                color = 'red'
            
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    quality, ha='center', va='bottom', fontsize=9,
                    color=color, fontweight='bold')
        
        plt.suptitle('Diversity & Repetition Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/diversity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_dir}/diversity_comparison.png")
    
    def _create_response_examples(self, save_dir, num_examples=10):
        """Create side-by-side comparison of example responses."""
        print(f"\n📝 Creating response examples ({num_examples} samples)...")
        
        combined_texts = self.results['combined']['generated_texts']
        separate_texts = self.results['separate']['generated_texts']
        reference_texts = self.results['reference_texts']
        input_texts = self.results['input_texts']
        
        # Create text file with examples
        with open(f'{save_dir}/response_examples.txt', 'w') as f:
            f.write("="*100 + "\n")
            f.write("RESPONSE EXAMPLES: COMBINED vs SEPARATE INPUT APPROACHES\n")
            f.write("="*100 + "\n\n")
            
            for i in range(min(num_examples, len(input_texts))):
                f.write(f"\nExample {i+1}:\n")
                f.write("-" * 100 + "\n")
                f.write(f"INPUT (Customer):  {input_texts[i]}\n")
                f.write(f"REFERENCE (Human): {reference_texts[i]}\n")
                f.write(f"\nCOMBINED INPUT:    {combined_texts[i]}\n")
                f.write(f"SEPARATE INPUT:    {separate_texts[i]}\n")
                f.write("-" * 100 + "\n")
                
                # Add quality assessment
                combined_rep = self.diversity_calculator.compute_repetition_rate(combined_texts[i])
                separate_rep = self.diversity_calculator.compute_repetition_rate(separate_texts[i])
                
                f.write(f"\nQuality Assessment:\n")
                f.write(f"  Combined - Repetition: {combined_rep:.4f} ")
                f.write(f"{'✅' if combined_rep < 0.1 else '⚠️' if combined_rep < 0.3 else '❌'}\n")
                f.write(f"  Separate - Repetition: {separate_rep:.4f} ")
                f.write(f"{'✅' if separate_rep < 0.1 else '⚠️' if separate_rep < 0.3 else '❌'}\n")
                
                better = "Combined" if combined_rep < separate_rep else "Separate"
                f.write(f"  Winner: {better} (lower repetition)\n")
                f.write("\n")
        
        print(f"  ✓ Saved: {save_dir}/response_examples.txt")
        
        # Create visual comparison figure
        fig, axes = plt.subplots(num_examples, 1, figsize=(16, 2.5 * num_examples))
        if num_examples == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes[:num_examples]):
            ax.axis('off')
            
            # Format text for display
            input_text = input_texts[i][:80] + "..." if len(input_texts[i]) > 80 else input_texts[i]
            ref_text = reference_texts[i][:80] + "..." if len(reference_texts[i]) > 80 else reference_texts[i]
            combined_text = combined_texts[i][:80] + "..." if len(combined_texts[i]) > 80 else combined_texts[i]
            separate_text = separate_texts[i][:80] + "..." if len(separate_texts[i]) > 80 else separate_texts[i]
            
            text_content = (
                f"Example {i+1}\n\n"
                f"INPUT:      {input_text}\n"
                f"REFERENCE:  {ref_text}\n\n"
                f"COMBINED:   {combined_text}\n"
                f"SEPARATE:   {separate_text}"
            )
            
            ax.text(0.05, 0.5, text_content, transform=ax.transAxes,
                   fontsize=9, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Response Examples Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/response_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_dir}/response_examples.png")
    
    def _create_metrics_table(self, save_dir):
        """Create detailed metrics comparison table."""
        print("\n📋 Creating metrics table...")
        
        # Prepare data
        metrics_data = []
        
        metric_names = {
            'bleu-1': 'BLEU-1',
            'bleu-2': 'BLEU-2',
            'bleu-3': 'BLEU-3',
            'bleu-4': 'BLEU-4',
            'rouge-1': 'ROUGE-1',
            'rouge-2': 'ROUGE-2',
            'rouge-l': 'ROUGE-L',
            'distinct_1': 'Distinct-1',
            'distinct_2': 'Distinct-2',
            'distinct_3': 'Distinct-3',
            'repetition_rate': 'Repetition Rate',
            'entropy': 'Entropy',
            'quality_score': 'Quality Score'
        }
        
        for key, name in metric_names.items():
            combined_val = self.results['combined'].get(key, 0)
            separate_val = self.results['separate'].get(key, 0)
            
            # Calculate difference and determine winner
            diff = separate_val - combined_val
            if key == 'repetition_rate':
                # Lower is better for repetition
                winner = 'Separate' if separate_val < combined_val else 'Combined'
                improvement = ((combined_val - separate_val) / combined_val * 100) if combined_val > 0 else 0
            else:
                # Higher is better for other metrics
                winner = 'Separate' if separate_val > combined_val else 'Combined'
                improvement = ((separate_val - combined_val) / combined_val * 100) if combined_val > 0 else 0
            
            metrics_data.append({
                'Metric': name,
                'Combined Input': f'{combined_val:.4f}',
                'Separate Input': f'{separate_val:.4f}',
                'Difference': f'{diff:+.4f}',
                'Improvement (%)': f'{improvement:+.1f}%',
                'Winner': winner
            })
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Save as CSV
        df.to_csv(f'{save_dir}/metrics_comparison.csv', index=False)
        print(f"  ✓ Saved: {save_dir}/metrics_comparison.csv")
        
        # Create formatted table image
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if j == len(df.columns) - 1:  # Winner column
                    winner = df.iloc[i-1]['Winner']
                    if winner == 'Separate':
                        table[(i, j)].set_facecolor('#e8f5e9')
                    else:
                        table[(i, j)].set_facecolor('#ffebee')
        
        plt.title('Detailed Metrics Comparison: Combined vs Separate Input',
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(f'{save_dir}/metrics_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_dir}/metrics_table.png")
    
    def print_summary(self):
        """Print comprehensive summary of comparison."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        combined = self.results['combined']
        separate = self.results['separate']
        
        print("\n📊 KEY METRICS:")
        print(f"\n{'Metric':<20} {'Combined':<12} {'Separate':<12} {'Winner':<10} {'Improvement'}")
        print("-" * 80)
        
        metrics_to_show = [
            ('bleu-4', 'BLEU-4', False),
            ('rouge-l', 'ROUGE-L', False),
            ('distinct_1', 'Distinct-1', False),
            ('distinct_2', 'Distinct-2', False),
            ('repetition_rate', 'Repetition Rate', True),
            ('quality_score', 'Quality Score', False)
        ]
        
        for key, name, lower_better in metrics_to_show:
            c_val = combined.get(key, 0)
            s_val = separate.get(key, 0)
            
            if lower_better:
                winner = 'Separate' if s_val < c_val else 'Combined'
                improvement = ((c_val - s_val) / c_val * 100) if c_val > 0 else 0
            else:
                winner = 'Separate' if s_val > c_val else 'Combined'
                improvement = ((s_val - c_val) / c_val * 100) if c_val > 0 else 0
            
            print(f"{name:<20} {c_val:<12.4f} {s_val:<12.4f} {winner:<10} {improvement:+.1f}%")
        
        # Overall winner
        print("\n" + "="*80)
        combined_wins = sum(1 for _, _, lower in metrics_to_show
                           if (lower and combined.get(_[0], 0) < separate.get(_[0], 0)) or
                              (not lower and combined.get(_[0], 0) > separate.get(_[0], 0)))
        separate_wins = len(metrics_to_show) - combined_wins
        
        print(f"\n🏆 OVERALL WINNER: ", end='')
        if separate_wins > combined_wins:
            print(f"SEPARATE INPUT ({separate_wins}/{len(metrics_to_show)} metrics)")
        elif combined_wins > separate_wins:
            print(f"COMBINED INPUT ({combined_wins}/{len(metrics_to_show)} metrics)")
        else:
            print(f"TIE ({combined_wins}/{len(metrics_to_show)} each)")
        
        print("\n" + "="*80)


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("INPUT APPROACH COMPARISON: COMBINED vs SEPARATE")
    print("="*80)
    
    # Load data
    print("\n📦 Loading data and tokenizer...")
    with open('processed_data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    val_customer = np.load('processed_data/val_customer.npy')
    val_agent = np.load('processed_data/val_agent.npy')
    
    print(f"✓ Loaded {len(val_customer):,} validation samples")
    
    # Create comparator
    comparator = InputApproachComparator(tokenizer, val_customer, val_agent)
    
    # Build models
    comparator.build_models()
    
    # Compare approaches
    print("\n" + "="*80)
    print("STARTING COMPARISON")
    print("="*80)
    results = comparator.compare_approaches(num_samples=500)
    
    # Create visualizations
    comparator.create_comparison_visualizations()
    
    # Print summary
    comparator.print_summary()
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETED!")
    print("="*80)
    print("\n📁 Results saved to: results/input_comparison/")
    print("  • metrics_comparison.png - All metrics bar chart")
    print("  • diversity_comparison.png - Diversity & repetition analysis")
    print("  • response_examples.txt - Side-by-side text examples")
    print("  • response_examples.png - Visual examples")
    print("  • metrics_table.png - Detailed comparison table")
    print("  • metrics_comparison.csv - Metrics data")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
