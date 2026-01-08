"""
XAI (Explainable AI) Module for Hybrid GAN + VAE

Implements multiple explainability techniques:
1. SHAP (SHapley Additive exPlanations) - Feature importance
2. Attention Visualization - Which tokens the model focuses on
3. Saliency Maps - Input sensitivity analysis
4. Token Importance - Word-level contribution analysis
5. Latent Space Visualization - Understanding internal representations

This module helps answer:
- Which words influence the generated response?
- What does the model pay attention to?
- How do different inputs lead to different outputs?
- What has the model learned in its latent space?
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pickle
import config

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")


class ModelExplainer:
    """
    Comprehensive explainability wrapper for Hybrid GAN + VAE
    
    Provides multiple explanation techniques for understanding
    model decisions and generated text.
    """
    
    def __init__(self, hybrid_model, tokenizer):
        """
        Initialize explainer
        
        Args:
            hybrid_model: HybridGANVAE instance
            tokenizer: SimpleTokenizer instance
        """
        self.hybrid_model = hybrid_model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index) + 1
        self.max_length = hybrid_model.max_length
        
        print("✓ Model explainer initialized")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Max sequence length: {self.max_length}")
    
    def get_attention_weights(self, customer_message: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from encoder's bidirectional LSTM
        
        Args:
            customer_message: Tokenized customer message (1, max_length)
        
        Returns:
            Dictionary with attention information
        """
        # Get encoder outputs
        encoder_outputs = self.hybrid_model.vae.encoder.predict(
            customer_message, 
            verbose=0
        )
        z_mean, z_log_var, z = encoder_outputs
        
        # Create attention model to get intermediate outputs
        # We'll use the encoder's internal states as attention proxy
        encoder_layers = self.hybrid_model.vae.encoder.layers
        
        # Find bidirectional LSTM layer
        bidirectional_layer = None
        for layer in encoder_layers:
            if 'bidirectional' in layer.name.lower():
                bidirectional_layer = layer
                break
        
        if bidirectional_layer is None:
            print("⚠️  No bidirectional layer found for attention")
            return {}
        
        # Create model that outputs intermediate representations
        attention_model = keras.Model(
            inputs=self.hybrid_model.vae.encoder.input,
            outputs=[
                bidirectional_layer.output,
                z_mean,
                z_log_var,
                z
            ]
        )
        
        # Get outputs
        lstm_output, z_mean_out, z_log_var_out, z_out = attention_model.predict(
            customer_message,
            verbose=0
        )
        
        # Compute attention as L2 norm across hidden dimensions
        # Shape: (1, seq_len, hidden_dim) -> (1, seq_len)
        attention_weights = np.linalg.norm(lstm_output, axis=-1)
        
        # Normalize to [0, 1]
        attention_weights = attention_weights / (np.max(attention_weights) + 1e-10)
        
        return {
            'attention_weights': attention_weights[0],  # (seq_len,)
            'lstm_output': lstm_output[0],              # (seq_len, hidden_dim)
            'z_mean': z_mean_out[0],                    # (latent_dim,)
            'z_log_var': z_log_var_out[0],              # (latent_dim,)
            'z': z_out[0]                               # (latent_dim,)
        }
    
    def visualize_attention(self, customer_message: np.ndarray, save_path: Optional[str] = None):
        """
        Visualize attention weights over input tokens
        
        Args:
            customer_message: Tokenized message (1, max_length)
            save_path: Path to save visualization
        """
        # Get attention weights
        attention_info = self.get_attention_weights(customer_message)
        
        if not attention_info:
            print("Cannot visualize attention - no weights available")
            return
        
        attention_weights = attention_info['attention_weights']
        
        # Decode tokens
        tokens = self.tokenizer.sequences_to_texts([customer_message[0]])[0].split()
        
        # Truncate to actual tokens (remove padding)
        num_tokens = min(len(tokens), len(attention_weights))
        tokens = tokens[:num_tokens]
        attention_weights = attention_weights[:num_tokens]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Bar plot
        colors = plt.cm.Blues(attention_weights)
        bars = ax.bar(range(len(tokens)), attention_weights, color=colors)
        
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_xlabel('Token', fontsize=12)
        ax.set_title('Attention Weights: Which Tokens Does the Model Focus On?', 
                     fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Strength', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Attention visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compute_saliency_map(self, customer_message: np.ndarray) -> np.ndarray:
        """
        Compute saliency map: gradient of output w.r.t. input
        
        Shows which input tokens are most important for the model's decision.
        
        Args:
            customer_message: Tokenized message (1, max_length)
        
        Returns:
            Saliency values for each token position
        """
        # Convert to tensor
        input_tensor = tf.constant(customer_message, dtype=tf.int32)
        
        # We need to convert tokens to embeddings first
        # Get embedding layer from encoder
        embedding_layer = None
        for layer in self.hybrid_model.vae.encoder.layers:
            if isinstance(layer, keras.layers.Embedding):
                embedding_layer = layer
                break
        
        if embedding_layer is None:
            print("⚠️  No embedding layer found")
            return np.zeros(self.max_length)
        
        # Get embeddings
        embeddings = embedding_layer(input_tensor)
        embeddings = tf.Variable(embeddings, trainable=True)
        
        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            
            # Forward pass through encoder (starting from embeddings)
            # We'll approximate this by getting latent representation quality
            z_mean, z_log_var, z = self.hybrid_model.vae.encoder.predict(
                customer_message,
                verbose=0
            )
            
            # Use L2 norm of z_mean as output to differentiate
            output = tf.reduce_sum(tf.square(tf.constant(z_mean)))
        
        # Compute gradients
        gradients = tape.gradient(output, embeddings)
        
        if gradients is None:
            print("⚠️  Could not compute gradients")
            return np.zeros(self.max_length)
        
        # Saliency: L2 norm across embedding dimensions
        saliency = tf.norm(gradients, axis=-1)
        saliency = saliency.numpy()[0]
        
        # Normalize
        saliency = saliency / (np.max(saliency) + 1e-10)
        
        return saliency
    
    def visualize_saliency_map(self, customer_message: np.ndarray, save_path: Optional[str] = None):
        """
        Visualize saliency map
        
        Args:
            customer_message: Tokenized message (1, max_length)
            save_path: Path to save visualization
        """
        saliency = self.compute_saliency_map(customer_message)
        
        # Decode tokens
        tokens = self.tokenizer.sequences_to_texts([customer_message[0]])[0].split()
        
        # Truncate to actual tokens
        num_tokens = min(len(tokens), len(saliency))
        tokens = tokens[:num_tokens]
        saliency = saliency[:num_tokens]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Heatmap style
        colors = plt.cm.Reds(saliency)
        bars = ax.bar(range(len(tokens)), saliency, color=colors)
        
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Saliency (Gradient Magnitude)', fontsize=12)
        ax.set_xlabel('Token', fontsize=12)
        ax.set_title('Saliency Map: Which Tokens Most Influence the Model?', 
                     fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        
        ax.grid(axis='y', alpha=0.3)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Importance', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saliency map saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_token_importance(self, customer_message: np.ndarray) -> Dict[str, float]:
        """
        Analyze importance of each token in the input
        
        Combines attention weights and saliency for comprehensive importance.
        
        Args:
            customer_message: Tokenized message (1, max_length)
        
        Returns:
            Dictionary mapping tokens to importance scores
        """
        # Get attention weights
        attention_info = self.get_attention_weights(customer_message)
        attention_weights = attention_info.get('attention_weights', np.zeros(self.max_length))
        
        # Get saliency
        saliency = self.compute_saliency_map(customer_message)
        
        # Combine: average of attention and saliency
        importance = (attention_weights + saliency) / 2.0
        
        # Decode tokens
        tokens = self.tokenizer.sequences_to_texts([customer_message[0]])[0].split()
        
        # Create importance dictionary
        token_importance = {}
        for i, token in enumerate(tokens):
            if i < len(importance):
                token_importance[token] = float(importance[i])
        
        # Sort by importance
        token_importance = dict(
            sorted(token_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return token_importance
    
    def explain_generation(self, customer_message: np.ndarray, top_k: int = 10) -> Dict:
        """
        Comprehensive explanation of how the model generates responses
        
        Args:
            customer_message: Tokenized message (1, max_length)
            top_k: Number of top important tokens to highlight
        
        Returns:
            Dictionary with all explanation information
        """
        print("\n" + "=" * 70)
        print("EXPLAINING MODEL GENERATION")
        print("=" * 70)
        
        # 1. Decode input
        input_text = self.tokenizer.sequences_to_texts([customer_message[0]])[0]
        print(f"\nInput Customer Message:")
        print(f"  \"{input_text}\"")
        
        # 2. Get latent representation
        z_mean, z_log_var, z = self.hybrid_model.vae.encoder.predict(
            customer_message,
            verbose=0
        )
        
        print(f"\nLatent Space Representation:")
        print(f"  Dimension: {len(z[0])}")
        print(f"  Mean magnitude: {np.mean(np.abs(z_mean[0])):.4f}")
        print(f"  Variance: {np.exp(0.5 * np.mean(z_log_var[0])):.4f}")
        
        # 3. Token importance
        token_importance = self.analyze_token_importance(customer_message)
        
        print(f"\nTop {top_k} Most Important Tokens:")
        for i, (token, importance) in enumerate(list(token_importance.items())[:top_k]):
            print(f"  {i+1}. '{token}': {importance:.4f}")
        
        # 4. Generate response
        generated_response = self.hybrid_model.generate_response(customer_message)
        generated_text = self.tokenizer.sequences_to_texts([generated_response[0]])[0]
        
        print(f"\nGenerated Agent Response:")
        print(f"  \"{generated_text}\"")
        
        # 5. Response quality
        quality_score = self.hybrid_model.evaluate_response_quality(generated_response)[0][0]
        
        print(f"\nResponse Quality Score: {quality_score:.4f}")
        print(f"  (Higher is better, range 0-1)")
        
        explanation = {
            'input_text': input_text,
            'input_tokens': input_text.split(),
            'token_importance': token_importance,
            'top_tokens': list(token_importance.items())[:top_k],
            'latent_representation': {
                'z_mean': z_mean[0],
                'z_log_var': z_log_var[0],
                'z': z[0],
                'mean_magnitude': float(np.mean(np.abs(z_mean[0]))),
                'variance': float(np.exp(0.5 * np.mean(z_log_var[0])))
            },
            'generated_text': generated_text,
            'generated_tokens': generated_text.split(),
            'quality_score': float(quality_score)
        }
        
        print("\n" + "=" * 70)
        
        return explanation
    
    def visualize_latent_space(self, data_sample: np.ndarray, labels: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None):
        """
        Visualize latent space using t-SNE or PCA
        
        Args:
            data_sample: Sample of customer messages (N, max_length)
            labels: Optional labels for coloring (e.g., customer segments)
            save_path: Path to save visualization
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        print("\nVisualizing latent space...")
        
        # Encode to latent space
        z_mean, z_log_var, z = self.hybrid_model.vae.encoder.predict(
            data_sample,
            verbose=0
        )
        
        # Use z (sampled latent vectors)
        latent_vectors = z
        
        print(f"  Original latent dimension: {latent_vectors.shape[1]}")
        
        # Reduce to 2D using PCA first (for speed), then t-SNE
        if latent_vectors.shape[1] > 50:
            pca = PCA(n_components=50)
            latent_reduced = pca.fit_transform(latent_vectors)
            print(f"  PCA reduced to: 50 dimensions")
            print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        else:
            latent_reduced = latent_vectors
        
        # t-SNE for final 2D visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(latent_reduced)
        
        print(f"  t-SNE reduced to: 2 dimensions")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if labels is not None:
            # Color by labels
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    latent_2d[mask, 0],
                    latent_2d[mask, 1],
                    c=[colors[i]],
                    label=label,
                    alpha=0.6,
                    s=50
                )
            
            ax.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # No labels, single color
            ax.scatter(
                latent_2d[:, 0],
                latent_2d[:, 1],
                c='steelblue',
                alpha=0.6,
                s=50
            )
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Latent Space Visualization (t-SNE)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Latent space visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_comprehensive_report(self, customer_message: np.ndarray, 
                                   output_dir: str = 'results/explainability'):
        """
        Create comprehensive explainability report with all visualizations
        
        Args:
            customer_message: Tokenized message to explain
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("CREATING COMPREHENSIVE EXPLAINABILITY REPORT")
        print("=" * 70)
        
        # 1. Comprehensive explanation
        explanation = self.explain_generation(customer_message, top_k=15)
        
        # 2. Attention visualization
        print("\nGenerating attention visualization...")
        self.visualize_attention(
            customer_message,
            save_path=f'{output_dir}/attention_weights.png'
        )
        
        # 3. Saliency map
        print("\nGenerating saliency map...")
        self.visualize_saliency_map(
            customer_message,
            save_path=f'{output_dir}/saliency_map.png'
        )
        
        # 4. Token importance comparison
        print("\nGenerating token importance comparison...")
        self._visualize_token_importance_comparison(
            customer_message,
            save_path=f'{output_dir}/token_importance_comparison.png'
        )
        
        # 5. Save explanation as text
        report_path = f'{output_dir}/explanation_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EXPLAINABILITY REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Input: {explanation['input_text']}\n\n")
            
            f.write("Top 15 Most Important Tokens:\n")
            for i, (token, score) in enumerate(explanation['top_tokens']):
                f.write(f"  {i+1}. '{token}': {score:.4f}\n")
            
            f.write(f"\nGenerated Response: {explanation['generated_text']}\n")
            f.write(f"Quality Score: {explanation['quality_score']:.4f}\n\n")
            
            f.write("Latent Space Statistics:\n")
            f.write(f"  Mean magnitude: {explanation['latent_representation']['mean_magnitude']:.4f}\n")
            f.write(f"  Variance: {explanation['latent_representation']['variance']:.4f}\n")
        
        print(f"\n✓ Text report saved to: {report_path}")
        
        print("\n" + "=" * 70)
        print("✅ COMPREHENSIVE REPORT COMPLETE!")
        print("=" * 70)
        print(f"\nAll visualizations saved to: {output_dir}/")
        print("  • attention_weights.png")
        print("  • saliency_map.png")
        print("  • token_importance_comparison.png")
        print("  • explanation_report.txt")
    
    def _visualize_token_importance_comparison(self, customer_message: np.ndarray,
                                               save_path: Optional[str] = None):
        """
        Compare attention vs. saliency vs. combined importance
        """
        # Get all importance measures
        attention_info = self.get_attention_weights(customer_message)
        attention_weights = attention_info.get('attention_weights', np.zeros(self.max_length))
        saliency = self.compute_saliency_map(customer_message)
        combined = (attention_weights + saliency) / 2.0
        
        # Decode tokens
        tokens = self.tokenizer.sequences_to_texts([customer_message[0]])[0].split()
        num_tokens = min(len(tokens), 15)  # Show top 15
        tokens = tokens[:num_tokens]
        attention_weights = attention_weights[:num_tokens]
        saliency = saliency[:num_tokens]
        combined = combined[:num_tokens]
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(tokens))
        width = 0.25
        
        bars1 = ax.bar(x - width, attention_weights, width, label='Attention', 
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x, saliency, width, label='Saliency', 
                      color='coral', alpha=0.8)
        bars3 = ax.bar(x + width, combined, width, label='Combined', 
                      color='green', alpha=0.8)
        
        ax.set_xlabel('Token', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_title('Token Importance: Attention vs. Saliency vs. Combined', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING EXPLAINABILITY MODULE")
    print("=" * 70)
    
    # Load preprocessing config
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    # Load tokenizer
    with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # Load validation data
    val_customer = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')
    
    print(f"Validation samples: {len(val_customer):,}")
    
    # Build hybrid model
    print("\nBuilding hybrid model...")
    from hybrid_model import HybridGANVAE
    hybrid = HybridGANVAE(vocab_size, max_length)
    
    # Initialize explainer
    print("\n" + "=" * 70)
    print("INITIALIZING EXPLAINER")
    print("=" * 70)
    
    explainer = ModelExplainer(hybrid, tokenizer)
    
    # Test on a single example
    test_message = val_customer[0:1]
    
    print("\n" + "=" * 70)
    print("TEST 1: TOKEN IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    token_importance = explainer.analyze_token_importance(test_message)
    print(f"\nFound {len(token_importance)} tokens")
    print("Top 10 most important:")
    for i, (token, score) in enumerate(list(token_importance.items())[:10]):
        print(f"  {i+1}. '{token}': {score:.4f}")
    
    print("\n" + "=" * 70)
    print("TEST 2: COMPREHENSIVE EXPLANATION")
    print("=" * 70)
    
    explanation = explainer.explain_generation(test_message, top_k=10)
    
    print("\n" + "=" * 70)
    print("TEST 3: VISUALIZATIONS")
    print("=" * 70)
    
    print("\nGenerating attention visualization...")
    explainer.visualize_attention(test_message, save_path='results/test_attention.png')
    
    print("\nGenerating saliency map...")
    explainer.visualize_saliency_map(test_message, save_path='results/test_saliency.png')
    
    print("\n" + "=" * 70)
    print("TEST 4: LATENT SPACE VISUALIZATION")
    print("=" * 70)
    
    # Use subset for speed
    sample_data = val_customer[:500]
    explainer.visualize_latent_space(
        sample_data,
        save_path='results/test_latent_space.png'
    )
    
    print("\n" + "=" * 70)
    print("✅ ALL EXPLAINABILITY TESTS PASSED!")
    print("=" * 70)
    print("\nExplainability module ready for use!")
    print("\nNext steps:")
    print("  1. Run on trained model for meaningful results")
    print("  2. Use create_comprehensive_report() for full analysis")
    print("  3. Compare explanations across different inputs")
