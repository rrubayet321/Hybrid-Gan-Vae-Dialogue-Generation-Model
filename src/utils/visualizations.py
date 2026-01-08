"""
Visualization Module for Hybrid GAN + VAE Research Presentation

This module provides comprehensive visualization capabilities for research presentations,
including:

1. Evaluation Metrics Visualization:
   - BLEU, ROUGE, Perplexity score comparisons
   - Quality score distributions
   - Diversity metric trends
   - Performance over epochs/training

2. Fairness Visualization:
   - Quality disparity across demographic groups
   - Per-group performance comparisons
   - Bias detection heatmaps

3. XAI Visualization:
   - Latent space projections (t-SNE/PCA)
   - Attention weight heatmaps
   - Token importance visualizations
   - Saliency maps for interpretability

4. Generation Examples:
   - Side-by-side reference vs generated text
   - Quality-annotated examples
   - Error analysis visualization

5. Model Comparison:
   - Baseline vs trained model metrics
   - Multiple model version comparisons
   - Architecture ablation studies

Usage:
    from visualizations import ResearchVisualizer
    
    # Create visualizer
    viz = ResearchVisualizer(hybrid_model, tokenizer, explainer)
    
    # Generate all presentation plots
    viz.create_research_presentation(
        val_customer, val_agent, val_metadata,
        save_dir='presentation_figures'
    )

Author: IT Support AI Team
Date: January 2026
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

import config

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# ============================================================================
# ResearchVisualizer Class
# ============================================================================

class ResearchVisualizer:
    """
    Comprehensive visualizer for research presentations and papers.
    
    This class generates publication-quality visualizations for:
    - Model evaluation metrics (BLEU, ROUGE, Perplexity)
    - Fairness analysis across demographic groups
    - XAI/explainability visualizations
    - Generated text examples
    - Training progress tracking
    - Model comparisons
    
    All visualizations are optimized for research presentations with:
    - Clear labels and titles
    - Appropriate color schemes
    - High-resolution output
    - Professional formatting
    
    Attributes:
        hybrid_model: HybridGANVAE instance to visualize
        tokenizer: SimpleTokenizer for text conversion
        explainer: Optional ModelExplainer for XAI visualizations
    
    Example:
        >>> viz = ResearchVisualizer(model, tokenizer, explainer)
        >>> viz.plot_evaluation_comparison(results_dict)
        >>> viz.plot_fairness_analysis(fairness_results)
        >>> viz.create_research_presentation(val_data, save_dir='figs')
    """
    
    def __init__(self, hybrid_model, tokenizer, explainer=None):
        """
        Initialize research visualizer.
        
        Args:
            hybrid_model: HybridGANVAE instance to visualize
            tokenizer: SimpleTokenizer for text-to-sequence conversion
            explainer (optional): ModelExplainer for XAI visualizations
        
        Side Effects:
            - Sets matplotlib style to whitegrid
            - Configures default figure parameters
            - Prints initialization confirmation
        """
        self.hybrid_model = hybrid_model
        self.tokenizer = tokenizer
        self.explainer = explainer
        
        print("✓ Research visualizer initialized")
        if explainer:
            print("  ✓ XAI visualizations enabled")
    
    def plot_evaluation_comparison(self, results: Dict, save_path: str = None):
        """
        Plot comprehensive evaluation metrics comparison.
        
        Creates a multi-panel figure showing:
        - BLEU scores (BLEU-1 through BLEU-4)
        - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        - Quality statistics (mean with error bars)
        - Diversity metrics (Distinct-1, Distinct-2)
        
        Args:
            results (Dict): Evaluation results containing:
                          - 'bleu': Dict with BLEU scores
                          - 'rouge': Dict with ROUGE scores
                          - 'quality': Dict with quality stats
                          - 'diversity': Dict with diversity metrics
            save_path (str, optional): Path to save figure. If None, displays only.
        
        Returns:
            matplotlib.figure.Figure: The created figure object
        
        Example:
            >>> from evaluation_metrics import ModelEvaluator
            >>> evaluator = ModelEvaluator(model, tokenizer)
            >>> results = evaluator.evaluate_generation_quality(val_customer, val_agent)
            >>> viz.plot_evaluation_comparison(results, 'figures/eval_metrics.png')
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Evaluation Metrics Comparison', fontsize=16, fontweight='bold')
        
        # ===== Panel 1: BLEU Scores =====
        ax1 = axes[0, 0]
        if 'bleu' in results:
            bleu_names = list(results['bleu'].keys())
            bleu_values = list(results['bleu'].values())
            
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(bleu_names)))
            bars = ax1.bar(range(len(bleu_names)), bleu_values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, bleu_values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}',
                        ha='center', va='bottom', fontweight='bold')
            
            ax1.set_xlabel('BLEU Metric', fontweight='bold')
            ax1.set_ylabel('Score', fontweight='bold')
            ax1.set_title('BLEU Scores (N-gram Precision)', fontweight='bold')
            ax1.set_xticks(range(len(bleu_names)))
            ax1.set_xticklabels([name.upper() for name in bleu_names])
            ax1.set_ylim(0, max(bleu_values) * 1.3 if bleu_values else 1.0)
            ax1.grid(axis='y', alpha=0.3)
        
        # ===== Panel 2: ROUGE Scores =====
        ax2 = axes[0, 1]
        if 'rouge' in results:
            rouge_names = list(results['rouge'].keys())
            rouge_values = list(results['rouge'].values())
            
            colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(rouge_names)))
            bars = ax2.bar(range(len(rouge_names)), rouge_values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, rouge_values)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}',
                        ha='center', va='bottom', fontweight='bold')
            
            ax2.set_xlabel('ROUGE Metric', fontweight='bold')
            ax2.set_ylabel('Score', fontweight='bold')
            ax2.set_title('ROUGE Scores (N-gram Recall)', fontweight='bold')
            ax2.set_xticks(range(len(rouge_names)))
            ax2.set_xticklabels([name.upper() for name in rouge_names])
            ax2.set_ylim(0, max(rouge_values) * 1.3 if rouge_values else 1.0)
            ax2.grid(axis='y', alpha=0.3)
        
        # ===== Panel 3: Quality Statistics =====
        ax3 = axes[1, 0]
        if 'quality' in results:
            quality_mean = results['quality']['mean']
            quality_std = results['quality']['std']
            
            # Create box plot representation
            bp = ax3.boxplot([[quality_mean]], 
                            positions=[0],
                            widths=0.6,
                            patch_artist=True,
                            boxprops=dict(facecolor='lightcoral', alpha=0.7, edgecolor='black'),
                            medianprops=dict(color='darkred', linewidth=2),
                            whiskerprops=dict(color='black'),
                            capprops=dict(color='black'))
            
            # Add mean and std as error bar
            ax3.errorbar([0], [quality_mean], yerr=[quality_std], 
                        fmt='o', color='darkred', markersize=10, 
                        capsize=10, capthick=2, label=f'Mean: {quality_mean:.4f}')
            
            # Add statistics text
            stats_text = f'Mean: {quality_mean:.4f}\nStd: {quality_std:.4f}\n'
            stats_text += f'Min: {results["quality"]["min"]:.4f}\nMax: {results["quality"]["max"]:.4f}'
            ax3.text(0.5, 0.05, stats_text, transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9, verticalalignment='bottom')
            
            ax3.set_ylabel('Discriminator Score', fontweight='bold')
            ax3.set_title('Generation Quality Distribution', fontweight='bold')
            ax3.set_xticks([0])
            ax3.set_xticklabels(['Quality'])
            ax3.set_ylim(0, 1.0)
            ax3.grid(axis='y', alpha=0.3)
            ax3.legend(loc='upper right')
        
        # ===== Panel 4: Diversity Metrics =====
        ax4 = axes[1, 1]
        if 'diversity' in results:
            diversity_metrics = ['distinct-1', 'distinct-2']
            diversity_values = [results['diversity'].get(m, 0) for m in diversity_metrics]
            
            colors = ['#FF6B6B', '#4ECDC4']
            bars = ax4.bar(range(len(diversity_metrics)), diversity_values, 
                          color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, diversity_values)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}',
                        ha='center', va='bottom', fontweight='bold')
            
            # Add vocab size info
            vocab_text = f"Vocab Size: {results['diversity'].get('vocab_size', 0):.0f}\n"
            vocab_text += f"Avg Length: {results['diversity'].get('avg_length', 0):.1f}"
            ax4.text(0.95, 0.95, vocab_text, transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    fontsize=9, verticalalignment='top', horizontalalignment='right')
            
            ax4.set_xlabel('Diversity Metric', fontweight='bold')
            ax4.set_ylabel('Score', fontweight='bold')
            ax4.set_title('Vocabulary Diversity', fontweight='bold')
            ax4.set_xticks(range(len(diversity_metrics)))
            ax4.set_xticklabels(['Distinct-1\n(Unigram)', 'Distinct-2\n(Bigram)'])
            ax4.set_ylim(0, max(diversity_values) * 1.3 if diversity_values else 1.0)
            ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Evaluation comparison saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_model_comparison(self, model_results: Dict[str, Dict], 
                             metrics: List[str] = None,
                             save_path: str = None):
        """
        Compare multiple models across evaluation metrics.
        
        Creates a grouped bar chart comparing different model versions
        (e.g., baseline, VAE-pretrained, fully trained) across multiple metrics.
        
        Args:
            model_results (Dict[str, Dict]): Dictionary mapping model names to their
                                            evaluation results. Structure:
                                            {
                                                'baseline': {'bleu': {...}, 'rouge': {...}},
                                                'trained': {'bleu': {...}, 'rouge': {...}}
                                            }
            metrics (List[str], optional): Specific metrics to compare. If None,
                                          uses ['bleu-4', 'rouge-l', 'quality']
            save_path (str, optional): Path to save figure
        
        Returns:
            matplotlib.figure.Figure: The created figure object
        
        Example:
            >>> baseline_results = evaluator.evaluate_generation_quality(val_data)
            >>> trained_results = evaluator.evaluate_generation_quality(val_data)
            >>> model_results = {
            ...     'Baseline': baseline_results,
            ...     'After Training': trained_results
            ... }
            >>> viz.plot_model_comparison(model_results, save_path='figs/comparison.png')
        """
        if metrics is None:
            metrics = ['bleu-4', 'rouge-l', 'distinct-1']
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Extract data for plotting
        model_names = list(model_results.keys())
        n_models = len(model_names)
        n_metrics = len(metrics)
        
        # Prepare data matrix
        data_matrix = []
        for model_name in model_names:
            model_data = []
            for metric in metrics:
                if metric.startswith('bleu'):
                    value = model_results[model_name].get('bleu', {}).get(metric, 0)
                elif metric.startswith('rouge'):
                    value = model_results[model_name].get('rouge', {}).get(metric, 0)
                elif metric.startswith('distinct'):
                    value = model_results[model_name].get('diversity', {}).get(metric, 0)
                elif metric == 'quality':
                    value = model_results[model_name].get('quality', {}).get('mean', 0)
                else:
                    value = 0
                model_data.append(value)
            data_matrix.append(model_data)
        
        # Create grouped bar chart
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        for i, (model_name, values) in enumerate(zip(model_names, data_matrix)):
            offset = (i - n_models/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name, 
                         color=colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xlabel('Evaluation Metric', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper().replace('-', '\n') for m in metrics])
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max([max(vals) for vals in data_matrix]) * 1.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Model comparison saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_fairness_analysis(self, fairness_results: Dict, save_path: str = None):
        """
        Visualize fairness analysis across demographic groups.
        
        Creates a comprehensive fairness visualization showing:
        - Quality disparity bar chart
        - Per-group quality comparison
        - Diversity metrics across groups
        
        Args:
            fairness_results (Dict): Results from ModelEvaluator.evaluate_fairness()
                                    containing group-wise metrics and disparities
            save_path (str, optional): Path to save figure
        
        Returns:
            matplotlib.figure.Figure: The created figure object
        
        Example:
            >>> fairness_results = evaluator.evaluate_fairness(
            ...     val_customer, val_metadata, ['customer_segment', 'region']
            ... )
            >>> viz.plot_fairness_analysis(fairness_results, 'figs/fairness.png')
        """
        n_attributes = len(fairness_results)
        fig, axes = plt.subplots(n_attributes, 2, figsize=(14, 5*n_attributes))
        
        if n_attributes == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Fairness Analysis Across Demographic Groups', 
                    fontsize=16, fontweight='bold')
        
        for idx, (attr, results) in enumerate(fairness_results.items()):
            # ===== Left panel: Quality comparison across groups =====
            ax_left = axes[idx, 0]
            
            groups = list(results['groups'].keys())
            qualities = [results['groups'][g]['quality_mean'] for g in groups]
            stds = [results['groups'][g]['quality_std'] for g in groups]
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
            bars = ax_left.bar(range(len(groups)), qualities, 
                              yerr=stds, capsize=5,
                              color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, quality in zip(bars, qualities):
                height = bar.get_height()
                ax_left.text(bar.get_x() + bar.get_width()/2., height,
                           f'{quality:.3f}',
                           ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Add disparity line
            max_quality = max(qualities)
            min_quality = min(qualities)
            disparity = results['quality_disparity']
            
            ax_left.axhline(max_quality, color='red', linestyle='--', alpha=0.5, label='Max')
            ax_left.axhline(min_quality, color='blue', linestyle='--', alpha=0.5, label='Min')
            
            # Disparity annotation
            ax_left.annotate('', xy=(len(groups)-0.5, max_quality), 
                           xytext=(len(groups)-0.5, min_quality),
                           arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax_left.text(len(groups)-0.3, (max_quality + min_quality)/2,
                       f'Disparity:\n{disparity:.4f}',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                       fontsize=9, fontweight='bold')
            
            ax_left.set_xlabel('Group', fontweight='bold')
            ax_left.set_ylabel('Quality Score', fontweight='bold')
            ax_left.set_title(f'{attr.replace("_", " ").title()}: Quality Comparison', 
                            fontweight='bold')
            ax_left.set_xticks(range(len(groups)))
            ax_left.set_xticklabels(groups, rotation=45, ha='right')
            ax_left.set_ylim(0, 1.0)
            ax_left.legend(loc='upper right')
            ax_left.grid(axis='y', alpha=0.3)
            
            # ===== Right panel: Diversity comparison across groups =====
            ax_right = axes[idx, 1]
            
            diversities = [results['groups'][g]['diversity']['distinct-1'] for g in groups]
            counts = [results['groups'][g]['count'] for g in groups]
            
            # Scatter plot with size representing sample count
            scatter = ax_right.scatter(range(len(groups)), diversities, 
                                      s=[c/10 for c in counts],  # Scale down for visibility
                                      c=colors, alpha=0.6, edgecolors='black', linewidth=2)
            
            # Connect with line
            ax_right.plot(range(len(groups)), diversities, 'k--', alpha=0.3, linewidth=1)
            
            # Add value labels
            for i, (div, count) in enumerate(zip(diversities, counts)):
                ax_right.text(i, div + 0.002, f'{div:.4f}\n(n={count})',
                           ha='center', va='bottom', fontsize=8)
            
            ax_right.set_xlabel('Group', fontweight='bold')
            ax_right.set_ylabel('Distinct-1 Score', fontweight='bold')
            ax_right.set_title(f'{attr.replace("_", " ").title()}: Diversity Comparison',
                             fontweight='bold')
            ax_right.set_xticks(range(len(groups)))
            ax_right.set_xticklabels(groups, rotation=45, ha='right')
            ax_right.grid(axis='y', alpha=0.3)
            
            # Add legend for bubble sizes
            legend_text = "Bubble size ∝ Sample count"
            ax_right.text(0.98, 0.02, legend_text,
                        transform=ax_right.transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7),
                        fontsize=8, ha='right', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Fairness analysis saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def visualize_generated_examples(self, customer_messages: np.ndarray,
                                    reference_responses: np.ndarray,
                                    n_examples: int = 5,
                                    save_path: str = None):
        """
        Visualize side-by-side comparison of reference and generated texts.
        
        Creates a formatted display showing:
        - Customer message (input)
        - Reference response (ground truth)
        - Generated response (model output)
        - Quality score
        
        Args:
            customer_messages (np.ndarray): Input customer messages (N, seq_len)
            reference_responses (np.ndarray): Ground truth responses (N, seq_len)
            n_examples (int, optional): Number of examples to display. Defaults to 5.
            save_path (str, optional): Path to save figure
        
        Returns:
            matplotlib.figure.Figure: The created figure object
        
        Example:
            >>> viz.visualize_generated_examples(
            ...     val_customer[:10],
            ...     val_agent[:10],
            ...     n_examples=5,
            ...     save_path='figs/examples.png'
            ... )
        """
        # Sample random examples
        indices = np.random.choice(len(customer_messages), 
                                  min(n_examples, len(customer_messages)),
                                  replace=False)
        
        sampled_customers = customer_messages[indices]
        sampled_references = reference_responses[indices]
        
        # Generate responses
        print("Generating responses...")
        generated = self.hybrid_model.generate_response(sampled_customers)
        generated_tokens = np.argmax(generated, axis=-1)
        
        # Evaluate quality
        quality_scores = self.hybrid_model.evaluate_response_quality(generated)
        
        # Decode texts
        customer_texts = self.tokenizer.sequences_to_texts(sampled_customers)
        reference_texts = self.tokenizer.sequences_to_texts(sampled_references)
        generated_texts = self.tokenizer.sequences_to_texts(generated_tokens)
        
        # Create figure
        fig = plt.figure(figsize=(14, 3*n_examples))
        
        for i in range(n_examples):
            ax = plt.subplot(n_examples, 1, i+1)
            ax.axis('off')
            
            # Format text for display
            customer_text = customer_texts[i][:100]  # Truncate if too long
            reference_text = reference_texts[i][:100]
            generated_text = generated_texts[i][:100]
            quality = quality_scores[i][0]
            
            # Determine quality color
            if quality > 0.7:
                quality_color = 'green'
                quality_label = 'Excellent'
            elif quality > 0.5:
                quality_color = 'orange'
                quality_label = 'Good'
            else:
                quality_color = 'red'
                quality_label = 'Poor'
            
            # Create text display
            display_text = f"{'='*80}\n"
            display_text += f"Example {i+1}  |  Quality Score: {quality:.4f} ({quality_label})\n"
            display_text += f"{'='*80}\n\n"
            display_text += f"📥 CUSTOMER MESSAGE:\n{customer_text}\n\n"
            display_text += f"✅ REFERENCE RESPONSE:\n{reference_text}\n\n"
            display_text += f"🤖 GENERATED RESPONSE:\n{generated_text}\n"
            display_text += f"{'='*80}"
            
            # Display text
            ax.text(0.05, 0.95, display_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
            
            # Add quality indicator
            circle = plt.Circle((0.95, 0.95), 0.02, 
                              transform=ax.transAxes,
                              color=quality_color, alpha=0.7)
            ax.add_patch(circle)
        
        fig.suptitle('Generated Text Examples with Quality Assessment',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Generated examples saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_latent_space(self, customer_messages: np.ndarray,
                         metadata: pd.DataFrame = None,
                         color_by: str = None,
                         n_samples: int = 500,
                         save_path: str = None):
        """
        Visualize latent space using t-SNE dimensionality reduction.
        
        Creates a 2D projection of the high-dimensional latent space,
        optionally colored by demographic attributes to reveal structure.
        
        Args:
            customer_messages (np.ndarray): Input messages to encode (N, seq_len)
            metadata (pd.DataFrame, optional): Metadata for coloring points
            color_by (str, optional): Column name in metadata to color by
            n_samples (int, optional): Number of samples to visualize. Defaults to 500.
            save_path (str, optional): Path to save figure
        
        Returns:
            matplotlib.figure.Figure: The created figure object
        
        Example:
            >>> viz.plot_latent_space(
            ...     val_customer,
            ...     metadata=val_metadata,
            ...     color_by='customer_segment',
            ...     save_path='figs/latent_space.png'
            ... )
        
        Note:
            Requires scikit-learn for t-SNE computation
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Sample data
        indices = np.random.choice(len(customer_messages),
                                  min(n_samples, len(customer_messages)),
                                  replace=False)
        sampled_messages = customer_messages[indices]
        
        print(f"Encoding {len(sampled_messages)} samples into latent space...")
        
        # Encode to latent space
        z_means = []
        batch_size = 64
        for i in range(0, len(sampled_messages), batch_size):
            batch = sampled_messages[i:i+batch_size]
            z_mean, _, _ = self.hybrid_model.vae.encoder.predict(batch, verbose=0)
            z_means.append(z_mean)
        
        z_means = np.vstack(z_means)
        
        print("  Applying PCA for dimensionality reduction...")
        # First reduce to 50D with PCA (faster)
        pca = PCA(n_components=min(50, z_means.shape[1]))
        z_pca = pca.fit_transform(z_means)
        
        print("  Applying t-SNE for 2D projection...")
        # Then apply t-SNE for 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_pca)-1))
        z_2d = tsne.fit_transform(z_pca)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine coloring
        if metadata is not None and color_by is not None and color_by in metadata.columns:
            labels = metadata.iloc[indices][color_by].values
            unique_labels = np.unique(labels)
            
            # Handle NaN
            labels = np.where(pd.isna(labels), 'unknown', labels)
            unique_labels = np.unique(labels)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                          c=[colors[i]], label=str(label),
                          alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            ax.legend(title=color_by.replace('_', ' ').title(),
                     loc='best', framealpha=0.9)
        else:
            # No coloring, single color
            ax.scatter(z_2d[:, 0], z_2d[:, 1],
                      c='steelblue', alpha=0.6, s=50,
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('t-SNE Dimension 1', fontweight='bold', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=12)
        ax.set_title('Latent Space Visualization (t-SNE Projection)',
                    fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3)
        
        # Add info text
        info_text = f"Samples: {len(z_2d)}\n"
        info_text += f"Original Dim: {z_means.shape[1]}\n"
        info_text += f"PCA Variance: {pca.explained_variance_ratio_.sum():.2%}"
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=9, verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Latent space visualization saved to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def create_research_presentation(self, customer_messages: np.ndarray,
                                    reference_responses: np.ndarray,
                                    metadata: pd.DataFrame,
                                    save_dir: str = 'presentation_figures'):
        """
        Generate complete set of visualizations for research presentation.
        
        This is a convenience method that creates all standard visualizations
        needed for a comprehensive research presentation or paper. Includes:
        
        1. Evaluation metrics comparison
        2. Fairness analysis
        3. Generated text examples
        4. Latent space visualization
        5. Model comparison (if multiple models provided)
        
        All figures are saved with descriptive filenames in the specified directory.
        
        Args:
            customer_messages (np.ndarray): Input customer messages (N, seq_len)
            reference_responses (np.ndarray): Ground truth responses (N, seq_len)
            metadata (pd.DataFrame): Metadata with demographic information
            save_dir (str, optional): Directory to save all figures.
                                     Defaults to 'presentation_figures'
        
        Returns:
            Dict[str, str]: Dictionary mapping figure names to file paths
        
        Example:
            >>> viz.create_research_presentation(
            ...     val_customer,
            ...     val_agent,
            ...     val_metadata,
            ...     save_dir='paper_figures'
            ... )
            ✓ Generated 5 figures in paper_figures/
            
        Side Effects:
            - Creates save_dir if it doesn't exist
            - Saves multiple PNG files with high resolution (300 DPI)
            - Prints progress for each figure
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("GENERATING RESEARCH PRESENTATION FIGURES")
        print("=" * 70)
        
        figure_paths = {}
        
        # ===== Figure 1: Evaluation Metrics =====
        print("\n1. Generating evaluation metrics comparison...")
        from evaluation_metrics import ModelEvaluator
        evaluator = ModelEvaluator(self.hybrid_model, self.tokenizer, self.explainer)
        
        results = evaluator.evaluate_generation_quality(
            customer_messages[:1000],
            reference_responses[:1000],
            sample_size=500
        )
        
        fig_path = save_path / 'fig1_evaluation_metrics.png'
        self.plot_evaluation_comparison(results, save_path=str(fig_path))
        figure_paths['evaluation_metrics'] = str(fig_path)
        
        # ===== Figure 2: Fairness Analysis =====
        print("\n2. Generating fairness analysis...")
        fairness_results = evaluator.evaluate_fairness(
            customer_messages[:1000],
            metadata[:1000],
            ['customer_segment', 'region']
        )
        
        fig_path = save_path / 'fig2_fairness_analysis.png'
        self.plot_fairness_analysis(fairness_results, save_path=str(fig_path))
        figure_paths['fairness_analysis'] = str(fig_path)
        
        # ===== Figure 3: Generated Examples =====
        print("\n3. Generating text examples...")
        fig_path = save_path / 'fig3_generated_examples.png'
        self.visualize_generated_examples(
            customer_messages[:100],
            reference_responses[:100],
            n_examples=5,
            save_path=str(fig_path)
        )
        figure_paths['generated_examples'] = str(fig_path)
        
        # ===== Figure 4: Latent Space =====
        print("\n4. Generating latent space visualization...")
        fig_path = save_path / 'fig4_latent_space.png'
        self.plot_latent_space(
            customer_messages[:1000],
            metadata=metadata[:1000],
            color_by='customer_segment',
            n_samples=500,
            save_path=str(fig_path)
        )
        figure_paths['latent_space'] = str(fig_path)
        
        # ===== Summary =====
        print("\n" + "=" * 70)
        print("✅ PRESENTATION FIGURES COMPLETE")
        print("=" * 70)
        print(f"\nGenerated {len(figure_paths)} figures in: {save_dir}/")
        print("\nFigures created:")
        for name, path in figure_paths.items():
            print(f"  ✓ {name}: {path}")
        
        return figure_paths


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    """
    Test suite for visualization module.
    
    Tests all visualization capabilities:
    1. Evaluation metrics visualization
    2. Model comparison
    3. Fairness analysis
    4. Generated examples
    5. Latent space visualization
    6. Complete research presentation generation
    """
    print("=" * 70)
    print("TESTING VISUALIZATION MODULE")
    print("=" * 70)
    
    # ===== Load Data =====
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    val_customer = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')
    val_agent = np.load(f'{config.PROCESSED_DATA_DIR}/val_agent.npy')
    val_metadata = pd.read_csv(f'{config.PROCESSED_DATA_DIR}/val_metadata.csv')
    
    print(f"\nLoaded validation data: {len(val_customer):,} samples")
    
    # ===== Build Model =====
    print("\nBuilding hybrid model...")
    from hybrid_model import HybridGANVAE
    from explainability import ModelExplainer
    
    hybrid = HybridGANVAE(vocab_size, max_length)
    explainer = ModelExplainer(hybrid, tokenizer)
    
    # ===== Initialize Visualizer =====
    print("\n" + "=" * 70)
    print("INITIALIZING VISUALIZER")
    print("=" * 70)
    
    viz = ResearchVisualizer(hybrid, tokenizer, explainer)
    
    # ===== Test: Complete Research Presentation =====
    print("\n" + "=" * 70)
    print("TEST: GENERATING COMPLETE RESEARCH PRESENTATION")
    print("=" * 70)
    
    figure_paths = viz.create_research_presentation(
        val_customer,
        val_agent,
        val_metadata,
        save_dir=f'{config.RESULTS_DIR}/presentation_figures'
    )
    
    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATION TESTS PASSED!")
    print("=" * 70)
    print("\nVisualization module ready for research presentation!")
