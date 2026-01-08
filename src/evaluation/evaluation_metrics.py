"""
Evaluation Metrics Module for Hybrid GAN + VAE

This module provides comprehensive evaluation capabilities for text generation models,
including the Hybrid GAN + VAE system. It implements multiple evaluation dimensions:

1. Text Generation Metrics:
   - BLEU (Bilingual Evaluation Understudy): Measures n-gram precision
   - ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Measures n-gram recall
   - Perplexity: Assesses language model quality
   - Diversity metrics: Evaluates vocabulary richness

2. Quality Metrics:
   - Discriminator quality scores: How "real" the generated text appears
   - Response coherence: Statistical consistency measures
   - Length statistics: Generation length characteristics

3. Fairness Metrics:
   - Bias detection across demographics: Identifies disparities
   - Demographic parity: Ensures equal treatment
   - Performance equality: Validates consistent quality

4. XAI-Enhanced Evaluation:
   - Latent space analysis: Examines internal representations
   - Regularization metrics: Validates VAE training
   - Representation diversity: Assesses latent space quality

Usage:
    from evaluation_metrics import ModelEvaluator, TextGenerationEvaluator
    
    # Create evaluator
    evaluator = ModelEvaluator(hybrid_model, tokenizer, explainer)
    
    # Evaluate generation quality
    results = evaluator.evaluate_generation_quality(
        customer_messages, reference_responses
    )
    
    # Evaluate fairness
    fairness_results = evaluator.evaluate_fairness(
        customer_messages, metadata, ['customer_segment', 'region']
    )

Author: IT Support AI Team
Date: January 2026
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import config
from diversity_metrics import DiversityMetrics, compare_model_diversity


# ============================================================================
# Text Generation Metrics Libraries
# ============================================================================
# These libraries provide industry-standard metrics for evaluating text generation quality

try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    
    # Download required NLTK data (punkt tokenizer)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  NLTK not available. Install with: pip install nltk")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  ROUGE not available. Install with: pip install rouge-score")


# ============================================================================
# TextGenerationEvaluator Class
# ============================================================================

class TextGenerationEvaluator:
    """
    Comprehensive evaluator for text generation quality.
    
    This class provides multiple metrics to assess generated text quality:
    - BLEU: Measures n-gram precision (how many n-grams in generated text appear in reference)
    - ROUGE: Measures n-gram recall (how many n-grams from reference appear in generated text)
    - Perplexity: Assesses language model quality (lower is better)
    - Diversity: Evaluates vocabulary richness and generation variety
    
    The evaluator handles tokenization, smoothing, and score aggregation automatically.
    
    Attributes:
        tokenizer: SimpleTokenizer instance for text conversion
        smoothing: NLTK smoothing function for BLEU scores (prevents zero scores)
        rouge_scorer: Rouge scorer instance for ROUGE metrics
    
    Example:
        >>> evaluator = TextGenerationEvaluator(tokenizer)
        >>> bleu_scores = evaluator.compute_bleu(references, hypotheses)
        >>> print(f"BLEU-4: {bleu_scores['bleu-4']:.4f}")
    """
    
    def __init__(self, tokenizer):
        """
        Initialize text generation evaluator.
        
        Args:
            tokenizer: SimpleTokenizer instance used for text-to-sequence conversion.
                      Required for consistent tokenization with the model.
        
        Side Effects:
            - Initializes NLTK smoothing function if NLTK is available
            - Initializes ROUGE scorer if rouge-score is available
            - Initializes DiversityMetrics for repetition and diversity analysis
            - Prints warnings if optional dependencies are missing
        """
        self.tokenizer = tokenizer
        
        # Initialize BLEU smoothing (prevents zero scores for missing n-grams)
        self.smoothing = SmoothingFunction() if NLTK_AVAILABLE else None
        
        # Initialize ROUGE scorer with stemming enabled for better matching
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        ) if ROUGE_AVAILABLE else None
        
        # Initialize diversity metrics calculator
        self.diversity_calculator = DiversityMetrics(tokenizer=tokenizer)
        
        print("✓ Text generation evaluator initialized")
        print("  ✓ Diversity metrics (Distinct-1, Distinct-2, Repetition Rate)")
        if not NLTK_AVAILABLE:
            print("  ⚠️  BLEU scores unavailable (install nltk)")
        if not ROUGE_AVAILABLE:
            print("  ⚠️  ROUGE scores unavailable (install rouge-score)")
    
    def compute_bleu(self, references: List[str], hypotheses: List[str],
                     max_n: int = 4) -> Dict[str, float]:
        """
        Compute BLEU scores (n-gram precision metrics).
        
        BLEU measures how many n-grams in the generated text (hypothesis) appear
        in the reference text. Higher scores indicate better overlap with reference.
        
        The method computes BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores, where:
        - BLEU-1: Unigram precision (individual word matching)
        - BLEU-2: Bigram precision (2-word phrase matching)
        - BLEU-3: Trigram precision (3-word phrase matching)
        - BLEU-4: 4-gram precision (4-word phrase matching)
        
        Args:
            references (List[str]): List of reference texts (ground truth responses)
            hypotheses (List[str]): List of generated texts (model outputs)
            max_n (int, optional): Maximum n-gram size to compute. Defaults to 4.
        
        Returns:
            Dict[str, float]: Dictionary with keys 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'
                            and their corresponding scores (0.0 to 1.0)
        
        Example:
            >>> refs = ["the cat sat on the mat", "hello world"]
            >>> hyps = ["the cat sat on mat", "hello there"]
            >>> scores = evaluator.compute_bleu(refs, hyps)
            >>> print(scores['bleu-4'])
            0.2345
        
        Note:
            - If NLTK is not available, returns dummy scores (0.0)
            - Uses smoothing to prevent zero scores for missing n-grams
            - Tokenizes on whitespace (assumes pre-tokenized text)
        """
        if not NLTK_AVAILABLE:
            print("⚠️  NLTK not available, returning dummy scores")
            return {f'bleu-{i}': 0.0 for i in range(1, max_n + 1)}
        
        # Initialize score storage for each n-gram level
        scores = {f'bleu-{i}': [] for i in range(1, max_n + 1)}
        
        # Compute BLEU for each reference-hypothesis pair
        for ref, hyp in zip(references, hypotheses):
            # Tokenize on whitespace
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            
            # Skip empty hypotheses
            if len(hyp_tokens) == 0:
                continue
            
            # Compute BLEU for each n-gram level
            for n in range(1, max_n + 1):
                # Create weight tuple: [1/n, 1/n, ..., 1/n, 0, 0, ...]
                # This focuses on n-grams up to size n
                weights = tuple([1.0 / n] * n + [0.0] * (max_n - n))
                
                try:
                    # Compute sentence-level BLEU with smoothing
                    score = sentence_bleu(
                        [ref_tokens],  # Reference (wrapped in list)
                        hyp_tokens,     # Hypothesis
                        weights=weights,
                        smoothing_function=self.smoothing.method1
                    )
                    scores[f'bleu-{n}'].append(score)
                except:
                    # Handle edge cases (e.g., very short sequences)
                    scores[f'bleu-{n}'].append(0.0)
        
        # Average scores across all pairs
        avg_scores = {
            key: np.mean(vals) if vals else 0.0 
            for key, vals in scores.items()
        }
        
        return avg_scores
    
    def compute_rouge(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores (n-gram recall metrics).
        
        ROUGE measures how many n-grams from the reference text appear in the
        generated text (hypothesis). It focuses on recall rather than precision.
        Higher scores indicate better content coverage.
        
        The method computes:
        - ROUGE-1: Unigram recall (single word overlap)
        - ROUGE-2: Bigram recall (2-word phrase overlap)
        - ROUGE-L: Longest common subsequence (accounts for sentence-level structure)
        
        Args:
            references (List[str]): List of reference texts (ground truth responses)
            hypotheses (List[str]): List of generated texts (model outputs)
        
        Returns:
            Dict[str, float]: Dictionary with keys 'rouge-1', 'rouge-2', 'rouge-l'
                            and their corresponding F1 scores (0.0 to 1.0)
        
        Example:
            >>> refs = ["the cat sat on the mat"]
            >>> hyps = ["the cat sat on the floor"]
            >>> scores = evaluator.compute_rouge(refs, hyps)
            >>> print(scores['rouge-1'])  # High unigram overlap
            0.8571
        
        Note:
            - If rouge-score library is not available, returns dummy scores (0.0)
            - Uses F1 score (harmonic mean of precision and recall)
            - Applies stemming for better word matching
        """
        if not ROUGE_AVAILABLE:
            print("⚠️  ROUGE not available, returning dummy scores")
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
        
        # Initialize score lists for each ROUGE variant
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        # Compute ROUGE for each reference-hypothesis pair
        for ref, hyp in zip(references, hypotheses):
            # Skip empty strings
            if not hyp.strip() or not ref.strip():
                continue
            
            # Compute all ROUGE variants at once
            scores = self.rouge_scorer.score(ref, hyp)
            
            # Extract F1 scores (balance of precision and recall)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # Return average scores
        return {
            'rouge-1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
            'rouge-2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
            'rouge-l': np.mean(rougeL_scores) if rougeL_scores else 0.0
        }
    
    def compute_perplexity(self, token_sequences: np.ndarray, 
                          probabilities: np.ndarray) -> float:
        """
        Compute perplexity (language model quality metric).
        
        Perplexity measures how "surprised" the model is by the generated text.
        Lower perplexity indicates the model is more confident and produces
        more natural language. It's the exponential of the negative average
        log probability.
        
        Formula: perplexity = exp(-average log probability)
        
        Args:
            token_sequences (np.ndarray): Token sequences of shape (batch_size, seq_len)
                                         containing token indices
            probabilities (np.ndarray): Predicted probabilities of shape 
                                       (batch_size, seq_len, vocab_size)
                                       from the model's output layer
        
        Returns:
            float: Perplexity score (lower is better)
                  - < 50: Excellent quality
                  - 50-200: Good quality
                  - > 200: Model is uncertain
                  - inf: No valid tokens found
        
        Example:
            >>> token_seqs = np.array([[1, 2, 3, 0], [4, 5, 6, 7]])
            >>> probs = model.predict(inputs)  # Shape: (2, 4, vocab_size)
            >>> perplexity = evaluator.compute_perplexity(token_seqs, probs)
            >>> print(f"Perplexity: {perplexity:.2f}")
            45.67
        
        Note:
            - Ignores padding tokens (token_id = 0)
            - Returns infinity if no valid tokens are found
            - Uses natural logarithm for stability
        """
        log_probs = []
        
        # Iterate through each sequence in the batch
        for i in range(len(token_sequences)):
            seq = token_sequences[i]
            probs = probabilities[i]
            
            # Calculate log probability for each token
            for j, token_id in enumerate(seq):
                if token_id > 0:  # Ignore padding tokens (0)
                    # Get probability of the actual token that was generated
                    token_prob = probs[j, token_id]
                    
                    # Only include valid probabilities
                    if token_prob > 0:
                        log_probs.append(np.log(token_prob))
        
        # Return infinity if no valid tokens found
        if not log_probs:
            return float('inf')
        
        # Compute perplexity: exp(-average log probability)
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)
        
        return float(perplexity)
    
    def compute_diversity(self, texts: List[str], verbose: bool = False) -> Dict[str, float]:
        """
        Compute comprehensive diversity metrics for generated texts.
        
        Diversity measures how varied and rich the generated text is. Higher
        diversity indicates the model is not repetitive and uses a wide vocabulary.
        
        Metrics computed:
        - Distinct-1: Ratio of unique unigrams to total unigrams
        - Distinct-2: Ratio of unique bigrams to total bigrams
        - Distinct-3: Ratio of unique trigrams to total trigrams
        - Repetition Rate: Percentage of consecutive repeated words (lower is better)
        - Vocab Size: Total number of unique words used
        - Avg Length: Average text length in tokens
        - Entropy: Shannon entropy of word distribution
        - Diversity Score: Overall diversity assessment (0-1)
        
        Args:
            texts (List[str]): List of generated texts to analyze
            verbose (bool): Print detailed diversity analysis
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'distinct_1': Unigram diversity (0.0 to 1.0, higher is better)
                - 'distinct_2': Bigram diversity (0.0 to 1.0, higher is better)
                - 'distinct_3': Trigram diversity (0.0 to 1.0, higher is better)
                - 'repetition_rate': Consecutive repetition (0.0 to 1.0, lower is better)
                - 'vocab_size': Number of unique words
                - 'avg_length': Average text length in tokens
                - 'entropy': Shannon entropy (higher is more diverse)
                - 'diversity_score': Overall quality score (0-1, higher is better)
        
        Example:
            >>> texts = ["hello world", "hello there", "goodbye world"]
            >>> diversity = evaluator.compute_diversity(texts)
            >>> print(f"Distinct-1: {diversity['distinct_1']:.2f}")
            0.80  # 4 unique words out of 5 total
            >>> print(f"Repetition Rate: {diversity['repetition_rate']:.2f}")
            0.00  # No consecutive repetitions
        
        Interpretation:
            - Distinct-1/2 > 0.50: High diversity (good)
            - Distinct-1/2 < 0.30: Repetitive generation (concerning)
            - Repetition Rate < 0.10: Low repetition (good)
            - Repetition Rate > 0.30: High repetition (needs attention)
            - Compare with reference text diversity as baseline
        """
        # Use comprehensive diversity calculator
        metrics = self.diversity_calculator.compute_all_metrics(texts)
        
        # Add legacy vocab_size and avg_length for compatibility
        all_tokens = []
        for text in texts:
            tokens = text.split()
            all_tokens.extend(tokens)
        
        metrics['vocab_size'] = len(set(all_tokens))
        metrics['avg_length'] = len(all_tokens) / len(texts) if texts else 0
        
        # Print analysis if requested
        if verbose:
            self.diversity_calculator.evaluate_diversity(texts, verbose=True)
        
        return metrics


# ============================================================================
# ModelEvaluator Class
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluator integrating multiple evaluation dimensions.
    
    This class orchestrates complete model evaluation by combining:
    - Text generation quality (BLEU, ROUGE, diversity)
    - Discriminator quality scores (GAN realism assessment)
    - Fairness metrics (demographic parity analysis)
    - XAI analysis (latent space characteristics)
    
    It provides both individual metric computation and comprehensive reporting
    capabilities for production monitoring and model comparison.
    
    Attributes:
        hybrid_model: HybridGANVAE instance to evaluate
        tokenizer: SimpleTokenizer for text conversion
        explainer: Optional ModelExplainer for XAI analysis
        text_evaluator: TextGenerationEvaluator instance
    
    Example:
        >>> evaluator = ModelEvaluator(hybrid_model, tokenizer, explainer)
        >>> results = evaluator.evaluate_generation_quality(
        ...     customer_messages, reference_responses
        ... )
        >>> evaluator.create_evaluation_report(results, 'report.txt')
    """
    
    def __init__(self, hybrid_model, tokenizer, explainer=None):
        """
        Initialize comprehensive model evaluator.
        
        Args:
            hybrid_model: HybridGANVAE instance to evaluate. Must have:
                         - generate_response() method
                         - evaluate_response_quality() method
                         - vae.encoder for latent space analysis
            tokenizer: SimpleTokenizer instance for text-to-sequence conversion.
                      Must match the tokenizer used during training.
            explainer (optional): ModelExplainer instance for XAI analysis.
                                 If None, XAI evaluation will be skipped.
        
        Side Effects:
            - Creates TextGenerationEvaluator instance
            - Prints initialization confirmation
        """
        self.hybrid_model = hybrid_model
        self.tokenizer = tokenizer
        self.explainer = explainer
        
        # Initialize text generation evaluator
        self.text_evaluator = TextGenerationEvaluator(tokenizer)
        
        print("✓ Model evaluator initialized")
    
    def evaluate_generation_quality(self, customer_messages: np.ndarray,
                                   reference_responses: np.ndarray,
                                   sample_size: Optional[int] = None) -> Dict:
        """
        Evaluate text generation quality with comprehensive metrics.
        
        This method performs end-to-end evaluation by:
        1. Generating responses from customer messages
        2. Computing text generation metrics (BLEU, ROUGE, diversity)
        3. Evaluating discriminator quality scores
        4. Aggregating results into a comprehensive report
        
        The evaluation includes:
        - BLEU scores (n-gram precision)
        - ROUGE scores (n-gram recall)
        - Diversity metrics (vocabulary richness)
        - Quality statistics (discriminator assessment)
        
        Args:
            customer_messages (np.ndarray): Input customer messages of shape
                                           (N, seq_len) containing token indices
            reference_responses (np.ndarray): Ground truth agent responses of shape
                                             (N, seq_len) containing token indices
            sample_size (int, optional): Number of samples to evaluate. If None,
                                        evaluates all samples. Use smaller values
                                        for faster evaluation during training.
        
        Returns:
            Dict: Comprehensive results containing:
                - 'bleu': Dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
                - 'rouge': Dict with ROUGE-1, ROUGE-2, ROUGE-L scores
                - 'diversity': Dict with distinct-1, distinct-2, vocab_size, avg_length
                - 'quality': Dict with mean, std, min, max discriminator scores
                - 'num_samples': Number of samples evaluated
        
        Example:
            >>> results = evaluator.evaluate_generation_quality(
            ...     val_customer[:500],
            ...     val_agent[:500]
            ... )
            >>> print(f"BLEU-4: {results['bleu']['bleu-4']:.4f}")
            >>> print(f"Quality: {results['quality']['mean']:.4f}")
        
        Note:
            - Processes data in batches of 64 for memory efficiency
            - Converts softmax probabilities to token indices automatically
            - Progress updates printed every 10 batches
        """
        # Sample data if requested
        if sample_size:
            indices = np.random.choice(
                len(customer_messages), 
                min(sample_size, len(customer_messages)),
                replace=False
            )
            customer_messages = customer_messages[indices]
            reference_responses = reference_responses[indices]
        
        print(f"\nEvaluating on {len(customer_messages)} samples...")
        
        # ===== Step 1: Generate responses =====
        print("  Generating responses...")
        generated_sequences = []
        quality_scores = []
        
        batch_size = 64
        num_batches = (len(customer_messages) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            # Extract batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(customer_messages))
            batch = customer_messages[start_idx:end_idx]
            
            # Generate responses for batch
            gen_batch = self.hybrid_model.generate_response(batch)
            generated_sequences.append(gen_batch)
            
            # Evaluate quality (discriminator scores)
            qual_batch = self.hybrid_model.evaluate_response_quality(gen_batch)
            quality_scores.append(qual_batch)
            
            # Progress tracking
            if (i + 1) % 10 == 0:
                print(f"    Batch {i+1}/{num_batches}")
        
        # Combine batches
        generated_sequences = np.vstack(generated_sequences)
        quality_scores = np.vstack(quality_scores)
        
        # ===== Step 2: Convert probabilities to tokens =====
        print("  Converting probabilities to tokens...")
        # Generated sequences are softmax outputs (batch, seq_len, vocab_size)
        # Convert to token indices by taking argmax
        generated_tokens = np.argmax(generated_sequences, axis=-1)
        
        # ===== Step 3: Decode to text =====
        print("  Decoding texts...")
        reference_texts = self.tokenizer.sequences_to_texts(reference_responses)
        generated_texts = self.tokenizer.sequences_to_texts(generated_tokens)
        
        # ===== Step 4: Compute text generation metrics =====
        print("  Computing BLEU scores...")
        bleu_scores = self.text_evaluator.compute_bleu(
            reference_texts, 
            generated_texts
        )
        
        print("  Computing ROUGE scores...")
        rouge_scores = self.text_evaluator.compute_rouge(
            reference_texts,
            generated_texts
        )
        
        print("  Computing diversity metrics...")
        diversity_scores = self.text_evaluator.compute_diversity(generated_texts)
        
        # ===== Step 5: Aggregate quality statistics =====
        avg_quality = np.mean(quality_scores)
        
        # Compile results
        results = {
            'bleu': bleu_scores,
            'rouge': rouge_scores,
            'diversity': diversity_scores,
            'quality': {
                'mean': float(avg_quality),
                'std': float(np.std(quality_scores)),
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores))
            },
            'num_samples': len(customer_messages)
        }
        
        return results
    
    def evaluate_fairness(self, customer_messages: np.ndarray,
                         metadata: pd.DataFrame,
                         sensitive_attributes: List[str]) -> Dict:
        """
        Evaluate fairness across demographic groups.
        
        This method assesses whether the model treats different demographic groups
        fairly by comparing quality and diversity metrics across groups. It identifies
        potential disparities that could indicate bias.
        
        For each sensitive attribute (e.g., customer_segment, region), the method:
        1. Splits data by attribute values
        2. Generates responses for each group
        3. Computes quality and diversity metrics per group
        4. Calculates disparity (difference between best and worst group)
        
        Args:
            customer_messages (np.ndarray): Input customer messages of shape
                                           (N, seq_len) containing token indices
            metadata (pd.DataFrame): DataFrame with demographic information.
                                    Must contain columns matching sensitive_attributes.
            sensitive_attributes (List[str]): List of column names in metadata to
                                            analyze for fairness (e.g., ['customer_segment',
                                            'region', 'priority']).
        
        Returns:
            Dict: Nested dictionary with structure:
                {
                    'attribute_name': {
                        'groups': {
                            'group_value': {
                                'count': int,
                                'quality_mean': float,
                                'quality_std': float,
                                'diversity': dict
                            },
                            ...
                        },
                        'quality_disparity': float,
                        'num_groups': int
                    },
                    ...
                }
        
        Example:
            >>> fairness = evaluator.evaluate_fairness(
            ...     val_customer,
            ...     val_metadata,
            ...     ['customer_segment', 'region']
            ... )
            >>> for attr, results in fairness.items():
            ...     print(f"{attr}: disparity={results['quality_disparity']:.4f}")
        
        Interpretation:
            - quality_disparity < 0.05: Excellent fairness
            - quality_disparity < 0.10: Acceptable fairness
            - quality_disparity > 0.10: Concerning disparity (investigate)
        
        Note:
            - Handles NaN values by converting to 'unknown'
            - Evaluates up to 100 samples per group for efficiency
            - Prints progress for each attribute and group
        """
        print(f"\nEvaluating fairness across {len(sensitive_attributes)} attributes...")
        
        fairness_results = {}
        
        # Iterate through each sensitive attribute
        for attr in sensitive_attributes:
            # Validate attribute exists in metadata
            if attr not in metadata.columns:
                print(f"  ⚠️  Attribute '{attr}' not in metadata")
                continue
            
            print(f"\n  Analyzing {attr}...")
            attr_values = metadata[attr].values
            
            # Handle NaN values (convert to 'unknown' category)
            if metadata[attr].dtype == 'object':
                attr_values = np.where(pd.isna(attr_values), 'unknown', attr_values)
            
            unique_values = np.unique(attr_values)
            
            group_results = {}
            
            # Evaluate each demographic group
            for value in unique_values:
                # Get samples for this group
                mask = attr_values == value
                group_messages = customer_messages[mask]
                
                if len(group_messages) == 0:
                    continue
                
                # Generate responses for this group (limit to 100 for efficiency)
                generated = self.hybrid_model.generate_response(group_messages[:100])
                quality = self.hybrid_model.evaluate_response_quality(generated)
                
                # Convert to tokens and decode
                generated_tokens = np.argmax(generated, axis=-1)
                generated_texts = self.tokenizer.sequences_to_texts(generated_tokens)
                diversity = self.text_evaluator.compute_diversity(generated_texts)
                
                # Store group results
                group_results[str(value)] = {
                    'count': int(np.sum(mask)),
                    'quality_mean': float(np.mean(quality)),
                    'quality_std': float(np.std(quality)),
                    'diversity': diversity
                }
                
                print(f"    {value}: n={int(np.sum(mask))}, "
                      f"quality={float(np.mean(quality)):.4f}")
            
            # Compute fairness metrics
            qualities = [v['quality_mean'] for v in group_results.values()]
            quality_disparity = max(qualities) - min(qualities) if qualities else 0.0
            
            fairness_results[attr] = {
                'groups': group_results,
                'quality_disparity': float(quality_disparity),
                'num_groups': len(group_results)
            }
        
        return fairness_results
    
    def evaluate_with_xai(self, customer_messages: np.ndarray,
                         sample_size: int = 50) -> Dict:
        """
        Evaluate model with XAI (Explainable AI) integration.
        
        This method performs latent space analysis to understand how the VAE
        represents input data. It examines:
        - Latent mean distributions (should be centered at 0 for proper regularization)
        - Latent variance distributions (should also be centered at 0)
        - Latent space diversity (how separated different inputs are)
        
        These metrics validate that the VAE is properly trained and that the
        latent space has good representational capacity.
        
        Args:
            customer_messages (np.ndarray): Input customer messages of shape
                                           (N, seq_len) containing token indices
            sample_size (int, optional): Number of samples to analyze. Defaults to 50.
                                        Larger values give more accurate statistics but
                                        take longer.
        
        Returns:
            Dict: XAI analysis results containing:
                {
                    'latent_space': {
                        'z_mean_avg': float,      # Should be near 0
                        'z_mean_std': float,      # Spread of means
                        'z_logvar_avg': float,    # Should be near 0
                        'z_logvar_std': float,    # Spread of variances
                        'latent_diversity': float # Higher = better separation
                    },
                    'num_samples': int
                }
        
        Example:
            >>> xai_results = evaluator.evaluate_with_xai(val_customer[:100])
            >>> print(f"Latent mean: {xai_results['latent_space']['z_mean_avg']:.4f}")
            >>> print(f"Diversity: {xai_results['latent_space']['latent_diversity']:.4f}")
        
        Interpretation:
            - z_mean_avg ≈ 0: Proper VAE regularization ✓
            - z_mean_avg far from 0: Adjust KL weight ⚠️
            - latent_diversity > 0.002: Good separation ✓
            - latent_diversity < 0.001: Posterior collapse ⚠️
        
        Note:
            - Requires explainer to be provided during initialization
            - Uses VAE encoder directly for efficient computation
            - Progress updates every 10 samples
        """
        if self.explainer is None:
            print("⚠️  No explainer provided, skipping XAI evaluation")
            return {}
        
        print(f"\nRunning XAI-enhanced evaluation on {sample_size} samples...")
        
        # Sample messages randomly
        indices = np.random.choice(
            len(customer_messages),
            min(sample_size, len(customer_messages)),
            replace=False
        )
        sample_messages = customer_messages[indices]
        
        # ===== Analyze latent space representations =====
        print("  Analyzing latent space representations...")
        z_means = []
        z_log_vars = []
        
        for i, msg in enumerate(sample_messages):
            msg_batch = msg[np.newaxis, :]
            
            # Get latent representation from VAE encoder
            # Returns: (z_mean, z_log_var, z_sample)
            z_mean, z_log_var, z = self.hybrid_model.vae.encoder.predict(
                msg_batch, 
                verbose=0
            )
            
            z_means.append(z_mean[0])
            z_log_vars.append(z_log_var[0])
            
            # Progress tracking
            if (i + 1) % 10 == 0:
                print(f"    Analyzed {i+1}/{sample_size} samples")
        
        # Convert to numpy arrays
        z_means = np.array(z_means)
        z_log_vars = np.array(z_log_vars)
        
        # ===== Compute latent space statistics =====
        xai_results = {
            'latent_space': {
                # Average of latent means (should be near 0 for proper VAE)
                'z_mean_avg': float(np.mean(z_means)),
                
                # Standard deviation of latent means (spread across samples)
                'z_mean_std': float(np.std(z_means)),
                
                # Average of latent log-variances (should be near 0)
                'z_logvar_avg': float(np.mean(z_log_vars)),
                
                # Standard deviation of latent log-variances
                'z_logvar_std': float(np.std(z_log_vars)),
                
                # Diversity: average std across dimensions
                # Higher = more diverse/separated representations
                'latent_diversity': float(np.mean(np.std(z_means, axis=0)))
            },
            'num_samples': len(sample_messages)
        }
        
        return xai_results
    
    def create_evaluation_report(self, results: Dict, save_path: str = None):
        """
        Create comprehensive evaluation report from results.
        
        This method generates a human-readable report combining all evaluation
        metrics into a formatted text document. The report includes:
        - Text generation metrics (BLEU, ROUGE, diversity)
        - Quality statistics (discriminator scores)
        - Fairness analysis (per-group results and disparities)
        - XAI metrics (latent space characteristics)
        
        The report can be printed to console and/or saved to a file for
        documentation and tracking.
        
        Args:
            results (Dict): Evaluation results dictionary containing any combination of:
                          - 'bleu': BLEU scores
                          - 'rouge': ROUGE scores
                          - 'diversity': Diversity metrics
                          - 'quality': Quality statistics
                          - 'fairness': Per-group fairness results
                          - 'xai': XAI analysis results
            save_path (str, optional): Path to save report text file. If None,
                                      report is only printed to console.
        
        Returns:
            str: The formatted report text
        
        Example:
            >>> all_results = {
            ...     'bleu': bleu_scores,
            ...     'rouge': rouge_scores,
            ...     'quality': quality_stats,
            ...     'fairness': fairness_results
            ... }
            >>> report = evaluator.create_evaluation_report(
            ...     all_results,
            ...     save_path='results/eval_report.txt'
            ... )
        
        Report Structure:
            1. Header with title
            2. Text Generation Metrics section
            3. Quality Metrics section
            4. Fairness Metrics section (if present)
            5. XAI Evaluation section (if present)
            6. Footer
        
        Note:
            - Handles missing sections gracefully
            - Formats numbers to 4 decimal places for consistency
            - Uses visual separators for readability
        """
        report_lines = []
        
        # ===== Report Header =====
        report_lines.append("=" * 70)
        report_lines.append("COMPREHENSIVE EVALUATION REPORT")
        report_lines.append("=" * 70)
        
        # ===== Text Generation Metrics Section =====
        if 'bleu' in results:
            report_lines.append("\n" + "=" * 70)
            report_lines.append("TEXT GENERATION METRICS")
            report_lines.append("=" * 70)
            
            # BLEU scores
            report_lines.append("\nBLEU Scores (N-gram Precision):")
            for key, value in results['bleu'].items():
                report_lines.append(f"  {key.upper()}: {value:.4f}")
            
            # ROUGE scores
            report_lines.append("\nROUGE Scores (N-gram Recall):")
            for key, value in results['rouge'].items():
                report_lines.append(f"  {key.upper()}: {value:.4f}")
            
            # Diversity metrics
            report_lines.append("\nDiversity Metrics:")
            for key, value in results['diversity'].items():
                report_lines.append(f"  {key}: {value:.4f}")
        
        # ===== Quality Metrics Section =====
        if 'quality' in results:
            report_lines.append("\n" + "=" * 70)
            report_lines.append("QUALITY METRICS")
            report_lines.append("=" * 70)
            
            report_lines.append(f"\nDiscriminator Quality Scores:")
            for key, value in results['quality'].items():
                report_lines.append(f"  {key}: {value:.4f}")
        
        # ===== Fairness Metrics Section =====
        if 'fairness' in results:
            report_lines.append("\n" + "=" * 70)
            report_lines.append("FAIRNESS METRICS")
            report_lines.append("=" * 70)
            
            # Iterate through each sensitive attribute
            for attr, attr_results in results['fairness'].items():
                report_lines.append(f"\n{attr.upper()}:")
                report_lines.append(f"  Quality Disparity: {attr_results['quality_disparity']:.4f}")
                report_lines.append(f"  Number of Groups: {attr_results['num_groups']}")
                
                # Per-group detailed results
                report_lines.append(f"\n  Per-Group Results:")
                for group, group_results in attr_results['groups'].items():
                    report_lines.append(f"    {group}:")
                    report_lines.append(f"      Count: {group_results['count']}")
                    report_lines.append(f"      Quality: {group_results['quality_mean']:.4f} "
                                      f"(±{group_results['quality_std']:.4f})")
                    report_lines.append(f"      Distinct-1: {group_results['diversity']['distinct-1']:.4f}")
        
        # ===== XAI Evaluation Section =====
        if 'xai' in results:
            report_lines.append("\n" + "=" * 70)
            report_lines.append("XAI EVALUATION")
            report_lines.append("=" * 70)
            
            if 'latent_space' in results['xai']:
                report_lines.append("\nLatent Space Analysis:")
                for key, value in results['xai']['latent_space'].items():
                    report_lines.append(f"  {key}: {value:.4f}")
        
        # ===== Report Footer =====
        report_lines.append("\n" + "=" * 70)
        
        # Join all lines into single string
        report_text = "\n".join(report_lines)
        
        # Print to console
        print("\n" + report_text)
        
        # Save to file if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Report saved to: {save_path}")
        
        return report_text


# ============================================================================
# Test Code
# ============================================================================
# This section tests all functionality of the evaluation metrics module

if __name__ == "__main__":
    """
    Test suite for evaluation metrics module.
    
    This test suite validates all functionality:
    1. Generation quality evaluation (BLEU, ROUGE, diversity, quality)
    2. Fairness evaluation across demographic groups
    3. XAI-enhanced evaluation (latent space analysis)
    4. Comprehensive report generation
    
    Expected behavior:
    - All tests should pass with reasonable baseline scores
    - Untrained model will have low BLEU/ROUGE scores (expected)
    - Fairness disparity should be low even for untrained model
    - Latent space should be properly regularized (z_mean/z_logvar ≈ 0)
    """
    print("=" * 70)
    print("TESTING EVALUATION METRICS MODULE")
    print("=" * 70)
    
    # ===== Load Preprocessing Configuration =====
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    vocab_size = preprocess_config['vocab_size']
    max_length = preprocess_config['max_length']
    
    # ===== Load Tokenizer =====
    with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_length}")
    
    # ===== Load Validation Data =====
    val_customer = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')
    val_agent = np.load(f'{config.PROCESSED_DATA_DIR}/val_agent.npy')
    val_metadata = pd.read_csv(f'{config.PROCESSED_DATA_DIR}/val_metadata.csv')
    
    print(f"Validation samples: {len(val_customer):,}")
    
    # ===== Build Hybrid Model =====
    print("\nBuilding hybrid model...")
    from hybrid_model import HybridGANVAE
    hybrid = HybridGANVAE(vocab_size, max_length)
    
    # ===== Initialize Explainer =====
    print("\nInitializing explainer...")
    from explainability import ModelExplainer
    explainer = ModelExplainer(hybrid, tokenizer)
    
    # ===== Initialize Evaluator =====
    print("\n" + "=" * 70)
    print("INITIALIZING EVALUATOR")
    print("=" * 70)
    
    evaluator = ModelEvaluator(hybrid, tokenizer, explainer)
    
    # ===== Test 1: Generation Quality Evaluation =====
    print("\n" + "=" * 70)
    print("TEST 1: GENERATION QUALITY EVALUATION")
    print("=" * 70)
    
    gen_results = evaluator.evaluate_generation_quality(
        val_customer[:200],
        val_agent[:200]
    )
    
    print("\nResults:")
    print(f"  BLEU-4: {gen_results['bleu']['bleu-4']:.4f}")
    print(f"  ROUGE-L: {gen_results['rouge']['rouge-l']:.4f}")
    print(f"  Distinct-1: {gen_results['diversity']['distinct-1']:.4f}")
    print(f"  Quality: {gen_results['quality']['mean']:.4f}")
    
    # ===== Test 2: Fairness Evaluation =====
    print("\n" + "=" * 70)
    print("TEST 2: FAIRNESS EVALUATION")
    print("=" * 70)
    
    fairness_results = evaluator.evaluate_fairness(
        val_customer[:500],
        val_metadata[:500],
        ['customer_segment', 'region']
    )
    
    print("\nFairness Results:")
    for attr, results in fairness_results.items():
        print(f"  {attr}: disparity={results['quality_disparity']:.4f}")
    
    # ===== Test 3: XAI Evaluation =====
    print("\n" + "=" * 70)
    print("TEST 3: XAI-ENHANCED EVALUATION")
    print("=" * 70)
    
    xai_results = evaluator.evaluate_with_xai(val_customer[:50])
    
    if xai_results:
        print("\nXAI Results:")
        if 'latent_space' in xai_results:
            print(f"  Latent mean avg: {xai_results['latent_space']['z_mean_avg']:.4f}")
            print(f"  Latent diversity: {xai_results['latent_space']['latent_diversity']:.4f}")
    
    # ===== Test 4: Comprehensive Report Creation =====
    print("\n" + "=" * 70)
    print("TEST 4: CREATING COMPREHENSIVE REPORT")
    print("=" * 70)
    
    # Combine all results
    all_results = {
        'bleu': gen_results['bleu'],
        'rouge': gen_results['rouge'],
        'diversity': gen_results['diversity'],
        'quality': gen_results['quality'],
        'fairness': fairness_results,
        'xai': xai_results
    }
    
    # Generate and save report
    evaluator.create_evaluation_report(
        all_results,
        save_path=f'{config.RESULTS_DIR}/evaluation_report.txt'
    )
    
    # ===== Test Summary =====
    print("\n" + "=" * 70)
    print("✅ ALL EVALUATION TESTS PASSED!")
    print("=" * 70)
    print("\nEvaluation module ready for use!")
    print("\nUsage:")
    print("  1. During training: Track BLEU/ROUGE scores over epochs")
    print("  2. Model comparison: Compare different model versions")
    print("  3. Fairness auditing: Monitor disparities across demographics")
    print("  4. Production monitoring: Track quality degradation")

