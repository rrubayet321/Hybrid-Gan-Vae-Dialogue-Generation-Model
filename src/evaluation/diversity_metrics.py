"""
Diversity Metrics for Text Generation
Measures repetition and diversity in generated responses
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Union
import pandas as pd


class DiversityMetrics:
    """
    Compute diversity and repetition metrics for generated text.
    
    Metrics:
    - Distinct-1: Ratio of unique unigrams to total unigrams
    - Distinct-2: Ratio of unique bigrams to total bigrams
    - Distinct-3: Ratio of unique trigrams to total trigrams (bonus)
    - Repetition Rate: Percentage of repeated consecutive words
    - Self-BLEU: Average BLEU score between generated texts (lower is more diverse)
    - Entropy: Shannon entropy of word distribution
    """
    
    def __init__(self, tokenizer=None):
        """
        Initialize diversity metrics calculator.
        
        Args:
            tokenizer: Optional tokenizer for text processing
        """
        self.tokenizer = tokenizer
    
    def compute_distinct_n(self, texts: List[str], n: int = 1) -> float:
        """
        Compute Distinct-n score: ratio of unique n-grams to total n-grams.
        
        Args:
            texts: List of generated texts
            n: N-gram size (1 for unigrams, 2 for bigrams, etc.)
        
        Returns:
            Distinct-n score (0 to 1, higher is more diverse)
        """
        if not texts:
            return 0.0
        
        all_ngrams = []
        
        for text in texts:
            # Tokenize text
            if isinstance(text, str):
                tokens = text.lower().split()
            else:
                tokens = [str(t) for t in text]
            
            # Skip if not enough tokens
            if len(tokens) < n:
                continue
            
            # Extract n-grams
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
        
        if not all_ngrams:
            return 0.0
        
        # Calculate distinct score
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def compute_distinct_1(self, texts: List[str]) -> float:
        """
        Compute Distinct-1: ratio of unique words to total words.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Distinct-1 score (0 to 1, higher is more diverse)
        """
        return self.compute_distinct_n(texts, n=1)
    
    def compute_distinct_2(self, texts: List[str]) -> float:
        """
        Compute Distinct-2: ratio of unique bigrams to total bigrams.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Distinct-2 score (0 to 1, higher is more diverse)
        """
        return self.compute_distinct_n(texts, n=2)
    
    def compute_distinct_3(self, texts: List[str]) -> float:
        """
        Compute Distinct-3: ratio of unique trigrams to total trigrams.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Distinct-3 score (0 to 1, higher is more diverse)
        """
        return self.compute_distinct_n(texts, n=3)
    
    def compute_repetition_rate(self, text: str) -> float:
        """
        Compute repetition rate: percentage of consecutive repeated words.
        
        Args:
            text: Generated text
        
        Returns:
            Repetition rate (0 to 1, lower is better)
        """
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = [str(t) for t in text]
        
        if len(tokens) <= 1:
            return 0.0
        
        # Count consecutive repetitions
        repetitions = 0
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                repetitions += 1
        
        return repetitions / (len(tokens) - 1) if len(tokens) > 1 else 0.0
    
    def compute_batch_repetition_rate(self, texts: List[str]) -> float:
        """
        Compute average repetition rate across multiple texts.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Average repetition rate (0 to 1, lower is better)
        """
        if not texts:
            return 0.0
        
        rates = [self.compute_repetition_rate(text) for text in texts]
        return np.mean(rates)
    
    def compute_sequence_repetition(self, text: str, max_ngram: int = 4) -> Dict[str, float]:
        """
        Compute repetition for n-grams of different lengths.
        
        Args:
            text: Generated text
            max_ngram: Maximum n-gram size to check
        
        Returns:
            Dictionary with repetition rates for each n-gram size
        """
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = [str(t) for t in text]
        
        results = {}
        
        # Always initialize all n-gram keys
        for n in range(1, max_ngram + 1):
            results[f'repetition_{n}gram'] = 0.0
        
        # Compute repetition for each n-gram size
        for n in range(1, min(max_ngram + 1, len(tokens) + 1)):
            # Extract n-grams
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            
            if not ngrams:
                continue
            
            # Count duplicates
            ngram_counts = Counter(ngrams)
            total_ngrams = len(ngrams)
            repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
            
            results[f'repetition_{n}gram'] = repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        
        return results
    
    def compute_entropy(self, texts: List[str]) -> float:
        """
        Compute Shannon entropy of word distribution.
        Higher entropy indicates more diverse vocabulary usage.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Entropy score (higher is more diverse)
        """
        if not texts:
            return 0.0
        
        all_tokens = []
        for text in texts:
            if isinstance(text, str):
                tokens = text.lower().split()
            else:
                tokens = [str(t) for t in text]
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return 0.0
        
        # Calculate word frequencies
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # Calculate entropy
        entropy = 0.0
        for count in token_counts.values():
            prob = count / total_tokens
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def compute_inter_text_similarity(self, texts: List[str]) -> float:
        """
        Compute average Jaccard similarity between texts.
        Lower similarity indicates more diversity.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Average similarity (0 to 1, lower is more diverse)
        """
        if len(texts) < 2:
            return 0.0
        
        # Convert texts to sets of tokens
        token_sets = []
        for text in texts:
            if isinstance(text, str):
                tokens = set(text.lower().split())
            else:
                tokens = set(str(t) for t in text)
            token_sets.append(tokens)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                
                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def compute_all_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute all diversity metrics for a list of texts.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Dictionary with all diversity metrics
        """
        metrics = {
            'distinct_1': self.compute_distinct_1(texts),
            'distinct_2': self.compute_distinct_2(texts),
            'distinct_3': self.compute_distinct_3(texts),
            'repetition_rate': self.compute_batch_repetition_rate(texts),
            'entropy': self.compute_entropy(texts),
            'inter_text_similarity': self.compute_inter_text_similarity(texts)
        }
        
        # Add sequence repetition stats (average across texts)
        seq_reps = []
        for text in texts:
            seq_rep = self.compute_sequence_repetition(text, max_ngram=4)
            seq_reps.append(seq_rep)
        
        # Average sequence repetitions
        if seq_reps:
            for key in seq_reps[0].keys():
                values = [sr[key] for sr in seq_reps]
                metrics[key] = np.mean(values)
        
        return metrics
    
    def evaluate_diversity(self, texts: List[str], verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate diversity with detailed analysis and interpretation.
        
        Args:
            texts: List of generated texts
            verbose: Print detailed analysis
        
        Returns:
            Dictionary with diversity metrics and quality assessment
        """
        metrics = self.compute_all_metrics(texts)
        
        # Quality assessment
        quality_scores = {
            'distinct_1': metrics['distinct_1'],
            'distinct_2': metrics['distinct_2'],
            'distinct_3': metrics['distinct_3'],
            'low_repetition': 1.0 - metrics['repetition_rate'],
            'entropy': min(metrics['entropy'] / 10.0, 1.0),  # Normalize to 0-1
        }
        
        # Overall diversity score
        metrics['diversity_score'] = np.mean(list(quality_scores.values()))
        
        if verbose:
            print("\n" + "="*80)
            print("DIVERSITY METRICS ANALYSIS")
            print("="*80)
            print(f"\n📊 Distinct Metrics (higher is better):")
            print(f"  • Distinct-1 (unique words):     {metrics['distinct_1']:.4f}")
            print(f"  • Distinct-2 (unique bigrams):   {metrics['distinct_2']:.4f}")
            print(f"  • Distinct-3 (unique trigrams):  {metrics['distinct_3']:.4f}")
            
            print(f"\n🔄 Repetition Analysis (lower is better):")
            print(f"  • Consecutive repetition rate:   {metrics['repetition_rate']:.4f}")
            print(f"  • Unigram repetition:            {metrics.get('repetition_1gram', 0.0):.4f}")
            print(f"  • Bigram repetition:             {metrics.get('repetition_2gram', 0.0):.4f}")
            print(f"  • Trigram repetition:            {metrics.get('repetition_3gram', 0.0):.4f}")
            print(f"  • 4-gram repetition:             {metrics.get('repetition_4gram', 0.0):.4f}")
            
            print(f"\n📈 Other Diversity Measures:")
            print(f"  • Vocabulary entropy:            {metrics['entropy']:.4f}")
            print(f"  • Inter-text similarity:         {metrics['inter_text_similarity']:.4f}")
            
            print(f"\n⭐ Overall Diversity Score:        {metrics['diversity_score']:.4f}")
            
            # Quality interpretation
            print(f"\n💡 Interpretation:")
            if metrics['diversity_score'] >= 0.7:
                print("  ✅ Excellent diversity! Responses are varied and non-repetitive.")
            elif metrics['diversity_score'] >= 0.5:
                print("  ✓ Good diversity. Some room for improvement.")
            elif metrics['diversity_score'] >= 0.3:
                print("  ⚠ Moderate diversity. Consider increasing temperature or diversity penalties.")
            else:
                print("  ❌ Low diversity. Responses are highly repetitive. Action needed.")
            
            print("="*80 + "\n")
        
        return metrics


def compare_model_diversity(model_texts: Dict[str, List[str]], 
                           verbose: bool = True) -> pd.DataFrame:
    """
    Compare diversity metrics across multiple models.
    
    Args:
        model_texts: Dictionary mapping model names to lists of generated texts
        verbose: Print comparison table
    
    Returns:
        DataFrame with comparison results
    """
    calculator = DiversityMetrics()
    results = []
    
    for model_name, texts in model_texts.items():
        metrics = calculator.compute_all_metrics(texts)
        metrics['model'] = model_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model', 'distinct_1', 'distinct_2', 'distinct_3', 
            'repetition_rate', 'entropy', 'inter_text_similarity']
    df = df[[col for col in cols if col in df.columns]]
    
    if verbose:
        print("\n" + "="*80)
        print("MODEL DIVERSITY COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
    
    return df


def demo_diversity_metrics():
    """Demonstrate diversity metrics with example texts."""
    print("\n" + "="*80)
    print("DIVERSITY METRICS DEMO")
    print("="*80)
    
    # Example 1: Diverse responses
    diverse_texts = [
        "please check your network connection and try again",
        "have you tried resetting your password recently",
        "let me help you troubleshoot this issue step by step",
        "could you provide more details about the error message",
        "i recommend updating your software to the latest version"
    ]
    
    # Example 2: Repetitive responses
    repetitive_texts = [
        "please check please check please check",
        "try again try again try again try again",
        "the the the the system system system",
        "error error error message message message",
        "password password reset reset reset"
    ]
    
    calculator = DiversityMetrics()
    
    print("\n📋 EXAMPLE 1: Diverse Responses")
    print("-" * 80)
    for i, text in enumerate(diverse_texts, 1):
        print(f"{i}. {text}")
    metrics_diverse = calculator.evaluate_diversity(diverse_texts, verbose=True)
    
    print("\n📋 EXAMPLE 2: Repetitive Responses")
    print("-" * 80)
    for i, text in enumerate(repetitive_texts, 1):
        print(f"{i}. {text}")
    metrics_repetitive = calculator.evaluate_diversity(repetitive_texts, verbose=True)
    
    # Compare
    comparison = compare_model_diversity({
        'Diverse Model': diverse_texts,
        'Repetitive Model': repetitive_texts
    }, verbose=True)
    
    print("\n✅ Demo completed!")
    return metrics_diverse, metrics_repetitive, comparison


if __name__ == "__main__":
    demo_diversity_metrics()
