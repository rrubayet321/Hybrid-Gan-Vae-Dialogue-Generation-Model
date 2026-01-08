"""
Enhanced Inference with Diversity Sampling
==========================================

Deployment-ready inference script for generating agent responses with:
1. Temperature sampling for diverse generation
2. Nucleus (top-p) sampling
3. Repetition penalty
4. Quality assessment
5. Batch processing
6. API-ready structure

Uses the fine-tuned Separate Input model with diversity optimization.
"""

import os
import sys
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

# Import project modules
from config import *
from simple_tokenizer import SimpleTokenizer
from compare_input_approaches import SeparateInputHybridGANVAE
from evaluation_metrics import TextGenerationEvaluator
from diversity_metrics import DiversityMetrics


# ============================================================================
# Inference Engine with Diversity Sampling
# ============================================================================

class EnhancedInferenceEngine:
    """
    Enhanced inference engine with diversity sampling strategies.
    """
    
    def __init__(self,
                 model: SeparateInputHybridGANVAE,
                 tokenizer: SimpleTokenizer,
                 temperature: float = 1.5,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.2,
                 repetition_window: int = 20,
                 max_length: int = 100):
        """
        Initialize inference engine.
        
        Args:
            model: Trained SeparateInputHybridGANVAE model
            tokenizer: SimpleTokenizer instance
            temperature: Sampling temperature (higher = more diverse)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            repetition_window: Look-back window for repetition penalty
            max_length: Maximum generation length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.repetition_window = repetition_window
        self.max_length = max_length
        
        # Evaluators
        self.text_evaluator = TextGenerationEvaluator(tokenizer)
        self.diversity_metrics = DiversityMetrics(tokenizer)
        
        # Special tokens
        self.pad_token = 0
        self.start_token = tokenizer.word_index.get('<start>', 1)
        self.end_token = tokenizer.word_index.get('<end>', 2)
    
    def generate_response(self,
                         customer_message: str,
                         return_quality: bool = True,
                         verbose: bool = False) -> Dict:
        """
        Generate agent response for a customer message.
        
        Args:
            customer_message: Customer input text
            return_quality: Whether to compute quality metrics
            verbose: Print generation details
        
        Returns:
            Dictionary with:
                - response: Generated text
                - quality_metrics: Quality assessment (if return_quality=True)
                - generation_info: Generation metadata
        """
        if verbose:
            print(f"\n{'='*80}")
            print("GENERATING RESPONSE")
            print(f"{'='*80}")
            print(f"Customer: {customer_message}")
        
        # Preprocess input
        customer_seq = self.tokenizer.texts_to_sequences([customer_message])
        customer_seq = keras.preprocessing.sequence.pad_sequences(
            customer_seq, maxlen=self.max_length, padding='post'
        )
        
        # Generate with diversity sampling
        generated_tokens = self._generate_with_diversity_sampling(
            customer_seq[0],
            verbose=verbose
        )
        
        # Decode
        response = self.tokenizer.sequences_to_texts([generated_tokens])[0]
        
        if verbose:
            print(f"Agent: {response}")
        
        # Quality assessment
        result = {
            'response': response,
            'generation_info': {
                'temperature': self.temperature,
                'top_p': self.top_p,
                'repetition_penalty': self.repetition_penalty,
                'length': len(generated_tokens)
            }
        }
        
        if return_quality:
            quality = self._assess_quality([response])
            result['quality_metrics'] = quality
            
            if verbose:
                print(f"\n📊 Quality Assessment:")
                print(f"  Distinct-1: {quality['distinct_1']:.3f}")
                print(f"  Distinct-2: {quality['distinct_2']:.3f}")
                print(f"  Repetition Rate: {quality['repetition_rate']:.3f}")
                print(f"  Quality Score: {quality['quality_score']:.3f}")
        
        if verbose:
            print(f"{'='*80}\n")
        
        return result
    
    def generate_batch(self,
                      customer_messages: List[str],
                      return_quality: bool = True,
                      verbose: bool = False) -> List[Dict]:
        """
        Generate responses for multiple customer messages.
        
        Args:
            customer_messages: List of customer input texts
            return_quality: Whether to compute quality metrics
            verbose: Print progress
        
        Returns:
            List of response dictionaries
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"BATCH GENERATION: {len(customer_messages)} messages")
            print(f"{'='*80}\n")
        
        results = []
        for i, message in enumerate(customer_messages):
            if verbose and (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{len(customer_messages)}...")
            
            result = self.generate_response(
                message,
                return_quality=return_quality,
                verbose=False
            )
            results.append(result)
        
        # Aggregate quality metrics
        if return_quality:
            responses = [r['response'] for r in results]
            aggregate_quality = self._assess_quality(responses)
            
            if verbose:
                print(f"\n📊 Aggregate Quality Metrics:")
                print(f"  Distinct-1: {aggregate_quality['distinct_1']:.3f}")
                print(f"  Distinct-2: {aggregate_quality['distinct_2']:.3f}")
                print(f"  Repetition Rate: {aggregate_quality['repetition_rate']:.3f}")
                print(f"  Avg Quality Score: {aggregate_quality['quality_score']:.3f}")
                print(f"{'='*80}\n")
        
        return results
    
    def _generate_with_diversity_sampling(self,
                                         customer_input: np.ndarray,
                                         verbose: bool = False) -> List[int]:
        """
        Generate sequence with diversity sampling strategies.
        
        Args:
            customer_input: Customer input sequence [max_length]
            verbose: Print generation steps
        
        Returns:
            List of generated token IDs
        """
        # Initialize with start token
        generated = [self.start_token]
        
        # Prepare customer input batch
        customer_batch = np.expand_dims(customer_input, axis=0)
        
        # Generate token by token
        for step in range(self.max_length - 1):
            # Prepare agent input (current generation)
            agent_input = generated + [self.pad_token] * (self.max_length - len(generated))
            agent_input = agent_input[:self.max_length]
            agent_batch = np.expand_dims(agent_input, axis=0)
            
            # Get model predictions
            predictions = self.model.hybrid_model.predict(
                [customer_batch, agent_batch],
                verbose=0
            )[0]
            
            # Get logits for current position
            current_pos = len(generated) - 1
            logits = predictions[current_pos]
            
            # Apply repetition penalty
            if self.repetition_penalty > 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated, self.repetition_penalty, self.repetition_window
                )
            
            # Sample next token with diversity strategies
            if self.top_p < 1.0:
                # Nucleus sampling
                next_token = self._nucleus_sampling(logits, self.top_p, self.temperature)
            else:
                # Temperature sampling
                next_token = self._temperature_sampling(logits, self.temperature)
            
            # Add to sequence
            generated.append(int(next_token))
            
            # Stop if end token generated
            if next_token == self.end_token:
                break
        
        return generated
    
    def _temperature_sampling(self, logits: np.ndarray, temperature: float) -> int:
        """Apply temperature scaling and sample."""
        if temperature <= 0:
            return np.argmax(logits)
        
        # Scale by temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        # Sample
        return np.random.choice(len(probs), p=probs)
    
    def _nucleus_sampling(self, logits: np.ndarray, top_p: float, temperature: float) -> int:
        """Nucleus (top-p) sampling."""
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        # Sort in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Cumulative sum
        cumsum = np.cumsum(sorted_probs)
        
        # Find cutoff
        cutoff_idx = np.searchsorted(cumsum, top_p)
        
        # Nucleus
        nucleus_indices = sorted_indices[:cutoff_idx + 1]
        nucleus_probs = sorted_probs[:cutoff_idx + 1]
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
        
        # Sample
        sampled_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
        return nucleus_indices[sampled_idx]
    
    def _apply_repetition_penalty(self,
                                 logits: np.ndarray,
                                 generated: List[int],
                                 penalty: float,
                                 window: int) -> np.ndarray:
        """Apply repetition penalty to logits."""
        if not generated:
            return logits
        
        # Recent tokens
        recent = generated[-window:] if len(generated) > window else generated
        
        # Penalize
        penalized = logits.copy()
        for token_id in set(recent):
            if 0 <= token_id < len(penalized):
                penalized[token_id] /= penalty
        
        return penalized
    
    def _assess_quality(self, texts: List[str]) -> Dict:
        """Assess quality of generated texts."""
        metrics = self.diversity_metrics.compute_all_metrics(texts)
        
        # Quality score
        quality_score = (
            0.4 * metrics['distinct_2'] +
            0.3 * metrics['distinct_1'] +
            0.3 * (1.0 - metrics['repetition_rate'])
        )
        
        return {
            'distinct_1': metrics['distinct_1'],
            'distinct_2': metrics['distinct_2'],
            'distinct_3': metrics['distinct_3'],
            'repetition_rate': metrics['repetition_rate'],
            'entropy': metrics['entropy'],
            'quality_score': quality_score
        }


# ============================================================================
# API-Ready Wrapper
# ============================================================================

class InferenceAPI:
    """
    API-ready inference wrapper for deployment.
    """
    
    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 config: Optional[Dict] = None):
        """
        Initialize API.
        
        Args:
            model_path: Path to saved model weights
            tokenizer_path: Path to tokenizer pickle
            config: Optional configuration dictionary
        """
        print("🚀 Initializing Inference API...")
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"✓ Loaded tokenizer: {len(self.tokenizer.word_index)} tokens")
        
        # Build model
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model = SeparateInputHybridGANVAE(
            vocab_size=vocab_size,
            max_length=MAX_SEQUENCE_LENGTH,
            latent_dim=VAE_LATENT_DIM
        )
        print("✓ Model architecture built")
        
        # Load weights
        self.model.hybrid_model.load_weights(model_path)
        print(f"✓ Loaded weights from: {model_path}")
        
        # Configuration
        default_config = {
            'temperature': 1.5,
            'top_p': 0.9,
            'repetition_penalty': 1.2,
            'repetition_window': 20,
            'max_length': MAX_SEQUENCE_LENGTH
        }
        if config:
            default_config.update(config)
        
        # Initialize inference engine
        self.engine = EnhancedInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            **default_config
        )
        print("✓ Inference engine ready")
        print(f"\n⚙️  Configuration:")
        for key, value in default_config.items():
            print(f"  {key}: {value}")
        print()
    
    def generate(self, customer_message: str, **kwargs) -> Dict:
        """
        Generate agent response (single message).
        
        Args:
            customer_message: Customer input text
            **kwargs: Additional arguments for generation
        
        Returns:
            Response dictionary
        """
        return self.engine.generate_response(customer_message, **kwargs)
    
    def generate_batch(self, customer_messages: List[str], **kwargs) -> List[Dict]:
        """
        Generate agent responses (batch).
        
        Args:
            customer_messages: List of customer input texts
            **kwargs: Additional arguments for generation
        
        Returns:
            List of response dictionaries
        """
        return self.engine.generate_batch(customer_messages, **kwargs)
    
    def health_check(self) -> Dict:
        """API health check."""
        return {
            'status': 'healthy',
            'model': 'SeparateInputHybridGANVAE',
            'vocabulary_size': len(self.tokenizer.word_index),
            'max_length': self.engine.max_length,
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# Demo/Testing
# ============================================================================

def demo_inference():
    """Demo inference with examples."""
    print(f"\n{'='*80}")
    print("INFERENCE DEMO")
    print(f"{'='*80}\n")
    
    # Load model and tokenizer
    print("📦 Loading model and tokenizer...")
    
    model_path = 'models/diversity_optimized/final_model.weights.h5'
    tokenizer_path = 'processed_data/tokenizer.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run fine_tune_with_diversity.py first!")
        return
    
    # Initialize API
    api = InferenceAPI(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        config={
            'temperature': 1.5,
            'top_p': 0.9,
            'repetition_penalty': 1.2
        }
    )
    
    # Test examples
    examples = [
        "my account is locked after multiple failed login attempts",
        "cannot access the billing dashboard",
        "getting error when trying to export data",
        "password reset link not working",
        "2fa code not arriving via email"
    ]
    
    print(f"\n{'='*80}")
    print("GENERATING RESPONSES")
    print(f"{'='*80}\n")
    
    for i, customer_msg in enumerate(examples, 1):
        print(f"\nExample {i}/{len(examples)}")
        print(f"{'-'*80}")
        
        result = api.generate(customer_msg, return_quality=True, verbose=False)
        
        print(f"Customer: {customer_msg}")
        print(f"Agent:    {result['response']}")
        
        if 'quality_metrics' in result:
            q = result['quality_metrics']
            print(f"\nQuality:")
            print(f"  Distinct-2: {q['distinct_2']:.3f}")
            print(f"  Repetition: {q['repetition_rate']:.3f}")
            print(f"  Score: {q['quality_score']:.3f}")
    
    # Batch generation
    print(f"\n{'='*80}")
    print("BATCH GENERATION TEST")
    print(f"{'='*80}\n")
    
    batch_results = api.generate_batch(examples, return_quality=True, verbose=True)
    
    print(f"\n✅ Generated {len(batch_results)} responses")
    
    # Health check
    print(f"\n{'='*80}")
    print("API HEALTH CHECK")
    print(f"{'='*80}\n")
    
    health = api.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*80}")
    print("✅ INFERENCE DEMO COMPLETED!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    demo_inference()
