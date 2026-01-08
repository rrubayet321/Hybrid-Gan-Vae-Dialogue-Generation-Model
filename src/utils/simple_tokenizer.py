"""
Simple Tokenizer implementation to avoid TensorFlow's slow initialization
This tokenizer is compatible with Keras preprocessing but doesn't require TensorFlow import
"""

import pickle
from collections import Counter
from typing import List, Dict


class SimpleTokenizer:
    """A simple tokenizer that mimics Keras Tokenizer interface"""
    
    def __init__(self, num_words=None, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.filters = filters
        self.word_index = {}
        self.word_counts = Counter()
        self.document_count = 0
        
    def fit_on_texts(self, texts: List[str]):
        """Build vocabulary from texts"""
        self.document_count = len(texts)
        
        # Count words
        for text in texts:
            # Filter and split
            text = self._filter_text(text)
            words = text.split()
            self.word_counts.update(words)
        
        # Build word index
        # Reserve index 0 for padding
        # Index 1 for OOV token
        self.word_index = {self.oov_token: 1}
        
        # Sort by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size if specified
        if self.num_words:
            sorted_words = sorted_words[:self.num_words-2]  # -2 for padding and OOV
        
        for idx, (word, count) in enumerate(sorted_words, start=2):
            self.word_index[word] = idx
    
    def _filter_text(self, text: str) -> str:
        """Filter text using the filter string"""
        for char in self.filters:
            text = text.replace(char, ' ')
        return text.lower()
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to sequences of integers"""
        sequences = []
        oov_index = self.word_index.get(self.oov_token, 1)
        
        for text in texts:
            text = self._filter_text(text)
            words = text.split()
            sequence = []
            for word in words:
                if word in self.word_index:
                    idx = self.word_index[word]
                    # Respect num_words limit
                    if self.num_words is None or idx < self.num_words:
                        sequence.append(idx)
                    else:
                        sequence.append(oov_index)
                else:
                    sequence.append(oov_index)
            sequences.append(sequence)
        
        return sequences
    
    def sequences_to_texts(self, sequences: List[List[int]]) -> List[str]:
        """Convert sequences back to texts"""
        # Create reverse index
        reverse_word_index = {v: k for k, v in self.word_index.items()}
        
        texts = []
        for sequence in sequences:
            words = []
            for idx in sequence:
                if idx == 0:  # Padding
                    continue
                word = reverse_word_index.get(idx, self.oov_token)
                words.append(word)
            texts.append(' '.join(words))
        
        return texts


def pad_sequences_simple(sequences: List[List[int]], maxlen: int = None, 
                         padding='post', truncating='post', value=0) -> List[List[int]]:
    """Simple implementation of sequence padding"""
    import numpy as np
    
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded = []
    for sequence in sequences:
        # Truncate if necessary
        if len(sequence) > maxlen:
            if truncating == 'post':
                sequence = sequence[:maxlen]
            else:  # 'pre'
                sequence = sequence[-maxlen:]
        
        # Pad if necessary
        if len(sequence) < maxlen:
            pad_length = maxlen - len(sequence)
            if padding == 'post':
                sequence = sequence + [value] * pad_length
            else:  # 'pre'
                sequence = [value] * pad_length + sequence
        
        padded.append(sequence)
    
    return np.array(padded)
