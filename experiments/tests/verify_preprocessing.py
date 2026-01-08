"""
Verification script to inspect preprocessed data
"""

import numpy as np
import pandas as pd
import pickle
import config


def main():
    print("="*60)
    print("Verifying Preprocessed Data")
    print("="*60)
    
    # Load tokenizer
    with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load config
    with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
        preprocess_config = pickle.load(f)
    
    print("\nPreprocessing Configuration:")
    for key, value in preprocess_config.items():
        print(f"  {key}: {value}")
    
    # Load data
    train_customer = np.load(f'{config.PROCESSED_DATA_DIR}/train_customer.npy')
    train_agent = np.load(f'{config.PROCESSED_DATA_DIR}/train_agent.npy')
    val_customer = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')
    val_agent = np.load(f'{config.PROCESSED_DATA_DIR}/val_agent.npy')
    
    print(f"\n" + "="*60)
    print("Data Shapes:")
    print("="*60)
    print(f"Training customer sequences: {train_customer.shape}")
    print(f"Training agent sequences: {train_agent.shape}")
    print(f"Validation customer sequences: {val_customer.shape}")
    print(f"Validation agent sequences: {val_agent.shape}")
    
    # Show vocabulary statistics
    print(f"\n" + "="*60)
    print("Vocabulary Statistics:")
    print("="*60)
    print(f"Total unique tokens: {len(tokenizer.word_index)}")
    print(f"\nMost common words:")
    # Sort by index (lower index = more frequent)
    sorted_words = sorted(tokenizer.word_index.items(), key=lambda x: x[1])[:30]
    for word, idx in sorted_words:
        count = tokenizer.word_counts.get(word, 0)
        print(f"  {idx:3d}. {word:20s} (count: {count})")
    
    # Show sample decoded sequences
    print(f"\n" + "="*60)
    print("Sample Decoded Sequences (First 5 from validation):")
    print("="*60)
    
    for i in range(5):
        customer_seq = val_customer[i]
        agent_seq = val_agent[i]
        
        # Decode sequences
        customer_text = tokenizer.sequences_to_texts([customer_seq.tolist()])[0]
        agent_text = tokenizer.sequences_to_texts([agent_seq.tolist()])[0]
        
        print(f"\nExample {i+1}:")
        print(f"Customer ({len([x for x in customer_seq if x != 0])} tokens): {customer_text}")
        print(f"Agent ({len([x for x in agent_seq if x != 0])} tokens): {agent_text}")
    
    # Statistics on sequence lengths
    print(f"\n" + "="*60)
    print("Sequence Length Statistics:")
    print("="*60)
    
    def non_zero_lengths(sequences):
        return [len([x for x in seq if x != 0]) for seq in sequences]
    
    train_customer_lengths = non_zero_lengths(train_customer)
    train_agent_lengths = non_zero_lengths(train_agent)
    
    print(f"\nCustomer Message Lengths (excluding padding):")
    print(f"  Mean: {np.mean(train_customer_lengths):.2f}")
    print(f"  Median: {np.median(train_customer_lengths):.2f}")
    print(f"  Min: {np.min(train_customer_lengths)}")
    print(f"  Max: {np.max(train_customer_lengths)}")
    print(f"  Std: {np.std(train_customer_lengths):.2f}")
    
    print(f"\nAgent Response Lengths (excluding padding):")
    print(f"  Mean: {np.mean(train_agent_lengths):.2f}")
    print(f"  Median: {np.median(train_agent_lengths):.2f}")
    print(f"  Min: {np.min(train_agent_lengths)}")
    print(f"  Max: {np.max(train_agent_lengths)}")
    print(f"  Std: {np.std(train_agent_lengths):.2f}")
    
    # Load metadata
    print(f"\n" + "="*60)
    print("Metadata Information:")
    print("="*60)
    
    train_metadata = pd.read_csv(f'{config.PROCESSED_DATA_DIR}/train_metadata.csv')
    
    print(f"\nCustomer Segment Distribution:")
    print(train_metadata['customer_segment'].value_counts())
    
    print(f"\nRegion Distribution:")
    print(train_metadata['region'].value_counts())
    
    print(f"\nPriority Distribution:")
    print(train_metadata['priority'].value_counts())
    
    print(f"\nSentiment Distribution:")
    print(train_metadata['customer_sentiment'].value_counts())
    
    print("\n" + "="*60)
    print("Verification Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
