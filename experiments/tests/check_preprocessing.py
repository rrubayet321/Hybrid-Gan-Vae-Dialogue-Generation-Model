"""
Check and verify the preprocessing results
Shows tokenized examples and data statistics
"""

import numpy as np
import pandas as pd
import pickle
from simple_tokenizer import SimpleTokenizer

def load_preprocessed_data():
    """Load all preprocessed data"""
    print("=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)
    
    # Load tokenizer
    with open('processed_data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load preprocessing config
    with open('processed_data/preprocessing_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    # Load training data
    train_customer = np.load('processed_data/train_customer.npy')
    train_agent = np.load('processed_data/train_agent.npy')
    train_metadata = pd.read_csv('processed_data/train_metadata.csv')
    
    # Load validation data
    val_customer = np.load('processed_data/val_customer.npy')
    val_agent = np.load('processed_data/val_agent.npy')
    val_metadata = pd.read_csv('processed_data/val_metadata.csv')
    
    print(f"\n✓ Successfully loaded all preprocessed data")
    
    return {
        'tokenizer': tokenizer,
        'config': config,
        'train_customer': train_customer,
        'train_agent': train_agent,
        'train_metadata': train_metadata,
        'val_customer': val_customer,
        'val_agent': val_agent,
        'val_metadata': val_metadata
    }

def print_data_statistics(data):
    """Print statistics about the preprocessed data"""
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    
    print(f"\n📊 Dataset Split:")
    print(f"  Training samples: {len(data['train_customer']):,}")
    print(f"  Validation samples: {len(data['val_customer']):,}")
    print(f"  Total samples: {len(data['train_customer']) + len(data['val_customer']):,}")
    
    print(f"\n📝 Sequence Properties:")
    print(f"  Vocabulary size: {data['config']['vocab_size']:,}")
    print(f"  Max sequence length: {data['config']['max_length']}")
    
    print(f"\n📐 Data Shapes:")
    print(f"  Train customer shape: {data['train_customer'].shape}")
    print(f"  Train agent shape: {data['train_agent'].shape}")
    print(f"  Val customer shape: {data['val_customer'].shape}")
    print(f"  Val agent shape: {data['val_agent'].shape}")
    
    # Get tokenizer info
    tokenizer = data['tokenizer']
    print(f"\n🔤 Tokenizer Information:")
    print(f"  Total unique words: {len(tokenizer.word_index):,}")
    print(f"  OOV token: {tokenizer.oov_token}")
    
    # Show some metadata statistics
    print(f"\n📋 Metadata Columns:")
    print(f"  {', '.join(data['train_metadata'].columns.tolist())}")

def show_tokenized_examples(data, num_examples=5):
    """Show examples of tokenized sequences"""
    print("\n" + "=" * 60)
    print("TOKENIZED EXAMPLES")
    print("=" * 60)
    
    tokenizer = data['tokenizer']
    
    # Create reverse word index for decoding
    reverse_word_index = {idx: word for word, idx in tokenizer.word_index.items()}
    
    def decode_sequence(sequence):
        """Convert token sequence back to text"""
        words = []
        for idx in sequence:
            if idx == 0:  # Padding
                continue
            word = reverse_word_index.get(idx, '<UNK>')
            words.append(word)
        return ' '.join(words)
    
    print(f"\nShowing {num_examples} random examples from training set:\n")
    
    # Get random indices
    random_indices = np.random.choice(len(data['train_customer']), num_examples, replace=False)
    
    for i, idx in enumerate(random_indices, 1):
        customer_seq = data['train_customer'][idx]
        agent_seq = data['train_agent'][idx]
        metadata = data['train_metadata'].iloc[idx]
        
        print(f"{'─' * 60}")
        print(f"Example {i} (Training Set Index: {idx}):")
        print(f"{'─' * 60}")
        
        print(f"\n📌 Metadata:")
        print(f"  Customer Segment: {metadata.get('customer_segment', 'N/A')}")
        print(f"  Region: {metadata.get('region', 'N/A')}")
        print(f"  Priority: {metadata.get('priority', 'N/A')}")
        print(f"  Customer Sentiment: {metadata.get('customer_sentiment', 'N/A')}")
        
        print(f"\n🔢 Customer Message (Tokenized):")
        customer_tokens = customer_seq[customer_seq != 0]  # Remove padding
        print(f"  Token IDs: {list(customer_tokens[:20])}...")
        print(f"  Number of tokens: {len(customer_tokens)}")
        
        print(f"\n📝 Customer Message (Decoded from tokens):")
        decoded_customer = decode_sequence(customer_seq)
        print(f"  {decoded_customer}")
        
        print(f"\n🔢 Agent Response (Tokenized):")
        agent_tokens = agent_seq[agent_seq != 0]  # Remove padding
        print(f"  Token IDs: {list(agent_tokens[:20])}...")
        print(f"  Number of tokens: {len(agent_tokens)}")
        
        print(f"\n📝 Agent Response (Decoded from tokens):")
        decoded_agent = decode_sequence(agent_seq)
        print(f"  {decoded_agent}")
        
        print()

def show_vocabulary_samples(data):
    """Show samples from the vocabulary"""
    print("\n" + "=" * 60)
    print("VOCABULARY SAMPLES")
    print("=" * 60)
    
    tokenizer = data['tokenizer']
    word_index = tokenizer.word_index
    
    # Get most common words (first 30)
    print("\n🔝 Top 30 Most Common Words:")
    common_words = list(word_index.items())[:30]
    for i, (word, idx) in enumerate(common_words, 1):
        print(f"  {i:2d}. '{word}' → Token ID: {idx}")
    
    # Show some random words from middle of vocabulary
    print("\n🎲 Sample Words from Vocabulary (Random):")
    all_words = list(word_index.items())
    if len(all_words) > 100:
        random_sample = np.random.choice(len(all_words), 20, replace=False)
        random_words = [all_words[i] for i in random_sample]
        for word, idx in sorted(random_words, key=lambda x: x[1])[:20]:
            print(f"  '{word}' → Token ID: {idx}")

def verify_data_integrity(data):
    """Verify data integrity and consistency"""
    print("\n" + "=" * 60)
    print("DATA INTEGRITY CHECKS")
    print("=" * 60)
    
    checks_passed = []
    checks_failed = []
    
    # Check 1: Customer and agent sequences have same length
    if len(data['train_customer']) == len(data['train_agent']):
        checks_passed.append("✓ Training customer-agent pairs are aligned")
    else:
        checks_failed.append("✗ Training customer-agent pairs misaligned")
    
    if len(data['val_customer']) == len(data['val_agent']):
        checks_passed.append("✓ Validation customer-agent pairs are aligned")
    else:
        checks_failed.append("✗ Validation customer-agent pairs misaligned")
    
    # Check 2: Sequences have correct max length
    max_len = data['config']['max_length']
    if data['train_customer'].shape[1] == max_len:
        checks_passed.append(f"✓ Customer sequences padded to {max_len}")
    else:
        checks_failed.append(f"✗ Customer sequences not padded correctly")
    
    if data['train_agent'].shape[1] == max_len:
        checks_passed.append(f"✓ Agent sequences padded to {max_len}")
    else:
        checks_failed.append(f"✗ Agent sequences not padded correctly")
    
    # Check 3: No NaN values
    if not np.isnan(data['train_customer']).any():
        checks_passed.append("✓ No NaN values in training customer data")
    else:
        checks_failed.append("✗ NaN values found in training customer data")
    
    if not np.isnan(data['train_agent']).any():
        checks_passed.append("✓ No NaN values in training agent data")
    else:
        checks_failed.append("✗ NaN values found in training agent data")
    
    # Check 4: Token IDs within vocabulary range
    vocab_size = data['config']['vocab_size']
    max_customer_token = data['train_customer'].max()
    max_agent_token = data['train_agent'].max()
    
    if max_customer_token < vocab_size:
        checks_passed.append(f"✓ Customer tokens within vocab range (max: {max_customer_token})")
    else:
        checks_failed.append(f"✗ Customer tokens exceed vocab range (max: {max_customer_token})")
    
    if max_agent_token < vocab_size:
        checks_passed.append(f"✓ Agent tokens within vocab range (max: {max_agent_token})")
    else:
        checks_failed.append(f"✗ Agent tokens exceed vocab range (max: {max_agent_token})")
    
    # Check 5: Metadata matches sequence count
    if len(data['train_metadata']) == len(data['train_customer']):
        checks_passed.append("✓ Training metadata matches sequence count")
    else:
        checks_failed.append("✗ Training metadata count mismatch")
    
    if len(data['val_metadata']) == len(data['val_customer']):
        checks_passed.append("✓ Validation metadata matches sequence count")
    else:
        checks_failed.append("✗ Validation metadata count mismatch")
    
    # Print results
    print("\n✅ Passed Checks:")
    for check in checks_passed:
        print(f"  {check}")
    
    if checks_failed:
        print("\n❌ Failed Checks:")
        for check in checks_failed:
            print(f"  {check}")
    else:
        print("\n🎉 All integrity checks passed!")
    
    return len(checks_failed) == 0

def main():
    """Main function to run all checks"""
    print("\n" + "=" * 60)
    print("PREPROCESSING VERIFICATION")
    print("=" * 60)
    
    try:
        # Load data
        data = load_preprocessed_data()
        
        # Show statistics
        print_data_statistics(data)
        
        # Show tokenized examples
        show_tokenized_examples(data, num_examples=3)
        
        # Show vocabulary samples
        show_vocabulary_samples(data)
        
        # Verify integrity
        integrity_ok = verify_data_integrity(data)
        
        # Final summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        if integrity_ok:
            print("\n✅ Preprocessing completed successfully!")
            print("✅ All data integrity checks passed!")
            print("✅ Dataset is ready for model training!")
        else:
            print("\n⚠️  Some integrity checks failed. Please review above.")
        
        print("\n📂 Preprocessed files location: processed_data/")
        print("  - tokenizer.pkl")
        print("  - train_customer.npy")
        print("  - train_agent.npy")
        print("  - val_customer.npy")
        print("  - val_agent.npy")
        print("  - train_metadata.csv")
        print("  - val_metadata.csv")
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
