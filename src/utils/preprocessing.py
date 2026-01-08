"""
Data Preprocessing Pipeline for Synthetic IT Support Tickets
Handles text cleaning, tokenization, padding, and train/validation splits
"""

import pandas as pd
import numpy as np
import pickle
import re
import string
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import config
from simple_tokenizer import SimpleTokenizer, pad_sequences_simple


class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self):
        self.tokenizer = None
        self.vocab_size = None
        self.max_length = None
        
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load the IT support tickets dataset"""
        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for null values in critical columns
        null_counts = df[['initial_message', 'agent_first_reply']].isnull().sum()
        print(f"\nNull values:")
        print(f"  initial_message: {null_counts['initial_message']}")
        print(f"  agent_first_reply: {null_counts['agent_first_reply']}")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by:
        - Converting to lowercase
        - Removing URLs
        - Removing special characters and numbers
        - Removing extra whitespace
        - Removing stopwords
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation but keep important ones for sentence structure
        # Keep periods, commas, question marks, exclamation marks
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Define stopwords (common English stopwords)
        stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once'
        }
        
        # Remove stopwords but keep sentence structure
        words = text.split()
        words = [word for word in words if word not in stopwords or word in ['.', ',', '!', '?']]
        text = ' '.join(words)
        
        return text.strip()
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text cleaning to the entire dataframe"""
        print("\nCleaning text data...")
        
        # Fill NaN values with empty strings
        df['initial_message'] = df['initial_message'].fillna('')
        df['agent_first_reply'] = df['agent_first_reply'].fillna('')
        
        # Clean customer messages
        df['cleaned_customer_message'] = df['initial_message'].apply(self.clean_text)
        
        # Clean agent responses
        df['cleaned_agent_response'] = df['agent_first_reply'].apply(self.clean_text)
        
        # Remove rows where either cleaned text is empty
        original_len = len(df)
        df = df[(df['cleaned_customer_message'] != '') & (df['cleaned_agent_response'] != '')]
        print(f"Removed {original_len - len(df)} rows with empty messages")
        print(f"Remaining records: {len(df)}")
        
        return df
    
    def build_tokenizer(self, texts: List[str], vocab_size: int = None):
        """
        Build a tokenizer from the combined corpus of customer messages and agent responses
        """
        if vocab_size is None:
            vocab_size = config.MAX_VOCAB_SIZE
        
        print(f"\nBuilding tokenizer with vocab size: {vocab_size}")
        
        # Create tokenizer
        tokenizer = SimpleTokenizer(
            num_words=vocab_size,
            oov_token='<OOV>',  # Out of vocabulary token
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        
        # Fit on texts
        tokenizer.fit_on_texts(texts)
        
        # Get actual vocabulary size
        actual_vocab_size = len(tokenizer.word_index) + 1
        print(f"Actual vocabulary size: {actual_vocab_size}")
        print(f"Using vocabulary size: {min(vocab_size, actual_vocab_size)}")
        
        self.tokenizer = tokenizer
        self.vocab_size = min(vocab_size, actual_vocab_size)
        
        return tokenizer
    
    def tokenize_and_pad(
        self, 
        customer_texts: List[str], 
        agent_texts: List[str],
        max_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize and pad both customer messages and agent responses
        """
        if max_length is None:
            max_length = config.MAX_SEQUENCE_LENGTH
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not built. Call build_tokenizer first.")
        
        print(f"\nTokenizing and padding sequences (max length: {max_length})...")
        
        # Convert texts to sequences
        customer_sequences = self.tokenizer.texts_to_sequences(customer_texts)
        agent_sequences = self.tokenizer.texts_to_sequences(agent_texts)
        
        # Determine actual max length if not specified
        if max_length is None:
            max_customer_len = max(len(seq) for seq in customer_sequences)
            max_agent_len = max(len(seq) for seq in agent_sequences)
            max_length = max(max_customer_len, max_agent_len)
            print(f"Auto-detected max length: {max_length}")
        
        self.max_length = max_length
        
        # Pad sequences
        customer_padded = pad_sequences_simple(
            customer_sequences, 
            maxlen=max_length, 
            padding='post',
            truncating='post'
        )
        
        agent_padded = pad_sequences_simple(
            agent_sequences, 
            maxlen=max_length, 
            padding='post',
            truncating='post'
        )
        
        print(f"Customer sequences shape: {customer_padded.shape}")
        print(f"Agent sequences shape: {agent_padded.shape}")
        
        return customer_padded, agent_padded
    
    def split_data(
        self, 
        customer_sequences: np.ndarray, 
        agent_sequences: np.ndarray,
        metadata_df: pd.DataFrame = None,
        test_size: float = None
    ) -> Tuple:
        """
        Split data into training and validation sets while keeping pairs intact
        """
        if test_size is None:
            test_size = 1 - config.TRAIN_VAL_SPLIT
        
        print(f"\nSplitting data: {test_size*100:.0f}% validation, {(1-test_size)*100:.0f}% training")
        
        # Create indices
        indices = np.arange(len(customer_sequences))
        
        # Split indices
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=config.RANDOM_SEED,
            shuffle=True
        )
        
        # Split sequences
        train_customer = customer_sequences[train_idx]
        val_customer = customer_sequences[val_idx]
        train_agent = agent_sequences[train_idx]
        val_agent = agent_sequences[val_idx]
        
        print(f"Training samples: {len(train_customer)}")
        print(f"Validation samples: {len(val_customer)}")
        
        # Split metadata if provided
        train_metadata = None
        val_metadata = None
        if metadata_df is not None:
            train_metadata = metadata_df.iloc[train_idx].reset_index(drop=True)
            val_metadata = metadata_df.iloc[val_idx].reset_index(drop=True)
        
        return (train_customer, train_agent, val_customer, val_agent, 
                train_metadata, val_metadata)
    
    def save_preprocessed_data(
        self, 
        train_customer: np.ndarray,
        train_agent: np.ndarray,
        val_customer: np.ndarray,
        val_agent: np.ndarray,
        train_metadata: pd.DataFrame = None,
        val_metadata: pd.DataFrame = None
    ):
        """Save preprocessed data and tokenizer"""
        print("\nSaving preprocessed data...")
        
        # Save numpy arrays
        np.save(f'{config.PROCESSED_DATA_DIR}/train_customer.npy', train_customer)
        np.save(f'{config.PROCESSED_DATA_DIR}/train_agent.npy', train_agent)
        np.save(f'{config.PROCESSED_DATA_DIR}/val_customer.npy', val_customer)
        np.save(f'{config.PROCESSED_DATA_DIR}/val_agent.npy', val_agent)
        
        # Save metadata
        if train_metadata is not None:
            train_metadata.to_csv(f'{config.PROCESSED_DATA_DIR}/train_metadata.csv', index=False)
        if val_metadata is not None:
            val_metadata.to_csv(f'{config.PROCESSED_DATA_DIR}/val_metadata.csv', index=False)
        
        # Save tokenizer
        with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save preprocessing config
        preprocessing_config = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'train_size': len(train_customer),
            'val_size': len(val_customer)
        }
        
        with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'wb') as f:
            pickle.dump(preprocessing_config, f)
        
        print(f"Data saved to {config.PROCESSED_DATA_DIR}/")
    
    def load_preprocessed_data(self) -> Tuple:
        """Load preprocessed data and tokenizer"""
        print("Loading preprocessed data...")
        
        train_customer = np.load(f'{config.PROCESSED_DATA_DIR}/train_customer.npy')
        train_agent = np.load(f'{config.PROCESSED_DATA_DIR}/train_agent.npy')
        val_customer = np.load(f'{config.PROCESSED_DATA_DIR}/val_customer.npy')
        val_agent = np.load(f'{config.PROCESSED_DATA_DIR}/val_agent.npy')
        
        with open(f'{config.PROCESSED_DATA_DIR}/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        with open(f'{config.PROCESSED_DATA_DIR}/preprocessing_config.pkl', 'rb') as f:
            preprocessing_config = pickle.load(f)
            self.vocab_size = preprocessing_config['vocab_size']
            self.max_length = preprocessing_config['max_length']
        
        print(f"Loaded {len(train_customer)} training samples and {len(val_customer)} validation samples")
        
        return train_customer, train_agent, val_customer, val_agent


def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("Data Preprocessing Pipeline")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load dataset
    df = preprocessor.load_dataset(config.DATA_PATH)
    
    # Show sample data
    print("\n" + "="*60)
    print("Sample Raw Data:")
    print("="*60)
    print(f"Customer: {df['initial_message'].iloc[0]}")
    print(f"Agent: {df['agent_first_reply'].iloc[0]}")
    
    # Preprocess dataframe
    df = preprocessor.preprocess_dataframe(df)
    
    # Show sample cleaned data
    print("\n" + "="*60)
    print("Sample Cleaned Data:")
    print("="*60)
    print(f"Customer: {df['cleaned_customer_message'].iloc[0]}")
    print(f"Agent: {df['cleaned_agent_response'].iloc[0]}")
    
    # Build tokenizer on combined corpus
    all_texts = (
        df['cleaned_customer_message'].tolist() + 
        df['cleaned_agent_response'].tolist()
    )
    tokenizer = preprocessor.build_tokenizer(all_texts)
    
    # Show sample vocabulary
    print("\n" + "="*60)
    print("Sample Vocabulary (first 20 words):")
    print("="*60)
    word_index = tokenizer.word_index
    sample_words = list(word_index.items())[:20]
    for word, idx in sample_words:
        print(f"  {word}: {idx}")
    
    # Tokenize and pad
    customer_padded, agent_padded = preprocessor.tokenize_and_pad(
        df['cleaned_customer_message'].tolist(),
        df['cleaned_agent_response'].tolist()
    )
    
    # Show tokenized examples
    print("\n" + "="*60)
    print("Sample Tokenized Sequences:")
    print("="*60)
    for i in range(3):
        print(f"\nExample {i+1}:")
        print(f"Original Customer: {df['initial_message'].iloc[i]}")
        print(f"Cleaned Customer: {df['cleaned_customer_message'].iloc[i]}")
        print(f"Tokenized Customer: {customer_padded[i][:20]}...")
        print(f"Original Agent: {df['agent_first_reply'].iloc[i]}")
        print(f"Cleaned Agent: {df['cleaned_agent_response'].iloc[i]}")
        print(f"Tokenized Agent: {agent_padded[i][:20]}...")
    
    # Prepare metadata for bias analysis
    metadata_columns = ['customer_segment', 'region', 'priority', 'customer_sentiment']
    metadata_df = df[metadata_columns].copy() if all(col in df.columns for col in metadata_columns) else None
    
    # Split data
    (train_customer, train_agent, val_customer, val_agent, 
     train_metadata, val_metadata) = preprocessor.split_data(
        customer_padded, 
        agent_padded,
        metadata_df
    )
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(
        train_customer, train_agent, val_customer, val_agent,
        train_metadata, val_metadata
    )
    
    # Summary statistics
    print("\n" + "="*60)
    print("Preprocessing Summary:")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    print(f"Max sequence length: {preprocessor.max_length}")
    print(f"Training samples: {len(train_customer)}")
    print(f"Validation samples: {len(val_customer)}")
    print(f"\nData saved to: {config.PROCESSED_DATA_DIR}/")
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
