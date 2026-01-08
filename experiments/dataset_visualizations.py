"""
Dataset Visualizations for Proof of Concept (PoC)
==================================================

This script generates key visualizations for understanding the IT support
tickets dataset structure and characteristics:

1. Word Frequency Distribution - Top 10 most frequent words in customer messages
2. Sentiment Distribution - Distribution of sentiments (positive, neutral, negative)

These visualizations are essential for:
- Understanding dataset characteristics
- Identifying common customer concerns
- Analyzing sentiment patterns
- Supporting PoC documentation

Usage:
------
```bash
python dataset_visualizations.py
```

Output:
-------
- word_freq.png: Bar chart of top 10 most frequent words
- sentiment_dist.png: Count plot of sentiment distribution

Author: Research Team
Date: January 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings

# Try to import NLTK
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Try to download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        try:
            nltk.download('punkt', quiet=True)
        except:
            print("⚠ Could not download NLTK data - using basic tokenization")
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        try:
            nltk.download('stopwords', quiet=True)
        except:
            print("⚠ Could not download stopwords - will use basic filtering")
    
    NLTK_AVAILABLE = True
    try:
        STOP_WORDS = set(stopwords.words('english'))
    except:
        STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                      'could', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it',
                      'we', 'they', 'them', 'their', 'this', 'that', 'these', 'those'}
        
except ImportError:
    print("⚠ NLTK not available - using basic tokenization")
    NLTK_AVAILABLE = False
    STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                  'could', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it',
                  'we', 'they', 'them', 'their', 'this', 'that', 'these', 'those'}

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class DatasetVisualizer:
    """
    Generate comprehensive visualizations for IT support tickets dataset.
    
    This class provides methods to analyze and visualize:
    - Word frequency distributions in customer messages
    - Sentiment distributions across the dataset
    - Additional statistical summaries
    
    Attributes:
        data_path (str): Path to the CSV dataset
        output_dir (str): Directory to save generated plots
        df (pd.DataFrame): Loaded dataset
    """
    
    def __init__(self, data_path: str = 'synthetic_it_support_tickets.csv',
                 output_dir: str = 'results/dataset_visualizations'):
        """
        Initialize the dataset visualizer.
        
        Args:
            data_path: Path to the IT support tickets CSV file
            output_dir: Directory to save generated visualizations
        """
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("DATASET VISUALIZATION - PROOF OF CONCEPT")
        print("="*70)
        print(f"Data source: {data_path}")
        print(f"Output directory: {output_dir}")
        
        # Load dataset
        self._load_data()
    
    def _load_data(self):
        """Load and validate the dataset."""
        print("\nLoading dataset...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Dataset loaded successfully")
            print(f"  Total records: {len(self.df):,}")
            print(f"  Columns: {list(self.df.columns)}")
            
            # Check for required columns
            required_cols = ['initial_message', 'sentiment']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                print(f"⚠ Missing columns: {missing_cols}")
                print("  Available columns:", list(self.df.columns))
                
                # Try to find alternative column names
                if 'customer_message' in self.df.columns and 'initial_message' not in self.df.columns:
                    print("  Using 'customer_message' as 'initial_message'")
                    self.df['initial_message'] = self.df['customer_message']
                
                if 'customer_sentiment' in self.df.columns and 'sentiment' not in self.df.columns:
                    print("  Using 'customer_sentiment' as 'sentiment'")
                    self.df['sentiment'] = self.df['customer_sentiment']
            
            # Display sample
            print("\nSample data:")
            print(self.df.head(3))
            
        except FileNotFoundError:
            print(f"✗ Dataset not found at: {self.data_path}")
            print("  Creating sample synthetic data for demonstration...")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration if dataset not found."""
        print("\nGenerating sample dataset...")
        
        # Sample customer messages
        messages = [
            "I can't access my account",
            "Password reset not working",
            "System is very slow",
            "Unable to login to the portal",
            "Error message when uploading files",
            "Need help with password recovery",
            "Website keeps crashing",
            "Can't connect to the server",
            "Account locked after multiple attempts",
            "Software update failed",
        ] * 100  # Repeat to create 1000 samples
        
        # Sample sentiments
        sentiments = ['negative'] * 400 + ['neutral'] * 350 + ['positive'] * 250
        
        # Create DataFrame
        np.random.shuffle(sentiments)
        self.df = pd.DataFrame({
            'initial_message': np.random.choice(messages, 1000),
            'sentiment': sentiments,
            'customer_segment': np.random.choice(['enterprise', 'individual', 'education'], 1000),
            'priority': np.random.choice(['high', 'medium', 'low'], 1000)
        })
        
        print(f"✓ Sample dataset created: {len(self.df)} records")
    
    def tokenize_text(self, text: str) -> list:
        """
        Tokenize text using NLTK if available, otherwise use basic tokenization.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens (words)
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except:
                # Fallback to basic split
                tokens = text.split()
        else:
            tokens = text.split()
        
        # Remove stopwords and short words
        tokens = [word for word in tokens 
                 if word not in STOP_WORDS and len(word) > 2]
        
        return tokens
    
    def plot_word_frequency(self, top_n: int = 10,
                           save_path: str = None) -> str:
        """
        Generate word frequency distribution plot.
        
        Creates a bar chart showing the most frequent words in customer messages.
        Uses NLTK for tokenization and removes stopwords for meaningful analysis.
        
        Args:
            top_n: Number of top frequent words to display
            save_path: Custom path to save the plot (default: word_freq.png)
            
        Returns:
            Path to the saved figure
        """
        print("\n" + "="*70)
        print("GENERATING WORD FREQUENCY DISTRIBUTION")
        print("="*70)
        
        # Collect all words from customer messages
        print(f"Analyzing {len(self.df)} customer messages...")
        all_words = []
        
        for message in self.df['initial_message']:
            tokens = self.tokenize_text(message)
            all_words.extend(tokens)
        
        print(f"  Total words found: {len(all_words):,}")
        print(f"  Unique words: {len(set(all_words)):,}")
        
        # Count word frequencies
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(top_n)
        
        print(f"\nTop {top_n} most frequent words:")
        for i, (word, count) in enumerate(most_common, 1):
            print(f"  {i:2d}. {word:15s} - {count:4d} occurrences")
        
        # Create visualization
        words, counts = zip(*most_common)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(words)), counts, color='steelblue', 
                      alpha=0.8, edgecolor='navy', linewidth=1.5)
        
        # Customize the plot
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Number of Occurrences)', 
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Words', fontsize=13, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Frequent Words in Customer Messages\n' +
                    f'(Total Messages: {len(self.df):,} | Total Words Analyzed: {len(all_words):,})',
                    fontsize=15, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,}',
                   ha='left', va='center', fontsize=11, fontweight='bold')
        
        # Add grid for readability
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Invert y-axis so most frequent is at top
        ax.invert_yaxis()
        
        # Add statistics box
        total_words = len(all_words)
        unique_words = len(set(all_words))
        avg_words_per_msg = total_words / len(self.df)
        
        stats_text = (
            f'Dataset Statistics:\n'
            f'  • Total Words: {total_words:,}\n'
            f'  • Unique Words: {unique_words:,}\n'
            f'  • Avg Words/Message: {avg_words_per_msg:.1f}\n'
            f'  • Vocabulary Coverage: {(sum(counts)/total_words)*100:.1f}%'
        )
        
        ax.text(0.98, 0.02, stats_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'word_freq.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n✓ Word frequency plot saved: {save_path}")
        
        return save_path
    
    def plot_sentiment_distribution(self, save_path: str = None) -> str:
        """
        Generate sentiment distribution plot.
        
        Creates a count plot showing the distribution of sentiments
        (positive, neutral, negative) in customer queries.
        
        Args:
            save_path: Custom path to save the plot (default: sentiment_dist.png)
            
        Returns:
            Path to the saved figure
        """
        print("\n" + "="*70)
        print("GENERATING SENTIMENT DISTRIBUTION")
        print("="*70)
        
        # Check if sentiment column exists
        if 'sentiment' not in self.df.columns:
            print("✗ 'sentiment' column not found in dataset")
            return None
        
        # Count sentiments
        sentiment_counts = self.df['sentiment'].value_counts()
        print("\nSentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {sentiment:10s}: {count:5d} ({percentage:5.1f}%)")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define color palette
        sentiment_colors = {
            'positive': '#2ecc71',   # Green
            'neutral': '#3498db',    # Blue
            'negative': '#e74c3c'    # Red
        }
        
        # Order sentiments for consistent display
        sentiment_order = ['positive', 'neutral', 'negative']
        # Filter to only existing sentiments in data
        sentiment_order = [s for s in sentiment_order if s in sentiment_counts.index]
        
        # Create count plot
        sns.countplot(
            data=self.df,
            x='sentiment',
            order=sentiment_order,
            palette=[sentiment_colors.get(s, '#95a5a6') for s in sentiment_order],
            ax=ax,
            edgecolor='black',
            linewidth=2
        )
        
        # Customize the plot
        ax.set_xlabel('Sentiment', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count (Number of Messages)', fontsize=14, fontweight='bold')
        ax.set_title('Sentiment Distribution in Customer Queries\n' +
                    f'(Total Messages: {len(self.df):,})',
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add percentage labels on bars
        total = len(self.df)
        for i, sentiment in enumerate(sentiment_order):
            count = sentiment_counts[sentiment]
            percentage = (count / total) * 100
            
            # Get bar position
            bar = ax.patches[i]
            height = bar.get_height()
            
            # Add count label
            ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                   f'{count:,}\n({percentage:.1f}%)',
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
        
        # Customize x-axis labels
        ax.set_xticklabels([s.capitalize() for s in sentiment_order],
                          fontsize=13, fontweight='bold')
        
        # Add grid for readability
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add statistics box
        stats_text = (
            f'Sentiment Analysis:\n'
            f'  • Most Common: {sentiment_counts.idxmax().capitalize()}\n'
            f'  • Least Common: {sentiment_counts.idxmin().capitalize()}\n'
            f'  • Sentiment Ratio:\n'
        )
        
        for sentiment in sentiment_order:
            count = sentiment_counts[sentiment]
            percentage = (count / total) * 100
            stats_text += f'    - {sentiment.capitalize()}: {percentage:.1f}%\n'
        
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='top',
               horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'sentiment_dist.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n✓ Sentiment distribution plot saved: {save_path}")
        
        return save_path
    
    def generate_all_visualizations(self):
        """
        Generate all dataset visualizations for PoC.
        
        Creates:
        1. Word frequency distribution (word_freq.png)
        2. Sentiment distribution (sentiment_dist.png)
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        print("\n" + "="*70)
        print("GENERATING ALL DATASET VISUALIZATIONS")
        print("="*70)
        
        visualizations = {}
        
        # Generate word frequency plot
        print("\n1. Word Frequency Distribution...")
        visualizations['word_frequency'] = self.plot_word_frequency(top_n=10)
        
        # Generate sentiment distribution plot
        print("\n2. Sentiment Distribution...")
        visualizations['sentiment_distribution'] = self.plot_sentiment_distribution()
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL VISUALIZATIONS COMPLETE")
        print("="*70)
        print(f"\nGenerated {len(visualizations)} visualizations in: {self.output_dir}/\n")
        
        print("Files created:")
        for name, path in visualizations.items():
            if path:
                print(f"  ✓ {name}: {path}")
        
        return visualizations
    
    def print_dataset_summary(self):
        """Print comprehensive dataset summary statistics."""
        print("\n" + "="*70)
        print("DATASET SUMMARY STATISTICS")
        print("="*70)
        
        print(f"\nBasic Information:")
        print(f"  • Total records: {len(self.df):,}")
        print(f"  • Number of columns: {len(self.df.columns)}")
        print(f"  • Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\nColumn Information:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            non_null = self.df[col].notna().sum()
            null_count = self.df[col].isna().sum()
            unique_count = self.df[col].nunique()
            print(f"  • {col}:")
            print(f"      - Type: {dtype}")
            print(f"      - Non-null: {non_null:,} | Null: {null_count:,}")
            print(f"      - Unique values: {unique_count:,}")
        
        # Message length statistics
        if 'initial_message' in self.df.columns:
            print(f"\nMessage Length Statistics:")
            msg_lengths = self.df['initial_message'].str.len()
            print(f"  • Min length: {msg_lengths.min()} characters")
            print(f"  • Max length: {msg_lengths.max()} characters")
            print(f"  • Mean length: {msg_lengths.mean():.1f} characters")
            print(f"  • Median length: {msg_lengths.median():.1f} characters")
        
        print("\n" + "="*70)


def main():
    """
    Main execution function for dataset visualizations.
    """
    print("\n" + "="*70)
    print("DATASET VISUALIZATION FOR PROOF OF CONCEPT (POC)")
    print("="*70)
    print("\nTask 2: Generate Key Visualizations")
    print("  1. Word Frequency Distribution (word_freq.png)")
    print("  2. Sentiment Distribution (sentiment_dist.png)")
    
    # Initialize visualizer
    visualizer = DatasetVisualizer(
        data_path='synthetic_it_support_tickets.csv',
        output_dir='results/dataset_visualizations'
    )
    
    # Print dataset summary
    visualizer.print_dataset_summary()
    
    # Generate all visualizations
    visualizations = visualizer.generate_all_visualizations()
    
    print("\n" + "="*70)
    print("✅ DATASET VISUALIZATION COMPLETE")
    print("="*70)
    print("\nVisualization files ready for Task 2 PoC documentation!")
    print("\nNext steps:")
    print("  1. Review generated plots in results/dataset_visualizations/")
    print("  2. Include visualizations in PoC presentation")
    print("  3. Use insights for model development discussion")


if __name__ == '__main__':
    main()
