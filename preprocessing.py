"""
Text Preprocessing Module
Handles text cleaning, tokenization, and normalization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Handle text preprocessing for spam detection."""
    
    def __init__(self):
        """Initialize the preprocessor with stopwords and lemmatizer."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean text by removing special characters, numbers, and converting to lowercase.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list of tokens
        """
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]
    
    def lemmatize(self, tokens):
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, remove_stop_words=True, apply_lemmatization=True):
        """
        Complete preprocessing pipeline for a text string.
        
        Args:
            text: Input text string
            remove_stop_words: Whether to remove stopwords
            apply_lemmatization: Whether to apply lemmatization
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if remove_stop_words:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        if apply_lemmatization:
            tokens = self.lemmatize(tokens)
        
        # Join back into string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='message', output_column='cleaned_message'):
        """
        Apply preprocessing to a dataframe column.
        
        Args:
            df: Input dataframe
            text_column: Name of the column containing text
            output_column: Name of the output column for cleaned text
            
        Returns:
            Dataframe with new cleaned text column
        """
        print(f"Preprocessing {len(df)} messages...")
        
        # Apply preprocessing to each message
        df[output_column] = df[text_column].apply(
            lambda x: self.preprocess(x) if isinstance(x, str) else ''
        )
        
        print(f"Preprocessing complete. Created '{output_column}' column.")
        
        return df


def example_preprocessing():
    """Demonstrate the preprocessing pipeline."""
    processor = TextPreprocessor()
    
    sample_texts = [
        "WINNER!!! You have won a FREE IPHONE!!! Click here: www.freephone.com",
        "Hey, how are you doing? Let's meet up tomorrow at 5pm.",
        "Limited time offer! Get 50% OFF on all products. Call 1-800-555-0123",
        "Hi John, just checking in. See you at the meeting.",
    ]
    
    print("="*60)
    print("PREPROCESSING EXAMPLES")
    print("="*60)
    
    for text in sample_texts:
        cleaned = processor.preprocess(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned: {cleaned}")


if __name__ == "__main__":
    example_preprocessing()
