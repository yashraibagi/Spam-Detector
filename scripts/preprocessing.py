"""
Text Preprocessing Module for Spam Email Detection

This module handles all text preprocessing steps including:
- Lowercasing
- Special character and punctuation removal
- Stopword removal
- Tokenization
- Lemmatization/Stemming
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd


# Download required NLTK data
def download_nltk_resources():
    """Download required NLTK data resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for email cleaning.
    
    Attributes:
        lemmatizer: NLTK WordNetLemmatizer for lemmatization
        stemmer: NLTK PorterStemmer for stemming
        stop_words: Set of English stopwords
        use_lemmatization: Whether to use lemmatization
    """
    
    def __init__(self, use_lemmatization=True):
        """
        Initialize the text preprocessor.
        
        Args:
            use_lemmatization (bool): Whether to use lemmatization (default: True)
        """
        download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.use_lemmatization = use_lemmatization
    
    def lowercase(self, text):
        """
        Convert text to lowercase.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Lowercased text
        """
        return text.lower()
    
    def remove_special_characters(self, text):
        """
        Remove special characters, punctuation, and numbers.
        
        Args:
            text (str): Input text
        
        Returns:
            str: Text without special characters
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters, keeping only alphabetic and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, tokens):
        """
        Remove common English stopwords from tokens.
        
        Args:
            tokens (list): List of word tokens
        
        Returns:
            list: Tokens with stopwords removed
        """
        return [word for word in tokens if word not in self.stop_words]
    
    def tokenize(self, text):
        """
        Split text into individual word tokens.
        
        Args:
            text (str): Input text
        
        Returns:
            list: List of word tokens
        """
        return word_tokenize(text)
    
    def lemmatize(self, tokens):
        """
        Apply lemmatization to reduce words to their root form.
        
        Args:
            tokens (list): List of word tokens
        
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def stem(self, tokens):
        """
        Apply stemming to reduce words to their root form.
        
        Args:
            tokens (list): List of word tokens
        
        Returns:
            list: Stemmed tokens
        """
        return [self.stemmer.stem(word) for word in tokens]
    
    def preprocess_text(self, text):
        """
        Apply all preprocessing steps to a single text sample.
        
        Args:
            text (str): Input email text
        
        Returns:
            str: Cleaned and preprocessed text
        """
        # Convert to lowercase
        text = self.lowercase(text)
        
        # Remove special characters and numbers
        text = self.remove_special_characters(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply lemmatization or stemming
        if self.use_lemmatization:
            tokens = self.lemmatize(tokens)
        else:
            tokens = self.stem(tokens)
        
        # Join tokens back into a string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def preprocess_dataframe(self, df, text_column, output_column='cleaned_text'):
        """
        Apply preprocessing to all texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the column containing text to preprocess
            output_column (str): Name of the column to store cleaned text
        
        Returns:
            pd.DataFrame: DataFrame with additional cleaned text column
        """
        print(f"\nPreprocessing {len(df)} texts...")
        print("This may take a moment...")
        
        # Apply preprocessing to each text
        df[output_column] = df[text_column].apply(self.preprocess_text)
        
        print(f"Preprocessing complete!")
        
        # Display sample of cleaned text
        print(f"\nSample of preprocessed text:")
        for idx in range(min(3, len(df))):
            print(f"\nOriginal: {df[text_column].iloc[idx][:100]}...")
            print(f"Cleaned: {df[output_column].iloc[idx][:100]}...")
        
        return df


def create_clean_dataset(df, text_column, label_column, use_lemmatization=True):
    """
    Create a clean, preprocessed version of the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        text_column (str): Name of the text column
        label_column (str): Name of the label column
        use_lemmatization (bool): Whether to use lemmatization
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_lemmatization=use_lemmatization)
    
    # Apply preprocessing
    df = preprocessor.preprocess_dataframe(df, text_column)
    
    # Keep only necessary columns
    df_clean = df[[text_column, label_column, 'cleaned_text']].copy()
    
    return df_clean, preprocessor
