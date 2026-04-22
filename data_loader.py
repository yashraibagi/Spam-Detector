"""
Data Loading and Exploration Module
Handles loading, exploring, and preprocessing the spam dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Load and explore spam email dataset."""
    
    def __init__(self, csv_path=None):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to the CSV file. If None, creates sample data.
        """
        self.csv_path = csv_path
        self.df = None
        
    def load_data(self):
        """
        Load the dataset from CSV file.
        If no file exists, create a sample dataset for demonstration.
        """
        if self.csv_path and Path(self.csv_path).exists():
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded data from {self.csv_path}")
        else:
            self.df = self._create_sample_data()
            print("Created sample spam dataset for demonstration")
        
        return self.df
    
    def _create_sample_data(self):
        """Create a sample spam dataset for demonstration purposes."""
        sample_data = {
            'v1': [
                'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
                'spam', 'ham'
            ] + ['ham'] * 20 + ['spam'] * 15,
            'v2': [
                'Go until jurong point', 'Crazy.. This is amazing',
                'Winner!! You have won a cash prize. Claim your prize now',
                'Please call me back', 'Free Money! Click here',
                'Thanks for checking in', 'CALL NOW for a FREE consultation',
                'See you later',
                'Call FREEPHONE 0800 542 0671 +1(202) 555 0123 Your Personalantine Secured Loan "No credit checks" "No fees" "Fast carey Visa Card" 0800 220022 or visit www.loanwiz.co.uk We guarantee the best rate!!',
                'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'
            ] + [f'Regular email message {i}' for i in range(20)] + 
            [f'You have won! Claim your prize {i}' for i in range(15)]
        }
        
        df = pd.DataFrame(sample_data)
        df.columns = ['label', 'message']
        return df
    
    def explore_data(self):
        """Display basic information about the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("DATASET EXPLORATION")
        print("="*60)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nColumn Names: {self.df.columns.tolist()}")
        print(f"\nData Types:\n{self.df.dtypes}")
        
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        
        print(f"\nDuplicate Rows: {self.df.duplicated().sum()}")
        
        print(f"\nClass Distribution:\n{self.df['label'].value_counts()}")
        
        class_ratio = self.df['label'].value_counts(normalize=True) * 100
        print(f"\nClass Distribution (%):\n{class_ratio}")
        
        return {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'class_distribution': self.df['label'].value_counts().to_dict()
        }
    
    def clean_data(self):
        """Remove duplicates and handle missing values."""
        initial_rows = len(self.df)
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Remove rows with missing values in critical columns
        self.df = self.df.dropna(subset=['label', 'message'])
        
        rows_removed = initial_rows - len(self.df)
        print(f"\nData Cleaning: Removed {rows_removed} rows (duplicates/missing)")
        
        return self.df
    
    def get_data(self):
        """Return the loaded dataframe."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    loader.load_data()
    loader.explore_data()
    loader.clean_data()
