"""
Data Loading Module for Spam Email Detection

This module handles loading and initial exploration of the spam email dataset.
It provides utilities for checking dataset structure, statistics, and class distribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import os


def download_dataset(dataset_url, save_path):
    """
    Download the spam dataset from a remote source.
    
    Args:
        dataset_url (str): URL to download the dataset from
        save_path (str): Local path to save the dataset
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        print(f"Downloading dataset from {dataset_url}...")
        urllib.request.urlretrieve(dataset_url, save_path)
        print(f"Dataset downloaded successfully to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def load_dataset(filepath):
    """
    Load the spam email dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        print(f"Dataset loaded successfully from {filepath}")
        return df
    except UnicodeDecodeError:
        # Try alternative encoding if UTF-8 fails
        df = pd.read_csv(filepath, encoding='latin-1')
        print(f"Dataset loaded with latin-1 encoding")
        return df


def explore_dataset(df):
    """
    Perform initial exploration of the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to explore
    
    Returns:
        dict: Dictionary containing exploration statistics
    """
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)
    
    # Basic information
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn Names and Types:")
    print(df.dtypes)
    
    # Check for missing values
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found")
    else:
        print(missing[missing > 0])
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    
    # Display first few rows
    print(f"\nFirst 5 Rows:")
    print(df.head())
    
    # Class distribution
    print(f"\nDataset Info:")
    print(df.info())
    
    exploration_stats = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': duplicates,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict()
    }
    
    return exploration_stats


def handle_missing_data(df, text_column, label_column):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): The dataset
        text_column (str): Name of the column containing email text
        label_column (str): Name of the column containing labels
    
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    initial_rows = len(df)
    
    # Remove rows with missing values in critical columns
    df = df.dropna(subset=[text_column, label_column])
    
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with missing critical values")
    
    return df


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): The dataset
    
    Returns:
        pd.DataFrame: Dataset with duplicates removed
    """
    initial_rows = len(df)
    df = df.drop_duplicates()
    rows_removed = initial_rows - len(df)
    
    if rows_removed > 0:
        print(f"Removed {rows_removed} duplicate rows")
    
    return df


def get_class_distribution(df, label_column):
    """
    Get the distribution of classes in the dataset.
    
    Args:
        df (pd.DataFrame): The dataset
        label_column (str): Name of the label column
    
    Returns:
        dict: Class distribution statistics
    """
    distribution = df[label_column].value_counts()
    percentages = df[label_column].value_counts(normalize=True) * 100
    
    print(f"\nClass Distribution:")
    for label in distribution.index:
        print(f"{label}: {distribution[label]} ({percentages[label]:.2f}%)")
    
    return {
        'counts': distribution.to_dict(),
        'percentages': percentages.to_dict()
    }
