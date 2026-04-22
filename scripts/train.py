"""
Main Training Script for Spam Email Detection

This script orchestrates the complete ML pipeline:
1. Data loading and exploration
2. Text preprocessing
3. Feature extraction with TF-IDF
4. Model training and optimization
5. Evaluation and visualization
6. Model persistence
"""

import sys
import os
import pandas as pd
from data_loader import (
    download_dataset, load_dataset, explore_dataset,
    handle_missing_data, remove_duplicates, get_class_distribution
)
from preprocessing import create_clean_dataset
from model_training import SpamDetectionModel
from visualization import create_all_visualizations
import warnings

warnings.filterwarnings('ignore')


def load_or_download_dataset():
    """
    Load dataset from local file or download if not available.
    
    Returns:
        pd.DataFrame: The dataset
    """
    # Try to load from local file first
    local_path = 'spam_dataset.csv'
    
    if os.path.exists(local_path):
        print(f"Loading dataset from {local_path}...")
        return load_dataset(local_path)
    
    # If not available locally, provide instructions
    print("\n" + "="*60)
    print("DATASET NOT FOUND")
    print("="*60)
    print("\nPlease download a spam dataset and save it as 'spam_dataset.csv'")
    print("\nRecommended datasets:")
    print("1. Kaggle Spam Ham Dataset:")
    print("   https://www.kaggle.com/datasets/mfaisalqureshi/spam-ham-dataset")
    print("\n2. UCI Spam Dataset:")
    print("   https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
    print("\n3. Sample dataset for testing (SpamAssassin):")
    print("   https://spamassassin.apache.org/public-corpus/")
    print("\nExpected CSV format:")
    print("   - Column with email/message text (e.g., 'message', 'email', 'text')")
    print("   - Column with label (e.g., 'label', 'category') containing 'spam' or 'ham'")
    print("\n" + "="*60)
    
    sys.exit(1)


def prepare_dataset(df, text_column, label_column):
    """
    Prepare the dataset for model training.
    
    Args:
        df (pd.DataFrame): Raw dataset
        text_column (str): Name of the text column
        label_column (str): Name of the label column
    
    Returns:
        tuple: (cleaned_df, class_distribution)
    """
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Explore initial dataset
    explore_dataset(df)
    
    # Handle missing values
    df = handle_missing_data(df, text_column, label_column)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Get class distribution
    class_dist = get_class_distribution(df, label_column)
    
    return df, class_dist


def main():
    """Main execution function for the spam detection pipeline."""
    
    print("\n" + "="*70)
    print(" "*15 + "SPAM EMAIL DETECTION - ML PIPELINE")
    print("="*70)
    
    # Step 1: Load Dataset
    print("\n[STEP 1] Loading Dataset...")
    df = load_or_download_dataset()
    
    # Detect column names (handle different naming conventions)
    columns = df.columns.str.lower()
    
    # Find text column
    text_column = None
    for col in ['message', 'email', 'text', 'content', 'body']:
        if col in columns:
            text_column = df.columns[columns == col][0]
            break
    
    if text_column is None:
        text_column = df.columns[0]
        print(f"Warning: Using first column '{text_column}' as text column")
    
    # Find label column
    label_column = None
    for col in ['label', 'category', 'class', 'spam', 'type']:
        if col in columns:
            label_column = df.columns[columns == col][0]
            break
    
    if label_column is None:
        label_column = df.columns[-1]
        print(f"Warning: Using last column '{label_column}' as label column")
    
    print(f"Text column: '{text_column}'")
    print(f"Label column: '{label_column}'")
    
    # Step 2: Data Preparation
    print("\n[STEP 2] Preparing Data...")
    df, class_dist = prepare_dataset(df, text_column, label_column)
    
    # Step 3: Text Preprocessing
    print("\n[STEP 3] Preprocessing Text...")
    df_clean, preprocessor = create_clean_dataset(df, text_column, label_column, use_lemmatization=True)
    
    # Prepare features and labels
    x = df_clean['cleaned_text'].values
    y = df_clean[label_column].values
    
    # Step 4: Initialize Model
    print("\n[STEP 4] Initializing Model...")
    model = SpamDetectionModel()
    
    # Create TF-IDF vectorizer
    model.vectorizer = model.create_tfidf_vectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Step 5: Split Data
    print("\n[STEP 5] Splitting Data...")
    x_train, x_test, y_train, y_test = model.split_data(x, y, test_size=0.2, random_state=42)
    
    # Step 6: Extract Features
    print("\n[STEP 6] Extracting TF-IDF Features...")
    x_train_tfidf, x_test_tfidf = model.extract_features(x_train, x_test)
    
    # Step 7: Train Baseline Model
    print("\n[STEP 7] Training Baseline Model...")
    model.train_baseline_model(x_train_tfidf, y_train)
    
    # Step 8: Hyperparameter Optimization
    print("\n[STEP 8] Optimizing Hyperparameters...")
    model.optimize_hyperparameters(x_train_tfidf, y_train, cv=5)
    
    # Step 9: Evaluate Model
    print("\n[STEP 9] Evaluating Model...")
    metrics = model.evaluate_model(x_test_tfidf, y_test)
    
    # Step 10: Save Model
    print("\n[STEP 10] Saving Model...")
    os.makedirs('models', exist_ok=True)
    model.save_model('models/spam_detector_model.pkl', 'models/tfidf_vectorizer.pkl')
    
    # Step 11: Create Visualizations
    print("\n[STEP 11] Creating Visualizations...")
    
    # Get spam and ham texts for word frequency analysis
    spam_mask = y_test == 'spam' if 'spam' in y_test else y_test == 1
    ham_mask = ~spam_mask
    
    spam_texts = [x_test[i] for i in range(len(x_test)) if spam_mask[i]]
    ham_texts = [x_test[i] for i in range(len(x_test)) if ham_mask[i]]
    
    os.makedirs('visualizations', exist_ok=True)
    create_all_visualizations(
        metrics,
        class_dist['counts'],
        spam_texts,
        ham_texts,
        output_dir='visualizations'
    )
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"\nModel Performance Summary:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nBest Hyperparameters: {model.best_params}")
    print(f"\nFiles Saved:")
    print(f"  - Model: models/spam_detector_model.pkl")
    print(f"  - Vectorizer: models/tfidf_vectorizer.pkl")
    print(f"  - Visualizations: visualizations/")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
