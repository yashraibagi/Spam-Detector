"""
Debug script to test each component of the training pipeline.
Run this to identify where the issue is occurring.
"""

import os
import sys

print("\n" + "="*70)
print("SPAM DETECTION - TRAINING DEBUG SCRIPT")
print("="*70)

# Check 1: Python version
print("\n[Check 1] Python Version:")
print(f"  Python {sys.version}")

# Check 2: Current directory
print("\n[Check 2] Current Directory:")
current_dir = os.getcwd()
print(f"  Working in: {current_dir}")
print(f"  Files present: {os.listdir('.')[:10]}...")

# Check 3: Import all modules
print("\n[Check 3] Testing Module Imports:")

try:
    print("  - Importing pandas...", end=" ")
    import pandas as pd
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing numpy...", end=" ")
    import numpy as np
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing sklearn...", end=" ")
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing nltk...", end=" ")
    import nltk
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing matplotlib...", end=" ")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing seaborn...", end=" ")
    import seaborn as sns
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing streamlit...", end=" ")
    import streamlit
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing joblib...", end=" ")
    import joblib
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Check 4: Import custom modules
print("\n[Check 4] Testing Custom Module Imports:")

try:
    print("  - Importing data_loader...", end=" ")
    from data_loader import DataLoader
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing preprocessing...", end=" ")
    from preprocessing import TextPreprocessor
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing model_training...", end=" ")
    from model_training import SpamDetectionModel
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

try:
    print("  - Importing evaluation...", end=" ")
    from evaluation import ModelEvaluator, ModelPersistence
    print("✓")
except ImportError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Check 5: Create directories
print("\n[Check 5] Creating Directories:")
try:
    os.makedirs('models', exist_ok=True)
    print("  - Created models/ ✓")
    os.makedirs('visualizations', exist_ok=True)
    print("  - Created visualizations/ ✓")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Check 6: Load data
print("\n[Check 6] Loading Data:")
try:
    loader = DataLoader()
    df = loader.load_data()
    print(f"  ✓ Loaded {len(df)} emails")
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 7: Explore data
print("\n[Check 7] Exploring Data:")
try:
    print(f"  ✓ Label distribution:")
    print(f"    {df['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Check 8: Clean data
print("\n[Check 8] Cleaning Data:")
try:
    df_clean = loader.clean_data()
    print(f"  ✓ Cleaned dataset: {len(df_clean)} emails")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Check 9: Preprocess text
print("\n[Check 9] Testing Text Preprocessing:")
try:
    preprocessor = TextPreprocessor()
    sample_text = df['message'].iloc[0]
    processed = preprocessor.preprocess(sample_text)
    print(f"  ✓ Sample preprocessing:")
    print(f"    Original:  {sample_text[:60]}...")
    print(f"    Processed: {processed[:60]}...")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 10: Model training (small scale test)
print("\n[Check 10] Testing Model Training:")
try:
    print("  - Initializing model...", end=" ")
    model = SpamDetectionModel()
    print("✓")
    
    print("  - Splitting data...", end=" ")
    model.split_data(df_clean, text_column='message', label_column='label')
    print(f"✓ (train: {len(model.X_train)}, test: {len(model.X_test)})")
    
    print("  - Building vectorizer...", end=" ")
    model.build_vectorizer(max_features=1000, ngram_range=(1, 2))
    print("✓")
    
    print("  - Extracting features...", end=" ")
    X_train_tfidf, X_test_tfidf = model.extract_features()
    print(f"✓ (shape: {X_train_tfidf.shape})")
    
    print("  - Training model...", end=" ")
    model.train_model(X_train_tfidf, C=1.0)
    print("✓")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 11: Save model
print("\n[Check 11] Testing Model Persistence:")
try:
    print("  - Saving model...", end=" ")
    ModelPersistence.save_model(
        model.model,
        model.vectorizer,
        'models/spam_model.joblib',
        'models/tfidf_vectorizer.joblib'
    )
    print("✓")
    
    # Verify files exist
    if os.path.exists('models/spam_model.joblib'):
        size = os.path.getsize('models/spam_model.joblib')
        print(f"  - Model file exists: {size} bytes ✓")
    else:
        print(f"  - ✗ Model file not found!")
        
    if os.path.exists('models/tfidf_vectorizer.joblib'):
        size = os.path.getsize('models/tfidf_vectorizer.joblib')
        print(f"  - Vectorizer file exists: {size} bytes ✓")
    else:
        print(f"  - ✗ Vectorizer file not found!")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("DEBUG COMPLETE - ALL CHECKS PASSED!")
print("="*70)
print("\nYour environment is ready. You can now run:")
print("  python train_model.py")
print("\nThen start the web app with:")
print("  streamlit run app.py")
print("\n" + "="*70 + "\n")
