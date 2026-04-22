"""
Example Usage of the Spam Detection System
Demonstrates how to use each component of the project independently.
"""

from data_loader import DataLoader
from preprocessing import TextPreprocessor
from model_training import SpamDetectionModel
from evaluation import ModelEvaluator, ModelPersistence
import pandas as pd


def example_1_data_exploration():
    """Example 1: Load and explore the dataset."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Data Loading and Exploration")
    print("="*60)
    
    loader = DataLoader()
    df = loader.load_data()
    stats = loader.explore_data()
    
    print("\nData loaded successfully!")
    print(f"Total emails: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")


def example_2_text_preprocessing():
    """Example 2: Preprocess email text."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Text Preprocessing")
    print("="*60)
    
    processor = TextPreprocessor()
    
    # Sample emails
    emails = [
        "WINNER!!! You have won a FREE IPHONE!!! Click here: www.freephone.com",
        "Hi John, are you free tomorrow at 5pm?",
        "LIMITED TIME OFFER! Get 50% OFF on all products. Call 1-800-555-0123",
        "Just checking in. Hope you're having a great day!",
    ]
    
    print("\nPreprocessing Examples:\n")
    
    for i, email in enumerate(emails, 1):
        cleaned = processor.preprocess(email)
        print(f"Email {i}:")
        print(f"  Original: {email}")
        print(f"  Cleaned:  {cleaned}")
        print()


def example_3_feature_extraction():
    """Example 3: Extract TF-IDF features."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Feature Extraction with TF-IDF")
    print("="*60)
    
    # Create sample data
    loader = DataLoader()
    df = loader.load_data()
    
    # Preprocess
    processor = TextPreprocessor()
    df = processor.preprocess_dataframe(df, 'message', 'cleaned_message')
    
    # Create and fit vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['cleaned_message'])
    
    print(f"\nTF-IDF Feature Matrix Shape: {X_tfidf.shape}")
    print(f"Number of features extracted: {X_tfidf.shape[1]}")
    
    # Show top features
    feature_names = vectorizer.get_feature_names_out()
    print(f"\nTop 10 Features (by index):")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i+1}. {name}")


def example_4_model_training():
    """Example 4: Train a model."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Model Training")
    print("="*60)
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_data()
    loader.clean_data()
    
    processor = TextPreprocessor()
    df = processor.preprocess_dataframe(df, 'message', 'cleaned_message')
    
    # Initialize model
    model = SpamDetectionModel()
    model.split_data(df, text_column='cleaned_message', label_column='label')
    model.build_vectorizer(max_features=2000)
    X_train_tfidf, X_test_tfidf = model.extract_features()
    
    # Train model
    model.train_model(X_train_tfidf, C=1.0)
    
    print(f"\nModel trained successfully!")
    print(f"Training set size: {X_train_tfidf.shape[0]}")
    print(f"Test set size: {X_test_tfidf.shape[0]}")
    print(f"Number of features: {X_train_tfidf.shape[1]}")


def example_5_model_evaluation():
    """Example 5: Evaluate model performance."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Model Evaluation")
    print("="*60)
    
    # Full pipeline
    loader = DataLoader()
    df = loader.load_data()
    loader.clean_data()
    
    processor = TextPreprocessor()
    df = processor.preprocess_dataframe(df, 'message', 'cleaned_message')
    
    model = SpamDetectionModel()
    model.split_data(df, text_column='cleaned_message', label_column='label')
    model.build_vectorizer(max_features=2000)
    X_train_tfidf, X_test_tfidf = model.extract_features()
    
    model.train_model(X_train_tfidf, C=1.0)
    
    # Evaluate
    evaluator = ModelEvaluator(model.model, model.vectorizer)
    metrics = evaluator.evaluate(X_test_tfidf, model.y_test)
    
    print(f"\nModel Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")


def example_6_single_prediction():
    """Example 6: Make predictions on new emails."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Single Email Prediction")
    print("="*60)
    
    # Train a model
    loader = DataLoader()
    df = loader.load_data()
    
    processor = TextPreprocessor()
    df = processor.preprocess_dataframe(df, 'message', 'cleaned_message')
    
    model = SpamDetectionModel()
    model.split_data(df, text_column='cleaned_message', label_column='label')
    model.build_vectorizer(max_features=2000)
    X_train_tfidf, X_test_tfidf = model.extract_features()
    model.train_model(X_train_tfidf, C=1.0)
    
    # Predict on new emails
    test_emails = [
        "Congratulations! You have won $1,000,000! Click here to claim your prize!",
        "Hi Sarah, can we schedule a meeting for next Tuesday?",
        "Free money! Get paid $500 daily by working from home!",
    ]
    
    print("\nPredicting on sample emails:\n")
    
    for email in test_emails:
        result = ModelPersistence.predict_email(
            model.model, model.vectorizer, email
        )
        
        print(f"Email: {email[:60]}...")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Spam Probability: {result['spam_probability']:.1%}")
        print(f"  Ham Probability: {result['ham_probability']:.1%}")
        print()


def example_7_custom_parameters():
    """Example 7: Train with custom parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Custom Model Parameters")
    print("="*60)
    
    loader = DataLoader()
    df = loader.load_data()
    
    processor = TextPreprocessor()
    df = processor.preprocess_dataframe(df, 'message', 'cleaned_message')
    
    model = SpamDetectionModel()
    model.split_data(df, text_column='cleaned_message', label_column='label', 
                     test_size=0.15)  # 85/15 split
    
    # Custom vectorizer parameters
    model.build_vectorizer(
        max_features=3000,      # More features
        ngram_range=(1, 3),     # Include trigrams
        min_df=1,
        max_df=0.98
    )
    
    X_train_tfidf, X_test_tfidf = model.extract_features()
    
    # Custom model parameters
    model.train_model(
        X_train_tfidf,
        C=10.0,                 # Higher regularization strength
        solver='liblinear',
        max_iter=2000
    )
    
    print("\nCustom model trained with:")
    print("  - 85/15 train/test split")
    print("  - Max 3000 features")
    print("  - N-gram range: (1, 3)")
    print("  - C = 10.0")
    print("  - Solver: liblinear")


def example_8_batch_prediction():
    """Example 8: Predict on multiple emails from a DataFrame."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Batch Prediction")
    print("="*60)
    
    # Train model
    loader = DataLoader()
    df = loader.load_data()
    
    processor = TextPreprocessor()
    df = processor.preprocess_dataframe(df, 'message', 'cleaned_message')
    
    model = SpamDetectionModel()
    model.split_data(df, text_column='cleaned_message', label_column='label')
    model.build_vectorizer(max_features=2000)
    X_train_tfidf, X_test_tfidf = model.extract_features()
    model.train_model(X_train_tfidf)
    
    # Batch prediction on test set
    predictions = model.model.predict(X_test_tfidf)
    probabilities = model.model.predict_proba(X_test_tfidf)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Actual': model.y_test.values,
        'Predicted': predictions,
        'Spam_Probability': probabilities[:, 1],
        'Ham_Probability': probabilities[:, 0]
    })
    
    print("\nBatch Prediction Results (first 5):\n")
    print(results.head())
    
    print(f"\nTotal predictions: {len(results)}")
    print(f"Accuracy: {(results['Actual'] == results['Predicted']).mean():.2%}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" SPAM DETECTION SYSTEM - COMPREHENSIVE EXAMPLES")
    print("="*70)
    
    try:
        example_1_data_exploration()
        example_2_text_preprocessing()
        example_3_feature_extraction()
        example_4_model_training()
        example_5_model_evaluation()
        example_6_single_prediction()
        example_7_custom_parameters()
        example_8_batch_prediction()
        
        print("\n" + "="*70)
        print(" ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
