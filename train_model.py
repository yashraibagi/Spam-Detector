"""
Main Training Script
Orchestrates the complete spam detection model training pipeline.
"""

import sys
from data_loader import DataLoader
from preprocessing import TextPreprocessor
from model_training import SpamDetectionModel
from evaluation import ModelEvaluator, ModelPersistence


def main():
    """Run the complete training pipeline."""
    
    print("\n" + "="*60)
    print("SPAM EMAIL DETECTION - MODEL TRAINING PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Load and explore data
        print("\n[Step 1/6] Loading Data...")
        loader = DataLoader()
        df = loader.load_data()
        print(f"✓ Data loaded: {len(df)} emails")
        
        stats = loader.explore_data()
        print("✓ Data exploration complete")
        
        df = loader.clean_data()
        print(f"✓ Data cleaned: {len(df)} emails remaining")
        
        # Step 2: Text preprocessing
        print("\n[Step 2/6] Text Preprocessing...")
        preprocessor = TextPreprocessor()
        print("✓ Preprocessing text...")
        df = preprocessor.preprocess_dataframe(
            df,
            text_column='message',
            output_column='cleaned_message'
        )
        print(f"✓ Text preprocessing complete")
        
        # Display sample preprocessed text
        print("\nSample Preprocessing Results:")
        for i in range(min(3, len(df))):
            print(f"\n  Original:  {df['message'].iloc[i][:80]}...")
            print(f"  Cleaned:   {df['cleaned_message'].iloc[i][:80]}...")
        
        # Step 3: Feature extraction and data split
        print("\n[Step 3/6] Feature Extraction...")
        model = SpamDetectionModel()
        print("✓ Splitting data into train/test...")
        model.split_data(df, text_column='cleaned_message', label_column='label')
        print(f"  Training samples: {len(model.X_train)}")
        print(f"  Test samples: {len(model.X_test)}")
        
        print("✓ Building TF-IDF vectorizer...")
        model.build_vectorizer(max_features=5000, ngram_range=(1, 2))
        
        print("✓ Extracting features...")
        X_train_tfidf, X_test_tfidf = model.extract_features()
        print(f"  Feature matrix shape: {X_train_tfidf.shape}")
        
        # Step 4: Train initial model
        print("\n[Step 4/6] Training Model...")
        model.train_model(X_train_tfidf, C=1.0)
        print("✓ Baseline model trained")
        
        # Step 5: Hyperparameter optimization
        print("\n[Step 5/6] Hyperparameter Optimization...")
        print("  (This may take a minute...)")
        model.optimize_hyperparameters(X_train_tfidf, cv=5)
        print("✓ Hyperparameter optimization complete")
        
        # Step 6: Evaluation
        print("\n[Step 6/6] Evaluating Model...")
        evaluator = ModelEvaluator(model.model, model.vectorizer)
        metrics = evaluator.evaluate(X_test_tfidf, model.y_test)
        print("✓ Model evaluation complete")
        
        # Detailed analysis
        evaluator.confusion_matrix_analysis(model.y_test)
        
        # Visualizations
        print("\nGenerating Visualizations...")
        evaluator.plot_confusion_matrix(model.y_test, 'visualizations/confusion_matrix.png')
        print("✓ Confusion matrix saved")
        
        evaluator.plot_metrics('visualizations/metrics.png')
        print("✓ Metrics chart saved")
        
        evaluator.plot_roc_curve(model.y_test, 'visualizations/roc_curve.png')
        print("✓ ROC curve saved")
        
        # Save model
        print("\nSaving Model...")
        ModelPersistence.save_model(
            model.model,
            model.vectorizer,
            'models/spam_model.joblib',
            'models/tfidf_vectorizer.joblib'
        )
        print("✓ Model saved to models/spam_model.joblib")
        print("✓ Vectorizer saved to models/tfidf_vectorizer.joblib")
        
        # Summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*60)
        print(f"\nDataset: {len(df)} emails")
        print(f"Training set: {len(model.X_train)} emails")
        print(f"Test set: {len(model.X_test)} emails")
        print(f"\nBest Model Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"\nModel saved to: models/spam_model.joblib")
        print(f"Vectorizer saved to: models/tfidf_vectorizer.joblib")
        print(f"Visualizations saved to: visualizations/")
        print("\n" + "="*60)
        print("\n✓ READY TO USE! Run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n[X] ERROR in step: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Create necessary directories
        import os
        print("\n[Setup] Creating directories...")
        os.makedirs('models', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        print("✓ Directories ready: models/, visualizations/")
        
        main()
        print("\n" + "="*60)
        print("SUCCESS! Your model is trained and ready.")
        print("="*60)
        print("\nNext step: Run this command to open the web app:")
        print("  streamlit run app.py")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR during training:")
        print(f"{'='*60}")
        print(f"\n[X] {str(e)}\n")
        import traceback
        print("Full error details:")
        traceback.print_exc()
        print(f"\n{'='*60}")
        print("Troubleshooting tips:")
        print("1. Check that all required packages are installed:")
        print("   pip install -r requirements.txt")
        print("2. Make sure you're in the correct directory")
        print("3. Check that you have write permissions")
        print(f"{'='*60}\n")
        sys.exit(1)
