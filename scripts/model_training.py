"""
Model Training Module for Spam Email Detection

This module handles feature extraction, model training, and hyperparameter optimization.
Uses TF-IDF vectorization and Logistic Regression with GridSearchCV for tuning.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
import numpy as np


class SpamDetectionModel:
    """
    A comprehensive spam detection model using TF-IDF and Logistic Regression.
    
    Attributes:
        vectorizer: TF-IDF vectorizer for feature extraction
        model: Logistic Regression classifier
        x_train: Training features
        x_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    """
    
    def __init__(self):
        """Initialize the spam detection model."""
        self.vectorizer = None
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.best_params = None
        self.cv_scores = None
    
    def create_tfidf_vectorizer(self, max_features=5000, ngram_range=(1, 2), 
                                 min_df=2, max_df=0.95):
        """
        Create and configure a TF-IDF vectorizer.
        
        Args:
            max_features (int): Maximum number of features to extract
            ngram_range (tuple): N-gram range (1-grams, 2-grams, etc.)
            min_df (int/float): Minimum document frequency
            max_df (float): Maximum document frequency
        
        Returns:
            TfidfVectorizer: Configured vectorizer
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            sublinear_tf=True
        )
        
        print("\nTF-IDF Vectorizer Configuration:")
        print(f"  Max Features: {max_features}")
        print(f"  N-gram Range: {ngram_range}")
        print(f"  Min DF: {min_df}")
        print(f"  Max DF: {max_df}")
        
        return vectorizer
    
    def split_data(self, x, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            x (array-like): Features
            y (array-like): Labels
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (x_train, x_test, y_train, y_test)
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nData Split (Train/Test: {1-test_size:.0%}/{test_size:.0%}):")
        print(f"  Training samples: {len(self.x_train)}")
        print(f"  Testing samples: {len(self.x_test)}")
        
        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def extract_features(self, x_train, x_test):
        """
        Extract TF-IDF features from text data.
        
        Args:
            x_train (array-like): Training text data
            x_test (array-like): Testing text data
        
        Returns:
            tuple: (x_train_tfidf, x_test_tfidf)
        """
        print("\nExtracting TF-IDF features...")
        
        # Fit vectorizer on training data
        x_train_tfidf = self.vectorizer.fit_transform(x_train)
        x_test_tfidf = self.vectorizer.transform(x_test)
        
        print(f"  Training features shape: {x_train_tfidf.shape}")
        print(f"  Testing features shape: {x_test_tfidf.shape}")
        print(f"  Number of features: {x_train_tfidf.shape[1]}")
        
        return x_train_tfidf, x_test_tfidf
    
    def train_baseline_model(self, x_train, y_train, max_iter=1000):
        """
        Train a baseline Logistic Regression model.
        
        Args:
            x_train: Training features
            y_train: Training labels
            max_iter (int): Maximum number of iterations
        """
        print("\nTraining baseline Logistic Regression model...")
        
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1
        )
        
        self.model.fit(x_train, y_train)
        
        # Get baseline accuracy
        train_accuracy = self.model.score(x_train, y_train)
        print(f"  Baseline training accuracy: {train_accuracy:.4f}")
    
    def optimize_hyperparameters(self, x_train, y_train, cv=5):
        """
        Optimize model hyperparameters using GridSearchCV.
        
        Args:
            x_train: Training features
            y_train: Training labels
            cv (int): Number of cross-validation folds
        """
        print(f"\nHyperparameter Optimization (GridSearchCV with {cv}-fold CV)...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000, 2000]
        }
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(x_train, y_train)
        
        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_scores = grid_search.cv_results_
        
        print(f"\n  Best parameters: {self.best_params}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
    
    def evaluate_model(self, x_test, y_test):
        """
        Evaluate the model on test data and generate metrics.
        
        Args:
            x_test: Testing features
            y_test: Testing labels
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred = self.model.predict(x_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Display metrics
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        class_report = classification_report(y_test, y_pred, output_dict=False)
        print(class_report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        return metrics
    
    def save_model(self, model_path, vectorizer_path):
        """
        Save the trained model and vectorizer to disk.
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str): Path to save the vectorizer
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path, vectorizer_path):
        """
        Load a trained model and vectorizer from disk.
        
        Args:
            model_path (str): Path to the model file
            vectorizer_path (str): Path to the vectorizer file
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"Model loaded from {model_path}")
        print(f"Vectorizer loaded from {vectorizer_path}")
    
    def predict_email(self, email_text):
        """
        Predict whether an email is spam or not.
        
        Args:
            email_text (str): The email text to classify
        
        Returns:
            dict: Prediction result with label and probability
        """
        # Vectorize the email
        email_tfidf = self.vectorizer.transform([email_text])
        
        # Get prediction
        prediction = self.model.predict(email_tfidf)[0]
        probability = self.model.predict_proba(email_tfidf)[0]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': max(probability) * 100
        }
