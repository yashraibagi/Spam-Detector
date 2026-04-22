"""
Model Evaluation and Visualization Module
Handles evaluation metrics, visualization, and model persistence.
"""

import joblib
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


class ModelEvaluator:
    """Evaluate and analyze model performance."""
    
    def __init__(self, model, vectorizer):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            vectorizer: TF-IDF vectorizer
        """
        self.model = model
        self.vectorizer = vectorizer
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
    
    def evaluate(self, X_test_tfidf, y_test):
        """
        Evaluate model performance on test set.
        
        Args:
            X_test_tfidf: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        self.y_pred = self.model.predict(X_test_tfidf)
        self.y_pred_proba = self.model.predict_proba(X_test_tfidf)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, self.y_pred)
        precision = precision_score(y_test, self.y_pred, pos_label='spam', zero_division=0)
        recall = recall_score(y_test, self.y_pred, pos_label='spam', zero_division=0)
        f1 = f1_score(y_test, self.y_pred, pos_label='spam', zero_division=0)
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, self.y_pred,
                                    target_names=['Ham', 'Spam'],
                                    zero_division=0))
        
        return self.metrics
    
    def confusion_matrix_analysis(self, y_test):
        """
        Analyze and display confusion matrix.
        
        Args:
            y_test: Test labels
            
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_test, self.y_pred)
        
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Ham  Spam")
        print(f"Actual Ham   [{cm[0,0]:5d} {cm[0,1]:5d}]")
        print(f"       Spam  [{cm[1,0]:5d} {cm[1,1]:5d}]")
        
        # Calculate derived metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nConfusion Matrix Analysis:")
        print(f"  True Negatives (Correctly classified ham):  {tn}")
        print(f"  True Positives (Correctly classified spam): {tp}")
        print(f"  False Positives (Ham classified as spam):   {fp}")
        print(f"  False Negatives (Spam classified as ham):   {fn}")
        print(f"  Sensitivity (True Positive Rate): {sensitivity:.4f}")
        print(f"  Specificity (True Negative Rate): {specificity:.4f}")
        
        return cm
    
    def plot_confusion_matrix(self, y_test, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_test: Test labels
            save_path: Path to save the figure
        """
        cm = confusion_matrix(y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Spam Detection Model')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix plot saved to '{save_path}'")
        plt.close()
    
    def plot_metrics(self, save_path='metrics.png'):
        """
        Plot performance metrics as bar chart.
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics:
            print("No metrics to plot. Run evaluate() first.")
            return
        
        metrics_names = list(self.metrics.keys())
        metrics_values = list(self.metrics.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to '{save_path}'")
        plt.close()
    
    def plot_roc_curve(self, y_test, save_path='roc_curve.png'):
        """
        Plot ROC curve.
        
        Args:
            y_test: Test labels
            save_path: Path to save the figure
        """
        if self.y_pred_proba is None:
            print("No probability predictions available.")
            return
        
        fpr, tpr, _ = roc_curve(y_test, self.y_pred_proba, pos_label='spam')
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Spam Detection Model', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to '{save_path}'")
        plt.close()


class ModelPersistence:
    """Handle saving and loading models."""
    
    @staticmethod
    def save_model(model, vectorizer, model_path='spam_model.joblib', 
                   vectorizer_path='tfidf_vectorizer.joblib'):
        """
        Save trained model and vectorizer.
        
        Args:
            model: Trained model
            vectorizer: TF-IDF vectorizer
            model_path: Path to save model
            vectorizer_path: Path to save vectorizer
        """
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"\nModel saved to '{model_path}'")
        print(f"Vectorizer saved to '{vectorizer_path}'")
    
    @staticmethod
    def load_model(model_path='spam_model.joblib',
                   vectorizer_path='tfidf_vectorizer.joblib'):
        """
        Load trained model and vectorizer.
        
        Args:
            model_path: Path to model file
            vectorizer_path: Path to vectorizer file
            
        Returns:
            Tuple of (model, vectorizer)
        """
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print(f"Model loaded from '{model_path}'")
        print(f"Vectorizer loaded from '{vectorizer_path}'")
        return model, vectorizer
    
    @staticmethod
    def predict_email(model, vectorizer, email_text):
        """
        Predict if an email is spam.
        
        Args:
            model: Trained model
            vectorizer: TF-IDF vectorizer
            email_text: Email text to classify
            
        Returns:
            Dictionary with prediction and probability
        """
        # Preprocess the email (note: should use the same preprocessing pipeline)
        email_tfidf = vectorizer.transform([email_text])
        
        prediction = model.predict(email_tfidf)[0]
        probability = model.predict_proba(email_tfidf)[0]
        
        return {
            'prediction': 'SPAM' if prediction == 'spam' else 'HAM',
            'spam_probability': float(probability[1]),
            'ham_probability': float(probability[0])
        }


if __name__ == "__main__":
    print("Evaluation module. Use this in the main pipeline.")
