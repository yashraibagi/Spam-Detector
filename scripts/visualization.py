"""
Visualization Module for Spam Email Detection

This module creates visualizations for:
- Confusion matrix heatmap
- Performance metrics bar chart
- Class distribution
- Word frequency distributions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os


def setup_plotting_style():
    """Configure matplotlib and seaborn styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    sns.set_context("notebook", font_scale=1.1)


def plot_confusion_matrix(confusion_matrix_data, labels=None, save_path=None):
    """
    Create and display a confusion matrix heatmap.
    
    Args:
        confusion_matrix_data (np.ndarray): Confusion matrix data
        labels (list): Class labels (e.g., ['Not Spam', 'Spam'])
        save_path (str): Path to save the figure (optional)
    """
    setup_plotting_style()
    
    if labels is None:
        labels = ['Not Spam', 'Spam']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        linewidths=2,
        linecolor='black'
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_performance_metrics(accuracy, precision, recall, f1_score, save_path=None):
    """
    Create a bar chart of performance metrics.
    
    Args:
        accuracy (float): Accuracy score
        precision (float): Precision score
        recall (float): Recall score
        f1_score (float): F1 score
        save_path (str): Path to save the figure (optional)
    """
    setup_plotting_style()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [accuracy, precision, recall, f1_score]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, scores, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{score:.4f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.ylim(0, 1.1)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value line at y=1
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics plot saved to {save_path}")
    
    plt.show()


def plot_class_distribution(class_counts, class_labels=None, save_path=None):
    """
    Create a bar chart showing class distribution.
    
    Args:
        class_counts (dict): Dictionary with class names and their counts
        class_labels (list): Custom labels for classes
        save_path (str): Path to save the figure (optional)
    """
    setup_plotting_style()
    
    if class_labels is None:
        class_labels = ['Not Spam', 'Spam']
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#2ECC71', '#E74C3C']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [class_labels[i] if i < len(class_labels) else str(cls) for i, cls in enumerate(classes)],
        counts,
        color=colors[:len(classes)],
        edgecolor='black',
        linewidth=2
    )
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(count)}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.ylabel('Number of Emails', fontsize=12)
    plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_word_frequency(texts, title, num_words=20, save_path=None):
    """
    Plot word frequency distribution.
    
    Args:
        texts (list): List of text strings
        title (str): Title for the plot
        num_words (int): Number of top words to display
        save_path (str): Path to save the figure (optional)
    """
    setup_plotting_style()
    
    # Combine all texts and split into words
    all_words = []
    for text in texts:
        words = str(text).split()
        all_words.extend(words)
    
    # Count word frequencies
    word_freq = Counter(all_words)
    most_common = dict(word_freq.most_common(num_words))
    
    plt.figure(figsize=(12, 6))
    words = list(most_common.keys())
    freqs = list(most_common.values())
    
    bars = plt.barh(words, freqs, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Add frequency labels
    for bar, freq in zip(bars, freqs):
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height()/2.,
            f' {int(freq)}',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.xlabel('Frequency', fontsize=12)
    plt.title(f'Word Frequency Distribution - {title}', fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Word frequency plot saved to {save_path}")
    
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """
    Plot ROC curve for binary classification.
    
    Args:
        fpr (np.ndarray): False positive rate
        tpr (np.ndarray): True positive rate
        roc_auc (float): Area under the ROC curve
        save_path (str): Path to save the figure (optional)
    """
    setup_plotting_style()
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to {save_path}")
    
    plt.show()


def create_all_visualizations(metrics, class_counts, spam_texts, ham_texts, output_dir='visualizations'):
    """
    Create all visualizations in one function.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        class_counts (dict): Class distribution counts
        spam_texts (list): List of spam email texts
        ham_texts (list): List of non-spam email texts
        output_dir (str): Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nCreating visualizations in {output_dir}...")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        labels=['Not Spam', 'Spam'],
        save_path=f'{output_dir}/confusion_matrix.png'
    )
    
    # Plot performance metrics
    plot_performance_metrics(
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        save_path=f'{output_dir}/performance_metrics.png'
    )
    
    # Plot class distribution
    plot_class_distribution(
        class_counts,
        class_labels=['Not Spam', 'Spam'],
        save_path=f'{output_dir}/class_distribution.png'
    )
    
    # Plot word frequency for spam and ham
    if spam_texts:
        plot_word_frequency(
            spam_texts,
            'Spam Emails',
            num_words=15,
            save_path=f'{output_dir}/spam_word_frequency.png'
        )
    
    if ham_texts:
        plot_word_frequency(
            ham_texts,
            'Non-Spam Emails',
            num_words=15,
            save_path=f'{output_dir}/ham_word_frequency.png'
        )
    
    print(f"All visualizations saved to {output_dir}/")
