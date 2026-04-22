"""
Streamlit Web Application for Spam Email Detection

This application provides an interactive interface for:
- Real-time email spam classification
- Probability visualization
- Batch prediction
- Model information and statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import TextPreprocessor


# Page configuration
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer from disk."""
    try:
        model_path = 'models/spam_detector_model.pkl'
        vectorizer_path = 'models/tfidf_vectorizer.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return None, None, "Models not found. Please train the model first."
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, None
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"


@st.cache_resource
def get_preprocessor():
    """Get the text preprocessor instance."""
    return TextPreprocessor(use_lemmatization=True)


def preprocess_email(email_text, preprocessor):
    """
    Preprocess email text before prediction.
    
    Args:
        email_text (str): Raw email text
        preprocessor: TextPreprocessor instance
    
    Returns:
        str: Cleaned email text
    """
    return preprocessor.preprocess_text(email_text)


def predict_email(email_text, model, vectorizer, preprocessor):
    """
    Predict if an email is spam or not.
    
    Args:
        email_text (str): Email text to classify
        model: Trained Logistic Regression model
        vectorizer: TF-IDF vectorizer
        preprocessor: Text preprocessor
    
    Returns:
        dict: Prediction result with label and probability
    """
    # Preprocess email
    cleaned_text = preprocess_email(email_text, preprocessor)
    
    # Vectorize
    email_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(email_tfidf)[0]
    probabilities = model.predict_proba(email_tfidf)[0]
    
    return {
        'prediction': prediction,
        'ham_probability': probabilities[0] if len(probabilities) > 0 else 0,
        'spam_probability': probabilities[1] if len(probabilities) > 1 else 0,
        'cleaned_text': cleaned_text,
        'confidence': max(probabilities) * 100
    }


def display_prediction_result(result):
    """Display prediction result with visualization."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if result['prediction'] == 'spam' or result['prediction'] == 1:
            st.error("🚨 **SPAM EMAIL DETECTED**", icon="⚠️")
            label = "SPAM"
            color = '#FF6B6B'
        else:
            st.success("✅ **LEGITIMATE EMAIL**", icon="✓")
            label = "LEGITIMATE"
            color = '#2ECC71'
    
    with col2:
        st.metric("Confidence", f"{result['confidence']:.2f}%")
    
    with col3:
        st.metric("Prediction", label)
    
    # Probability visualization
    st.subheader("Prediction Probabilities")
    
    prob_data = {
        'Legitimate': result['ham_probability'] * 100,
        'Spam': result['spam_probability'] * 100
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#2ECC71', '#FF6B6B']
        bars = ax.bar(prob_data.keys(), prob_data.values(), color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, value in zip(bars, prob_data.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probability (%)', fontsize=11, fontweight='bold')
        ax.set_title('Email Classification Probability', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        # Gauge chart using metric
        st.metric("Spam Probability", f"{result['spam_probability']*100:.2f}%")
        st.metric("Legitimate Probability", f"{result['ham_probability']*100:.2f}%")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("<h1 style='text-align: center; color: #2E86AB;'>📧 Spam Email Detector</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Classify emails as spam or legitimate using Machine Learning</p>", 
                unsafe_allow_html=True)
    
    st.divider()
    
    # Load model and vectorizer
    model, vectorizer, error = load_model_and_vectorizer()
    
    if error:
        st.error(f"❌ {error}")
        st.info("Please run the training script first: `python train.py`")
        return
    
    # Get preprocessor
    preprocessor = get_preprocessor()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        page = st.radio("Select Page", ["Single Email", "Batch Prediction", "Model Info"])
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This application uses:
        - **TF-IDF Vectorization** for feature extraction
        - **Logistic Regression** for classification
        - **NLTK** for text preprocessing
        
        The model was trained on a spam email dataset to classify emails as either:
        - ✅ Legitimate
        - 🚨 Spam
        """)
    
    # Page content
    if page == "Single Email":
        st.header("🔍 Single Email Classification")
        
        # Input area
        st.subheader("Enter Email Text")
        email_text = st.text_area(
            "Paste your email content here:",
            height=200,
            placeholder="Enter email subject and body...",
            label_visibility="collapsed"
        )
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            predict_button = st.button("🔎 Classify Email", use_container_width=True, type="primary")
        
        with col2:
            if st.button("📋 Clear", use_container_width=True):
                st.rerun()
        
        if predict_button and email_text.strip():
            with st.spinner("Analyzing email..."):
                result = predict_email(email_text, model, vectorizer, preprocessor)
            
            st.divider()
            
            # Display results
            display_prediction_result(result)
            
            # Show preprocessed text
            with st.expander("📄 Preprocessed Email Text"):
                st.code(result['cleaned_text'], language="text")
                st.caption(f"Original length: {len(email_text)} characters | Cleaned length: {len(result['cleaned_text'])} characters")
        
        elif predict_button:
            st.warning("⚠️ Please enter an email to classify.")
    
    elif page == "Batch Prediction":
        st.header("📊 Batch Email Prediction")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file with emails",
            type=['csv'],
            help="CSV should have a 'text' or 'email' column with email content"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Find text column
                text_column = None
                for col in ['text', 'email', 'message', 'content', 'body']:
                    if col in df.columns:
                        text_column = col
                        break
                
                if text_column is None:
                    text_column = df.columns[0]
                
                st.info(f"Using column '{text_column}' for email text")
                
                if st.button("🔎 Classify All Emails", type="primary", use_container_width=True):
                    with st.spinner("Classifying emails..."):
                        predictions = []
                        probabilities = []
                        
                        for idx, row in df.iterrows():
                            email = str(row[text_column])
                            result = predict_email(email, model, vectorizer, preprocessor)
                            predictions.append(result['prediction'])
                            probabilities.append(result['spam_probability'])
                        
                        df['prediction'] = predictions
                        df['spam_probability'] = probabilities
                    
                    # Display results
                    st.subheader("📈 Classification Results")
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Emails", len(df))
                    with col2:
                        spam_count = (df['prediction'] == 'spam').sum() if 'spam' in df['prediction'].values else (df['prediction'] == 1).sum()
                        st.metric("Spam Detected", spam_count)
                    with col3:
                        legit_count = len(df) - spam_count
                        st.metric("Legitimate", legit_count)
                    
                    st.divider()
                    
                    # Display results table
                    st.subheader("📋 Detailed Results")
                    display_df = df[[text_column, 'prediction', 'spam_probability']].copy()
                    display_df['spam_probability'] = display_df['spam_probability'].apply(lambda x: f"{x*100:.2f}%")
                    display_df.columns = ['Email', 'Classification', 'Spam Probability']
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"spam_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif page == "Model Info":
        st.header("ℹ️ Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Details")
            st.markdown(f"""
            - **Algorithm**: Logistic Regression
            - **Feature Extraction**: TF-IDF Vectorization
            - **Text Preprocessing**: 
              - Lowercasing
              - Special character removal
              - Stopword removal
              - Lemmatization
            - **N-gram Range**: (1, 2)
            - **Max Features**: 5000
            """)
        
        with col2:
            st.subheader("🎯 Model Purpose")
            st.markdown("""
            This model classifies emails as either:
            
            1. **Legitimate (Ham)** ✅
               - Regular emails from trusted sources
               - Contains genuine content
            
            2. **Spam** 🚨
               - Unsolicited bulk emails
               - Phishing attempts
               - Advertisements
               - Malicious content
            """)
        
        st.divider()
        
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Single Email**: Paste an email and click 'Classify Email' to get instant prediction
        2. **Batch Prediction**: Upload a CSV file with multiple emails for bulk classification
        3. The model will return:
           - Email classification (Spam or Legitimate)
           - Confidence score (0-100%)
           - Individual probability scores
        """)
        
        st.divider()
        
        st.subheader("⚠️ Important Notes")
        st.info("""
        - The model's accuracy depends on the training dataset
        - False positives/negatives are possible
        - Always review important emails manually
        - The model works best with longer emails with full content
        """)


if __name__ == "__main__":
    main()
