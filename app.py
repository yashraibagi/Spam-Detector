"""
Streamlit Web Application for Spam Email Detection
Allows users to input emails and get spam/ham predictions.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from evaluation import ModelPersistence
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
    .header {
        text-align: center;
        padding: 2rem 0;
    }
    .result-spam {
        background-color: #ff6b6b;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-ham {
        background-color: #51cf66;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metrics {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background-color: #f0f0f0;
        border-radius: 0.5rem;
        flex: 1;
        margin: 0 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    model_path = 'models/spam_model.joblib'
    vectorizer_path = 'models/tfidf_vectorizer.joblib'
    
    if not Path(model_path).exists() or not Path(vectorizer_path).exists():
        return None, None
    
    return ModelPersistence.load_model(model_path, vectorizer_path)


@st.cache_resource
def get_preprocessor():
    """Get the text preprocessor."""
    return TextPreprocessor()


def main():
    """Main Streamlit app."""
    
    # Initialize session state for email input if not exists
    if 'email_input' not in st.session_state:
        st.session_state.email_input = ""

    def set_email_example(example_text):
        st.session_state.email_input = example_text
    
    # Header
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    st.title("📧 Spam Email Detection System")
    st.markdown("""
    This application uses machine learning to classify emails as **SPAM** or **HAM** (not spam).
    The model was trained using TF-IDF vectorization and Logistic Regression.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Load model
    model, vectorizer = load_model_and_vectorizer()
    preprocessor = get_preprocessor()
    
    if model is None or vectorizer is None:
        st.error("""
        ❌ **Model files not found!**
        
        Please train the model first by running:
        ```bash
        python train_model.py
        ```
        
        This will generate the required model files in the `models/` directory.
        """)
        return
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **Model Information:**
        - Algorithm: Logistic Regression
        - Feature Extraction: TF-IDF
        - N-gram range: (1, 2)
        - Max features: 5000
        """)
        
        st.header("📊 How it works")
        st.markdown("""
        1. **Text Preprocessing**: Email text is cleaned, tokenized, and lemmatized
        2. **Feature Extraction**: Text is converted to numerical features using TF-IDF
        3. **Classification**: Logistic Regression predicts if email is spam or ham
        4. **Probability**: Confidence score is provided for the prediction
        """)
        
        st.header("⚙️ Options")
        show_details = st.checkbox("Show preprocessing details", value=False)
    
    # Main content area
    st.header("📝 Enter Email Content")
    
    # Text input
    email_text = st.text_area(
        "Paste your email text here:",
        placeholder="Enter the email content you want to classify...",
        height=200,
        key="email_input"
    )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button("🔍 Classify Email", use_container_width=True)
    
    # Make prediction
    if predict_button and email_text.strip():
        with st.spinner("Analyzing email..."):
            # Preprocess the email
            cleaned_text = preprocessor.preprocess(email_text)
            
            # Show preprocessing details if requested
            if show_details:
                st.subheader("📋 Preprocessing Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Text:**")
                    st.text(email_text[:300] + "..." if len(email_text) > 300 else email_text)
                with col2:
                    st.write("**Cleaned Text:**")
                    st.text(cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text)
            
            # Make prediction using the original email text
            result = ModelPersistence.predict_email(model, vectorizer, email_text)
            
            # Display results
            st.subheader("🎯 Prediction Result")
            
            if result['prediction'] == 'SPAM':
                st.markdown(f"""
                <div class='result-spam'>
                <h2>🚨 SPAM DETECTED</h2>
                <p>This email is likely SPAM with {result['spam_probability']*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-ham'>
                <h2>✅ LEGITIMATE EMAIL</h2>
                <p>This email is likely LEGITIMATE (HAM) with {result['ham_probability']*100:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed probabilities
            st.subheader("📊 Confidence Scores")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Spam Probability",
                    value=f"{result['spam_probability']*100:.2f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Ham Probability",
                    value=f"{result['ham_probability']*100:.2f}%",
                    delta=None
                )
            
            # Probability visualization
            st.subheader("📈 Probability Distribution")
            chart_data = pd.DataFrame(
                [result['spam_probability'], result['ham_probability']],
                index=['SPAM', 'HAM'],
                columns=['Probability']
            )
            st.bar_chart(chart_data, use_container_width=True)
    
    elif predict_button:
        st.warning("⚠️ Please enter some email text to classify.")
    
    # Examples
    st.divider()
    st.header("📚 Example Inputs")
    
    examples = {
        "Clear Spam": "WINNER!!! You have won a FREE IPHONE!!! Click here now: www.freephone.com to claim your prize!!!",
        "Clear Ham": "Hi John, I wanted to follow up on our conversation yesterday. Do you have time to meet next week?",
        "Suspicious": "Dear valued customer, we need to verify your account information. Click here immediately!"
    }
    
    cols = st.columns(len(examples))
    for idx, (label, example) in enumerate(examples.items()):
        with cols[idx]:
            st.button(
                f"Try: {label}", 
                use_container_width=True, 
                on_click=set_email_example, 
                args=(example,)
            )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>💡 Disclaimer: This is a demonstration model. For production use, consider using 
    established email filtering services.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
