import sys
import os
import json
import joblib
from pathlib import Path

# Add root directory to path to import local modules with priority
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from evaluation import ModelPersistence
from preprocessing import TextPreprocessor

def classify():
    # Load model and vectorizer
    model_path = root_dir / 'models' / 'spam_model.joblib'
    vectorizer_path = root_dir / 'models' / 'tfidf_vectorizer.joblib'
    
    if not model_path.exists() or not vectorizer_path.exists():
        print(json.dumps({"error": f"Model files not found at {model_path}. Please train the model first."}))
        sys.exit(1)
        
    try:
        model, vectorizer = ModelPersistence.load_model(str(model_path), str(vectorizer_path))
        preprocessor = TextPreprocessor()
        
        # Read from stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            print(json.dumps({"error": "No input provided"}))
            sys.exit(1)
            
        # Try both 'preprocess' and 'preprocess_text' for compatibility
        if hasattr(preprocessor, 'preprocess'):
            cleaned_text = preprocessor.preprocess(input_data)
        elif hasattr(preprocessor, 'preprocess_text'):
            cleaned_text = preprocessor.preprocess_text(input_data)
        else:
            # Fallback if no preprocess method exists
            cleaned_text = input_data
        
        # Predict
        result = ModelPersistence.predict_email(model, vectorizer, cleaned_text)
        
        # Add metadata
        result['input_length'] = len(input_data)
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    classify()
