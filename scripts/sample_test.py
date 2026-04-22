"""
Sample Test Script for Spam Detection Model

This script demonstrates how to use the trained model for predictions
on sample emails. Run this after training the model.
"""

import sys
import os
import joblib
from preprocessing import TextPreprocessor

# Sample emails for testing
SAMPLE_EMAILS = {
    "spam_1": """
    Subject: CLICK HERE FOR FREE MONEY!!!
    
    Hi Friend,
    
    I have an amazing opportunity for you! Earn $5000 per week 
    from home with NO EXPERIENCE NEEDED! 
    
    Click here: http://bit.ly/makemoney123
    
    This is a LIMITED TIME OFFER!
    
    Best regards,
    Nigeria Prince
    """,
    
    "spam_2": """
    Dear Valued Customer,
    
    Your account has been compromised! Click here immediately to verify 
    your details and restore access to your account.
    
    URGENT: Your account may be permanently closed if you don't respond.
    
    Click: malicious-site.com/verify
    
    PayPal Security Team
    """,
    
    "ham_1": """
    Hi Sarah,
    
    I hope you're doing well! I wanted to follow up on our meeting 
    scheduled for Thursday at 2 PM. 
    
    Please let me know if you can still make it, or if we need to 
    reschedule.
    
    Looking forward to hearing from you!
    
    Best,
    John
    """,
    
    "ham_2": """
    Team,
    
    Please find attached the quarterly report for Q3. The results show 
    a 15% increase in revenue compared to Q2.
    
    Key highlights:
    - Customer satisfaction up by 8%
    - Operating costs reduced by 12%
    - New product launch successful
    
    Let's discuss these results in tomorrow's meeting at 10 AM.
    
    Thanks,
    Manager
    """,
    
    "spam_3": """
    CONGRATULATIONS! YOU'VE WON!!!
    
    You have been randomly selected as a winner in the International 
    Lottery. You have won $2,500,000!
    
    To claim your prize, please reply with your banking details.
    
    All our lawyers are standing by to help you!
    """,
    
    "ham_3": """
    Hello,
    
    Thank you for ordering from us! Your order #12345 has been 
    shipped and is on its way.
    
    Tracking number: ABC123DEF456
    Expected delivery: 5-7 business days
    
    You can track your package here: www.shipping-tracker.com
    
    Thank you for your business!
    
    Customer Service Team
    """
}


def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    model_path = 'models/spam_detector_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("❌ Models not found!")
        print("Please run 'python train.py' first to train the model.")
        return None, None
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("✅ Models loaded successfully!")
        return model, vectorizer
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None


def test_emails(model, vectorizer):
    """Test the model on sample emails."""
    
    if model is None or vectorizer is None:
        return
    
    preprocessor = TextPreprocessor()
    
    print("\n" + "="*70)
    print("SPAM DETECTION - SAMPLE EMAIL TESTING")
    print("="*70)
    
    correct_predictions = 0
    total_predictions = 0
    
    for email_name, email_text in SAMPLE_EMAILS.items():
        # Determine expected category from name
        expected = "SPAM" if "spam" in email_name else "HAM"
        
        # Preprocess
        cleaned_text = preprocessor.preprocess_text(email_text)
        
        # Vectorize
        email_tfidf = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(email_tfidf)[0]
        probabilities = model.predict_proba(email_tfidf)[0]
        
        # Map prediction to label
        if isinstance(prediction, str):
            predicted_label = prediction.upper()
        else:
            predicted_label = "SPAM" if prediction == 1 else "HAM"
        
        # Get confidence
        confidence = max(probabilities) * 100
        
        # Check if correct
        is_correct = expected == predicted_label
        correct_predictions += is_correct
        total_predictions += 1
        
        # Display result
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        
        print(f"\n{status}")
        print(f"Email: {email_name.upper()}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted_label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Email Preview: {email_text[:100]}...")
        print(f"Probabilities - HAM: {probabilities[0]*100:.2f}%, SPAM: {probabilities[1]*100:.2f}%")
        print("-" * 70)
    
    # Summary
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {total_predictions}")
    print(f"Correct: {correct_predictions}")
    print(f"Incorrect: {total_predictions - correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*70}\n")


def interactive_test(model, vectorizer):
    """Interactive testing mode."""
    
    if model is None or vectorizer is None:
        return
    
    preprocessor = TextPreprocessor()
    
    print("\n" + "="*70)
    print("INTERACTIVE SPAM DETECTION TEST")
    print("="*70)
    print("Enter email text (type 'quit' to exit):\n")
    
    while True:
        email_text = input("Enter email text (or 'quit' to exit): ").strip()
        
        if email_text.lower() == 'quit':
            print("Exiting...")
            break
        
        if not email_text:
            print("Please enter some text.\n")
            continue
        
        try:
            # Preprocess
            cleaned_text = preprocessor.preprocess_text(email_text)
            
            # Vectorize
            email_tfidf = vectorizer.transform([cleaned_text])
            
            # Predict
            prediction = model.predict(email_tfidf)[0]
            probabilities = model.predict_proba(email_tfidf)[0]
            
            # Map prediction to label
            if isinstance(prediction, str):
                predicted_label = prediction
            else:
                predicted_label = "spam" if prediction == 1 else "ham"
            
            confidence = max(probabilities) * 100
            
            # Display result
            if predicted_label.lower() == "spam":
                print(f"\n🚨 SPAM DETECTED - Confidence: {confidence:.2f}%")
            else:
                print(f"\n✅ LEGITIMATE EMAIL - Confidence: {confidence:.2f}%")
            
            print(f"Probabilities - Legitimate: {probabilities[0]*100:.2f}%, Spam: {probabilities[1]*100:.2f}%\n")
        
        except Exception as e:
            print(f"Error processing email: {e}\n")


def main():
    """Main test function."""
    
    print("\nLoading models...")
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        return
    
    print("\nChoose test mode:")
    print("1. Test with sample emails (automatic)")
    print("2. Interactive test (manual input)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        test_emails(model, vectorizer)
    
    if choice in ['2', '3']:
        interactive_test(model, vectorizer)


if __name__ == "__main__":
    main()
