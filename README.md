# Spam Email Detection - Machine Learning Project

A comprehensive end-to-end machine learning project that classifies emails as spam or legitimate using Natural Language Processing (NLP), TF-IDF vectorization, and Logistic Regression.

## 📋 Project Overview

This project demonstrates the complete ML pipeline:
- **Data Collection & Exploration**: Load and analyze spam/ham datasets
- **Text Preprocessing**: Clean, tokenize, and normalize email text
- **Feature Extraction**: Convert text to numerical features using TF-IDF
- **Model Building**: Train Logistic Regression classifier
- **Hyperparameter Optimization**: Use GridSearchCV for optimal parameters
- **Evaluation**: Comprehensive metrics and visualizations
- **Deployment**: Streamlit web application for predictions

## 🎯 Key Features

- ✅ Complete text preprocessing pipeline with NLTK
- ✅ TF-IDF vectorization with configurable parameters
- ✅ Logistic Regression with hyperparameter tuning
- ✅ Cross-validation and GridSearchCV optimization
- ✅ Detailed evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ Confusion matrix and ROC curve visualization
- ✅ Model persistence with joblib
- ✅ Interactive Streamlit web application

## 🏗️ Project Structure

```
.
├── data_loader.py              # Data loading and exploration
├── preprocessing.py            # Text preprocessing pipeline
├── model_training.py           # Model building and optimization
├── evaluation.py               # Model evaluation and visualization
├── train_model.py              # Main training orchestrator
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── models/                     # Saved model and vectorizer
│   ├── spam_model.joblib
│   └── tfidf_vectorizer.joblib
├── visualizations/             # Generated plots and charts
│   ├── confusion_matrix.png
│   ├── metrics.png
│   └── roc_curve.png
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository** (if applicable)
```bash
git clone <repository-url>
cd spam-detection
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Step 1: Train the Model

Run the main training script to build and optimize the model:

```bash
python train_model.py
```

This will:
- Load and explore the dataset
- Preprocess all email text
- Extract TF-IDF features
- Train the initial model
- Optimize hyperparameters using GridSearchCV
- Evaluate performance on test set
- Generate visualizations
- Save the trained model

**Output:**
- Trained model: `models/spam_model.joblib`
- TF-IDF vectorizer: `models/tfidf_vectorizer.joblib`
- Visualizations: `visualizations/` directory

### Step 2: Run the Web Application

Launch the Streamlit app for interactive predictions:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

The app allows you to:
- Paste email text for classification
- View spam/ham probability scores
- See preprocessing details
- Try example emails

## 📊 Model Details

### Algorithm: Logistic Regression
- Probabilistic classifier for binary classification
- Fast training and prediction
- Interpretable results

### Feature Extraction: TF-IDF
- **Max features**: 5,000 most important terms
- **N-gram range**: (1, 2) - unigrams and bigrams
- **Min document frequency**: 2
- **Max document frequency**: 95%

### Hyperparameter Optimization
GridSearchCV searches over:
- **C** (regularization): [0.001, 0.01, 0.1, 1, 10, 100]
- **penalty**: ['l2']
- **solver**: ['lbfgs', 'liblinear']
- **max_iter**: [500, 1000, 2000]

### Cross-Validation
- 5-fold cross-validation
- Scoring metric: F1-Score (balances precision and recall)

## 📈 Evaluation Metrics

The model provides:

1. **Accuracy**: Overall correctness of predictions
2. **Precision**: Of emails flagged as spam, how many are actually spam
3. **Recall**: Of actual spam emails, how many are correctly identified
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: True positives, false positives, false negatives, true negatives
6. **ROC Curve**: Trade-off between true and false positive rates

## 🔍 Text Preprocessing Pipeline

The preprocessing module applies these steps:

1. **Lowercasing**: Convert all text to lowercase
2. **URL/Email Removal**: Remove URLs and email addresses
3. **Special Character Removal**: Remove punctuation and numbers
4. **Tokenization**: Split text into individual words
5. **Stopword Removal**: Remove common words (the, a, is, etc.)
6. **Lemmatization**: Reduce words to root form (running → run)

## 📝 Using the Model Programmatically

```python
from evaluation import ModelPersistence
from preprocessing import TextPreprocessor

# Load model
model, vectorizer = ModelPersistence.load_model()

# Predict on new email
email = "You have won a free prize! Click here now!"
result = ModelPersistence.predict_email(model, vectorizer, email)

print(f"Prediction: {result['prediction']}")
print(f"Spam probability: {result['spam_probability']:.2%}")
```

## 🎓 Understanding the Code

### data_loader.py
- `DataLoader`: Main class for data operations
  - `load_data()`: Load or create dataset
  - `explore_data()`: Display dataset statistics
  - `clean_data()`: Remove duplicates and missing values

### preprocessing.py
- `TextPreprocessor`: Text cleaning and normalization
  - `preprocess()`: Complete preprocessing pipeline
  - `preprocess_dataframe()`: Apply to entire column

### model_training.py
- `SpamDetectionModel`: Model building and training
  - `split_data()`: Train/test split
  - `extract_features()`: TF-IDF vectorization
  - `train_model()`: Train initial model
  - `optimize_hyperparameters()`: GridSearchCV tuning

### evaluation.py
- `ModelEvaluator`: Performance evaluation
  - `evaluate()`: Calculate metrics
  - `plot_confusion_matrix()`: Visualize predictions
  - `plot_metrics()`: Bar chart of metrics
  - `plot_roc_curve()`: ROC curve visualization

- `ModelPersistence`: Save/load models
  - `save_model()`: Persist trained model
  - `load_model()`: Load saved model
  - `predict_email()`: Predict on new data

## 🔧 Customization

### Change Model Parameters

Edit `model_training.py`:
```python
# Change TF-IDF parameters
model.build_vectorizer(
    max_features=3000,      # Reduce features
    ngram_range=(1, 3),     # Include trigrams
    min_df=5                # Higher minimum frequency
)

# Change model parameters
model.optimize_hyperparameters(X_train_tfidf, cv=10)  # 10-fold CV
```

### Use Different Dataset

Provide your own CSV file:
```python
loader = DataLoader('path/to/your/emails.csv')
```

The CSV should have columns: `label` (ham/spam) and `message` (email text)

## 📦 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: ML algorithms and metrics
- **nltk**: NLP preprocessing
- **matplotlib & seaborn**: Visualization
- **streamlit**: Web application
- **joblib**: Model persistence

## ⚠️ Important Notes

1. **Data Privacy**: Never share your model with sensitive training data
2. **Model Performance**: Performance depends on training data quality
3. **Production Use**: This is a demonstration model; use established email services for production
4. **False Positives/Negatives**: Always consider the cost of misclassification

## 🐛 Troubleshooting

### Model files not found
```bash
python train_model.py  # Retrain the model
```

### Missing NLTK data
The preprocessing module automatically downloads required data on first run.

### Streamlit port already in use
```bash
streamlit run app.py --server.port=8502
```

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/
- NLTK Documentation: https://www.nltk.org/
- TF-IDF Vectorization: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Logistic Regression: https://en.wikipedia.org/wiki/Logistic_regression
- Streamlit Documentation: https://docs.streamlit.io/

## 📄 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Feel free to fork, modify, and improve this project. Some ideas:
- Try different classifiers (SVM, Random Forest, etc.)
- Implement deep learning approaches (LSTM, BERT)
- Add more sophisticated text preprocessing
- Expand dataset with more examples
- Improve the web UI with more features

## 📧 Contact

For questions or suggestions, feel free to reach out!

---

Happy spam detecting! 🎯
