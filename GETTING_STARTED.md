# Getting Started with Spam Email Detection

Quick start guide to get the spam detection project up and running.

## ⚡ Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r scripts/requirements.txt
```

### 2. Get a Dataset

Download a spam dataset and save it as `spam_dataset.csv` in the root directory.

**Quick options:**
- **Kaggle Spam Ham Dataset** (Recommended for beginners)
  - Download: https://www.kaggle.com/datasets/mfaisalqureshi/spam-ham-dataset
  - Save as: `spam_dataset.csv`

- **UCI SMS Spam Collection**
  - Download: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
  - Convert to CSV format with 'text' and 'label' columns

### 3. Train the Model

```bash
cd scripts
python train.py
```

This will:
- Load and analyze your dataset
- Preprocess all emails
- Train the ML model
- Generate performance visualizations
- Save the trained model

⏱️ Training time: 2-15 minutes (depending on dataset size)

### 4. Run the Web App

```bash
cd scripts
streamlit run streamlit_app.py
```

Open your browser and visit: `http://localhost:8501`

---

## 📊 Understanding Your Results

After training, you'll see:

### 1. Training Console Output
```
Model Performance Summary:
  Accuracy:  0.9750
  Precision: 0.9823
  Recall:    0.9654
  F1-Score:  0.9738
```

**What this means:**
- **Accuracy**: 97.5% of all predictions correct
- **Precision**: 98.23% of spam predictions are correct
- **Recall**: 96.54% of actual spam caught
- **F1-Score**: Balanced performance (good!)

### 2. Generated Files

```
models/
├── spam_detector_model.pkl       ← Your trained model
└── tfidf_vectorizer.pkl          ← Feature transformer

visualizations/
├── confusion_matrix.png           ← True/False positives
├── performance_metrics.png        ← Accuracy, Precision, Recall
├── class_distribution.png         ← Spam vs Legitimate
├── spam_word_frequency.png        ← Most common spam words
└── ham_word_frequency.png         ← Most common legitimate words
```

### 3. Web App Interface

When you open the Streamlit app, you'll find three sections:

**📧 Single Email Classification**
- Paste one email
- Get instant prediction
- View confidence score

**📊 Batch Prediction**
- Upload CSV with multiple emails
- Process all at once
- Download results

**ℹ️ Model Info**
- Learn how the model works
- Understand preprocessing steps
- See classification categories

---

## 🧪 Test Your Model

### Option 1: Sample Emails (Automated)

```bash
cd scripts
python sample_test.py
```

Choose option 1 to test on 6 pre-loaded sample emails.

### Option 2: Interactive Test

```bash
cd scripts
python sample_test.py
```

Choose option 2 to manually enter emails and test.

### Option 3: Web App (Visual)

Use the Streamlit app and paste emails directly!

---

## 🎯 Common Tasks

### Check Dataset Statistics

```python
import pandas as pd
df = pd.read_csv('spam_dataset.csv')
print(df.info())
print(df['label'].value_counts())  # Adjust 'label' to your column name
```

### Change Dataset Column Names

If your CSV has different column names, edit `scripts/train.py`:

```python
# Around line 50, adjust these:
text_column = 'message'  # Change to your text column name
label_column = 'category'  # Change to your label column name
```

### Retrain with Different Settings

Edit `scripts/train.py`:

```python
# Fewer features (faster training, less memory)
max_features=2000  # Instead of 5000

# Different train/test split
test_size=0.15  # Instead of 0.2 (85/15 split)

# More cross-validation folds
cv=10  # Instead of 5 (more thorough but slower)
```

### Use Stemming Instead of Lemmatization

```python
df_clean, preprocessor = create_clean_dataset(
    df, text_column, label_column, 
    use_lemmatization=False  # Changes to PorterStemmer
)
```

---

## ❓ Troubleshooting

### "Dataset not found"
1. Download a spam CSV file
2. Save it as `spam_dataset.csv` in root directory
3. Ensure it has email text and spam/ham label columns

### "NLTK data missing"
Run this once:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### "No module named 'X'"
Install missing package:
```bash
pip install -r scripts/requirements.txt
```

### Model training is very slow
- Use smaller dataset for testing
- Reduce `max_features` in train.py (e.g., 2000 instead of 5000)
- Reduce CV folds: `cv=3` instead of `cv=5`

### Streamlit not connecting
- Make sure you're in the scripts directory
- Check port 8501 isn't blocked
- Try different port: `streamlit run streamlit_app.py --server.port 8502`

---

## 📈 Project Flow Visualization

```
1. RAW DATA (spam_dataset.csv)
        ↓
2. DATA EXPLORATION & CLEANING
   - Check columns and types
   - Remove missing values
   - Remove duplicates
        ↓
3. TEXT PREPROCESSING
   - Lowercase
   - Remove special chars
   - Remove stopwords
   - Lemmatize
        ↓
4. FEATURE EXTRACTION (TF-IDF)
   - Convert text to numbers
   - Create 5000 features
        ↓
5. TRAIN/TEST SPLIT
   - 80% training, 20% testing
        ↓
6. MODEL TRAINING
   - Baseline model
   - Hyperparameter tuning
   - Cross-validation
        ↓
7. EVALUATION
   - Accuracy, Precision, Recall
   - Confusion matrix
   - Visualizations
        ↓
8. SAVED MODEL & APP
   - spam_detector_model.pkl
   - streamlit_app.py
   - Web interface
```

---

## 🚀 Next Steps

After getting familiar with the basics:

1. **Try Different Datasets**: Experiment with different email datasets
2. **Tune Hyperparameters**: Modify C, solver, and other parameters
3. **Improve Preprocessing**: Experiment with different text cleaning techniques
4. **Deploy**: Share the web app with others
5. **Advanced**: Try other algorithms (SVM, Random Forest, Neural Networks)

---

## 📚 Key Files Explained

| File | Purpose |
|------|---------|
| `data_loader.py` | Loads and explores CSV files |
| `preprocessing.py` | Cleans and preprocesses text |
| `model_training.py` | Trains and evaluates the model |
| `visualization.py` | Creates charts and graphs |
| `train.py` | Main script that runs everything |
| `streamlit_app.py` | Interactive web application |
| `sample_test.py` | Test the trained model |

---

## 💡 Tips for Success

1. **Use a good dataset**: More data = better model
2. **Keep column names consistent**: Use 'text' and 'label' if possible
3. **Check visualizations**: They show if model is working well
4. **Test with different emails**: Verify predictions make sense
5. **Save your model**: Don't lose trained model - it took time to train!

---

## 📞 Need Help?

1. Check **README.md** for detailed documentation
2. Review **code comments** in the Python files
3. Run **sample_test.py** to verify everything works
4. Check the **Troubleshooting** section above

---

**You're ready! 🚀 Start with Step 1: Install Dependencies**
