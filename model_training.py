import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class SpamDetectionModel:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.model = None

    def split_data(self, df, text_column='cleaned_message', label_column='label'):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df[text_column], df[label_column], test_size=0.2, random_state=42
        )

    def build_vectorizer(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )

    def extract_features(self):
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        return X_train_tfidf, X_test_tfidf

    def train_model(self, X_train_tfidf, C=1.0):
        self.model = LogisticRegression(C=C, random_state=42, solver='lbfgs', max_iter=1000)
        self.model.fit(X_train_tfidf, self.y_train)

    def optimize_hyperparameters(self, X_train_tfidf, cv=5):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [500, 1000, 2000]
        }
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1
        )
        grid_search.fit(X_train_tfidf, self.y_train)
        self.model = grid_search.best_estimator_