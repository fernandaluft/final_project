# -*- coding: utf-8 -*-
"""sentiment_ml_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14ugGw-KUmrcdghyoq7D45jBQLHa1ZJBl
"""

# Commented out IPython magic to ensure Python compatibility.
import os
from pathlib import Path
# Check if the code is running on Google Colab
try:
    import google.colab
    IN_COLAB = True
    base_path = "/content/"
    if Path(f"{base_path}final_project").is_dir():
#       %cd {base_path}final_project
      !git pull
#       %cd {base_path}
    else:
      !git clone https://github.com/fernandaluft/final_project.git
except ImportError:
    IN_COLAB = False
    base_path = "/workspaces/"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from final_project.src.scraping import Scraping
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

scraping = Scraping(IN_COLAB, sentiment_ds = True)
scraping.kaggle_scrape()

class SentimentMLTrain():
  def __init__(self, dataset_limit):
    self.dataset_limit = dataset_limit
    self.stop_words = set(stopwords.words('english'))
    self.lemmatizer = WordNetLemmatizer()
    self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
    self.best_model = None

  def read_sentiment_dataset(self):
    !unzip -o -n /content/imdb-dataset-of-65k-movie-reviews-and-translation.zip -d {base_path}final_project/data
    os.system(f'rm -rf /content/imdb-dataset-of-65k-movie-reviews-and-translation.zip')
    self.sentiment_df = pd.read_csv(f"{base_path}final_project/data/IMDB-Dataset.csv").sample(5000)

  def preprocess_text(self, text):
    text = re.sub(r'\W|\d', ' ', str(text))  # Remove special characters and digits
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]  # Lemmatize and remove stopwords
    return ' '.join(lemmatized_tokens)

  def preprocess_data(self):
    self.sentiment_df['clean_text'] = self.sentiment_df['Reviews'].apply(self.preprocess_text)

  def split_data(self):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.tfidf_vectorizer.fit_transform(self.sentiment_df['clean_text']),
        self.sentiment_df['Ratings'],
        test_size=0.2,
        random_state=42)
    with open('/content/final_project/models/tf_idf.pickle', 'wb') as f:
      pickle.dump(self.tfidf_vectorizer, f)

  def train_model(self):
    svm_classifier = SVC(kernel='linear')
    param_grid = {'C': [0.1, 1, 10, 100]}  # Hyperparameter grid for tuning
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
    grid_search.fit(self.X_train, self.y_train)
    self.best_model = grid_search.best_estimator_

  def evaluate_model(self):
    y_pred = self.best_model.predict(self.X_test)
    accuracy = accuracy_score(self.y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(self.y_test, y_pred))

  def save_model(self, model_path):
    with open(model_path, 'wb') as f:
      pickle.dump(self.best_model, f)

sentiment_ml_train = SentimentMLTrain(None)
sentiment_ml_train.read_sentiment_dataset()

sentiment_ml_train.sentiment_df.info()

sentiment_ml_train.preprocess_data()
sentiment_ml_train.split_data()
sentiment_ml_train.train_model()
sentiment_ml_train.evaluate_model()

sentiment_ml_train.save_model("/content/final_project/models/sentiment_model.pkl")

!unzip -o -n /content/final_project/preprocessed_data/xaa_books_reviews.zip -d {base_path}final_project/data

import pickle

def calculate_sentiment_book(title):
  sentiment_ml = SentimentMLTrain(None)
  n_neg = 0
  n_pos = 0
  with open("/content/final_project/models/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

  with open("/content/final_project/models/tf_idf.pickle", "rb") as f:
    vec = pickle.load(f)
  books = pd.read_csv("/content/final_project/data/content/final_project/data/books_reviews.csv")


  books_subset = books[books.Title == title]['review/text']

  for rev, rev2 in books_subset.items():

    processed = sentiment_ml.preprocess_text(rev2)
    processed = vec.transform([processed])

    score = sentiment_model.predict(processed)
    if score >= 5:
      n_pos += 1
    else:
      n_neg += 1

  return [n_neg, n_pos]

print(calculate_sentiment_book('Run Baby Run'))