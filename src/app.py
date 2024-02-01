import os
os.environ['FLASK_ENV'] = 'production'

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from pickle import load
import pandas as pd
import re
from zipfile import ZipFile
#from sentiment_ml_train import SentimentMLTrain

app = Flask(__name__)

with ZipFile('books_processed.zip', 'r') as zip_file:
    with zip_file.open('books_processed.csv') as csv_file:
        df_rec = pd.read_csv(csv_file)

vector_rec = load(open("vector_books.sav", "rb"))
model_rec = load(open("knn_neighbors_books.sav", "rb"))

def preprocess_text(text):
    # remove special chars and digits
    text = re.sub(r'\W|\d', ' ', text).lower()
    text = re.sub(r' +', ' ', text)
    return text.strip()

def choose_model():
    if request.method == 'POST':
        chosen_model = request.form.get('choose_model')
        if chosen_model == 'recommendation':
            return render_template('book_recommendation.html')
        elif chosen_model == 'sentiment_analysis':
            return render_template('sentiment_analysis.html')

    return render_template('index.html')

def rec(book, df_rec, model_rec):
    recs=[]
    book = preprocess_text(book)
    book_index = df_rec[df_rec['Title'] == book].index[0]
    distances, indices = model_rec.kneighbors(vector_rec[book_index], n_neighbors=12)
    similar_books = [(df_rec['Title'][i], distances[0][j]) for j, i in enumerate(indices[0])]
    for m in range(len(similar_books)-1):
        recs.append((similar_books[1:][m][0]).capitalize())
    recs = [b for b in recs if preprocess_text(b) != book]
    return list(set(recs))[0:5]

'''
def calculate_sentiment_book(title):
  sentiment_ml = SentimentMLTrain(None)
  n_neg = 0
  n_pos = 0
  with open("/content/final_project/models/model1k.pkl", "rb") as f:
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
'''
@app.route('/', methods=['GET', 'POST'])
def index():
    return choose_model()

@app.route('/eda')
def eda(filename=None):
    render_template('eda.html')

@app.route('/book_recommendation', methods=['GET', 'POST'])
def recommendation_book():
    book_list = []
    book_title = ''
    if request.method == 'POST' and 'book' in request.form:
        book_title = request.form.get('book')
        book_list = rec(book_title, df_rec, model_rec)
        print(book_title)

    return render_template('book_recommendation.html', book_list=book_list, book_title=book_title)

@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def prediction():
    review = ''
    result = ''
    if request.method == 'POST' and 'review' in request.form:
        review = request.form.get('review')
        if review is None:
            result = 'Enter a review'
        else:
            result = sentiment_analysis(review)

    return render_template('sentiment_analysis.html', result=result)

