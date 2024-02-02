import os
os.environ['FLASK_ENV'] = 'production'

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from pickle import load
import pandas as pd
import re
from zipfile import ZipFile
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

with ZipFile('books_processed.zip', 'r') as zip_file:
    with zip_file.open('books_processed.csv') as csv_file:
        df_rec = pd.read_csv(csv_file)

vector_rec = load(open("vector_books.sav", "rb"))
model_rec = load(open("knn_neighbors_books.sav", "rb"))
vector_sentiment = load(open("tf_idf.sav", "rb"))
model_sentiment = load(open("model1k.pkl", "rb"))

def preprocess_text_sentiment_analysis(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\W|\d', ' ', str(text))  # Remove special characters and digits
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if
                         word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(lemmatized_tokens)

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

def rec(book, df_rec, model):
    recs=[]
    book = preprocess_text(book)
    if book not in df_rec['Title'].values:
        return ["Book not in database"]
    book_index = df_rec[df_rec['Title'] == book].index[0]
    distances, indices = model_rec.kneighbors(vector_rec[book_index], n_neighbors=12)
    similar_books = [(df_rec['Title'][i], distances[0][j]) for j, i in enumerate(indices[0])]
    for m in range(len(similar_books)-1):
        recs.append((similar_books[1:][m][0]).capitalize())
    recs = [b for b in recs if preprocess_text(b) != book]
    if not recs:
        return ["No recommendations"]
    else:
        return list(set(recs))[0:5]

def calculate_sentiment_book(title):
  n_neg = 0
  n_pos = 0

  books_subset = df_rec[df_rec.Title == title]['review/text']

  for rev, rev2 in books_subset.items():
    processed = preprocess_text_sentiment_analysis(rev2)
    processed = vector_sentiment.transform([processed])
    score = model_sentiment.predict(processed)

    if score >= 5:
      n_pos += 1
    else:
      n_neg += 1
  return [n_neg, n_pos]

@app.route('/', methods=['GET', 'POST'])
def index():
    return choose_model()

@app.route('/eda')
def eda():
    return render_template('eda.html')

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
def sentiment():
    title_sentiment = ''
    sentiments = []
    if request.method == 'POST' and 'title_sentiment' in request.form:
        title_sentiment = request.form.get('title_sentiment')
        sentiments = calculate_sentiment_book(title_sentiment)

    return render_template('sentiment_analysis.html', sentiments=sentiments)
'''
if __name__ == "__main__":
    app.run(debug=True)'''