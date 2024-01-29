from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from pickle import load

app = Flask(__name__)

#vector = load(open("vector_tfidf.sav", "rb"))
#model = load(open("svm_sentiment_analysis.sav", "rb"))

def choose_model():
    if request.method == 'POST':
        chosen_model = request.form.get('choose_model')
        if chosen_model == 'classification':
            return render_template('classification.html')
        elif chosen_model == 'sentiment_analysis':
            return render_template('sentiment_analysis.html')

    return render_template('index.html')

def sentiment_analysis(review):
    book_review = [review.strip().lower().replace('\t', ' ').replace('\n', ' ').replace('.', '')]
    book_review_vector = vector.transform(book_review).toarray()
    prediction = model.predict(book_review_vector)
    if prediction == 1:
        return 'Positive'
    else:
        return 'Negative'

@app.route('/', methods=['GET', 'POST'])
def index():
    return choose_model()

@app.route('/eda')
def eda(filename=None):
    render_template('eda.html')

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    book_category = ''
    if request.method == 'POST' and 'title' in request.form:
        book_title = request.form.get('title')
        print(book_title)
        #book_category = cat_classifier(book_title)

    return render_template('classification.html', book_title=book_title, book_category=book_category)

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

if __name__ == '__main__':
    app.run(debug=True)
