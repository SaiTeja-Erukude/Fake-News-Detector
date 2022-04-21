from flask import Flask, render_template, request
import pickle
from matplotlib.style import use
import pandas as pd
from model import word_drop
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/detect")
def detect():

    # getting the user input
    user_news = request.args.get("user_news")

    user_news = {"text": [user_news]}
    df = pd.DataFrame(user_news)

    # applying word drop functionality
    df["text"] = df["text"].apply(word_drop)
    user_news = df["text"]

    # vectorize the user input
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    user_news_v = vectorizer.transform(user_news)

    # loading the saved Logistic Regression model
    model = pickle.load(open("model.sav", "rb"))

    # predicting the user input
    prediction = model.predict(user_news_v)

    return str(prediction)
