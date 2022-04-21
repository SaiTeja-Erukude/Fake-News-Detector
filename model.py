"""
Python File to create, train and save a Vectorizer and Logistic Regression Model.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import string
import pickle


# Function to convert the text in lowercase, remove the extra space, special chr, ulr and links.
def word_drop(text):
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text


if __name__ == "__main__":

    # importing fake and true news
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    # Inserting a column called "class" for fake and real news dataset to categories fake and true news.
    df_fake["class"] = 0
    df_true["class"] = 1

    # merging both true and fake news and dropping unwanted columns
    df_merge = pd.concat([df_fake, df_true], axis=0)
    df = df_merge.drop(["title", "subject", "date"], axis=1)

    df.reset_index(inplace=True)
    df.drop(["index"], axis=1, inplace=True)

    # applying word drop function
    df["text"] = df["text"].apply(word_drop)

    # Defining dependent and independent variable as x and y
    x = df["text"]
    y = df["class"]

    # Convert text to vectors
    vectorizer = TfidfVectorizer()
    xv = vectorizer.fit_transform(x)

    # Create and fit model
    LR = LogisticRegression()
    LR.fit(xv, y)

    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
    pickle.dump(LR, open("model.sav", "wb"))
    print("Logistic Regression Model, Vectorizer Trained & Saved!")
