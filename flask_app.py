from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Load trained model
with open("/home/tanujakasani/mysite/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF Vectorizer
with open("/home/tanujakasani/mysite/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        review = request.form["review"]

        review = clean_text(review)

        vector = tfidf.transform([review])

        prediction = model.predict(vector)[0]

        confidence = round(model.predict_proba(vector).max() * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)