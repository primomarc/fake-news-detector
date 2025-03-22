from flask import Flask, render_template, request
import joblib
import os

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        news_text = request.form["news_text"]
        if news_text:
            # Transform input text using TF-IDF
            text_vectorized = vectorizer.transform([news_text])
            result = model.predict(text_vectorized)[0]
            prediction = "Fake News" if result == 1 else "Real News"
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
