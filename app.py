from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    confidence_label = None

    if request.method == "POST":
        article = request.form["article"].strip()

        if article:
            article_vector = vectorizer.transform([article])

            result = model.predict(article_vector)[0]
            probabilities = model.predict_proba(article_vector)[0]
            confidence = max(probabilities)

            if confidence >= 0.75:
                confidence_label = "High confidence"
            elif confidence >= 0.60:
                confidence_label = "Moderate confidence"
            else:
                confidence_label = "Low confidence"

            if result == 0:
                prediction = "FAKE NEWS"
            else:
                prediction = "REAL NEWS"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        confidence_label=confidence_label
    )


if __name__ == "__main__":
    app.run(debug=False)