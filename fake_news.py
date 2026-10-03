import pandas as pd
from sklearn.model_selection import train_test_split

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)

print(df.head())
print("\nDataset size:", len(df))

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining articles:", len(X_train))
print("Testing articles:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTraining data shape:", X_train_vec.shape)
print("Testing data shape:", X_test_vec.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

print("\nModel training complete!")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

while True:
    print("\n--- Fake News Detector ---")

    headline = input("Headline (or type 'quit' to exit): ")

    if headline.lower() == "quit":
        break

    article = input("Article: ")

    combined_text = headline + " " + article

    article_vec = vectorizer.transform([combined_text])

    prediction = model.predict(article_vec)[0]
    probabilities = model.predict_proba(article_vec)[0]

    confidence = max(probabilities) * 100

    if prediction == 1:
        result = "REAL"
    else:
        result = "FAKE"

    print("\nPrediction:", result)
    print(f"Confidence: {confidence:.1f}%")