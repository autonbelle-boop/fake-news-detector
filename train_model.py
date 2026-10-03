import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Use headlines
X = data["title"].fillna("")
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

# Convert text to numbers
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectors, y_train)

# Test model
y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)

# Show results
print("----- Fake News Detector -----")
print("Dataset size:", len(data))
print("Fake articles:", len(fake))
print("Real articles:", len(true))
print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))
print("Model accuracy:", accuracy)
print("Accuracy percentage:", round(accuracy * 100, 2), "%")

# Save trained model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")