import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine the datasets
data = pd.concat([fake, true], ignore_index=True)

# Use the article text
X = data["text"].fillna("")
y = data["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Turn words into numbers the computer can understand
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectors, y_train)

# Test the model
predictions = model.predict(X_test_vectors)

accuracy = accuracy_score(y_test, predictions)

print("Fake News Detector")
print("------------------")
print(f"Model accuracy: {accuracy:.2%}")
print()
print("The model is trained and ready!")