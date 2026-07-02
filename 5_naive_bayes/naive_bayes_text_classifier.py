# naive_bayes_text_classifier.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Example training data
# -----------------------------
texts = [
    "I love this movie",
    "This film was amazing",
    "What a great experience",
    "I really enjoyed this",
    "The acting was excellent",

    "I hate this movie",
    "This film was terrible",
    "What a bad experience",
    "I really disliked this",
    "The acting was awful"
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",

    "negative",
    "negative",
    "negative",
    "negative",
    "negative"
]


# -----------------------------
# 2. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 3. Build Naive Bayes pipeline
# -----------------------------
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])


# -----------------------------
# 4. Train model
# -----------------------------
model.fit(X_train, y_train)


# -----------------------------
# 5. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# 6. Predict user input
# -----------------------------
while True:
    user_text = input("\nEnter text to classify, or type 'quit' to stop: ")

    if user_text.lower() == "quit":
        break

    prediction = model.predict([user_text])[0]
    probability = model.predict_proba([user_text])[0]

    print("Prediction:", prediction)
    print("Class probabilities:")

    for label, prob in zip(model.classes_, probability):
        print(f"{label}: {prob:.4f}")