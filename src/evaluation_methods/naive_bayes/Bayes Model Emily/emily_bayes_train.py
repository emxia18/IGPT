from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load messages from files
with open("emily_messages.txt", "r", encoding="utf-8") as f:
    bruno_messages = f.read().split("\n\n")

with open("someone_else_messages.txt", "r", encoding="utf-8") as f:
    someone_else_messages = f.read().split("\n\n")

# Prepare data for training
messages = bruno_messages + someone_else_messages
labels = ["Emily"] * len(bruno_messages) + ["Someone Else"] * len(someone_else_messages)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# Create a Naive Bayes model pipeline with text vectorization
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model to a file
joblib.dump(model, "message_classifier_model_emily.joblib")
print("Model saved as 'message_classifier_model.joblib'")