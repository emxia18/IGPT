from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

input_file = "eric_messages.txt"
output_file = "eric_messages_clean.txt" 

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        cleaned_line = line.replace('"', '')
        outfile.write(cleaned_line)

with open("eric_messages_clean.txt", "r", encoding="utf-8") as f:
    eric_messages = f.read().split("\n\n")

with open("other_messages.txt", "r", encoding="utf-8") as f:
    someone_else_messages = f.read().split("\n\n")

messages = eric_messages + someone_else_messages
labels = ["Eric"] * len(eric_messages) + ["Someone Else"] * len(someone_else_messages)

X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "message_classifier_model_eric.joblib")
print("Model saved as 'message_classifier_model.joblib'")