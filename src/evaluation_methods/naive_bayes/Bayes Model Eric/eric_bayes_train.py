from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

input_file = "eric_messages.txt"  # Replace with your file name
output_file = "eric_messages_clean.txt"  # Replace with your desired output file name

# Open the input file for reading and the output file for writing
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Remove all double quotes from the line
        cleaned_line = line.replace('"', '')
        # Write the cleaned line to the output file
        outfile.write(cleaned_line)

# Load messages from files
with open("eric_messages_clean.txt", "r", encoding="utf-8") as f:
    bruno_messages = f.read().split("\n\n")

with open("other_messages.txt", "r", encoding="utf-8") as f:
    someone_else_messages = f.read().split("\n\n")

# Prepare data for training
messages = bruno_messages + someone_else_messages
labels = ["Eric"] * len(bruno_messages) + ["Someone Else"] * len(someone_else_messages)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# Create a Naive Bayes model pipeline with text vectorization
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model to a file
joblib.dump(model, "message_classifier_model_eric.joblib")
print("Model saved as 'message_classifier_model.joblib'")