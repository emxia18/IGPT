import joblib

model = joblib.load("message_classifier_model_eric.joblib")
print("Model loaded successfully.")

def classify_messages_with_probabilities(messages):
    probabilities = model.predict_proba(messages)
    class_labels = model.classes_
    results = []
    for message, prob in zip(messages, probabilities):
        result = {
            "message": message,
            class_labels[0]: prob[0],
            class_labels[1]: prob[1]
        }
        results.append(result)
    return results

messages_to_classify = [
    "chillin, u?",
    "probably work on some stuff, then vibe",
    "tiring lol i got like four hours of sleep every day for the week.",
    "sure haha when and where?",
    "nah. just existing probably.",
    "how i deserve a nap.",
    "ya sure where",
    "too much tbh, need a break",
    "I’m feeling alright, but not really that much energy. hbu?",
    "yea if u consider studying for finals fun lol",
    "i sleep til noon bro i dont rly have a “morning”",
    "it's been alright, could be worse",
    "ramen, always",
    "grinding and being tired",
    "ehhh not really. i just went to class and ate food",
    "depends what it is lol",
    "just lie down and watch dumb vids",
    "yeah sure, what's up?",
    "honestly, idk rn",
    "with as little effort as possible",
    "sleep lol",
    "everything lol",
    "play League, play random sports, talk with friends.",
    "yeah sure, where tho?"
]

results = classify_messages_with_probabilities(messages_to_classify)
for result in results:
    print(f"Message: {result['message']}\nProbabilities - Eric: {result['Eric']:.2f}, Someone Else: {result['Someone Else']:.2f}\n")
