import joblib

model = joblib.load("message_classifier_model_emily.joblib")
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

# Example usage
messages_to_classify = [
    "i'm doing well",
    "nothing crazy, just gonna vibe and see what happens",
    "lol it’s been alright, kinda flew by tbh",
    "i would love to but i can only make it at 5:30 instead of 6:00",
    "not much, u wanna go hang out ?",
    " i'M THINKING ABOUT HOW MUCH I HATE SCHOOL",
    "i would love to go out with you later, but i have a lot i need to take care of first, i will let you know when i am free",
    "ive j been contemplating my life decisions and how to be a good person",
    "i'M FEELING A LITTLE OFF TODAY",
    "not really, might just chill tbh",
    "scroll on my phone for way too long lol",
    "i had a great day, i went to the park with my friends and we played on the swings and i got a yummy ice cream",
    "yes yes im so down, what is a good time for you?",
    "i just wrap up the work i gotta do hehe",
    "uh i think there was a new yt video i wanna watch but i cant cuz i have to do my homework",
    "yesss theres this new coffee shop i wanna go try, do u wanna go tgt",
    "i like to just chill and do nothing",
    "yea for sure, what’s up?",
    "maybe the weekend? idk lol",
    "coffee first, then we’ll see",
    "yesss ima be visiting la soon hehe",
    "i just need to talk to someone, ive been feeling really down lately and i need someone to just listen to me and be there for me",
    "hang w friends or like go out for snacks lol",
    "maybe later, it sounds nice tho"
]

results = classify_messages_with_probabilities(messages_to_classify)
for result in results:
    print(f"Message: {result['message']}\nProbabilities - Emily: {result['Emily']:.2f}, Someone Else: {result['Someone Else']:.2f}\n")
