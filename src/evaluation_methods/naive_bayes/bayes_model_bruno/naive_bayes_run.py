import joblib

model = joblib.load("message_classifier_model.joblib")
print("Model loaded successfully.")

def classify_messages(messages):
    predictions = model.predict(messages)
    return predictions

messages_to_classify = [
    "Hey babe it was good I guess I got woken up a bit at night but nothing much though I had a pretty strange dream",
    "Yeah it happens a lot actually I try to move but I just can't like I'm in a pretty strange liquid or wtv like I'm moving slow as fuck",
    "Hell yeah its back day today but last time I got so destroyed I don't know if I can handle a full workout today",
    "We always go before its easier for me to eat after working out and the timing works better",
    "Yeah but we should go together after breakfast though",
    "see you baby",
    "Hey babe it was good I guess I got woken up a bit at night but nothing much though I had a pretty strange dream "+
    "Yeah it happens a lot actually I try to move but I just can't like I'm in a pretty strange liquid or wtv like I'm moving slow as fuck "+
    "Hell yeah its back day today but last time I got so destroyed I don't know if I can handle a full workout today "+
    "We always go before its easier for me to eat after working out and the timing works better "+
    "Yeah but we should go together after breakfast though "+
    "see you baby",
    "Heyy! It was pretty good, I had some weird dreams though lol. How’d you sleep? 😊",
    "Haha, it was like… I was back in class but couldn’t understand anything in the lecture, and then suddenly everyone turned into cats and started taking notes with their paws. 😂 I think you were there too, but you were the only human trying to help me figure out what was going on!",
    "Yeah, for sure! I’ll need a workout to shake off those weird dream vibes 😂 What time are you thinking?",
    "Hmm, maybe before? We’ll feel more energized after breakfast, and it’s always nice to start the day with a workout. But I’m good with whatever you prefer! 😊",
    "Sounds perfect! Let’s fuel up first then hit the gym at 8:30. 😊 What are you in the mood for breakfast?",
    "Haha, fair enough, we’ll see what’s on the menu today! Let’s get some good fuel for the workout. See you at 8:30!",
    "See you soon, bunny! ",
    "Heyy! It was pretty good, I had some weird dreams though lol. How’d you sleep? 😊 "+
    "Haha, it was like… I was back in class but couldn’t understand anything in the lecture, and then suddenly everyone turned into cats and started taking notes with their paws. 😂 I think you were there too, but you were the only human trying to help me figure out what was going on! "+
    "Yeah, for sure! I’ll need a workout to shake off those weird dream vibes 😂 What time are you thinking? "+
    "Hmm, maybe before? We’ll feel more energized after breakfast, and it’s always nice to start the day with a workout. But I’m good with whatever you prefer! 😊 "+
    "Sounds perfect! Let’s fuel up first then hit the gym at 8:30. 😊 What are you in the mood for breakfast? "+
    "Haha, fair enough, we’ll see what’s on the menu today! Let’s get some good fuel for the workout. See you at 8:30! "+
    "See you soon, bunny! ",
    "Are you having fun at your weekend? What are your"
]

predictions = classify_messages(messages_to_classify)
for message, prediction in zip(messages_to_classify, predictions):
    print(f"Message: {message}\nPredicted Sender: {prediction}\n")
