import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

with open("IGPT/src/evaluation_methods/Bayes Model Emily/emily_messages.txt", "r", encoding="utf-8") as f:
    emily = f.readlines()

with open("IGPT/src/evaluation_methods/Bayes Model Emily/someone_else_messages.txt", "r", encoding="utf-8") as f:
    other = f.readlines()

print(len(emily), len(other))
data = pd.DataFrame({
    "Text": emily + other, 
    "Label": ["emily"] * len(emily) + ["other"] * len(other)
})

print(data.head())

def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text.strip()

data["Text"] = data["Text"].apply(preprocess)

X = data["Text"]
y = data["Label"]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_texts = [
    "i'm doing well",
    "nothing crazy, just gonna vibe and see what happens",
    "lol it’s been alright, kinda flew by tbh",
    "i would love to but i can only make it at 5:30 instead of 6:00",
    "not much, u wanna go hang out ?",
    "i'M THINKING ABOUT HOW MUCH I HATE SCHOOL",
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
new_texts_vectorized = vectorizer.transform(new_texts)
predictions = nb_model.predict(new_texts_vectorized)

print("Predictions:", predictions)