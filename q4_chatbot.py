import pandas as pd
import string
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("reviews.csv")

def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

df['clean'] = df['review'].apply(preprocess)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean'])
y = df['sentiment']

model = MultinomialNB()
model.fit(X, y)

ranking = df.groupby('product')['sentiment'].value_counts().unstack().fillna(0)
ranking['score'] = ranking.get('Positive',0) - ranking.get('Negative',0)

best_product = ranking['score'].idxmax()
worst_product = ranking['score'].idxmin()

def predict(text):
    text_clean = preprocess(text)
    vec = vectorizer.transform([text_clean])
    return model.predict(vec)[0]

print("Chatbot: Ask anything about products, reviews, or type 'exit'")

while True:
    user = input("You: ")
    user_lower = user.lower()

    if "exit" in user_lower:
        print("Chatbot: Goodbye!")
        break

    elif any(word in user_lower for word in ["best", "top", "recommend"]):
        print("Chatbot: Best product is", best_product)

    elif any(word in user_lower for word in ["worst", "bad product", "not good product"]):
        print("Chatbot: Worst product is", worst_product)

    elif any(word in user_lower for word in ["which product", "suggest", "buy"]):
        print("Chatbot: I recommend", best_product)

    else:
        sentiment = predict(user)
        print("Chatbot: This sounds", sentiment)