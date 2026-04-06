import pandas as pd
import string
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------- LOAD DATA ----------
df = pd.read_csv("reviews.csv", low_memory=False)

# Use 15,000 rows
df = df.sample(15000, random_state=42)

# Select required columns
df = df[["name", "reviews.text", "reviews.rating"]]
df.columns = ["product", "review", "rating"]

# Drop missing values
df.dropna(inplace=True)

# ---------- SENTIMENT ----------
def get_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["rating"].apply(get_sentiment)

# Remove neutral
df = df[df["sentiment"] != "neutral"]

# ---------- BALANCE DATA ----------
pos = df[df["sentiment"] == "positive"]
neg = df[df["sentiment"] == "negative"]

min_len = min(len(pos), len(neg))

df = pd.concat([
    pos.sample(min_len, random_state=42),
    neg.sample(min_len, random_state=42)
])

# ---------- PREPROCESS ----------
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(w for w in text.split() if w not in stop_words)

df["clean_review"] = df["review"].apply(preprocess)

# ---------- VECTORIZE ----------
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

# ---------- TRAIN MODEL ----------
model = MultinomialNB()
model.fit(X, y)

# ---------- PRODUCT ANALYSIS ----------
product_ratings = df.groupby("product")["rating"].mean()

best_product = product_ratings.idxmax()
worst_product = product_ratings.idxmin()

# ---------- CHATBOT ----------
def chatbot():
    print("\nProduct Review Chatbot (type 'exit' to quit)\n")

    while True:
        user = input("You: ").lower()

        if user == "exit":
            print("Bot: Goodbye")
            break

        # ---------- INTENT FIX ----------
        elif "best" in user:
            print("Bot: Best reviewed product is:", best_product)

        elif "worst" in user:
            print("Bot: Worst reviewed product is:", worst_product)

        # ---------- SMART PRODUCT SEARCH ----------
        elif "review" in user or "about" in user:
            found = False

            for product in df["product"].unique():
                if any(word in product.lower() for word in user.split()):
                    reviews = df[df["product"] == product]

                    print("\nBot: Showing reviews for", product, ":\n")
                    print(reviews[["review", "sentiment"]].head(5))

                    avg = reviews["rating"].mean()
                    print("Average Rating:", round(avg, 2))

                    found = True
                    break

            if not found:
                print("Bot: Product not found in dataset")

        # ---------- NEGATION FIX ----------
        elif "not bad" in user:
            print("Bot: This review sounds positive")

        elif "not good" in user:
            print("Bot: This review sounds negative")

        # ---------- RULE BASE ----------
        elif any(word in user for word in ["bad", "worst", "terrible", "poor"]):
            print("Bot: This review sounds negative")

        elif any(word in user for word in ["good", "excellent", "amazing", "great", "awesome"]):
            print("Bot: This review sounds positive")

        # ---------- ML ----------
        else:
            clean = preprocess(user)
            vec = vectorizer.transform([clean])
            pred = model.predict(vec)[0]

            print("Bot: This review sounds", pred)

# ---------- RUN ----------
chatbot()