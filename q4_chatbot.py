import nltk
import string
from nltk.corpus import stopwords

# -------------------------------
# NLP Preprocessing
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()

    clean_words = []
    for w in words:
        if w not in stopwords.words('english'):
            clean_words.append(w)

    return clean_words

# -------------------------------
# Intents with multiple keywords
# -------------------------------
intents = {
    "greeting": {
        "keywords": ["hello", "hi", "hey"],
        "response": "Hello! I can help you with NLP and AI topics."
    },
    "nlp": {
        "keywords": ["nlp", "language", "processing"],
        "response": "NLP (Natural Language Processing) enables computers to understand and process human language."
    },
    "ai": {
        "keywords": ["ai", "artificial", "intelligence"],
        "response": "AI allows machines to learn, reason, and make decisions similar to humans."
    },
    "applications": {
        "keywords": ["application", "use", "uses", "where", "applied"],
        "response": "AI is used in healthcare, finance, chatbots, recommendation systems, and self-driving cars."
    },
    "difference": {
        "keywords": ["difference", "between", "nlp", "ai"],
        "response": "AI is a broad field, while NLP is a subfield of AI focused on language understanding."
    },
    "chatbot": {
        "keywords": ["chatbot", "bot"],
        "response": "A chatbot is an AI application that interacts with users using natural language."
    }
}

# -------------------------------
# Improved Intent Matching
# -------------------------------
def get_response(user_input):
    words = preprocess(user_input)

    best_intent = None
    max_score = 0

    for intent in intents:
        keywords = intents[intent]["keywords"]
        score = 0

        for i in range(len(words)):
            if words[i] in keywords:
                score += 1

        if score > max_score:
            max_score = score
            best_intent = intent

    if best_intent is not None and max_score > 0:
        return intents[best_intent]["response"]
    else:
        return "Sorry, I didn't understand. Please ask about NLP or AI."

# -------------------------------
# Chat Loop
# -------------------------------
print("Chatbot Ready! (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = get_response(user_input)
    print("Bot:", response)