import nltk
import string
from nltk.corpus import stopwords

//Preprocessing updated
def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    
    clean_words = []
    for w in words:
        if w not in stopwords.words('english'):
            clean_words.append(w)
    
    return " ".join(clean_words)