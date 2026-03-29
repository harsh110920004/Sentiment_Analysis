import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm

# ✅ Load SMALL model (VERY IMPORTANT)
model = api.load("glove-wiki-gigaword-50")

def cosine_similarity(w1, w2):
    try:
        return dot(model[w1], model[w2]) / (norm(model[w1]) * norm(model[w2]))
    except KeyError:
        return "Word not found"

pairs = [
    ("king", "queen"),
    ("doctor", "nurse"),
    ("car", "tree")
]

for i in range(len(pairs)):
    w1 = pairs[i][0]
    w2 = pairs[i][1]
    
    sim = cosine_similarity(w1, w2)
    print(w1, "-", w2, ":", sim)
    print("\nInterpretation:")

for i in range(len(pairs)):
    w1 = pairs[i][0]
    w2 = pairs[i][1]
    sim = cosine_similarity(w1, w2)
    
    if isinstance(sim, str):
        print(w1, "-", w2, ":", sim)
    elif sim > 0.7:
        print(w1, "-", w2, "→ Highly Similar")
    elif sim > 0.4:
        print(w1, "-", w2, "→ Moderately Similar")
    else:
        print(w1, "-", w2, "→ Not Similar")