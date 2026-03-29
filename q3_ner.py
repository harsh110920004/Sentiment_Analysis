import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# News article (use this or replace with any news text)
text = """
Prime Minister Narendra Modi met Apple CEO Tim Cook in New Delhi.
The meeting focused on investment opportunities in India.
"""

# Step 1: Tokenization
tokens = word_tokenize(text)

# Step 2: POS tagging
tags = pos_tag(tokens)

# Step 3: Named Entity Recognition
tree = ne_chunk(tags)

# Step 4: Print entities
print("\nNamed Entities:\n")

for subtree in tree:
    if hasattr(subtree, 'label'):
        entity = " ".join([word for word, tag in subtree])
        print(entity, "->", subtree.label())

# Step 5: Visualization (REQUIRED)
tree.draw()