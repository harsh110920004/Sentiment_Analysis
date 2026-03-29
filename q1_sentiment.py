import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# -------------------------------
# 1. DATASET
# -------------------------------
data = {
    "review": [
        "Amazing phone with great battery",
        "Worst product ever",
        "Good performance and camera",
        "Battery drains very fast",
        "Excellent quality and build",
        "Very bad experience",
        "Loved this product so much",
        "Waste of money totally",
        "Highly recommend this phone",
        "Not worth buying at all",
        "Fantastic sound quality",
        "Poor build and cheap material",
        "Great value for money",
        "Terrible customer service",
        "Super fast and smooth phone",
        "Stopped working after a week",
        "Very satisfied with purchase",
        "Horrible product experience",
        "Best purchase ever made",
        "Not good at all disappointed",
        "Awesome features and design",
        "Bad packaging and delivery",
        "Nice design and performance",
        "Very disappointing product",
        "Works perfectly fine",
        "Cheap quality material",
        "Excellent performance device",
        "Not as expected poor results",
        "Loved the battery life",
        "Worst experience ever"
    ],
    "sentiment": ["Positive","Negative"] * 15
}

df = pd.DataFrame(data)

# -------------------------------
# 2. DATASET VISUALIZATION
# -------------------------------
counts = df['sentiment'].value_counts()

# Bar Chart
plt.figure()
counts.plot(kind='bar')
plt.title("Dataset Distribution (Bar Chart)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# Pie Chart
plt.figure()
counts.plot(kind='pie', autopct='%1.1f%%')
plt.title("Dataset Distribution (Pie Chart)")
plt.ylabel("")
plt.show()

# -------------------------------
# 3. PREPROCESSING
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

df['clean'] = df['review'].apply(preprocess)

# -------------------------------
# 4. TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
X = vectorizer.fit_transform(df['clean'])
y = df['sentiment']

# -------------------------------
# 5. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# 6. NAÏVE BAYES (BEST MODEL)
# -------------------------------
grid = GridSearchCV(MultinomialNB(), {'alpha':[0.1,0.5,1.0]}, cv=3)
grid.fit(X, y)
nb_model = grid.best_estimator_

nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

# -------------------------------
# 7. CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

TN, FP = cm[0]
FN, TP = cm[1]

print("\nConfusion Matrix:\n", cm)

# -------------------------------
# 8. METRICS
# -------------------------------
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = 2*(precision*recall)/(precision+recall)

sensitivity = recall
specificity = TN/(TN+FP)

fpr = FP/(TN+FP)
fnr = FN/(TP+FN)

npv = TN/(TN+FN)
fdr = FP/(TP+FP)

mcc = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

print("\n--- Metrics ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("FPR:", fpr)
print("FNR:", fnr)
print("NPV:", npv)
print("FDR:", fdr)
print("MCC:", mcc)

# -------------------------------
# 9. FIGURE 9
# -------------------------------
plt.figure()
plt.bar(["Sensitivity","Specificity","Accuracy","F1"], 
        [sensitivity,specificity,accuracy,f1])
plt.title("Figure 9")
plt.show()

# -------------------------------
# 10. FIGURE 10
# -------------------------------
plt.figure()
plt.bar(["FDR","FNR","NPV","FPR"], 
        [fdr,fnr,npv,fpr])
plt.title("Figure 10")
plt.show()

# -------------------------------
# 11. FIGURE 11
# -------------------------------
plt.figure()
plt.bar(["Precision","MCC"], [precision,mcc])
plt.title("Figure 11")
plt.show()

# -------------------------------
# 12. HYPERPARAMETER GRAPH (NB)
# -------------------------------
alpha_vals = [0.1,0.5,1.0]
nb_acc = []

for a in alpha_vals:
    m = MultinomialNB(alpha=a)
    m.fit(X_train,y_train)
    nb_acc.append(accuracy_score(y_test,m.predict(X_test)))

plt.figure()
plt.plot(alpha_vals, nb_acc, marker='o')
plt.title("Naïve Bayes: Accuracy vs Alpha")
plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# -------------------------------
# 13. LOGISTIC REGRESSION GRAPH
# -------------------------------
C_vals = [0.1,1.0,10]
lr_acc = []

for c in C_vals:
    m = LogisticRegression(C=c, solver='liblinear')
    m.fit(X_train,y_train)
    lr_acc.append(accuracy_score(y_test,m.predict(X_test)))

plt.figure()
plt.plot(C_vals, lr_acc, marker='o')
plt.title("Logistic Regression: Accuracy vs C")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
