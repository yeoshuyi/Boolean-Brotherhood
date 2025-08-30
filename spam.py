import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import torch
import joblib

# ----------------------------
# Step 0: Load and clean data
# ----------------------------
# Try flexible encoding
try:
    df = pd.read_csv("my_predictions_on_wah.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("my_predictions_on_wah.csv", encoding="latin1")



# Lowercase & basic cleaning
df["text"] = df["text"].str.lower()
df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", str(x)))
df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())

# Stopwords + lemmatization
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

df["text"] = df["text"].apply(lambda x: " ".join(
    [lemmatizer.lemmatize(w) for w in x.split() if w not in stop_words]
))

# ----------------------------
# Step 1: Split data
# ----------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["predicted_classification"].tolist(),
    test_size=0.2,
    random_state=42,
)

# ----------------------------
# Step 2: Embeddings
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = embed_model.encode(train_texts, show_progress_bar=True)
test_embeddings = embed_model.encode(test_texts, show_progress_bar=True)

# ----------------------------
# Step 3: PCA for compression
# ----------------------------
pca = PCA(n_components=128)
X_train_reduced = pca.fit_transform(train_embeddings)
X_test_reduced = pca.transform(test_embeddings)

# ----------------------------
# Step 4: Train classifier
# ----------------------------
clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf.fit(X_train_reduced, train_labels)

# ----------------------------
# Step 5: Evaluation
# ----------------------------
preds = clf.predict(X_test_reduced)
print(classification_report(test_labels, preds, digits=3))

# ----------------------------
# Step 6: Save predictions
# ----------------------------
df["predicted_label"] = clf.predict(pca.transform(embed_model.encode(df["text"].tolist(), show_progress_bar=True)))

# Save to JSON
results = df[["text", "predicted_classification", "predicted_label"]].to_dict(orient="records")
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("✅ Predictions saved to predictions.json")
#joblib.dump(pca, r"C:\Users\leeje\Downloads\techjam\Big D\pca_2bigD.pkl")
#joblib.dump(clf, r"C:\Users\leeje\Downloads\techjam\Big D\2Big_D.pkl")
print("✅ Model saved to 2Big_D.pkl")
