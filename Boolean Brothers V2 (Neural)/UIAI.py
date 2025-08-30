import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import pandas as pd

# ----------------------------
# 1. Define PyTorch model
# ----------------------------
class ReviewClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # 4 classes

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------------------
# 2. Load trained model
# ----------------------------
input_dim = 384  # embedding size of all-MiniLM-L6-v2
loaded_model = ReviewClassifier(input_dim)

# Load state dict (weights)
loaded_model.load_state_dict(torch.load(r"3Big_D_weights.pth"))
loaded_model.eval()

# ----------------------------
# 3. Load embedding model
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# 4. Text cleaning function
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# 5. Prediction + similarity function
# ----------------------------
def predict_review_consistency(review_text, place_description):
    # Clean review
    review_clean = clean_text(review_text)
    
    # Embed review
    review_emb = embed_model.encode([review_clean], convert_to_tensor=True)
    
    # Predict class probabilities
    with torch.no_grad():
        logits = loaded_model(review_emb.float())
        pred_prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(pred_prob))
    
    # Map index to label
    label_map = {0: "spam", 1: "advertisement", 2: "rant", 3: "relevant"}
    pred_label = label_map[pred_idx]

    # Embed place description
    description_clean = clean_text(place_description)
    desc_emb = embed_model.encode([description_clean])
    
    # Cosine similarity
    similarity = cosine_similarity([review_emb.cpu().numpy()[0]], [desc_emb[0]])[0][0]
    
    # Normalize similarity
    if similarity <= 0.1:
        norm_similarity = 0
    elif similarity >= 0.3:
        norm_similarity = 1
    else:
        norm_similarity = (similarity - 0.1) / (0.3 - 0.1)
    
    # Combine classifier probability for "relevant" + similarity
    relevant_idx = 3
    combined_score = 0.6 * pred_prob[relevant_idx] + 0.4 * norm_similarity
    final_label = "relevant" if combined_score >= 0.5 else "irrelevant"
    
    if review_text == "":
        return {"final_combined_label": "Relevant"}

    return {
        "predicted_label": pred_label,
        "review_probabilities": pred_prob.tolist(),
        "review_description_similarity": round(float(similarity), 3),
        "normalized_similarity": round(float(norm_similarity), 3),
        "final_combined_label": final_label,
        "combined_score": round(float(combined_score), 3)
    }