import re
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models
pca = joblib.load(r"pca_BigD.pkl")
clf = joblib.load(r"Big_D.pkl")             # your pretrained classifier
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # embeddings

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction function
def predict_review_consistency(review_text, place_description):
    # Clean review
    review_clean = clean_text(review_text)
    
    # Embed review and reduce with PCA
    review_emb = embed_model.encode([review_clean])
    review_emb_reduced = pca.transform(review_emb)
    
    # Predict class probabilities
    pred_prob = clf.predict_proba(review_emb_reduced)[0]
    pred_label = clf.classes_[np.argmax(pred_prob)]
    
    # Compute similarity with place description
    description_clean = clean_text(place_description)
    desc_emb = embed_model.encode([description_clean])
    similarity = cosine_similarity(review_emb, desc_emb)[0][0]
    
    # Normalize similarity using thresholds
    # Example: similarity <0.1 -> 0, >0.3 -> 1, else scale proportionally
    if similarity <= 0.1:
        norm_similarity = 0
    elif similarity >= 0.3:
        norm_similarity = 1
    else:
        norm_similarity = (similarity - 0.1) / (0.3 - 0.1)
    
    # Combine classifier probability of "relevant" and similarity
    # Assuming 'relevant' class exists
    relevant_idx = list(clf.classes_).index("relevant") if "relevant" in clf.classes_ else 0
    combined_score = 0.6 * pred_prob[relevant_idx] + 0.4 * norm_similarity
    
    # Final prediction based on combined score threshold
    final_label = "relevant" if combined_score >= 0.6 else "irrelevant"

    if review_text == "":
        return {"final_combined_label": "Relevant"}
    
    return {
        "predicted_label": pred_label,
        "review_probabilities": pred_prob.tolist(),
        "review_description_similarity": round(similarity, 3),
        "normalized_similarity": round(norm_similarity, 3),
        "final_combined_label": final_label,
        "combined_score": round(combined_score, 3)
    }
