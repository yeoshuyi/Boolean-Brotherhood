# ----------------------------
# Imports
# ----------------------------
import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1. Load and clean data
# ----------------------------
try:
    df = pd.read_csv("my_predictions_on_wah.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("my_predictions_on_wah.csv", encoding="latin1")

df = df.dropna(subset=['text', 'predicted_classification'])

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# Encode labels as integers 0-3
label_map = {'spam': 0, 'advertisement': 1, 'rant': 2, 'relevant': 3}
df['predicted_classification'] = df['predicted_classification'].map(label_map)

# Convert labels to tensor
y_tensor = torch.tensor(df['predicted_classification'].values, dtype=torch.long)

# ----------------------------
# 2. Compute embeddings
# ----------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
X_text_embeddings = embed_model.encode(df['text'].tolist(), show_progress_bar=True)
X_combined = torch.tensor(X_text_embeddings, dtype=torch.float32)

# ----------------------------
# 3. Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_tensor, test_size=0.2, random_state=42
)

# ----------------------------
# 4. PyTorch Dataset
# ----------------------------
class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ReviewDataset(X_train, y_train)
test_dataset = ReviewDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ----------------------------
# 5. Define neural network
# ----------------------------
input_dim = X_combined.shape[1]

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

model = ReviewClassifier(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# 6. Train model
# ----------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# ----------------------------
# 7. Evaluate model
# ----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_y.tolist())

f1_per_class = f1_score(all_labels, all_preds, average=None)
accuracy = sum([p==l for p,l in zip(all_preds, all_labels)]) / len(all_labels)

print(f"Test Accuracy: {accuracy:.3f}")
print(f"F1 Scores per class [spam, advertisement, rant, relevant]: {f1_per_class}")
torch.save(model.state_dict(), "3Big_D_weights.pth")
print("✅ Model saved to 3Big_D.pth")
