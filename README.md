# Boolean-Brotherhood
TikTok Techjam 2025 Question 1 Submission

Google Places API key required in .env file

Run application using app.py

The repository contains three main components:  
- **Boolean Brothers V2** — Neural network–based review analyzer (current version).  
- **Boolean Brothers V1** — Machine learning–based review analyzer (earlier version).  
- **Trainer** — Training scripts for both V1 (ML) and V2 (Neural Network).  

Boolean Brothers V2
.env - Contains GMaps API Key
3Big_D_weights.pth - Neural Network Weights
app.py - Flask App Launcher
main.py - Main Functions
UIAI.py - Neural Network Implementation
templates - HTML Files

Boolean Brothers V1
.env - Contains GMaps API Key
Big_D.pkl - Machine Learning Files
pca_bigD.pkl - Machine Learning Files
app.py - Flask App Launcher
main.py - Main Functions
uiaitwo.py - Machine Learning Implementation
templates - HTML Files

Trainer
NNtik.py - Neural Network (V2 Trainer)
spam.py - Machine Learning (V1 Trainer)

API Used:
googlemaps

Libraries Used:
googlemaps
dotenv
sklearn
sentence_transformers
numpy
torch
flask
pandas

Install dependencies:
```bash
pip install -r requirements.txt
