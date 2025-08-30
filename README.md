# Boolean-Brotherhood

This project is the **Boolean Brotherhood’s** solution to TikTok TechJam 2025, Question 1.  
It leverages machine learning and neural network models, combined with Google Maps API to analyze and classify reviews.  

**Setup**
Boolean Brothers V1 and V2
1) Enter GMaps API keys into .env file
2) Run app.py
3) Access application through http://127.0.0.1:5000/

Trainer
1) Unzip my_predictions_on_wah.csv into the same file as NNtik.py or spam.py
2) Run NNtik.py for Neural Network or spam.py for Machine Learning Training

**Files**
The repository contains three main components:  
- **Boolean Brothers V2** — Neural network–based review analyzer (current version).  
- **Boolean Brothers V1** — Machine learning–based review analyzer (earlier version).  
- **Trainer** — Training scripts for both V1 (ML) and V2 (Neural Network).  

Boolean Brothers V2
- .env - Contains GMaps API Key
- 3Big_D_weights.pth - Neural Network Weights
- app.py - Flask App Launcher
- main.py - Main Functions
- UIAI.py - Neural Network Implementation
- templates - HTML Files

Boolean Brothers V1
- .env - Contains GMaps API Key
- Big_D.pkl - Machine Learning Files
- pca_bigD.pkl - Machine Learning Files
- app.py - Flask App Launcher
- main.py - Main Functions
- uiaitwo.py - Machine Learning Implementation
- templates - HTML Files

Trainer
- NNtik.py - Neural Network (V2 Trainer)
- spam.py - Machine Learning (V1 Trainer)
- my_predictions_on_wah.csv zipped (Training Data)

**Dependencies**
API Used:
- googlemaps

Libraries Used:
- googlemaps
- dotenv
- sklearn
- sentence_transformers
- numpy
- torch
- flask
- pandas
- joblib

Install dependencies:
```bash
pip install -r requirements.txt

**License**
Distributed under the Unlicense License. See LICENSE.txt for more information.
