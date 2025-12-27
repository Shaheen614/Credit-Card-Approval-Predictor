# Credit Card Approval Predictor 

This is a  machine learning–powered credit card approval system, wrapped in a FastAPI service.  
It uses a Kaggle dataset features to predict whether an applicant would be approved.  


## ✨ Features
1. **Synthetic dataset generator** (also included a kaggle dataset )
2. **Multiple ML model integretion**: Logistic Regression, Decision Tree, Random Forest.
3. **Preprocessing pipeline**: scaling and encoding handled automatically.
4. **FastAPI service** with `/predict` endpoint.
5. **Interactive Swagger docs** at `/docs`.
6. **Dockerized deployment** for portability.


## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Train models:
   python scripts/train_models.py

3. Run API:
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000




