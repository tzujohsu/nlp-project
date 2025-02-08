# fastapi dev main.py
from fastapi import FastAPI
from app.functions import returnClassifyResults
import joblib


# define model info
model_name = 'trained_model.joblib'
model_path = "app/data/" + model_name

# load model
model = joblib.load(model_path)

# load vectorizer
vectorizer_path = "app/data/" + 'vectorizer.joblib'
vectorizer = joblib.load(vectorizer_path)

# create FastAPI object
app = FastAPI()

# API operations
@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/info")
def info():
    return {'name': 'news-cateorization', 'description': "Categorization API for Tzu-Jo Hsu's NLP project demo."}

@app.get("/categorize")
def categorize(txt: str):
    topk_categories, topk_probabilities = returnClassifyResults(txt, model, vectorizer, k=3)
    response = {category: probability for category, probability in zip(topk_categories, topk_probabilities)}
    return response