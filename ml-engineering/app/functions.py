import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

def preprocess_text(text):
    '''Preprocess text by making it lowercase, removing text in square brackets,
    removing links, removing punctuation, and removing words containing numbers.'''
    return re.sub('\[.*?\]|\w*\d\w*|https?://\S+|www\.\S+|<.*?>+|[%s]' %
                re.escape(string.punctuation), '', str(text).lower())

def apply_stemming(sentence):
    stemmer = nltk.SnowballStemmer("english")
    return ' '.join(stemmer.stem(word) for word in sentence.split(' '))

def preprocess_and_clean(sentence):
    '''Preprocess and clean the text'''
    cleaned_text = preprocess_text(sentence)
    stop_words = stopwords.words('english')
    removed_stopwords_text = ' '.join(word for word in
        cleaned_text.split(' ') if word not in stop_words)
    stemmed_text = ' '.join(apply_stemming(word) for word in removed_stopwords_text.split(' '))
    return stemmed_text

# helper function
def returnClassifyResults(txt: str,
                            model, 
                            vectorizer,
                            k:int = 3) -> np.ndarray:
    """
        Function to return top k classification results based on given text
    """
    processed_txt = preprocess_and_clean(txt)
    x = pd.DataFrame([processed_txt], columns = ['preprocessed_text'])
    x_vectorized = vectorizer.transform(x['preprocessed_text'])
    y_pred_proba = model.predict_proba(x_vectorized)
    topk_indices = np.argsort(y_pred_proba[0])[::-1][:k]
    topk_categories = model.classes_[topk_indices]
    topk_probabilities = y_pred_proba[0][topk_indices]
    return topk_categories, topk_probabilities
