# vectorizer_loader.py
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import numpy as np

def identity_tokenizer(x):
    return x

def load_tfidf(json_path):
    # Create fresh instance
    tfidf = TfidfVectorizer(
        tokenizer=identity_tokenizer,
        preprocessor=identity_tokenizer,
        token_pattern=None,
        lowercase=False
    )
    
    # Load JSON data
    with open(json_path) as f:
        model_data = json.load(f)
    
    # Manually set vocabulary and IDF
    tfidf.vocabulary_ = model_data['vocabulary']
    tfidf.idf_ = np.array(model_data['idf'])
    
    return tfidf