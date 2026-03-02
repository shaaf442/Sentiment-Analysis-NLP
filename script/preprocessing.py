import numpy as np 
import pandas as pd
import spacy
import re

from spacy.cli import download
download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

def infer_intent(text):
    text_l = str(text).lower()
    if any(kw in text_l for kw in ['refund', 'return', 'money back']):
        return 'Refund Request'
    if any(kw in text_l for kw in ['late', 'delay', 'delivery', 'shipping']):
        return 'Delivery Issue'
    if any(kw in text_l for kw in ['complain', 'bad', 'poor', 'terrible', 'worst', 'awful', 'disappointed']):
        return 'Complaint'
    return 'General Query'

def reviews(x):
    x = int(x)
    if x > 3:
        return 'positive'
    elif x == 3:
        return 'neutral'
    else:
        return 'negative'

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    doc = nlp(text)
    
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha              
        and not token.is_stop          
        and len(token) > 2
    ]
    
    return " ".join(tokens)

    return cleaned

file_path = '../data/raw/reviews.csv'
output_path = '../data/processed/cleaned_reviews.csv'

df = pd.read_csv(file_path)

df = df[['Review Text', 'Rating']].copy()
df = df[df['Review Text'].notna() & df['Rating'].notna()]

df['clean_text'] = df['Review Text'].apply(clean_text)
df['intent'] = df['clean_text'].apply(infer_intent)
df['result'] = df['Rating'].apply(reviews)

df.to_csv(output_path, index=False)
print("Preprocessing completed. Cleaned data saved to:", output_path)