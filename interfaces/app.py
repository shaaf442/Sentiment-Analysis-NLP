
import os
import re

import gradio as gr
import joblib
import pandas as pd
import spacy
from spacy.cli import download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import NMF


# ------------------------
# Load language model
# ------------------------

download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


# ------------------------
# Simple text cleaning
# ------------------------

def clean_text(text: str) -> str:
    """Lowercase, remove links and extra characters, then lemmatize.

    This is a simplified version of the cleaning used in preprocessing.py.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and len(token) > 2
    ]

    return " ".join(tokens)


# ------------------------
# Load trained models
# ------------------------

MODELS_DIR = os.path.join("..", "models")

sentiment_model_path = os.path.join(MODELS_DIR, "sentiment_model.pkl")
intent_model_path = os.path.join(MODELS_DIR, "intent_model.pkl")
tfidf_vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
tfidf_dtm_path = os.path.join(MODELS_DIR, "tfidf_dtm.csv")

sentiment_model = joblib.load(sentiment_model_path)
intent_model = joblib.load(intent_model_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)


# ------------------------
# Topic model (NMF)
# ------------------------

tfidf_matrix_df = pd.read_csv(tfidf_dtm_path)

n_topics = 5
nmf_model = NMF(n_components=n_topics, random_state=42)
nmf_model.fit(tfidf_matrix_df.values)

feature_names = tfidf_vectorizer.get_feature_names_out()


# ------------------------
# VADER analyzer
# ------------------------

analyzer = SentimentIntensityAnalyzer()


sentiment_label_map = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
}


def analyze_review(review_text: str):
    """Analyze a single review and return sentiment, intent and topic."""

    # Rule-based sentiment (VADER)
    vader_scores = analyzer.polarity_scores(review_text)
    vader_compound = vader_scores["compound"]

    if vader_compound >= 0.05:
        vader_sentiment = "Positive"
    elif vader_compound <= -0.05:
        vader_sentiment = "Negative"
    else:
        vader_sentiment = "Neutral"

    # ML-based sentiment (LogReg + TF-IDF)
    cleaned = clean_text(review_text)
    X_new = tfidf_vectorizer.transform([cleaned])
    ml_sentiment = sentiment_model.predict(X_new)[0]
    ml_sentiment_readable = sentiment_label_map.get(ml_sentiment, ml_sentiment)

    # Intent prediction
    intent_pred = intent_model.predict(X_new)[0]

    # Topic / keywords from NMF
    topic_scores = nmf_model.transform(X_new)
    topic_idx = int(topic_scores.argmax(axis=1)[0])

    topic_components = nmf_model.components_[topic_idx]
    top_indices = topic_components.argsort()[-5:][::-1]
    top_keywords = [feature_names[i] for i in top_indices]
    topic_description = f"Topic {topic_idx}: " + ", ".join(top_keywords)

    return vader_sentiment, ml_sentiment_readable, intent_pred, topic_description


iface = gr.Interface(
    fn=analyze_review,
    inputs=gr.Textbox(lines=4, label="Enter customer review"),
    outputs=[
        gr.Textbox(label="Rule-based Sentiment (VADER)"),
        gr.Textbox(label="ML Sentiment (Logistic Regression + TF-IDF)"),
        gr.Textbox(label="Predicted Intent"),
        gr.Textbox(label="Identified Topic / Keywords"),
    ],
    title="Customer Reviews Intelligence System",
    description=(
        "Analyze customer reviews to predict sentiment, intent, and "
        "identify main topics using NLP and Machine Learning."
    ),
)


if __name__ == "__main__":
    iface.launch(share=False)