import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_PATH = '../data/processed/cleaned_reviews.csv'
MODELS_DIR = '../models'

def load_models():
	"""Load the saved models and the TF-IDF vectorizer."""
	sentiment_model_path = f"{MODELS_DIR}/sentiment_model.pkl"
	intent_model_path = f"{MODELS_DIR}/intent_model.pkl"
	tfidf_vectorizer_path = f"{MODELS_DIR}/tfidf_vectorizer.pkl"

	sentiment_model = joblib.load(sentiment_model_path)
	intent_model = joblib.load(intent_model_path)
	tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

	return sentiment_model, intent_model, tfidf_vectorizer


def evaluate_sentiment(df: pd.DataFrame, sentiment_model, tfidf_vectorizer) -> None:
	"""Evaluate the sentiment model on all available data."""
	X = tfidf_vectorizer.transform(df["clean_text"])
	y_true = df["result"]
	y_pred = sentiment_model.predict(X)

	print("===== SENTIMENT MODEL EVALUATION =====")
	print("Accuracy:", accuracy_score(y_true, y_pred))
	print("Classification Report:\n", classification_report(y_true, y_pred))
	print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


def evaluate_intent(df: pd.DataFrame, intent_model, tfidf_vectorizer) -> None:
	"""Evaluate the intent model on all available data."""
	X = tfidf_vectorizer.transform(df["clean_text"])
	y_true = df["intent"]
	y_pred = intent_model.predict(X)

	print("\n===== INTENT MODEL EVALUATION =====")
	print("Accuracy:", accuracy_score(y_true, y_pred))
	print("Classification Report:\n", classification_report(y_true, y_pred))
	print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


def main() -> None:
	df = pd.read_csv(DATA_PATH)
	sentiment_model, intent_model, tfidf_vectorizer = load_models()

	evaluate_sentiment(df, sentiment_model, tfidf_vectorizer)
	evaluate_intent(df, intent_model, tfidf_vectorizer)


if __name__ == "__main__":
	main()

