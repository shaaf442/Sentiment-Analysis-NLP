import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

file_path = '../data/processed/cleaned_reviews.csv'
df = pd.read_csv(file_path)

# CountVectorizer
cv = CountVectorizer(stop_words='english', min_df=2, token_pattern=r'[A-Za-z]+')
dtm = cv.fit_transform(df['Review Text'])
dtm_df = pd.DataFrame(dtm.toarray(), columns=cv.get_feature_names_out())
dtm_df.to_csv('../models/dtm.csv', index=False)
print(f'------------ Document-Term Matrix shape: {dtm_df.shape} --------------\nFirst 5 records:\n{dtm_df.head()}')

# TF-IDF Vectorizer
tv = TfidfVectorizer(stop_words='english', min_df=2, token_pattern=r'[A-Za-z]+')
tf_idf_dtm = tv.fit_transform(df['clean_text'])
tf_idf_dtm = pd.DataFrame(tf_idf_dtm.toarray(), columns=tv.get_feature_names_out())
tf_idf_dtm.to_csv('../models/tfidf_dtm.csv', index=False)
print(f'------------ TF-IDF Document-Term Matrix shape: {tf_idf_dtm.shape} --------------\nFirst 5 records:\n{tf_idf_dtm.head()}')

# VADER sentiment analysis
text = df.loc[0,'clean_text']
analyzer = SentimentIntensityAnalyzer()
print(f'------------ Polarity scores for the first cleaned review--------------\nText: {text} \nScores:{analyzer.polarity_scores(text)}')

# NMF Topic Modeling
# ==============================
n_topics = 5
nmf_model = NMF(n_components=n_topics, random_state=42)
W = nmf_model.fit_transform(tf_idf_dtm.values)
H = nmf_model.components_
feature_names = tv.get_feature_names_out()

print("\n----- TOPICS DISCOVERED -----")
for topic_idx, topic in enumerate(H):
    top_terms = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic {topic_idx + 1}: {', '.join(top_terms)}")

# Intent Classification
y_intent = df['intent']

X_train_intent, X_test_intent, y_train_intent, y_test_intent = train_test_split(tf_idf_dtm, y_intent, test_size=0.2, random_state=12)

intent_model = LogisticRegression(max_iter=1000, random_state=12)
intent_model.fit(X_train_intent, y_train_intent)
pred_intent = intent_model.predict(X_test_intent)

print("Intent Accuracy:", accuracy_score(y_test_intent, pred_intent))

# Logistic Regression 
y_sentiment = df['result']

X_train, X_test, y_train, y_test = train_test_split(tf_idf_dtm, y_sentiment, test_size=0.2, random_state=12)
sentiment_model = LogisticRegression(max_iter=1000, random_state=12)
sentiment_model.fit(X_train, y_train)
pred_sentiment = sentiment_model.predict(X_test)

print("\n----- ML SENTIMENT RESULTS -----")
print("Accuracy:", accuracy_score(y_test, pred_sentiment))


# Models downlaoding
os.makedirs("../models", exist_ok=True)

# Save models
joblib.dump(sentiment_model, "../models/sentiment_model.pkl")
joblib.dump(intent_model, "../models/intent_model.pkl")
joblib.dump(tv, "../models/tfidf_vectorizer.pkl")
joblib.dump(cv, "../models/bow_vectorizer.pkl")

print("Models saved successfully.")