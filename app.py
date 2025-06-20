import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

st.title("Fake News Detector")
st.write("Enter the details of a news article to predict if it's real or fake.")

title = st.text_input("Article Title")
author = st.text_input("Author")
text = st.text_area("Article Text")

if st.button("Predict"):
    if title.strip() == "" and author.strip() == "" and text.strip() == "":
        st.error("Please provide at least one field (title, author, or text).")
    else:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('LogisticRegression.pkl')
        
        combined_text = f"{title} {author} {text}"
        processed_text = preprocess_text(combined_text)
        text_vectorized = vectorizer.transform([processed_text])
        
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        st.subheader("Prediction")
        st.write(f"The article is predicted to be: **{prediction.upper()}**")
        st.write(f"Confidence: Real: {probabilities[0]:.2%}, Fake: {probabilities[1]:.2%}")
        
        st.subheader("Explanation")
        feature_names = vectorizer.get_feature_names_out()
        coef = model.coef_[0]
        top_n = 5
        top_indices = np.argsort(np.abs(coef))[-top_n:]
        top_words = [feature_names[i] for i in top_indices]
        top_coefs = [coef[i] for i in top_indices]
        
        st.write("The following words most influenced the prediction:")
        for word, coef in zip(top_words, top_coefs):
            impact = "indicates fake" if coef > 0 else "indicates real"
            st.write(f"- **{word}** ({impact}, weight: {coef:.4f})")