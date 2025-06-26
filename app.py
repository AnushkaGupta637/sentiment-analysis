import streamlit as st
import nltk
import re
import joblib
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words('english'))


#load the model and vectorizer 
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

#clean the text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]"," ",text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

#create a UI
st.set_page_config(page_title="Sentiment Analysis of MOvies",layout="centered")
st.title("sentiment analysis app for movies")
st.markdown("enter the review of a movie")

user_input = st.text_area("enter the review")
if st.button("predict sentiment"):
    cleaned=clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "positive" if prediction ==1 else "negative"
    st.success(f"Prediction is: {sentiment}")