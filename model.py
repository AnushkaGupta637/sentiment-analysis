import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report,precision_score,recall_score,f1_score,confusion_matrix
#import library required for nlp
import nltk  #natural lang. toolkit
import re #regular expression
from nltk.corpus import stopwords
import joblib

nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

df = pd.read_csv("IMDB Dataset.csv")

#mapping the sentiment to some neumerical value
df["sentiment"] = df["sentiment"].map({"positive":1,"negative":0})

#clean the text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]"," ",text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

#apply the clean text function on reviews 
#loop through entire sent. of reviews and single - single row will be passed at one time
df["clean_review"] = df["review"].apply(clean_text)

#feature extraction
vectorizer = CountVectorizer(max_features = 5000)
X = vectorizer.fit_transform(df["clean_review"])

y = df["sentiment"]

#divide the dataset into tarin test split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#train the model 
model = MultinomialNB()
model.fit(x_train,y_train)

#prediction
y_pred = model.predict(x_test)

#calculate the performance matrix
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
classification_rep = classification_report(y_test,y_pred)

#save the model and vectorizer
joblib.dump(model,"sentiment_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("model and vectorizer has been saved")