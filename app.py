
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer #steamer le root word patta lagauxa(playing = play)
from sklearn.pipeline import Pipeline #data lai algo ma rakhda kun kun step bata pass hunxa vanera vanne
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

st.title("Text classification")
df = pd.read_csv('cleaned_bbc_data.csv')
df
# --------------------------------model creation----------------------------------------
# Training model
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer()
X = df['text']
Y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15) #Splitting dataset


# #Creating Pipeline
#pipeline vaneko architecture nai ho
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=2000)),
                     ('clf', LogisticRegression())])


# #Training model
model = pipeline.fit(X_train, y_train)
# ---------------------------------model creation end-----------------------------------------------
# file = open('news.txt','r')
# news = file.read()
# file.close()

news = st.text_area("text to translate")
if st.button("Submit"):

# news = input("Enter news = ")
    news_data = {'predict_news':[news]}
    news_data_df = pd.DataFrame(news_data)
# news_data_df
    predict_news_cat = model.predict(news_data_df['predict_news']) #dataframe name ani kun key ni lekhne
    st.write("Predicted news category = ",predict_news_cat[0])
else:
    st.write("please enter news")