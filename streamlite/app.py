import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import string
import re
from stop_words import get_stop_words
from joblib import dump, load


st.title("Streamlit Trash Talk")
data = pd.read_csv('./labeled_data.csv', dtype={'text':str})
data['tweet'] = data['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', ' ', tweet.lower()))

st.dataframe(data.head())
data['tweet'] = data['tweet'].replace('[\!,]', '', regex=True)
tweet = data['tweet']
tweet.head()

clf = load('filename.joblib')
default_value_goes_here = 'I love my job'
user_input = st.text_input("label goes here", default_value_goes_here)

clf.predict_proba([user_input])[0]


