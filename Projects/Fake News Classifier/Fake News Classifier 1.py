# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:17:33 2021

    @author: Subham Roy
    https://www.codeingschool.com/
"""

import nltk
import pickle
import re
import pandas as pd
import numpy as np

# https://www.kaggle.com/c/fake-news/data
# 0--> Real News, 1 --> Fake news
df = pd.read_csv('./DataSet/data.csv')

#remove nan value
df = df.dropna()

messages=df.copy()

#reset index
messages.reset_index(inplace=True)

corpus = []

#apply text preprocessing on text column (Remove StopWord, then apply Lematization)
from nltk.stem import WordNetLemmatizer #Lemmalizer
from nltk.corpus import stopwords

wn = WordNetLemmatizer()

for i in range(len(messages.index)):
    review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ wn.lemmatize(word) for word in review if not word in set(stopwords.words('english')) ]
    review = ' '.join(review)
    corpus.append(review)