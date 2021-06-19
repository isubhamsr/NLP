# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 21:29:55 2021

    @author: Subham Roy
    https://www.codeingschool.com/
"""
message = []
sentence = input("Enter a sentence for predict spam or ham: ")

message.append(sentence)

# remove pantuation and apply lemmatigation on input sentence
import re
import nltk
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer #Lemmalizer
from nltk.corpus import stopwords   
import re   

wn = WordNetLemmatizer()

corpus = []

#apply Lematizatrion and Rermove Stopewords and puntuations
for i in range(len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message[i])
    review = review.lower()
    review = review.split()
    
    review = [ wn.lemmatize(word) for word in review if not word in set(stopwords.words('english')) ]
    review = ' '.join(review)
    corpus.append(review)
    
#apply TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()   

'''
Demo sentences
ham - Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
spam - Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
'''

import pickle
loaded_model = pickle.load(open('model_pickle', 'rb'))
predict = loaded_model.predict(X)
print(predict)