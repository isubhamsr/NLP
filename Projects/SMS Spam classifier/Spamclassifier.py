# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 13:17:31 2021

    @author: Subham Roy
    https://www.codeingschool.com/
"""

import pandas as pd

#dataset
#https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
massages = pd.read_csv('./DataSet/SMSSpamCollection.csv', sep='\t', names=["label", "message"])

import re
import nltk
import pickle

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer #Lemmalizer
from nltk.stem import PorterStemmer #Stemmer
from nltk.corpus import stopwords   
import re   

ps = PorterStemmer()
wn = WordNetLemmatizer()

corpus = []

#apply Lematizatrion and Rermove Stopewords and puntuations
for i in range(len(massages)):
    review = re.sub('[^a-zA-Z]', ' ', massages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ wn.lemmatize(word) for word in review if not word in set(stopwords.words('english')) ]
    review = ' '.join(review)
    corpus.append(review)
    
#create dependent variable y
y = pd.get_dummies(massages['label'])  
y = y.iloc[:,1].values  
    
#apply TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(max_features=6000, ngram_range=(2,2))
X = tf.fit_transform(corpus).toarray()  

filename = 'transform.pkl'
pickle.dump(tf, open(filename, 'wb'))  

#train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#creating model for naive_bayes
from sklearn.naive_bayes import MultinomialNB
spam_detect_model_NB = MultinomialNB().fit(X_train, y_train)
    
y_pred_NB = spam_detect_model_NB.predict(X_test)   

#confution matrix
from sklearn.metrics import confusion_matrix
confusion_m_NB = confusion_matrix(y_test, y_pred_NB) 

#accurecy
from sklearn.metrics import accuracy_score
accuracy_NB = accuracy_score(y_test, y_pred_NB)

#creating model for randomforest
from sklearn.ensemble import RandomForestClassifier
spam_detect_model_RF = RandomForestClassifier(n_estimators=200,criterion='entropy').fit(X_train, y_train)

y_pred_RF = spam_detect_model_RF.predict(X_test)   

#confution matrix for randomforest
confusion_m_RF = confusion_matrix(y_test, y_pred_RF) 

#accurecy for randomforest
accuracy_RF = accuracy_score(y_test, y_pred_RF)

#creating model for SMV
from sklearn import svm
spam_detect_model_SVM = svm.SVC(gamma='auto').fit(X_train, y_train)

y_pred_SVM = spam_detect_model_SVM.predict(X_test)   

#confution matrix for randomforest
confusion_m_SVM = confusion_matrix(y_test, y_pred_SVM) 

#accurecy for randomforest
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)


#save the model
filename = 'model_pickle.pkl'
pickle.dump(spam_detect_model_RF, open(filename, 'wb'))