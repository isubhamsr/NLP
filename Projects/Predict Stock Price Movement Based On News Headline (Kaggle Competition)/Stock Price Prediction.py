# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 05:26:40 2021

    @author: Subham Roy
    https://www.codeingschool.com/
"""

import nltk
import re
import pickle
import pandas as pd

df = pd.read_csv('./DataSet/Data.csv', encoding='ISO-8859-1')

data = df.iloc[:, 2:27] #take only the headline parts
y = df['Label']

#remane columns
new_index = [str(i) for i in range(25)]
data.columns = new_index

from nltk.stem import WordNetLemmatizer #Lemmalizer
from nltk.stem import PorterStemmer #Stemmer
from nltk.corpus import stopwords 

ps = PorterStemmer()
wn = WordNetLemmatizer()

headlines = []
corpus = []

# remove puntuations from text
df.replace('[^a-zA-Z]', ' ', regex=True, inplace=True)

#convert into lower case
for i in new_index:
    data[i] = data[i].str.lower()
    
' '.join(str(i) for i in data.iloc[1, 0:25] )

#store the headline into headline array
# make rows are one paragraphy [one paragraph == one row's data]
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(i) for i in data.iloc[row, 0:25] ))
    
headlines.split()    
    
#apply lematization an d remove stopwords from headlines
for i in range(len(headlines)):
    review = headlines[i].split()
    review = [ wn.lemmatize(word) for word in review if not word in set(stopwords.words('english')) ]  
    review = ' '.join(review)
    corpus.append(review)
    
    
#Create TF-IDF model    
from sklearn.feature_extraction.text import TfidfVectorizer

'''
    Prefered use ngram_range. I got mmory error because I have 8gb of RAM this is not enough for this
    amont of data
    this is due to ram memory overflow meaning your storage requirements are higher than ram space has. 
    I would recommend switching to google colab they will provide you 25 gb of ram. 
    You can also prefer to use sparse matrices.
'''

# tf = TfidfVectorizer(ngram_range=(2,2)) -- use this when you habe more amount of ram, this will give you more accurecy
tf = TfidfVectorizer()
X = tf.fit_transform(corpus).toarray()

filename = 'transform.pkl'
pickle.dump(tf, open(filename, 'wb'))  

#train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#creating model for naive_bayes
from sklearn.naive_bayes import MultinomialNB
stock_price_prediction_NB = MultinomialNB(alpha=0.1).fit(X_train, y_train)
    
y_pred_NB = stock_price_prediction_NB.predict(X_test)   

#confution matrix
from sklearn.metrics import confusion_matrix
confusion_m_NB = confusion_matrix(y_test, y_pred_NB) 

#accurecy
from sklearn.metrics import accuracy_score
accuracy_NB = accuracy_score(y_test, y_pred_NB)

from sklearn.ensemble import RandomForestClassifier
stock_price_prediction_RF = RandomForestClassifier(n_estimators=200,criterion='entropy').fit(X_train, y_train)

y_pred_RF = stock_price_prediction_RF.predict(X_test)   

#confution matrix for randomforest
confusion_m_RF = confusion_matrix(y_test, y_pred_RF) 

#accurecy for randomforest
accuracy_RF = accuracy_score(y_test, y_pred_RF)

#creating model for SMV
from sklearn import svm
stock_price_prediction_SVM = svm.SVC(gamma='auto').fit(X_train, y_train)

y_pred_SVM = stock_price_prediction_SVM.predict(X_test)   

#confution matrix for randomforest
confusion_m_SVM = confusion_matrix(y_test, y_pred_SVM) 

#accurecy for randomforest
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)

#save the model
filename = 'model_pickle.pkl'
pickle.dump(stock_price_predictionl_NB, open(filename, 'wb'))