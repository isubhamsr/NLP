# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 21:29:55 2021

    @author: Subham Roy
    https://www.codeingschool.com/
"""
import pickle
filename = 'model_pickle.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))

'''
    example:
        Ham Message: Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
        Spam Message: Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
'''

message = input("Enter message to check Ham or Spam: ")
data = [message]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)
if(my_prediction == 1):
    print("This Message is Spam")
else:
    print("This Message is not Spam (a Ham message)")
