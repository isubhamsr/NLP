# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 21:29:55 2021

    @author: Subham Roy
    https://www.codeingschool.com/
"""
import pickle
import re
filename = 'model_pickle.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))

'''
    example:
        0: A  hindrance to operations   extracts from the leaked reports
        1: Pilgrim knows how to progress
'''

headline = input("Enter a Headline: ")
headline = re.sub('[^a-zA-Z]', ' ', headline) # remove pantuations
data = [headline]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)
if(my_prediction == 1):
    print("Stock Price will goes Up")
else:
    print("Stock Price will goes Down")
