import nltk
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

df = pd.read_csv('./DataSet/data.csv')
df.head()

df.shape

df.info()

df.describe()

#remove nan value
df = df.dropna()

df.head(10)

messages=df.copy()

#reset index
messages.reset_index(inplace=True)

messages.head(10)

corpus = []

# apply text preprocessing on text column (Remove StopWord, then apply Lematization)
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

corpus[6]


# TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=10000,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()


filename = 'transform.pkl'
pickle.dump(tfidf_v, open(filename, 'wb'))

tfidf_v.get_feature_names()[:20]

X.shape

y = messages['label']

y.shape


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# MultinomialNB Algorithm
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


from sklearn import metrics
import numpy as np

MultinomialNB_accu = metrics.accuracy_score(y_test, y_pred)
print(f"accuracy: {MultinomialNB_accu}")

cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

#Passive Aggressive Classifier Algorithm
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier()

linear_clf.fit(X_train, y_train)
y_pred = linear_clf.predict(X_test)
PassiveAggressiveClassifier_accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"accuracy:  {PassiveAggressiveClassifier_accuracy}" )
cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])

filename = 'model_pickle.pkl'
pickle.dump(linear_clf, open(filename, 'wb'))

# Multinomial Classifier with Hyperparameter
classifier=MultinomialNB(alpha=0.1)

previous_score = 0
for alpha in np.arange(0,1,0.1):
    sub_clasifier = MultinomialNB(alpha=alpha)
    sub_clasifier.fit(X_train, y_train)
    y_pred = sub_clasifier.predict(X_test)
    sub_clasifier_accuracy = metrics.accuracy_score(y_test, y_pred)
    if sub_clasifier_accuracy > previous_score:
        classifier = sub_clasifier
        previous_score = sub_clasifier_accuracy
    print(f"Alpha --> {alpha}, Accuracy --> {sub_clasifier_accuracy}, Previous Score --> {previous_score}")


## Get Features names
feature_names = tfidf_v.get_feature_names()

linear_clf.coef_[0]

### Most real
sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]

### Most fake
sorted(zip(classifier.coef_[0], feature_names))[:20]
