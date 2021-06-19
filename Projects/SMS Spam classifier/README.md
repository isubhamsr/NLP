
# SMS Spam Classifier

SMS Spam Classifier using NLP. Machine Learning Algorithm used Naive Bayes, Random Forest, SVM


## In Details

To read more details of this project [Read Here](https://codeingschool.com)

  
## Roadmap of the Project

#### Data collection

- I downloaded data from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

#### Data Cleaning

- Remove all punctuations from the text.
- Then remove Stopwords.
- Apply Lemmatization.

#### Create TF-IDF model

- Create a TF-IDF model to convert the text into the vector because our model can't understand the plain text so we need to convert the text into vectors.
  
#### Creating ML Model
- I create three machine learning models. Models are Naive Bayes, Random Forest, SVM, the accuracy of these models are 95%, 96%, 85%.
- So we can observe that Random Forest has the highest accuracy among the algorithms.

#### Save ML Model
- I saved the Random Forest model (which has high accuracy) for model deployment.
## Deployment

- Deploy this project on the web, to predict new texts that it is Spam or Ham. 
- I use Heroku to host the backend (Django) and Netlify to host frontend (React Js)
[View Here](https://codeingschool.com)

  
## Authors

- [@isubhamsr](https://subhamroy.netlify.app/)

  