
# SMS Spam Classifier

SMS Spam Classifier using NLP. Machine Learning Algorithm used Naive Bayes, Random Forest, SVM


## Project Goal
SMS Spam Classifier using Natural Language Processing, with help of machine learning. Here we can predict a message Spam or not. It helps us to protect from potential scams.

## In Details

To read more details of this project [Read Here](https://www.codeingschool.com/2021/06/sms-spam-classifier-with-nlp-with-deployment-code.html)

  
## Roadmap of the Project

#### Data collection

- I downloaded data from [Here](https://github.com/isubhamsr/NLP/tree/master/Projects/Predict%20Stock%20Price%20Movement%20Based%20On%20News%20Headline%20(Kaggle%20Competition)/DataSet).

#### Data Cleaning

- Remove all punctuations from the text.
- Then remove Stopwords.
- Apply Lemmatization.

#### Create TF-IDF model

- Create a TF-IDF model to convert the text into the vector because our model can't understand the plain text so we need to convert the text into vectors.

Note: Prefered use ngram_range (2,2 or 4,4) for better accuracy. I got mmory error because I have 8gb of RAM this is not enough for this amont of data
  
#### Creating ML Model
- I create three machine learning models. Models are Naive Bayes, Random Forest, SVM, the accuracy of these models are 95%, 96%, 85%.
- So we can observe that Random Forest has the highest accuracy among the algorithms.

#### Save ML Model
- I saved the Random Forest model (which has high accuracy) for model deployment.


## Deployment
- Deploy this project on the web, I create the `model_pickle.pkl` file for prodiction and `transform.pkl` for transform the new input text.
- In `ClassificationfromUserInput.py` file you can prodict Ham or Spam with new text.
- Web live demo [Click Here](https://subhamroy.netlify.app/project/sms-span-classification) 

#### Technology Used

- `Django` for backend and host the model
- `React Js` for frontend
- `MongoDb` as database
- `Heroku` for host the backend
- `netlify` for host the frontend

#### Deployment Architecture

![Deployment Architecture](https://res.cloudinary.com/dkcwzsz7t/image/upload/v1624268730/Web_1280_1_mncmry.png)

## Authors

- [@Subham Roy](https://subhamroy.netlify.app/)