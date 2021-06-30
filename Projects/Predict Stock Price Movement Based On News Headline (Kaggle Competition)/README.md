
# Predict Stock Price Movement Based On News Headline

Predict Stock Price Movement Based On News Headline using NLP (Kaggle Competition). Machine Learning Algorithm used Naive Bayes, Random Forest, SVM


## Project Goal
Predict Stock Price Movement Based On News Headline using Natural Language Processing, with help of machine learning.Here we can predict a company's stock price goes up or down based on this day's news headline. With this, we can plan our stock market investments.

## In Details

To read more details of this project [Read Here](https://www.codeingschool.com/2021/06/sms-spam-classifier-with-nlp-with-deployment-code.html)

  
## Roadmap of the Project

#### Data collection

- I have used the Kaggle dataset. You can directly download it from [Here](https://github.com/isubhamsr/NLP/tree/master/Projects/Predict%20Stock%20Price%20Movement%20Based%20On%20News%20Headline%20(Kaggle%20Competition)/DataSet).
- There are 25 columns of top news headlines for each day in the data frame
- Class 1 -> The stock price increased
- Class 0 -> the stock price stayed the same or decreased

#### Data Cleaning

- Remove all punctuations from the text.
- Then remove Stopwords.
- Apply Lemmatization.

#### Create TF-IDF model

- Create a `TF-IDF model` to convert the text into the vector because our model can't understand the plain text so we need to convert the text into vectors.

Note: Prefered use ngram_range (2,2 or 4,4) for better accuracy. I got mmory error because I have 8gb of RAM this is not enough for this amont of data
  
#### Creating ML Model
- I create three machine learning models. Models are Naive Bayes, Random Forest, SVM.
- So we can observe that Random Forest has the highest accuracy among the algorithms.

#### Save ML Model
- I saved the Naive Bayes model (which has high accuracy) for model deployment.


## Deployment
- Deploy this project on the web, I create the `model_pickle.pkl` file for prodiction and `transform.pkl` for transform the new input text.
- In `Stock Price Prediction User Input.py` file you can prodict Ham or Spam with new text.
- Web live demo [Click Here](https://subhamroy.netlify.app/project/stock-price-prediction) 

#### Technology Used

- `Django` for backend and host the model
- `React Js` for frontend
- `MongoDb` as database
- `Heroku` for host the backend
- `netlify` for host the frontend

#### Deployment Architecture

![Deployment Architecture](https://res.cloudinary.com/dkcwzsz7t/image/upload/v1625037907/Web_1280_1_tl52ju.png)

## Authors

- [@Subham Roy](https://subhamroy.netlify.app/)
