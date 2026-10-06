# Fake News Detector

An AI-powered web application that analyzes news headlines and predicts whether they are **FAKE NEWS** or **REAL NEWS**.

## Project Overview

This project uses machine learning and natural language processing (NLP) to classify news headlines.

The application was built with **Python** and **Flask**, with a machine learning model using **TF-IDF** and **Logistic Regression**.

## Technologies Used

* Python
* Flask
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Joblib
* HTML/CSS

## Model Performance

The model was trained and tested using **44,898 news articles**.

| Metric            |     Result |
| ----------------- | ---------: |
| Total Articles    |     44,898 |
| Fake Articles     |     23,481 |
| Real Articles     |     21,417 |
| Training Articles |     35,918 |
| Testing Articles  |      8,980 |
| Accuracy          | **94.39%** |

The model uses the **headline/title** of each article for classification.

## How It Works

1. The application receives a news headline from the user.
2. TF-IDF converts the headline into numerical features.
3. A Logistic Regression model analyzes the features.
4. The model predicts whether the headline is fake or real.
5. The application displays the prediction and model confidence.

## Project Structure

```text
fake-news-detector/
│
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── fake_news.py
├── test.py
└── templates/
    └── index.html
```

## Running the Project

Install the required Python packages:

```bash
pip install flask pandas scikit-learn joblib
```

Train the model:

```bash
python train_model.py
```

Start the Flask application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## What I Learned

Through this project, I practiced:

* Python programming
* Machine learning
* Natural language processing
* Data preparation with Pandas
* TF-IDF text vectorization
* Logistic Regression
* Model evaluation
* Saving and loading trained machine learning models
* Building a web application with Flask
* Connecting a machine learning model to a web interface
* Using GitHub to document and showcase a software project
