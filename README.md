# Credit Score Prediction Model

## Project Overview

This project predicts a customer's credit score category using machine learning.

The model uses customer financial and personal information to predict the credit score category.

## Features

* Data preprocessing
* Missing value handling
* Categorical feature encoding
* Feature scaling
* Machine learning model training
* Credit score prediction
* Model and scaler saving using Pickle

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Jupyter Notebook / VS Code

## Project Structure

```text
Credit_Score_Model/
│
├── credit_score.csv
├── credit_score_model.py
├── model.pkl
├── requirements.txt
└── README.md
```

## Dataset

The project uses a Credit Score dataset containing customer financial information such as income, loans, credit history, payment behaviour, and other financial features.

The dataset file used in this project is:

```text
credit_score.csv
```

## Model

The machine learning model is trained using the data from `credit_score.csv`.

The trained model is saved as:

```text
model.pkl
```

The feature scaler used during preprocessing is saved as:

```text
scaler.pkl
```

## Installation

First, install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Make sure all project files are inside the same folder.

Then run:

```bash
python credit_score_model.py
```

The trained model and scaler will be saved automatically as:

```text
model.pkl
scaler.pkl
```

## Output

The model predicts the customer's credit score category based on the input financial information.

## Future Improvements

* Build a web application using Streamlit or Flask
* Add a user-friendly prediction interface
* Deploy the model online
* Improve model performance using hyperparameter tuning
* Add more financial features
* Compare multiple machine learning algorithms


