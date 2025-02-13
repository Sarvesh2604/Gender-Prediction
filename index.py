# Import the Necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data_file = pd.read_csv('GenderPrediction.csv')
print(data_file.head())

# Data Preprocessing
data_file_cleaned = data_file[['LastLetter','Gender']]
data_file_cleaned.dropna(inplace= True)
print(data_file_cleaned.head())

# Encode Catagorical Data
letter_encoder = LabelEncoder()
data_file_cleaned['LastLetter'] = letter_encoder.fit_transform(data_file_cleaned['LastLetter'])
gender_encoder = LabelEncoder()
data_file_cleaned['Gender'] = gender_encoder.fit_transform(data_file_cleaned['Gender'])
print(data_file_cleaned.head())

# Splitting Data into Training and Test Data
X = data_file_cleaned['LastLetter']
Y = data_file_cleaned['Gender']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state= 42)
print(X_train.shape, X_test.shape)

# Reshaping the values
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)
print(X_train.shape, X_test.shape)

# Train the Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)
print("Model Trained Successfully")

# Making Predictions
Y_pred = nb_model.predict(X_test)
print("Prediction:",Y_pred[:10])

# Checking Accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)
print(classification_report(Y_test, Y_pred, target_names=['Female', 'Male']))

# Predict Gender for new names
new_name = 'Rahul'
last_letter = new_name[-1].lower()
encoded_letter = letter_encoder.transform([last_letter])
predicted_gender = nb_model.predict([[encoded_letter[0]]])
gender_label = gender_encoder.inverse_transform(predicted_gender)
print(predicted_gender)