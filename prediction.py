# importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Load and read the datasets
heart = pd.read_csv("heart_cleveland_upload.csv")
diabetes = pd.read_csv("diabetes.csv")

# Preprocessing for Heart Disease Prediction
heart_df = heart.rename(columns={'condition': 'target'})
x_heart = heart_df.drop(columns='target')
y_heart = heart_df.target
x_train_heart, x_test_heart, y_train_heart, y_test_heart = train_test_split(x_heart, y_heart, test_size=0.25, random_state=42)
scaler_heart = StandardScaler()
x_train_scaler_heart = scaler_heart.fit_transform(x_train_heart)
x_test_scaler_heart = scaler_heart.transform(x_test_heart)

# Preprocessing for Diabetes Prediction
x_diabetes = diabetes.drop(columns='Outcome')
y_diabetes = diabetes.Outcome
x_train_diabetes, x_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(x_diabetes, y_diabetes, test_size=0.25, random_state=42)
scaler_diabetes = StandardScaler()
x_train_scaler_diabetes = scaler_diabetes.fit_transform(x_train_diabetes)
x_test_scaler_diabetes = scaler_diabetes.transform(x_test_diabetes)

# Heart Disease Prediction - Random Forest Classifier
heart_model = RandomForestClassifier(n_estimators=20)
heart_model.fit(x_train_scaler_heart, y_train_heart)
heart_model_accuracy = heart_model.score(x_test_scaler_heart, y_test_heart)
print('Heart Disease Prediction Model Accuracy: {}%\n'.format(round(heart_model_accuracy * 100, 2)))
pickle.dump(heart_model, open('heart-disease-prediction-model.pkl', 'wb'))

# Diabetes Prediction - Random Forest Classifier
diabetes_model = RandomForestClassifier(n_estimators=20)
diabetes_model.fit(x_train_scaler_diabetes, y_train_diabetes)
diabetes_model_accuracy = diabetes_model.score(x_test_scaler_diabetes, y_test_diabetes)
print('Diabetes Prediction Model Accuracy: {}%\n'.format(round(diabetes_model_accuracy * 100, 2)))
pickle.dump(diabetes_model, open('diabetes-prediction-model.pkl', 'wb'))

# Heart Disease Prediction - KNN Classifier
knn_model_heart = KNeighborsClassifier(n_neighbors=5)
knn_model_heart.fit(x_train_scaler_heart, y_train_heart)
knn_model_heart_accuracy = knn_model_heart.score(x_test_scaler_heart, y_test_heart)
print('Heart Disease Prediction KNN Model Accuracy: {}%\n'.format(round(knn_model_heart_accuracy * 100, 2)))
pickle.dump(knn_model_heart, open('knn-model-heart.pkl', 'wb'))

# Diabetes Prediction - KNN Classifier
knn_model_diabetes = KNeighborsClassifier(n_neighbors=5)
knn_model_diabetes.fit(x_train_scaler_diabetes, y_train_diabetes)
knn_model_diabetes_accuracy = knn_model_diabetes.score(x_test_scaler_diabetes, y_test_diabetes)
print('Diabetes Prediction KNN Model Accuracy: {}%\n'.format(round(knn_model_diabetes_accuracy * 100, 2)))
pickle.dump(knn_model_diabetes, open('knn-model-diabetes.pkl', 'wb'))
