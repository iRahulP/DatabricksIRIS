import os
import warnings
import sys

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics #for checking the model accuracy
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  
  #Read Dataset
  iris_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris.csv")
  data = pd.read_csv(iris_path)
  
  #Split the Dataset
  train, test = train_test_split(data, test_size = 0.3)
  # in this our main data is split into train and test
  # the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%
  train_X = train[['sepal_length','sepal_width','petal_length','petal_width']]# taking the training data features
  train_y=train.species# output of our training data
  test_X= test[['sepal_length','sepal_width','petal_length','petal_width']] # taking test data features
  test_y =test.species   #output value of test data
      
  with mlflow.start_run():
    model = svm.SVC() #select the algorithm
    model.fit(train_X,train_y) # we train the algorithm with the training data and the training output
    prediction=model.predict(test_X) #now we pass the testing data to the trained algorithm
    accuracy = metrics.accuracy_score(prediction,test_y)
    print("Accuracy :: "accuracy)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")