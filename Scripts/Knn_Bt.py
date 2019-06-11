import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras import utils as np_utils

def load_numeric_training():
    plant = pd.read_csv('Data/tr1.txt', skiprows=0)
    plant.head()
    y = plant.pop('species')
    # Since the labels are textual, so we encode them categorically
    encoder = LabelEncoder()
    encoder.fit(y)
    # convert integers to dummy variables
    encoded_Y = encoder.transform(y)
    # standardize the data by setting the mean to 0 and std to 1
    y = np_utils.to_categorical(encoded_Y)
    X = plant.iloc[:, 1:]
   
    return X,y 

def load_test_data():
    x, y = load_numeric_training()
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 1-0.2, test_size = 0.2, random_state =42)
    return (x_train, y_train), (x_test, y_test)

    
def Decesion_tree():
    (X_train, y_train),(X_test, y_test)=load_test_data()
    
    #Pour la transformation des entités en adaptant chaque entité à une plage donnée par la suite
    scaler = MinMaxScaler()
    
    #Fit() : Learn a vocabulary dictionary of all tokens in the raw documents
    #Fit_transform() : Learn the vocabulary dictionary and return term-document matrix
    X_train = scaler.fit_transform(X_train)
    
    #Transform(): Transform documents to document-term matrix
    X_test = scaler.transform(X_test)
    
    #Build a decision tree classifier from the training set 
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
         .format(clf.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
         .format(clf.score(X_test, y_test)))
    #print('Log-loss error : ', )

def Knn_():
    (X_train, y_train),(X_test, y_test)=load_test_data()
    
    #Build a knn classifior 
    knn = KNeighborsClassifier(n_neighbors=3)
    #Ajuster le modèle en utilisant X comme données d'apprentissage et y comme valeurs cibles
    knn.fit(X_train, y_train)
    
    print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

def main():   
    Knn_()
    Decesion_tree()
