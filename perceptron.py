#-------------------------------------------------------------------------
# AUTHOR: Kate Yuan
# FILENAME: preceptron.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test
highest_accuracy_perceptron = 0
highest_accuracy_mlp = 0


for lr in n: #iterates over n

    for b in r: #iterates over r

        #iterates over both algorithms
        #-->add your Python code here
        algorithms = ['Perceptron', 'MLP']

        for algo in algorithms: #iterates over the algorithms

            #Create a Neural Network classifier
            if 'Perceptron' == algo:
               clf = Perceptron(eta0=lr, shuffle=b, max_iter=1000)    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
               clf = MLPClassifier(activation='logistic', learning_rate_init=lr, hidden_layer_sizes=25, shuffle=b, max_iter=1000) #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Python code here

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            total = 0
            correct = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                total+=1
                p = clf.predict([x_testSample])
                if p == y_testSample:
                    correct+=1

            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here

            accuracy = float(correct)/total
            if 'Perceptron' == algo:
                if accuracy > highest_accuracy_perceptron:
                    highest_accuracy_perceptron = accuracy
                    print(f'Highest Perceptron accuracy so far: {highest_accuracy_perceptron}, Parameters: learning rate={lr}, shuffle={b}')

            else:
                if accuracy > highest_accuracy_mlp:
                    highest_accuracy_mlp = accuracy
                    print(f'Highest MLP accuracy so far: {highest_accuracy_mlp}, Parameters: learning rate={lr}, shuffle={b}')

            #check if the calculated accuracy is higher 
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here






