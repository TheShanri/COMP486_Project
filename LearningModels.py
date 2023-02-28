"""
This class contains the functions to create, run, and test an instance based k-neighbors model for
rental price prediction.

By Cole Koryto
"""
import pprint

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import tensorflow as tf


class LearningModels:

    # creates instance variables for k-neighbors model
    def __init__(self):

        # gets total dataset
        self.cleanDataDf = pd.read_csv("FINAL Rental Results.csv")
        X = self.cleanDataDf.iloc[:, 5:10]
        y = self.cleanDataDf.iloc[:, 10:]

        # splits data into training and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.10, random_state=42, shuffle=True)

        # creates a standard scaler
        self.std_scaler = StandardScaler()

        # scales data with standard scaler
        self.scaleData()

    # visualizes the data to understand structure of data
    def visualizeData(self):

        # plots histograms of attributes
        plt.rc('font', size=14)
        plt.rc('axes', labelsize=14, titlesize=14)
        plt.rc('legend', fontsize=14)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        self.cleanDataDf.hist(bins=50, figsize=(12, 8))
        plt.show()

        # plots correlation graphs
        attributes = ["Prices_num", "Beds_num", "Baths_num", "Square Footage_num", "Longitude", "Latitude"]
        scatter_matrix(self.cleanDataDf[attributes], figsize=(12, 8))
        plt.show()

        # displays correlation matrix with price
        corr_matrix = self.cleanDataDf.iloc[:, 5:].corr()
        pprint.pprint(corr_matrix["Prices_num"].sort_values(ascending=False))

    # scales data with standard scaler
    def scaleData(self):

        # scales all data
        print("\nScaling data")
        self.X_train = self.std_scaler.fit_transform(self.X_train)
        self.X_test = self.std_scaler.fit_transform(self.X_test)

    # outputs metrics for given predictions and actual data set
    def outputMetrics(self, y_actual, y_pred):
        mae = metrics.mean_absolute_error(y_actual, y_pred)
        mse = metrics.mean_squared_error(y_actual, y_pred)
        rmse = metrics.mean_squared_error(y_actual, y_pred, squared=False)
        r2 = metrics.r2_score(y_actual, y_pred)
        print("--------------------------------------")
        print('MAE is {}'.format(mae))
        print('MSE is {}'.format(mse))
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("--------------------------------------")

    # creates, tests, and visualizes a k-neighbors regression
    def createKNeighborsModel(self):

        # loops through a range of k to find the best model
        print("\n\nCreating k-neighbors regression model")
        parameters = [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}]
        K = 3
        k_neighbors_reg = GridSearchCV(KNeighborsRegressor(), parameters, cv=K, verbose=3, n_jobs=-1)
        k_neighbors_reg.fit(self.X_train, self.y_train)
        print(f"Best model parameters: {k_neighbors_reg.best_params_}")
        print("\nTraining Set Metrics")
        y_train_pred = k_neighbors_reg.predict(self.X_train)
        self.outputMetrics(self.y_train, y_train_pred)
        print("\nTest Set Metrics")
        y_test_pred = k_neighbors_reg.predict(self.X_test)
        self.outputMetrics(self.y_test, y_test_pred)

        # prints out the best model and a prediction on the first instance
        y_pred_first = k_neighbors_reg.predict([self.X_test[0]])
        print(f"First Instance: {self.X_test[0]}")
        print(f"Predicted price: {y_pred_first}")
        print(f"Actual price: {self.y_test.iloc[0, :]}")

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = k_neighbors_reg.predict(self.X_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1), alpha=0.1)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()

    # creates, tests, and visualizes a linear regression (elastic net regression)
    def createLinearModel(self):

        # creates a linear regression
        print("\nCreating linear regression model (elastic net)")
        parameters = [{'l1_ratio': [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}]
        K = 3
        linearRegElastic = GridSearchCV(ElasticNet(random_state=0, max_iter=200000), parameters, cv=K, verbose=3, n_jobs=-1)
        linearRegElastic.fit(self.X_train, self.y_train)
        print(f"Best model parameters: {linearRegElastic.best_params_}")
        print("\nTraining Set Metrics")
        y_train_pred = linearRegElastic.predict(self.X_train)
        self.outputMetrics(self.y_train, y_train_pred)
        print("\nTest Set Metrics")
        y_test_pred = linearRegElastic.predict(self.X_test)
        self.outputMetrics(self.y_test, y_test_pred)

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = linearRegElastic.predict(self.X_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1), alpha=0.1)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()


    # creates, tests, and visualizes a SVM polynomial regression
    def createSVMModel(self):

        # creates a SVM polynomial model
        print("\nCreating SVM polynomial regression model")
        parameters = [{'kernel': ['rbf'], 'gamma': [0.8], 'C': [5000], 'degree': [1], 'epsilon': [0.1]}]
        K = 3
        svm_poly_reg = GridSearchCV(SVR(kernel="poly"), parameters, cv=K, verbose=3, n_jobs=-1)
        svm_poly_reg.fit(self.X_train, self.y_train.values.ravel())
        print(f"Best model parameters: {svm_poly_reg.best_params_}")
        print("\nTraining Set Metrics")
        y_train_pred = svm_poly_reg.predict(self.X_train)
        self.outputMetrics(self.y_train, y_train_pred)
        print("\nTest Set Metrics")
        y_test_pred = svm_poly_reg.predict(self.X_test)
        self.outputMetrics(self.y_test, y_test_pred)

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = svm_poly_reg.predict(self.X_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1), alpha=0.1)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3,)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()

    # creates, tests, and visualizes a sequential ANN
    def createNeuralNetwork(self):

        # builds neural network
        tf.random.set_seed(42)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=[5]))
        model.add(tf.keras.layers.Dense(300,kernel_initializer='normal', activation="relu"))
        model.add(tf.keras.layers.Dense(300,kernel_initializer='normal', activation="relu"))
        model.add(tf.keras.layers.Dense(300,kernel_initializer='normal', activation="relu"))
        model.add(tf.keras.layers.Dense(300,kernel_initializer='normal', activation="relu"))
        model.add(tf.keras.layers.Dense(300,kernel_initializer='normal', activation="relu"))
        model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))

        # compiles neural network
        model.compile(loss="mean_squared_error",
                      optimizer="adam",
                      metrics=["mean_absolute_error"])

        # trains neural network
        history = model.fit(self.X_train, self.y_train, epochs=30)

        # evaluates neural network
        print(f"Loss and accuracy for test set: {model.evaluate(self.X_test, self.y_test)}")
        print("\nTraining Set Metrics")
        y_train_pred = model.predict(self.X_train)
        self.outputMetrics(self.y_train, y_train_pred)
        print("\nTest Set Metrics")
        y_test_pred = model.predict(self.X_test)
        self.outputMetrics(self.y_test, y_test_pred)

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = model.predict(self.X_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1), alpha=0.1)
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3, )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()