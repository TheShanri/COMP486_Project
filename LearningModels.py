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
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
        """plt.rc('font', size=14)
        plt.rc('axes', labelsize=14, titlesize=14)
        plt.rc('legend', fontsize=14)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        self.cleanDataDf.hist(bins=50, figsize=(12, 8))
        plt.show()"""

        # plots correlation graphs
        """attributes = ["Prices_num", "Beds_num", "Baths_num", "Square Footage_num", "Longitude", "Latitude"]
        scatter_matrix(self.cleanDataDf[attributes], figsize=(12, 8))
        plt.show()"""

        # displays correlation matrix with price
        """corr_matrix = self.cleanDataDf.iloc[:, 5:].corr()
        pprint.pprint(corr_matrix["Prices_num"].sort_values(ascending=False))"""

    # scales data with standard scaler
    def scaleData(self):

        # scales all data
        print("\nScaling data")
        self.X_train = self.std_scaler.fit_transform(self.X_train)
        self.X_test = self.std_scaler.fit_transform(self.X_test)
        #self.y_train = self.std_scaler.fit_transform(self.y_train)
        #self.y_test = self.std_scaler.fit_transform(self.y_test)
        """pprint.pprint(self.X_train)
        pprint.pprint(self.X_test)
        pprint.pprint(self.y_train)
        pprint.pprint(self.y_test)"""

    # creates, tests, and visualizes a k-neighbors regression
    def createKNeighborsModel(self):

        # loops through a range of k to find the best model
        print("\nCreating k-neighbors regression model")
        range_k = range(1, 11)
        scores = {}
        bestModel = None
        bestR2 = None
        for k in range_k:
            regression = KNeighborsRegressor(n_neighbors=k)
            regression.fit(self.X_train, self.y_train)
            y_pred = regression.predict(self.X_test)
            mae = metrics.mean_absolute_error(self.y_test, y_pred)
            mse = metrics.mean_squared_error(self.y_test, y_pred)
            rmse = metrics.mean_squared_error(self.y_test, y_pred, squared=False)
            r2 = metrics.r2_score(self.y_test, y_pred)
            scores[k] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
            if bestR2 is None or r2 > bestR2:
                bestModel = regression
                bestR2 = r2
            """print("The model performance for testing set")
            print("--------------------------------------")
            print('MAE is {}'.format(mae))
            print('MSE is {}'.format(mse))
            print('RMSE is {}'.format(rmse))
            print('R2 score is {}'.format(r2))"""

        # prints out the best model and a prediction on the first instance
        print(f"\nBest k-neighbors model parameters: {bestModel.get_params()}")
        print(f"Best k-neighbors model scores: {scores[bestModel.get_params()['n_neighbors']]}")
        y_pred_first = bestModel.predict([self.X_test[0]])
        print(f"First Instance: {self.X_test[0]}")
        print(f"Predicted price: {y_pred_first}")
        print(f"Actual price: {self.y_test.iloc[0, :]}")

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = bestModel.predict(self.X_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1))
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()

    # creates, tests, and visualizes a linear regression
    def createLinearModel(self):

        # creates a linear regression
        print("\nCreating linear regression model")
        linearReg = LinearRegression()
        linearReg.fit(self.X_train, self.y_train)
        y_pred = linearReg.predict(self.X_test)
        mae = metrics.mean_absolute_error(self.y_test, y_pred)
        mse = metrics.mean_squared_error(self.y_test, y_pred)
        rmse = metrics.mean_squared_error(self.y_test, y_pred, squared=False)
        r2 = metrics.r2_score(self.y_test, y_pred)
        print("The model performance for testing set")
        print("--------------------------------------")
        print('MAE is {}'.format(mae))
        print('MSE is {}'.format(mse))
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = linearReg.predict(self.X_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1))
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()


    # creates, tests, and visualizes a SVM polynomial regression
    def createSVMModel(self):

        self.X_train = self.X_train[0:1000]
        self.y_train = self.y_train[0:1000]
        self.X_test = self.X_test[0:1000]
        self.y_test = self.y_test[0:1000]


        # creates a SVM polynomial model
        print("\nCreating SVM polynomial regression model")
        svm_poly_reg = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
        svm_poly_reg.fit(self.X_train, self.y_train)
        y_pred = svm_poly_reg.predict(self.X_test)
        mae = metrics.mean_absolute_error(self.y_test, y_pred)
        mse = metrics.mean_squared_error(self.y_test, y_pred)
        rmse = metrics.mean_squared_error(self.y_test, y_pred, squared=False)
        r2 = metrics.r2_score(self.y_test, y_pred)
        print("The model performance for testing set")
        print("--------------------------------------")
        print('MAE is {}'.format(mae))
        print('MSE is {}'.format(mse))
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))

        # plots a graph comparing actual value versus predicted value
        fig, ax = plt.subplots()
        y_pred = svm_poly_reg.predict(self.X_test)
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1))
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()