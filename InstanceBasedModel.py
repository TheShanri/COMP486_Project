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

class InstanceBasedModel:

    # creates instance variables for k-neighbors model
    def __init__(self):

        # gets total dataset
        cleanDataDf = pd.read_csv("FINAL Rental Results.csv")
        X = cleanDataDf.iloc[:, 6:10]
        y = cleanDataDf.iloc[:, 10:]

        # splits data into training and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.10, random_state=42, shuffle=True)

        # creates a standard scaler
        self.std_scaler = StandardScaler()

        # scales data with standard scaler
        self.scaleData()

        # creates
        self.createModel()


    # scales data with standard scaler
    def scaleData(self):

        # scales all data
        self.X_train = self.std_scaler.fit_transform(self.X_train)
        self.X_test = self.std_scaler.fit_transform(self.X_test)
        self.y_train = self.std_scaler.fit_transform(self.y_train)
        self.y_test = self.std_scaler.fit_transform(self.y_test)



        # converts data back into properly labeled dataframe
        #pd.DataFrame(admissions_std_scaled, columns=["Serial No.", "GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research", "Chance of Admit"])
        pprint.pprint(self.X_train)
        pprint.pprint(self.X_test)
        pprint.pprint(self.y_train)
        pprint.pprint(self.y_test)


    def createModel(self):

        regression = KNeighborsRegressor(n_neighbors=5)
        regression.fit(self.X_train, self.y_train)
        y_pred = regression.predict(self.X_test)
        mae = metrics.mean_absolute_error(self.y_test, y_pred)
        mse = metrics.mean_squared_error(self.y_test, y_pred)
        r2 = metrics.r2_score(self.y_test, y_pred)
        print("The model performance for testing set")
        print("--------------------------------------")
        print('MAE is {}'.format(mae))
        print('MSE is {}'.format(mse))
        print('R2 score is {}'.format(r2))
        #print(regression.score(self.X_test, self.y_test))
        fig, ax = plt.subplots()
        ax.scatter(y_pred, self.y_test, edgecolors=(0, 0, 1))
        ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=3)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.show()
