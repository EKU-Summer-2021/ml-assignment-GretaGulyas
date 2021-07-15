"""
    Concrete strength prediction with linear regression.
"""

# pylint: disable=too-few-public-methods
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from src.strength_prediction_with_lr import ConcreteStrength
import os
import pandas as pd
import matplotlib.pyplot as plt


class LRExample:

    def __init__(self, fit_intercept=True, normalize=False):
        self.model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)

    def split_data(self, cs: ConcreteStrength):
        """
            Function for train-test split.
        """

        X = cs.concrete_strength_dataset.iloc[:, :-1]
        y = cs.concrete_strength_dataset.iloc[:, -1:]
        return train_test_split(X, y, test_size=0.2)

    def train(self, x_train, y_train):
        """
            Function for train-test split.
        """

        self.model.fit(x_train, y_train)
        return self.model

    def model_score(self, x_test, y_test):
        """
            Function for train-test split.
        """

        return self.model.score(x_test, y_test)

    def grid_search(self, params_space, x_train, y_train):
        """
            Function that finds the best model.
        """
        search = GridSearchCV(self.model, params_space)
        search.fit(x_train, y_train)
        self.search = search
        return search.best_estimator_

    def save_result(self):
        """

        """
        directory = 'Results'
        parent_dir = 'D:\Pycharm Projects\ml-assignment-GretaGulyas'
        path = os.path.join(parent_dir, directory)
        isdir = os.path.isdir(path)
        if not isdir:
            os.mkdir(path)
        directory_lr = 'LR'
        parent_dir = 'D:\Pycharm Projects\ml-assignment-GretaGulyas\Results'
        path = os.path.join(parent_dir, directory_lr)
        isdir = os.path.isdir(path)
        if not isdir:
            os.mkdir(path)
        file_location = os.path.join(path,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        parent_dir = 'D:\Pycharm Projects\ml-assignment-GretaGulyas\Results'
        path = os.path.join(parent_dir, directory)
        isdir = os.path.isdir(file_location)
        os.mkdir(file_location)
        self.__save_output_to_csv(file_location)
        self.__save_output_to_plot(file_location)

    def __save_output_to_csv(self, file_location):
        file_name = file_location + "/" + 'result'
        params = self.search.cv_results_
        params = pd.DataFrame(params)
        params.to_csv(file_name)

    def __save_output_to_plot(self, x_test, y_test):
        plt.plot(x_test, y_test, color='k', label='Regression model')
        plt.show()
