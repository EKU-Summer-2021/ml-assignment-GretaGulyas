"""
    Support vector machine solution for the wine quality prediction problem.
"""

import os
# pylint: disable=too-few-public-methods
from src.wine_quality_prediction_with_svm_read import ReadData
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SVMExample:
    """
        The class that contains the methods for predicting the wine quality.
    """

    def __init__(self, kernel='rbf', c=1.0, gamma='auto'):
        self.model = SVC(kernel=kernel, C=c, gamma=gamma)
        self.search: GridSearchCV

    @classmethod
    def split_data(cls, cs_read: ReadData):
        """
            Method for splitting the dataset.
        """

        features = cs_read.dataset_read().iloc[:, :-1]
        target = cs_read.dataset_read().iloc[:, -1:].values
        print(target)
        return train_test_split(features, target, test_size=0.2)

    def train(self, x_train, y_train):
        """
            Method to train-test split.
        """
        scale = StandardScaler()
        x_scaled = scale.fit_transform(x_train)
        y_scaled = scale.fit_transform(y_train.reshape(-1, 1))
        self.model.fit(x_train, y_train)
        return self.model

    def model_score(self, x_test, y_test):
        """
            Method that returns with the score of the model.
        """

        return self.model.score(x_test, y_test)

    def grid_search(self, x_train, y_train, param_grid):
        """
            Method that finds the best model.
        """

        svc = SVC()
        search = GridSearchCV(svc, param_grid, cv=2, verbose=3)
        search.fit(x_train, y_train)
        self.search = search
        return self.search.best_estimator_

    def save_result(self, x_test, y_test):
        """
            Method that creates a directory for every single output, and saves it in a csv file with its plot.
        """
        directory = 'Results'
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, directory)
        isdir = os.path.isdir(path)
        if not isdir:
            os.mkdir(path)
        directory_svm = 'SVM'
        path = os.path.join(path, directory_svm)
        isdir = os.path.isdir(path)
        if not isdir:
            os.mkdir(path)
        file_location = os.path.join(path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.mkdir(file_location)
        self.__save_output_to_csv(file_location)
        self.__save_output_comparison_to_plot(file_location, self.search.best_estimator_, x_test, y_test)
        self.__save_plot_error(file_location, self.search.best_estimator_, x_test, y_test)

    def __save_output_to_csv(self, file_location):
        """
            Private method that saves the output into a csv file.
        """
        file_name = file_location + "/" + 'result.csv'
        params = self.search.cv_results_
        params = pd.DataFrame(params)
        params.to_csv(file_name)

    @classmethod
    def __save_output_comparison_to_plot(cls, file_location, model, x_test, y_test):
        """
            Private method that saves the output plot, which compares the actual data with the predicted.
        """

        plt.scatter(x_test.iloc[:, 5], y_test, color='red')
        plt.scatter(x_test.iloc[:, 5], model.predict(x_test), color='blue')
        plt.savefig(file_location + "/ResultPlot2")
        plt.show()

    @classmethod
    def __save_plot_error(cls, file_location, model, x_test, y_test):
        """
            Private method that saves the output plot, which shows the errorline.
        """

        plt.scatter(y_test, model.predict(x_test), color='red')
        plt.savefig(file_location + "/ResultPlot3")
        plt.show()
