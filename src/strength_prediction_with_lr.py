"""
    Concrete strength prediction with linear regression.
"""

# pylint: disable=too-few-public-methods

import pandas as pd
from sklearn.model_selection import train_test_split


class ReadConcreteStrength:
    """
        Class with method that reads the concrete strength dataset.
    """

    def __init__(self, dataset_location):
        self.dataset = dataset_location
        self.concrete_strength_dataset = self.concrete_strength_dataset_read()

    def concrete_strength_dataset_read(self):
        """
            Function for reading the CSV file.
        """
        return pd.read_csv(self.dataset, names=['cement', 'slag', 'flyash', 'water', 'superplasticizer',
                                                'coarseaggregate', 'fineaggregate', 'age', 'csMPa'])

    def train_test_split(self):
        """
            Function for train-test split.
        """

        training_data, testing_data = train_test_split(self.concrete_strength_dataset, test_size=0.2, random_state=25)
        print('Train', training_data.shape)
        print('Test', testing_data.shape)
