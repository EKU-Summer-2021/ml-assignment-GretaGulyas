"""
    Concrete strength prediction with linear regression.
"""

# pylint: disable=too-few-public-methods

import pandas as pd


class ReadDataset:
    """
        Class with method that reads the dataset.
    """

    def __init__(self, dataset_location):
        self.dataset = dataset_location
        self.concrete_strength_dataset = self.concrete_strength_dataset_read()

    def concrete_strength_dataset_read(self):
        """
            Function for reading the CSV file.
        """
        return pd.read_csv(self.dataset, names=['cement', 'slag', 'fly ash', 'water', 'superplasticizer',
                                                'coarseaggregate', 'fineaggregate', 'age', 'csMPa'])
