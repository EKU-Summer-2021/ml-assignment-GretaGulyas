"""
    Concrete strength prediction with linear regression.
"""

import pandas as pd


class ReadConcreteStrength:
    """
        Class with method that reads the concrete strength dataset.
    """
    def __init__(self, concrete_strength):
        self.concrete_strength = concrete_strength

    def concrete_strength_dataset_read(self):
        """
            Function for reading the CSV file.
        """
        return pd.read_csv(self.concrete_strength, names=['cement', 'slag', 'flyash', 'water', 'superplasticizer',
                                                             'coarseaggregate', 'fineaggregate', 'age', 'csMPa'])
