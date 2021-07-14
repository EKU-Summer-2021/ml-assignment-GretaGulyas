"""
    Main class.
"""
from src.strength_prediction_with_lr import ReadConcreteStrength

if __name__ == '__main__':
    concrete_strength = ReadConcreteStrength(r'datasets/Concrete_Data_Yeh.csv')
    print(concrete_strength.concrete_strength_dataset_read())
