"""
    Main class.
"""
from src.strength_prediction_with_lr import ReadConcreteStrength

if __name__ == '__main__':
    cs = ReadConcreteStrength(r'datasets/Concrete_Data_Yeh.csv')
    print(cs.concrete_strength_dataset_read())
    print(cs.train_test_split())
