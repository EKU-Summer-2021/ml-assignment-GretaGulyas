"""
    Main class.
"""

from src.concrete_strength_dataset_read import ReadDataset
from src.linear_regression import LRExample

if __name__ == '__main__':
    cs = ReadDataset(r'datasets/Concrete_Data_Yeh.csv')
    lre = LRExample()
    x_train, x_test, y_train, y_test = lre.split_data(cs)
    lre.train(x_train, y_train)
    param_grid = [
        {'fit_intercept': [True, False],
         'normalize': [True, False]}
    ]
    print(lre.model_score(x_test, y_test))
    best_model = lre.grid_search(param_grid, x_train, y_train)
    print(best_model.score(x_test, y_test))
    lre.save_result(x_test, y_test)
