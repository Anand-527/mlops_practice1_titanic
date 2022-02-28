
from sklearn.model_selection import train_test_split
import pandas as pd
import Load_Data #import data

import yaml

with open('config.yaml','r') as file:
    y = yaml.safe_load(file)


data = Load_Data.data()
def test_train(X):
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('survived', axis=1),  # predictors
        data['survived'],  # target
        test_size= y['Test_Train']['test_size'],  # percentage of obs in test set
        random_state=y['Test_Train']['Random_State'])  # seed to ensure reproducibility

    if X == 'train':
        return X_train, y_train

    if X == 'test':
        return X_test, y_test

if __name__=='__main__':
    print(test_train('train'))
    print(test_train('test'))

# python Train_Test_split.py
