
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np



class ReplacingWithNan(BaseEstimator,TransformerMixin):
    # Extract fist letter of variable

    def __init__(self,var = None):
        self.var = None
    
    def fit(self, X, y=None):
        return self

    def transform(self,X):
        X = X.copy()

        X = X.replace('?',np.nan)
        
        return X



class FeatureDropping(BaseEstimator,TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables,list):
            raise ValueError('variables should be a list')

        self.variables = variables
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):
        X = X.copy()
        X = X.drop(labels = self.variables, axis=1)
        
        return X



class CovertingToFloat(BaseEstimator,TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables,list):
            raise ValueError('variables should be a list')

        self.variables = variables
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):
        X = X.copy()

        for i in self.variables:
            X[i] = X[i].astype('float')
        
        return X



class SalutationExtraction(BaseEstimator,TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables,list):
            raise ValueError('variables should be a list')

        self.variables = variables
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):
        X = X.copy()
    
        X['title']  = X[self.variables].apply(lambda x:
        'Mrs' if 'Mrs' in x else(
            'Mr' if 'Mr' in x else(
                'Miss' if 'Miss' else(
                    'Master' if 'Master' in X else 'Other'
        ))))
        
        return X




class ExtractLetterTransformer(BaseEstimator,TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables,list):
            raise ValueError('variables should be a list')

        self.variables = variables
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].str[0]
        
        return X

if __name__ == '__main__':
    pass