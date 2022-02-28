
from sklearn.pipeline import Pipeline
import Features as Fea

from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

import yaml


with open('config.yaml','r') as file:
    y=yaml.safe_load(file)


def titanic_pipe():

    # set up the pipeline
    pipeline = Pipeline([

    ('Replacing_?_with_nan',Fea.ReplacingWithNan()),

    ('Converting_the_numerical_to_float',Fea.CovertingToFloat(variables=y['Pipeline']['NUMERICAL_VARIABLES'])),

    ('salutation_extraction',Fea.SalutationExtraction(variables=y['Pipeline']['SALUTATION'])),

    ('dropping_features',Fea.FeatureDropping(variables=y['Pipeline']['DROP'])),    

    # ===== IMPUTATION =====
    # impute categorical variables with string 'missing'
    ('categorical_imputation', CategoricalImputer(variables=y['Pipeline']['CATEGORICAL_VARIABLES'])),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables=y['Pipeline']['NUMERICAL_VARIABLES'])),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(variables=y['Pipeline']['NUMERICAL_VARIABLES'])),


    # Extract first letter from cabin
    ('extract_letter', Fea.ExtractLetterTransformer(variables=y['Pipeline']['CABIN'])),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(variables=y['Pipeline']['CATEGORICAL_VARIABLES'])),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(drop_last=True,variables=y['Pipeline']['CATEGORICAL_VARIABLES'])),

    # scale using standardization
    ('scaler', StandardScaler()),

    # logistic regression (use C=0.0005 and random_state=0)
    ('Logit', LogisticRegression(C = y['Model_Params']['C'],random_state=y['Model_Params']['Random_State']))])

    return pipeline

if __name__ == '__main__':
    #pass
    print(y['Pipeline']['NUMERICAL_VARIABLES'])


# python Pipeline.py