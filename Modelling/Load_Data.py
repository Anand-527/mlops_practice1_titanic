from importlib.resources import path
import pandas as pd
import yaml
import pathlib
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline

with open('config.yaml','r') as file:
    y = yaml.safe_load(file)

def data():
    data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    return data

def save_model_pipeline(model_pipeline_to_save:Pipeline):
    file_name = 'Trained_models\\' + y['Model_name']+'.pkl'
    complete_file_name = Path.joinpath(Path.cwd(),file_name)
    remove_old_model_pipeline()
    joblib.dump(model_pipeline_to_save,complete_file_name)

def load_model_pipeline():
    file_name = 'Trained_models\\' + y['Model_name']+'.pkl'
    complete_file_name = Path.joinpath(Path.cwd(),file_name)
    return joblib.load(complete_file_name)

def remove_old_model_pipeline():
    path_name = Path.joinpath(Path.cwd(),'Trained_models')
    #print(path_name)
    for i in Path(path_name).iterdir():
        print(i)
        if '.pkl' in str(i):
            i.unlink()


if __name__ == '__main__':
    #print(remove_model_pipeline())
    pass


# python Load_Data.py