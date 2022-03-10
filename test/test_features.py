import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import model_pack.Features as fea
from model_pack.Config.config_validations import Config


@pytest.mark.fea_test
def test_ReplacingWithNan(sample_data):
    feature_call = fea.ReplacingWithNan()
    feature_op = feature_call.fit_transform(sample_data)

    manual_op = sample_data.copy()
    manual_op = manual_op.replace("?", np.nan)

    difference_op = manual_op.compare(feature_op)

    if difference_op.shape[0] == 0:
        feature_op.to_csv("sample_data1.csv")

    assert difference_op.shape[0] == 0, difference_op


@pytest.mark.fea_test
def test_CovertingToFloat():
    sample_data = pd.read_csv("sample_data1.csv")

    feature_call = fea.CovertingToFloat(
        variables=Config.pipe_params.NUMERICAL_VARIABLES
    )
    feature_op = feature_call.fit_transform(sample_data)

    manual_op = sample_data.copy()
    for i in Config.pipe_params.NUMERICAL_VARIABLES:
        manual_op[i] = manual_op[i].astype("float")

    difference_op = manual_op.compare(feature_op)

    if difference_op.shape[0] == 0:
        feature_op.to_csv("sample_data2.csv")

    assert difference_op.shape[0] == 0, difference_op


@pytest.mark.fea_test
def test_SalutationExtraction():
    sample_data = pd.read_csv("sample_data2.csv")

    feature_call = fea.SalutationExtraction(variables=Config.pipe_params.SALUTATION)
    feature_op = feature_call.fit_transform(sample_data)

    manual_op = sample_data.copy()

    def get_title(passenger):
        line = passenger
        if re.search("Mrs", line):
            return "Mrs"
        elif re.search("Mr", line):
            return "Mr"
        elif re.search("Miss", line):
            return "Miss"
        elif re.search("Master", line):
            return "Master"
        else:
            return "Other"

    manual_op["title"] = manual_op[Config.pipe_params.SALUTATION[0]].apply(get_title)

    difference_op = manual_op.compare(feature_op)

    if difference_op.shape[0] == 0:
        feature_op.to_csv("sample_data3.csv")

    assert difference_op.shape[0] == 0, difference_op


@pytest.mark.fea_test
def test_FeatureDropping():
    sample_data = pd.read_csv("sample_data3.csv")
    feature_call = fea.FeatureDropping(variables=Config.pipe_params.DROP)
    feature_op = feature_call.fit_transform(sample_data)

    manual_op = sample_data.copy()
    manual_op.drop(labels=Config.pipe_params.DROP, axis=1, inplace=True)

    difference_op = manual_op.compare(feature_op)

    if difference_op.shape[0] == 0:
        feature_op.to_csv("sample_data4.csv")

    assert difference_op.shape[0] == 0, difference_op


@pytest.mark.fea_test
def test_ExtractLetterTransformer():
    sample_data = pd.read_csv("sample_data4.csv")
    feature_call = fea.ExtractLetterTransformer(variables=Config.pipe_params.CABIN)
    feature_op = feature_call.fit_transform(sample_data)

    manual_op = sample_data.copy()

    def get_first_cabin(row):
        try:
            return row.split()[0][0]
        except:
            return np.nan

    manual_op[Config.pipe_params.CABIN[0]] = manual_op[
        Config.pipe_params.CABIN[0]
    ].apply(get_first_cabin)

    difference_op = manual_op.compare(feature_op)

    if difference_op.shape[0] == 0:
        feature_op.to_csv("sample_data4.csv")

    assert difference_op.shape[0] == 0, difference_op


@pytest.mark.scrap_delete
def test_delete_scrap():
    cd = Path(__file__).parent
    for i in Path(cd).iterdir():
        if ".csv" in str(i):
            i.unlink()
