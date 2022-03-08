from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    cd = Path(__file__).parent
    sample_file = Path.joinpath(cd.parent, "Datasets\\sample_data.csv")
    # importing dataframe
    data = pd.read_csv(sample_file)

    for i in data.columns:
        if "." in i:
            i_new = i.replace(".", "_")
            data.rename(columns={i: i_new}, inplace=True)

    return data
