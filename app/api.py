import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model_pack.Load_Data import load_model_pipeline

# from fastapi.encoders import jsonable_encoder
# adding Folder_2 to the system path
# sys.path.insert(0, '"Total path"/titanic-assignment/Artifacts/Production/Modelling')

app = FastAPI()

titanic_pipeline = load_model_pipeline()


class input_data(BaseModel):
    pclass: int
    name: str
    sex: str
    age: float
    sibsp: int
    parch: int
    ticket: str
    fare: float
    cabin: str
    embarked: str
    boat: str
    body: str
    home_dest: str


@app.get("/")
def root():
    return {"Title": "Titanic Survival"}


@app.post("/predict")
async def predict_data(input: input_data):
    idata = input.dict()

    data = pd.DataFrame(idata, index=[0])

    prediction = titanic_pipeline.predict(data).tolist()
    probability = titanic_pipeline.predict_proba(data)[:, 1].tolist()

    return {"prediction": prediction, "probability": probability}


if __name__ == "__main__":
    # Use this for debugging purposes only
    # logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000, log_level="debug")
