import pickle
import argparse

import uvicorn
import pandas as pd
from fastapi import FastAPI

from Utils.utils import InModel, OutModel

with open("Model.pkl", "rb") as file:
    Model = pickle.load(file)

api = FastAPI()

@api.get("/model/info")
def get_model_info():
    return Model.__str__()

@api.post("/model/score", response_model=OutModel)
def get_model_score(input: InModel):
    input = pd.DataFrame([input.dict()])
    return OutModel(scores=Model.predict_proba(input).squeeze().tolist())


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--host", type=str, default="0.0.0.0", required=False)
    argument_parser.add_argument("--port", type=int, default=8080, required=False)

    arguments = argument_parser.parse_args()

    uvicorn.run(api, host=arguments.host, port=arguments.port)