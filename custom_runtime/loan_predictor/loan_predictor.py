import uuid
import kserve
from kserve import InferRequest, InferResponse, InferOutput
from kserve.logging import logger
import pandas as pd
import numpy as np
from typing import Dict, Union

from autogluon.tabular import TabularPredictor


class LoanPredictor(kserve.Model):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.model_dir = model_dir
        self.model = None

    def load(self):
        print("Loading TabularPredictor from %s", self.model_dir)
        self.model = TabularPredictor.load(self.model_dir)
        self.ready = True
        print("Model loaded successfully")

    def predict(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferResponse]:

        # Build a DataFrame from the v2 InferRequest inputs
        data = {inp.name: inp.data for inp in payload.inputs}
        df = pd.DataFrame(data)
        print("Input dataframe shape: %s", df.shape)

        proba = self.model.predict_proba(df)
        # Pick the class with the highest probability and convert True->1, False->0
        predictions = proba.idxmax(axis=1).astype(int).tolist()
        print("Predictions: %s", predictions)

        return InferResponse(
            response_id=payload.id or str(uuid.uuid4()),
            model_name=self.name,
            infer_outputs=[
                InferOutput(
                    name="predictions",
                    datatype="INT64",
                    shape=[len(predictions)],
                    data=predictions,
                )
            ],
        )
