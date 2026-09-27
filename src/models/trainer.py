"""
=========================================================
Model Trainer

Reusable model training framework.

Author:
Gyanvi
=========================================================
"""
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import joblib
import pandas as pd

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score

)

class ModelTrainer:

    def __init__(self):

        self.model = None

        self.results = {}

    def train(

        self,

        model,

        X_train,

        y_train

    ):

        self.model = model

        self.model.fit(

            X_train,

            y_train

        )

        return self
    def predict(

        self,

        X_test

    ):

        return self.model.predict(

            X_test

        )

    def evaluate(
            self,
            X_test,
            y_test
    ):
        predictions = self.predict(X_test)

        self.results = {

            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "Precision": precision_score(
                y_test,
                predictions,
                average="weighted"
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                average="weighted"
            ),

            "F1 Score": f1_score(
                y_test,
                predictions,
                average="weighted"
            )

        }

        return pd.DataFrame(
            self.results,
            index=[type(self.model).__name__]
        )
    def save(

        self,

        filepath

    ):

        joblib.dump(

            self.model,

            filepath

        )
    @staticmethod

    def load(filepath):

        return joblib.load(

            filepath

        )