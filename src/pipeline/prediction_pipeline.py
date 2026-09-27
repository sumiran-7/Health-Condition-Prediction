"""
=========================================================
Prediction Pipeline

Loads trained artifacts and performs inference.

Author:
Gyanvi
=========================================================
"""

from pathlib import Path

import joblib
import pandas as pd


class PredictionPipeline:
    """
    Prediction pipeline used during deployment.
    """

    def __init__(self):

        project_root = Path(__file__).resolve().parents[2]

        artifacts_path = (
            project_root / "artifacts"
        )

        self.model = joblib.load(
            artifacts_path / "best_model.pkl"
        )

        self.preprocessor = joblib.load(
            artifacts_path / "preprocessor.pkl"
        )

        self.label_encoder = joblib.load(
            artifacts_path / "label_encoder.pkl"
        )

    def preprocess(
            self,
            input_df: pd.DataFrame
    ):
        processed = self.preprocessor.transform(
            input_df
        )

        feature_names = self.preprocessor.get_feature_names_out()

        processed = pd.DataFrame(
            processed,
            columns=feature_names
        )

        return processed

    def predict(
        self,
        input_df: pd.DataFrame
    ):

        processed = self.preprocess(
            input_df
        )

        prediction = self.model.predict(
            processed
        )

        return self.label_encoder.inverse_transform(
            prediction
        )

    def predict_proba(
        self,
        input_df: pd.DataFrame
    ):

        processed = self.preprocess(
            input_df
        )

        return self.model.predict_proba(
            processed
        )

    def get_prediction(
        self,
        input_df: pd.DataFrame
    ):

        prediction = self.predict(
            input_df
        )[0]

        probability = self.predict_proba(
            input_df
        )[0]

        confidence = probability.max()

        return {

            "prediction": prediction,

            "probability": probability,

            "confidence": confidence

        }