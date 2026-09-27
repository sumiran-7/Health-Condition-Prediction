"""
=========================================================
Feature Engineering Module

Author:
Gyanvi
=========================================================
"""

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
class DataPreprocessor:

    def __init__(self):

        self.pipeline = None

        self.numerical_features = None

        self.categorical_features = None

    def identify_features(self, X):

        self.numerical_features = X.select_dtypes(

            include=np.number

        ).columns.tolist()

        self.categorical_features = X.select_dtypes(

            exclude=np.number

        ).columns.tolist()

    def numerical_pipeline(self):

        return Pipeline(

            steps=[

                (

                    "imputer",

                    SimpleImputer(

                        strategy="median"

                    )

                ),

                (

                    "scaler",

                    StandardScaler()

                )

            ]

        )
    def categorical_pipeline(self):

        return Pipeline(

            steps=[

                (

                    "imputer",

                    SimpleImputer(

                        strategy="most_frequent"

                    )

                ),

                (

                    "encoder",

                    OneHotEncoder(

                        handle_unknown="ignore"

                    )

                )

            ]

        )
    def build_pipeline(self, X):

        self.identify_features(X)

        self.pipeline = ColumnTransformer(

            transformers=[

                (

                    "numerical",

                    self.numerical_pipeline(),

                    self.numerical_features

                ),

                (

                    "categorical",

                    self.categorical_pipeline(),

                    self.categorical_features

                )

            ]

        )

        return self.pipeline
    def fit(self, X):

        self.pipeline.fit(X)

        return self
    def transform(self, X):

        return self.pipeline.transform(X)
    def fit_transform(self, X):

        return self.pipeline.fit_transform(X)
    def get_feature_names(self):

        encoded = self.pipeline.named_transformers_[

            "categorical"

        ].named_steps[

            "encoder"

        ].get_feature_names_out(

            self.categorical_features

        )

        return self.numerical_features + list(encoded)
    def save(self, filepath):

        joblib.dump(

            self.pipeline,

            filepath

        )
    @staticmethod
    def load(filepath):

        return joblib.load(filepath)
