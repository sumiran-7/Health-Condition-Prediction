"""
=========================================================
Data Profiler

Reusable profiling utility for tabular datasets.

Author:
Gyanvi
=========================================================
"""

import numpy as np
import pandas as pd


class DataProfiler:

    """
    Data profiling class.
    """

    def __init__(self, df: pd.DataFrame):

        self.df = df


    def overview(self):
        """
        Display overall dataset information.
        """

        print("=" * 80)
        print("DATASET OVERVIEW")
        print("=" * 80)

        print(f"Rows             : {self.df.shape[0]:,}")
        print(f"Columns          : {self.df.shape[1]}")

        memory = self.df.memory_usage(
            deep=True
        ).sum()/1024**2

        print(f"Memory Usage     : {memory:.2f} MB")

        print(
            f"Duplicate Rows   : {self.df.duplicated().sum()}"
        )

        print(
            f"Missing Values   : {self.df.isnull().sum().sum()}"
        )

    def shape(self):

        return pd.DataFrame({

            "Metric":[

                "Rows",

                "Columns"

            ],

            "Value":[

                self.df.shape[0],

                self.df.shape[1]

            ]

        })

    def feature_lists(self):

        numerical = self.df.select_dtypes(

            include=np.number

        ).columns.tolist()

        categorical = self.df.select_dtypes(

            exclude=np.number

        ).columns.tolist()

        return numerical, categorical

    def column_summary(self):

        summary = pd.DataFrame({

            "Data Type":self.df.dtypes,

            "Missing Values":

            self.df.isnull().sum(),

            "Missing (%)":

            (self.df.isnull().mean()*100).round(2),

            "Unique Values":

            self.df.nunique()

        })

        summary["Cardinality (%)"]=(

            summary["Unique Values"]

            /len(self.df)

            *100

        ).round(2)

        return summary

    def duplicate_summary(self):

        duplicate_rows = self.df.duplicated().sum()

        duplicate_percentage = round(

            duplicate_rows/

            len(self.df)

            *100,

            2

        )

        return pd.DataFrame({

            "Duplicate Rows":[

                duplicate_rows

            ],

            "Duplicate Percentage":[

                duplicate_percentage

            ]

        })
    def numerical_summary(self):

        return self.df.describe().T

    def categorical_summary(self):

        return self.df.describe(

            include="object"

        ).T

    def missing_summary(self):
        """
        Returns missing value statistics for each feature.
        """

        missing = pd.DataFrame({

            "Missing Values": self.df.isnull().sum(),

            "Missing (%)": (
                    self.df.isnull().mean() * 100
            ).round(2)

        })

        missing = missing[
            missing["Missing Values"] > 0
            ].sort_values(
            by="Missing Values",
            ascending=False
        )

        return missing

    def outlier_summary(self):
        """
        Detect outliers using the IQR method.
        """

        numerical = self.df.select_dtypes(include=np.number)

        results = []

        for column in numerical.columns:
            q1 = numerical[column].quantile(0.25)

            q3 = numerical[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            outliers = (
                    (numerical[column] < lower) |
                    (numerical[column] > upper)
            ).sum()

            results.append({

                "Feature": column,

                "Outliers": outliers,

                "Percentage": round(
                    outliers / len(self.df) * 100,
                    2
                )

            })

        return (
            pd.DataFrame(results)
            .sort_values("Outliers", ascending=False)
        )

    def data_quality_report(self):
        """
        Generate an overall data quality report.
        """

        report = {

            "Rows": self.df.shape[0],

            "Columns": self.df.shape[1],

            "Duplicate Rows":
                self.df.duplicated().sum(),

            "Missing Values":
                self.df.isnull().sum().sum(),

            "Numerical Features":
                len(
                    self.df.select_dtypes(
                        include=np.number
                    ).columns
                ),

            "Categorical Features":
                len(
                    self.df.select_dtypes(
                        exclude=np.number
                    ).columns
                )

        }

        return pd.DataFrame(
            report.items(),
            columns=["Metric", "Value"]
        )
