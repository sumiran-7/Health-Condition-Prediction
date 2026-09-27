"""
=========================================================
Dataset Manager

Handles saving and loading processed datasets.

Author:
Gyanvi
=========================================================
"""

from pathlib import Path
import pandas as pd


class DatasetManager:
    """
    Utility class for saving and loading processed datasets.
    """

    def __init__(self):

        project_root = Path(__file__).resolve().parents[2]

        self.artifacts_dir = (
            project_root
            / "artifacts"
            / "processed"
        )

        self.artifacts_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_processed_data(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        """
        Save processed datasets.
        """

        X_train.to_csv(
            self.artifacts_dir / "X_train.csv",
            index=False
        )

        X_test.to_csv(
            self.artifacts_dir / "X_test.csv",
            index=False
        )

        pd.DataFrame(y_train).to_csv(
            self.artifacts_dir / "y_train.csv",
            index=False
        )

        pd.DataFrame(y_test).to_csv(
            self.artifacts_dir / "y_test.csv",
            index=False
        )

        print("✅ Processed datasets saved successfully.")

    def load_processed_data(self):
        """
        Load processed datasets.
        """

        X_train = pd.read_csv(
            self.artifacts_dir / "X_train.csv"
        )

        X_test = pd.read_csv(
            self.artifacts_dir / "X_test.csv"
        )

        y_train = pd.read_csv(
            self.artifacts_dir / "y_train.csv"
        ).squeeze("columns")

        y_test = pd.read_csv(
            self.artifacts_dir / "y_test.csv"
        ).squeeze("columns")

        print("✅ Processed datasets loaded successfully.")

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )