"""
Utility functions used across the project.

Author:
Gyanvi

Project:
Health Condition Prediction
"""

import pandas as pd


def print_section(title: str):
    """
    Print a formatted section header.
    """

    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)


def dataset_shape(df: pd.DataFrame):
    """
    Print dataset shape.
    """

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")


def memory_usage(df: pd.DataFrame):
    """
    Print dataset memory usage.
    """

    memory = df.memory_usage(deep=True).sum() / 1024**2

    print(f"Memory Usage : {memory:.2f} MB")