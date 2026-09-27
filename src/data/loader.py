"""
Dataset loading functions.
"""

import pandas as pd


def load_train_data():

    return pd.read_csv("../data/raw/train.csv")


def load_test_data():

    return pd.read_csv("../data/raw/test.csv")