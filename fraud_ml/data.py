"""Loading the training and holdout datasets."""

import pandas as pd

from fraud_ml import config


def load_dataset(url: str, sep: str = config.CSV_SEPARATOR, encoding: str = config.CSV_ENCODING) -> pd.DataFrame:
    """Load a pipe-separated CSV dataset from a local path or remote URL."""
    return pd.read_csv(url, sep=sep, encoding=encoding)


def load_training_data() -> pd.DataFrame:
    """Load the training dataset from `config.TRAIN_DATA_URL`."""
    return load_dataset(config.TRAIN_DATA_URL)


def load_holdout_data() -> pd.DataFrame:
    """Load the unlabeled holdout dataset from `config.HOLDOUT_DATA_URL`."""
    return load_dataset(config.HOLDOUT_DATA_URL)
