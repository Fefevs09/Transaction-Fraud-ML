"""Feature engineering and categorical encoding."""

from collections.abc import Sequence

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fraud_ml.config import UNKNOWN_CATEGORY_VALUE


def add_datetime_features(data: pd.DataFrame) -> None:
    """Extract hour/minute/seconds/year/month/day from trans_time/trans_date in place and drop the originals."""
    time_transaction = pd.to_datetime(data["trans_time"], format="%H:%M:%S")
    date_transaction = pd.to_datetime(data["trans_date"], format="%Y-%m-%d")

    data["hour"] = time_transaction.dt.hour
    data["minute"] = time_transaction.dt.minute
    data["seconds"] = time_transaction.dt.second
    data["year"] = date_transaction.dt.year
    data["month"] = date_transaction.dt.month
    data["day"] = date_transaction.dt.day
    data.drop(["trans_time", "trans_date"], axis=1, inplace=True)


def add_age_feature(data: pd.DataFrame) -> None:
    """Compute cardholder `age` from `dob` and the `year` column (in place); drop `dob`.

    Must be called after `add_datetime_features()`, which produces `year`.
    """
    year_of_birth = pd.to_datetime(data["dob"], format="%Y-%m-%d")
    data["age"] = data["year"] - year_of_birth.dt.year
    data.drop(["dob"], axis=1, inplace=True)


def fit_label_encoders(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, LabelEncoder]:
    """Fit one LabelEncoder per column in `columns` on `df`'s current values."""
    encoders = {}
    for column in columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder
    return encoders


def apply_label_encoders(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    columns: Sequence[str],
    unknown_value: int = UNKNOWN_CATEGORY_VALUE,
) -> None:
    """Encode `columns` of `df` using previously fitted `encoders` (in place).

    Categories unseen at fit time are mapped to `unknown_value` instead of
    raising `ValueError`, unlike calling `LabelEncoder.transform` directly.
    """
    for column in columns:
        encoder = encoders[column]
        class_to_code = dict(zip(encoder.classes_, range(len(encoder.classes_))))
        df[column] = df[column].map(class_to_code).fillna(unknown_value).astype(int)
