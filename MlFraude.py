"""Credit card fraud detection pipeline.

Trains and evaluates baseline classifiers, tunes a Random Forest via
RandomizedSearchCV, and writes predictions for the holdout test set.
Pipeline logic lives in the `fraud_ml` package; this script only orchestrates it.
"""

import json

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from fraud_ml import config, data, evaluation, features, modeling


def build_features(df):
    """Apply datetime and age feature engineering to a dataframe already stripped of PII columns."""
    features.add_datetime_features(df)
    features.add_age_feature(df)
    return df


def main() -> None:
    df = data.load_training_data()
    df = df.dropna(subset=config.TARGET_COLUMN)
    y = df[config.TARGET_COLUMN]

    df_train = df.drop(columns=[*config.PII_AND_HIGH_CARDINALITY_COLUMNS, config.TARGET_COLUMN])
    build_features(df_train)
    encoders = features.fit_label_encoders(df_train, config.CATEGORICAL_COLUMNS)

    X_train, X_test, y_train, y_test = train_test_split(
        df_train, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )

    imputer = SimpleImputer()
    X_train_transform = imputer.fit_transform(X_train)
    X_test_transform = imputer.transform(X_test)

    target_encoder = LabelEncoder()
    y_train_transformed = target_encoder.fit_transform(y_train)
    y_test_transform = target_encoder.transform(y_test)

    models = modeling.build_models(config.RANDOM_SEED)
    baseline_scores = evaluation.fit_and_score(
        models, X_train_transform, X_test_transform, y_train_transformed, y_test_transform
    )
    print(json.dumps(baseline_scores, indent=2))

    search = modeling.tune_random_forest(X_train_transform, y_train_transformed, config.RANDOM_SEED)
    y_pred_tuned = search.predict(X_test_transform)
    y_proba_tuned = search.predict_proba(X_test_transform)[:, 1]
    tuned_metrics = evaluation.evaluate_model(y_test_transform, y_pred_tuned, y_proba_tuned)
    tuned_metrics["best_params"] = search.best_params_
    print(json.dumps(tuned_metrics, indent=2, default=str))

    evaluation.save_metrics(
        {"baseline": baseline_scores, "tuned_random_forest": tuned_metrics}, config.METRICS_OUTPUT_PATH
    )

    # Holdout inference: reuse the encoders and imputer fitted on training data
    # (not refit on the holdout) so train/test preprocessing stay consistent.
    df_holdout_raw = data.load_holdout_data()
    df_holdout = df_holdout_raw.drop(columns=config.PII_AND_HIGH_CARDINALITY_COLUMNS)
    build_features(df_holdout)
    features.apply_label_encoders(df_holdout, encoders, config.CATEGORICAL_COLUMNS)
    df_holdout_transform = imputer.transform(df_holdout)

    predictions = search.predict(df_holdout_transform)
    evaluation.save_submission(df_holdout_raw["trans_num"], predictions, config.SUBMISSION_OUTPUT_PATH)


if __name__ == "__main__":
    main()
