"""Model construction and hyperparameter tuning."""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": np.arange(10, 1000, 50),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2),
}


def build_models(random_seed: int) -> dict[str, ClassifierMixin]:
    """Build the baseline classifiers compared before hyperparameter tuning."""
    return {
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=random_seed),
    }


def tune_random_forest(X_train, y_train, random_seed: int, cv: int = 5, n_iter: int = 20) -> RandomizedSearchCV:
    """Run RandomizedSearchCV over a RandomForestClassifier and return the fitted search object."""
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_seed),
        param_distributions=RANDOM_FOREST_PARAM_GRID,
        cv=cv,
        n_iter=n_iter,
        verbose=True,
        random_state=random_seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search
