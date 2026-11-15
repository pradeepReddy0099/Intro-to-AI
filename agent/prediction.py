"""Prediction: local, bounded AutoML-style model selection.

A small fixed set of scikit-learn models is compared with cross-validation
under a fixed seed. No external service is called and no model is
downloaded. The stage runs only when the user names a target column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agent.guardrails import check_prediction_ready


def run_prediction(frame: pd.DataFrame, target: str, config) -> dict:
    """Select and evaluate models for `target`. Raises RefusalError if not defensible."""
    task_type = check_prediction_ready(frame, target, config)

    feature_columns = [c for c in frame.columns if c != target]
    numeric_features = [
        c for c in feature_columns if pd.api.types.is_numeric_dtype(frame[c])
    ]
    categorical_features = [c for c in feature_columns if c not in numeric_features]
    # Datetime and high-cardinality text columns are not usable as features
    # without open-ended feature engineering, so they are excluded by rule.
    usable_categorical = []
    dropped_features = []
    for column in categorical_features:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            dropped_features.append(column)
        elif frame[column].nunique(dropna=True) > 50:
            dropped_features.append(column)
        else:
            usable_categorical.append(column)
    categorical_features = usable_categorical

    if not numeric_features and not categorical_features:
        from agent.guardrails import RefusalError

        raise RefusalError(
            "No usable feature columns remain after excluding the target, "
            "date columns, and high-cardinality text columns."
        )

    features = frame[numeric_features + categorical_features]
    labels = frame[target]
    if task_type == "classification":
        labels = labels.astype(str)

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=labels if task_type == "classification" else None,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        sparse_threshold=0,
    )

    if task_type == "classification":
        candidates = [
            ("Dummy (baseline)", DummyClassifier(strategy="stratified", random_state=config.random_seed)),
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=200, random_state=config.random_seed
                ),
            ),
            (
                "Gradient Boosting",
                GradientBoostingClassifier(random_state=config.random_seed),
            ),
        ]
        cv = StratifiedKFold(
            n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed
        )
        metric = "accuracy"
    else:
        candidates = [
            ("Dummy (baseline)", DummyRegressor(strategy="mean")),
            ("Ridge Regression", Ridge()),
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=200, random_state=config.random_seed
                ),
            ),
            (
                "Gradient Boosting",
                GradientBoostingRegressor(random_state=config.random_seed),
            ),
        ]
        cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)
        metric = "r2"

    model_results = []
    for name, model in candidates:
        steps = [("preprocess", preprocessor)]
        if "Dummy" not in name and task_type == "classification" and name == "Logistic Regression":
            steps.append(("scale", StandardScaler()))
        if "Dummy" not in name and task_type == "regression" and name == "Ridge Regression":
            steps.append(("scale", StandardScaler()))
        pipeline = Pipeline(steps + [("model", model)])
        cv_scores = cross_val_score(
            pipeline, train_x, train_y, cv=cv, scoring=metric
        )
        pipeline.fit(train_x, train_y)
        test_score = float(pipeline.score(test_x, test_y))
        model_results.append(
            {
                "model": name,
                "cv_mean": round(float(np.mean(cv_scores)), 4),
                "cv_std": round(float(np.std(cv_scores)), 4),
                "test_score": round(test_score, 4),
            }
        )

    best = max(model_results, key=lambda item: item["cv_mean"])
    feature_importances = _feature_importances(
        best["model"], candidates, preprocessor, train_x, train_y, numeric_features, categorical_features
    )

    notes = []
    if dropped_features:
        notes.append(
            "Excluded from features by rule (dates / high-cardinality text): "
            + ", ".join(dropped_features)
            + "."
        )
    return {
        "task_type": task_type,
        "target": target,
        "metric": metric,
        "models": model_results,
        "best_model": best["model"],
        "best_cv_score": best["cv_mean"],
        "best_test_score": best["test_score"],
        "feature_importances": feature_importances,
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "notes": notes,
    }


def _feature_importances(best_name, candidates, preprocessor, train_x, train_y, numeric_features, categorical_features) -> list:
    """Top 10 feature importances for tree models; empty list otherwise."""
    if "Forest" not in best_name and "Boosting" not in best_name:
        return []
    try:
        model = dict(candidates)[best_name]
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(train_x, train_y)
        importances = pipeline.named_steps["model"].feature_importances_
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        ranked = sorted(
            zip(names, importances), key=lambda pair: pair[1], reverse=True
        )
        return [
            {"feature": str(name), "importance": round(float(value), 4)}
            for name, value in ranked[:10]
        ]
    except Exception:
        return []
