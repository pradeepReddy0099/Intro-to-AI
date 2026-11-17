"""Compare a small fixed set of models; pick the winner by cross-validation."""
from __future__ import annotations

import pandas as pd

from agent.analysis import _round as _safe_float
from agent.guardrails import RefusalError, check_prediction_ready


def run_prediction(frame: pd.DataFrame, target: str, config) -> dict:
    """Select and evaluate models for `target`. Raises RefusalError if not defensible."""
    # scikit-learn is only needed when a prediction is actually run, so it
    # is imported here instead of at module level: importing the package
    # (and the Streamlit app) stays fast for runs without a target.
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    task_type = check_prediction_ready(frame, target, config)

    feature_columns = [c for c in frame.columns if c != target]
    numeric_features = [
        c for c in feature_columns if pd.api.types.is_numeric_dtype(frame[c])
    ]
    categorical_features = [c for c in feature_columns if c not in numeric_features]
    usable_categorical = []
    dropped_features = []
    for column in categorical_features:
        if pd.api.types.is_datetime64_any_dtype(
            frame[column]
        ) or frame[column].nunique(dropna=True) > 50:
            dropped_features.append(column)
        else:
            usable_categorical.append(column)
    categorical_features = usable_categorical

    if not numeric_features and not categorical_features:
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

    candidates, cv, metric = _model_setup(task_type, config)

    scale_models = {"Logistic Regression", "Ridge Regression"}
    model_results = []
    fitted_pipelines = {}
    for name, model in candidates:
        steps = [("preprocess", preprocessor)]
        if name in scale_models:
            steps.append(("scale", StandardScaler()))
        pipeline = Pipeline(steps + [("model", model)])
        cv_scores = cross_val_score(
            pipeline, train_x, train_y, cv=cv, scoring=metric
        )
        pipeline.fit(train_x, train_y)
        fitted_pipelines[name] = pipeline
        test_score = float(pipeline.score(test_x, test_y))
        model_results.append(
            {
                "model": name,
                "cv_mean": _safe_float(np.mean(cv_scores)),
                "cv_std": _safe_float(np.std(cv_scores)),
                "test_score": _safe_float(test_score),
            }
        )

    best = max(model_results, key=lambda item: item["cv_mean"])
    feature_importances = _feature_importances(best["model"], fitted_pipelines)

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


def _model_setup(task_type: str, config):
    """Candidate models, cross-validation splitter, and metric for one task type.

    The two task types share the same four-model shape (dummy baseline,
    linear model, random forest, gradient boosting); only the model classes,
    the linear model's name, the splitter, and the metric differ.
    """
    # Imported here, like the rest of scikit-learn in this module, so the
    # package stays fast to import for runs without a target.
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold

    seed = config.random_seed
    if task_type == "classification":
        dummy = DummyClassifier(strategy="stratified", random_state=seed)
        linear_name, linear = "Logistic Regression", LogisticRegression(max_iter=1000)
        forest = RandomForestClassifier(n_estimators=200, random_state=seed)
        boosting = GradientBoostingClassifier(random_state=seed)
        cv = StratifiedKFold(
            n_splits=config.cv_folds, shuffle=True, random_state=seed
        )
        metric = "accuracy"
    else:
        dummy = DummyRegressor(strategy="mean")
        linear_name, linear = "Ridge Regression", Ridge()
        forest = RandomForestRegressor(n_estimators=200, random_state=seed)
        boosting = GradientBoostingRegressor(random_state=seed)
        cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
        metric = "r2"
    candidates = [
        ("Dummy (baseline)", dummy),
        (linear_name, linear),
        ("Random Forest", forest),
        ("Gradient Boosting", boosting),
    ]
    return candidates, cv, metric


def _feature_importances(best_name, fitted_pipelines) -> list:
    """Top 10 feature importances for tree models; empty list otherwise."""
    if "Forest" not in best_name and "Boosting" not in best_name:
        return []
    try:
        pipeline = fitted_pipelines[best_name]
        importances = pipeline.named_steps["model"].feature_importances_
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        ranked = sorted(
            zip(names, importances), key=lambda pair: pair[1], reverse=True
        )
        return [
            {"feature": str(name), "importance": _safe_float(value)}
            for name, value in ranked[:10]
        ]
    except Exception:
        return []
