"""Training pipeline with blocked/purged CV and generational artifacts."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.core.io import ensure_directory, save_json
from app.core.registry import registry
from app.core.seeds import seed_everything


@dataclass
class CVResult:
    params: Dict[str, float]
    scores: List[float]
    mean_score: float


class BlockedTimeSeriesSplit:
    """Simple blocked CV for time series."""

    def __init__(self, n_splits: int) -> None:
        self.n_splits = n_splits

    def split(self, X: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_end = train_end + fold_size
            yield np.arange(train_end), np.arange(train_end, test_end)


class PurgedTimeSeriesSplit:
    """Purged CV removing overlapping observations."""

    def __init__(self, n_splits: int, purge: int = 5) -> None:
        self.n_splits = n_splits
        self.purge = purge

    def split(self, X: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.purge
            test_end = test_start + fold_size
            yield np.arange(train_end), np.arange(test_start, test_end)


MODEL_GRID: Dict[str, Dict[str, List[float]]] = {
    "logistic_regression": {
        "model__C": [0.1, 1.0, 10.0],
    },
    "random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__n_estimators": [100, 200],
    },
}


MODEL_FACTORY = {
    "logistic_regression": LogisticRegression(
        solver="lbfgs", max_iter=1000
    ),
    "random_forest": RandomForestClassifier(random_state=settings.random_seed),
    "gradient_boosting": GradientBoostingClassifier(random_state=settings.random_seed),
}


def _build_pipeline(model_name: str, params: Dict[str, float]) -> Pipeline:
    model = MODEL_FACTORY[model_name]
    pipeline = Pipeline(
        steps=[("scaler", StandardScaler()), ("model", model)]
    )
    pipeline.set_params(**params)
    return pipeline


def _cross_validate(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_scheme: str,
    n_splits: int,
    max_iter: int,
) -> Tuple[Pipeline, CVResult, List[float]]:
    seed_everything()
    cv_cls = BlockedTimeSeriesSplit if cv_scheme == "blocked" else PurgedTimeSeriesSplit
    splitter = cv_cls(n_splits=n_splits)
    best_score = -np.inf
    best_model: Pipeline | None = None
    best_result: CVResult | None = None
    learning_curve: List[float] = []
    for params in ParameterGrid(MODEL_GRID[model_name]):
        scores = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline = _build_pipeline(model_name, params)
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, proba)
            scores.append(score)
        mean_score = float(np.mean(scores))
        learning_curve.append(mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_model = _build_pipeline(model_name, params)
            best_model.fit(X, y)
            best_result = CVResult(params=params, scores=scores, mean_score=mean_score)
        if len(learning_curve) >= max_iter:
            break
    if best_model is None or best_result is None:
        raise RuntimeError("No valid model found during CV")
    return best_model, best_result, learning_curve


def _save_learning_curve(curve: List[float], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(curve) + 1), curve, marker="o")
    ax.set_title("Learning Curve (Mean ROC AUC)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ROC AUC")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(fig)


def _save_confusion_matrix(model: Pipeline, X: pd.DataFrame, y: pd.Series, output_dir: Path) -> None:
    preds = model.predict(X)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y, preds, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _save_feature_importance(model: Pipeline, feature_names: List[str], output_dir: Path) -> Dict[str, float]:
    ensure_directory(output_dir)
    if hasattr(model.named_steps["model"], "feature_importances_"):
        importance = model.named_steps["model"].feature_importances_
    elif hasattr(model.named_steps["model"], "coef_"):
        importance = np.abs(model.named_steps["model"].coef_).ravel()
    else:
        importance = np.ones(len(feature_names)) / len(feature_names)
    series = pd.Series(importance, index=feature_names, name="importance")
    fig, ax = plt.subplots(figsize=(8, 4))
    series.sort_values(ascending=False).head(20).plot(kind="bar", ax=ax)
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    series.to_frame().reset_index().rename(columns={"index": "feature"}).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    return series.to_dict()


def train_generation(
    generation_id: str,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "gradient_boosting",
    cv_scheme: str | None = None,
    max_iter: int = 10,
) -> Path:
    cv_scheme = cv_scheme or settings.training.default_cv
    models_dir = ensure_directory(settings.storage.models_dir / generation_id)
    start = time.time()
    model, result, curve = _cross_validate(
        model_name=model_name,
        X=X,
        y=y,
        cv_scheme=cv_scheme,
        n_splits=settings.training.n_splits,
        max_iter=max_iter,
    )
    duration = time.time() - start
    model_path = models_dir / "model.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "cv_scheme": cv_scheme,
        "scores": result.scores,
        "mean_score": result.mean_score,
        "accuracy": accuracy_score(y, model.predict(X)),
        "f1": f1_score(y, model.predict(X), average="macro"),
    }
    save_json(models_dir / "cv_metrics.json", metrics)
    _save_learning_curve(curve, models_dir)
    _save_confusion_matrix(model, X, y, models_dir)
    importance = _save_feature_importance(model, list(X.columns), models_dir)

    model_card = {
        "model_type": model_name,
        "params": result.params,
        "features": list(X.columns),
        "dataset_rows": len(X),
        "generation_id": generation_id,
        "seed": settings.random_seed,
        "train_time_sec": duration,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_json(models_dir / "model_card.json", model_card)

    registry.register_generation(
        generation_id,
        {
            "model_path": str(model_path),
            "model_type": model_name,
            "metrics": metrics,
        },
    )

    drift_payload = {
        "feature_importance": importance,
    }
    save_json(models_dir / "feature_importance.json", drift_payload)
    return model_path
