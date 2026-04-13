from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from data import get_stock_data
from features import FEATURE_COLUMNS, add_technical_indicators, create_sequences, fit_transform_sequences, inverse_return_to_price
from model import get_model_builder


ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = ARTIFACT_DIR / "models"
RESULTS_DIR = ARTIFACT_DIR / "results"
PLOT_DIR = ARTIFACT_DIR / "plots"


@dataclass
class TrainConfig:
    ticker: str = "AAPL"
    period: str = "10y"
    interval: str = "1d"
    look_back: int = 60
    initial_train_size: int = 500
    test_window_size: int = 63
    retrain_every: int = 63
    epochs: int = 50
    batch_size: int = 32
    regression_model_name: str = "lstm"
    classification_model_name: str = "bilstm_attention"
    mode: str = "fast_ml"
    random_seed: int = 42


def ensure_dirs() -> None:
    for directory in [MODEL_DIR, RESULTS_DIR, PLOT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    import random
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def walk_forward_splits(
    n_samples: int,
    initial_train_size: int,
    test_window_size: int,
    retrain_every: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    train_end = initial_train_size
    while train_end < n_samples:
        test_end = min(train_end + test_window_size, n_samples)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        if len(test_idx) == 0:
            break
        splits.append((train_idx, test_idx))
        train_end += retrain_every
    return splits


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def evaluate_classification(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def trading_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    strategy_returns = strategy_returns.fillna(0.0)
    benchmark_returns = benchmark_returns.fillna(0.0)

    cumulative_return = float((1 + strategy_returns).prod() - 1)
    annualized_sharpe = 0.0
    if strategy_returns.std() > 0:
        annualized_sharpe = float((strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252))

    equity_curve = (1 + strategy_returns).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_drawdown = float(drawdown.min())

    benchmark_cum_return = float((1 + benchmark_returns).prod() - 1)

    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": annualized_sharpe,
        "max_drawdown": max_drawdown,
        "buy_and_hold_return": benchmark_cum_return,
    }


def plot_predictions(results_df: pd.DataFrame, ticker: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df["date"], results_df["actual_next_close"], label="Actual Next Close")
    ax.plot(results_df["date"], results_df["predicted_next_close"], label="Predicted Next Close")
    ax.set_title(f"{ticker} Actual vs Predicted Next-Day Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    path = PLOT_DIR / f"{ticker}_actual_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_equity_curve(results_df: pd.DataFrame, ticker: str) -> Tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df["date"], results_df["strategy_equity"], label="Strategy")
    ax.plot(results_df["date"], results_df["buy_hold_equity"], label="Buy & Hold")
    ax.set_title(f"{ticker} Equity Curve")
    ax.legend()
    fig.autofmt_xdate()
    equity_path = PLOT_DIR / f"{ticker}_equity_curve.png"
    fig.tight_layout()
    fig.savefig(equity_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(results_df["date"], results_df["drawdown"], 0, color="tomato", alpha=0.35)
    ax.set_title(f"{ticker} Drawdown")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    drawdown_path = PLOT_DIR / f"{ticker}_drawdown.png"
    fig.tight_layout()
    fig.savefig(drawdown_path, dpi=150)
    plt.close(fig)
    return equity_path, drawdown_path


def train_and_evaluate(config: TrainConfig) -> dict:
    if config.mode == "fast_ml":
        return train_and_evaluate_fast(config)

    ensure_dirs()
    set_seed(config.random_seed)

    raw_df = get_stock_data(ticker=config.ticker, period=config.period, interval=config.interval)
    feature_df = add_technical_indicators(raw_df)

    regression_data = create_sequences(feature_df, FEATURE_COLUMNS, "target_return", look_back=config.look_back)
    classification_data = create_sequences(feature_df, FEATURE_COLUMNS, "target_direction", look_back=config.look_back)

    if len(regression_data.X) < config.initial_train_size + 10:
        raise ValueError("Not enough samples after feature engineering. Reduce look_back or initial_train_size.")

    splits = walk_forward_splits(
        n_samples=len(regression_data.X),
        initial_train_size=config.initial_train_size,
        test_window_size=config.test_window_size,
        retrain_every=config.retrain_every,
    )

    regression_builder = get_model_builder(config.regression_model_name)
    classification_builder = get_model_builder(config.classification_model_name)

    regression_preds, regression_truth = [], []
    classification_probs, classification_truth = [], []
    dates, current_close, actual_next_close = [], [], []

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    regression_model = None
    classification_model = None
    last_scaler = None

    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train_raw = regression_data.X[train_idx]
        X_test_raw = regression_data.X[test_idx]
        X_train_scaled, X_test_scaled, scaler = fit_transform_sequences(X_train_raw, X_test_raw)

        y_train_reg = regression_data.y[train_idx]
        y_test_reg = regression_data.y[test_idx]
        y_train_clf = classification_data.y[train_idx]
        y_test_clf = classification_data.y[test_idx]

        regression_model = regression_builder(input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), task_type="regression")
        regression_model.fit(
            X_train_scaled,
            y_train_reg,
            validation_split=0.1,
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            shuffle=False,
            verbose=0,
        )

        classification_model = classification_builder(
            input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
            task_type="classification",
        )
        classification_model.fit(
            X_train_scaled,
            y_train_clf,
            validation_split=0.1,
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            shuffle=False,
            verbose=0,
        )

        reg_pred = regression_model.predict(X_test_scaled, verbose=0).reshape(-1)
        clf_prob = classification_model.predict(X_test_scaled, verbose=0).reshape(-1)

        regression_preds.extend(reg_pred.tolist())
        regression_truth.extend(y_test_reg.tolist())
        classification_probs.extend(clf_prob.tolist())
        classification_truth.extend(y_test_clf.tolist())
        dates.extend(regression_data.dates[test_idx].tolist())
        current_close.extend(regression_data.current_close[test_idx].tolist())
        actual_next_close.extend(regression_data.next_close[test_idx].tolist())
        last_scaler = scaler

        print(
            f"Completed split {split_id}/{len(splits)} "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

    results_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "current_close": np.array(current_close, dtype=float),
            "actual_next_close": np.array(actual_next_close, dtype=float),
            "actual_return": np.array(regression_truth, dtype=float),
            "predicted_return": np.array(regression_preds, dtype=float),
            "predicted_prob_up": np.array(classification_probs, dtype=float),
            "actual_direction": np.array(classification_truth, dtype=int),
        }
    ).sort_values("date")

    results_df["predicted_direction"] = (results_df["predicted_prob_up"] >= 0.5).astype(int)
    results_df["predicted_next_close"] = inverse_return_to_price(
        results_df["current_close"].values,
        results_df["predicted_return"].values,
    )
    results_df["market_return"] = results_df["actual_return"]
    results_df["strategy_position"] = results_df["predicted_direction"]
    results_df["strategy_return"] = results_df["strategy_position"] * results_df["market_return"]
    results_df["strategy_equity"] = (1 + results_df["strategy_return"]).cumprod()
    results_df["buy_hold_equity"] = (1 + results_df["market_return"]).cumprod()
    results_df["drawdown"] = results_df["strategy_equity"] / results_df["strategy_equity"].cummax() - 1

    regression_metrics = evaluate_regression(results_df["actual_return"].values, results_df["predicted_return"].values)
    classification_metrics = evaluate_classification(
        results_df["actual_direction"].values,
        results_df["predicted_prob_up"].values,
    )
    trade_metrics = trading_metrics(results_df["strategy_return"], results_df["market_return"])

    regression_model_path = MODEL_DIR / f"{config.ticker}_regression_{config.regression_model_name}.keras"
    classification_model_path = MODEL_DIR / f"{config.ticker}_classification_{config.classification_model_name}.keras"
    results_path = RESULTS_DIR / f"{config.ticker}_walk_forward_predictions.csv"
    metrics_path = RESULTS_DIR / f"{config.ticker}_metrics.json"
    scaler_path = MODEL_DIR / f"{config.ticker}_feature_scaler.npz"

    if regression_model is not None and classification_model is not None and last_scaler is not None:
        regression_model.save(regression_model_path)
        classification_model.save(classification_model_path)
        np.savez(
            scaler_path,
            mean=last_scaler.mean_,
            scale=last_scaler.scale_,
            var=last_scaler.var_,
            n_features_in=last_scaler.n_features_in_,
        )

    results_df.to_csv(results_path, index=False)
    plot_predictions(results_df, config.ticker)
    plot_equity_curve(results_df, config.ticker)

    output = {
        "config": asdict(config),
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_metrics,
        "trading_metrics": trade_metrics,
        "paths": {
            "regression_model": str(regression_model_path),
            "classification_model": str(classification_model_path),
            "results_csv": str(results_path),
            "metrics_json": str(metrics_path),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    return output


def train_and_evaluate_fast(config: TrainConfig) -> dict:
    ensure_dirs()
    set_seed(config.random_seed)

    raw_df = get_stock_data(ticker=config.ticker, period=config.period, interval=config.interval)
    feature_df = add_technical_indicators(raw_df).copy()

    X_all = feature_df.loc[:, FEATURE_COLUMNS].astype(float).values
    y_reg_all = feature_df["target_return"].astype(float).values
    y_clf_all = feature_df["target_direction"].astype(int).values
    dates_all = feature_df.index.to_numpy()
    current_close_all = feature_df["Close"].astype(float).values
    next_close_all = feature_df["target_close"].astype(float).values

    if len(X_all) < config.initial_train_size + 10:
        raise ValueError("Not enough samples after feature engineering. Reduce initial_train_size.")

    splits = walk_forward_splits(
        n_samples=len(X_all),
        initial_train_size=config.initial_train_size,
        test_window_size=config.test_window_size,
        retrain_every=config.retrain_every,
    )

    regression_preds, regression_truth = [], []
    classification_probs, classification_truth = [], []
    dates, current_close, actual_next_close = [], [], []

    reg_model = None
    clf_model = None

    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train_reg = y_reg_all[train_idx]
        y_test_reg = y_reg_all[test_idx]
        y_train_clf = y_clf_all[train_idx]
        y_test_clf = y_clf_all[test_idx]

        reg_model = RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=5,
            random_state=config.random_seed,
            n_jobs=-1,
        )
        clf_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=5,
            random_state=config.random_seed,
            n_jobs=-1,
        )

        reg_model.fit(X_train, y_train_reg)
        clf_model.fit(X_train, y_train_clf)

        reg_pred = reg_model.predict(X_test)
        clf_prob = clf_model.predict_proba(X_test)[:, 1]

        regression_preds.extend(reg_pred.tolist())
        regression_truth.extend(y_test_reg.tolist())
        classification_probs.extend(clf_prob.tolist())
        classification_truth.extend(y_test_clf.tolist())
        dates.extend(dates_all[test_idx].tolist())
        current_close.extend(current_close_all[test_idx].tolist())
        actual_next_close.extend(next_close_all[test_idx].tolist())

        print(
            f"Completed split {split_id}/{len(splits)} "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

    results_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "current_close": np.array(current_close, dtype=float),
            "actual_next_close": np.array(actual_next_close, dtype=float),
            "actual_return": np.array(regression_truth, dtype=float),
            "predicted_return": np.array(regression_preds, dtype=float),
            "predicted_prob_up": np.array(classification_probs, dtype=float),
            "actual_direction": np.array(classification_truth, dtype=int),
        }
    ).sort_values("date")

    results_df["predicted_direction"] = (results_df["predicted_prob_up"] >= 0.5).astype(int)
    results_df["predicted_next_close"] = inverse_return_to_price(
        results_df["current_close"].values,
        results_df["predicted_return"].values,
    )
    results_df["market_return"] = results_df["actual_return"]
    results_df["strategy_position"] = results_df["predicted_direction"]
    results_df["strategy_return"] = results_df["strategy_position"] * results_df["market_return"]
    results_df["strategy_equity"] = (1 + results_df["strategy_return"]).cumprod()
    results_df["buy_hold_equity"] = (1 + results_df["market_return"]).cumprod()
    results_df["drawdown"] = results_df["strategy_equity"] / results_df["strategy_equity"].cummax() - 1

    regression_metrics = evaluate_regression(results_df["actual_return"].values, results_df["predicted_return"].values)
    classification_metrics = evaluate_classification(
        results_df["actual_direction"].values,
        results_df["predicted_prob_up"].values,
    )
    trade_metrics = trading_metrics(results_df["strategy_return"], results_df["market_return"])

    regression_model_path = MODEL_DIR / f"{config.ticker}_regression_random_forest.pkl"
    classification_model_path = MODEL_DIR / f"{config.ticker}_classification_random_forest.pkl"
    results_path = RESULTS_DIR / f"{config.ticker}_walk_forward_predictions.csv"
    metrics_path = RESULTS_DIR / f"{config.ticker}_metrics.json"

    if reg_model is not None and clf_model is not None:
        with open(regression_model_path, "wb") as fp:
            pickle.dump(reg_model, fp)
        with open(classification_model_path, "wb") as fp:
            pickle.dump(clf_model, fp)

    results_df.to_csv(results_path, index=False)
    plot_predictions(results_df, config.ticker)
    plot_equity_curve(results_df, config.ticker)

    output = {
        "config": asdict(config),
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_metrics,
        "trading_metrics": trade_metrics,
        "paths": {
            "regression_model": str(regression_model_path),
            "classification_model": str(classification_model_path),
            "results_csv": str(results_path),
            "metrics_json": str(metrics_path),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    return output


def load_saved_models(config: TrainConfig):
    regression_model_path = MODEL_DIR / f"{config.ticker}_regression_{config.regression_model_name}.keras"
    classification_model_path = MODEL_DIR / f"{config.ticker}_classification_{config.classification_model_name}.keras"
    return load_model(regression_model_path), load_model(classification_model_path)


if __name__ == "__main__":
    config = TrainConfig()
    results = train_and_evaluate(config)
    print(json.dumps(results, indent=2))
