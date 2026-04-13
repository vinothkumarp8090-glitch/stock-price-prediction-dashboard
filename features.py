from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr_14",
    "obv",
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "volatility_10",
    "volatility_20",
    "rolling_mean_5",
    "rolling_mean_10",
    "rolling_std_5",
    "rolling_std_10",
    "high_low_spread",
    "close_open_spread",
    "volume_change",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    return (direction * df["Volume"]).cumsum()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sma_20"] = out["Close"].rolling(window=20, min_periods=20).mean()
    out["sma_50"] = out["Close"].rolling(window=50, min_periods=50).mean()
    out["sma_200"] = out["Close"].rolling(window=200, min_periods=200).mean()

    out["ema_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["Close"].ewm(span=26, adjust=False).mean()

    out["rsi_14"] = _rsi(out["Close"], period=14)

    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["bb_mid"] = out["Close"].rolling(window=20, min_periods=20).mean()
    bb_std = out["Close"].rolling(window=20, min_periods=20).std()
    out["bb_upper"] = out["bb_mid"] + 2 * bb_std
    out["bb_lower"] = out["bb_mid"] - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]

    out["atr_14"] = _atr(out, period=14)
    out["obv"] = _obv(out)

    for lag in [1, 2, 3, 5, 10]:
        out[f"ret_{lag}"] = out["Close"].pct_change(lag)

    out["volatility_10"] = out["Close"].pct_change().rolling(10, min_periods=10).std()
    out["volatility_20"] = out["Close"].pct_change().rolling(20, min_periods=20).std()
    out["rolling_mean_5"] = out["Close"].rolling(5, min_periods=5).mean()
    out["rolling_mean_10"] = out["Close"].rolling(10, min_periods=10).mean()
    out["rolling_std_5"] = out["Close"].rolling(5, min_periods=5).std()
    out["rolling_std_10"] = out["Close"].rolling(10, min_periods=10).std()
    out["high_low_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["close_open_spread"] = (out["Close"] - out["Open"]) / out["Open"]
    out["volume_change"] = out["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    out["target_return"] = out["Close"].pct_change().shift(-1)
    out["target_direction"] = (out["target_return"] > 0).astype(int)
    out["target_close"] = out["Close"].shift(-1)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().copy()
    return out


@dataclass
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    current_close: np.ndarray
    next_close: np.ndarray
    feature_names: List[str]


def create_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    look_back: int = 60,
) -> SequenceDataset:
    values = df.loc[:, feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    dates = df.index.to_numpy()
    current_close = df["Close"].values.astype(np.float32)
    next_close = df["target_close"].values.astype(np.float32)

    X_list, y_list, date_list, current_close_list, next_close_list = [], [], [], [], []
    for idx in range(look_back, len(df)):
        X_list.append(values[idx - look_back : idx])
        y_list.append(target[idx])
        date_list.append(dates[idx])
        current_close_list.append(current_close[idx])
        next_close_list.append(next_close[idx])

    return SequenceDataset(
        X=np.array(X_list, dtype=np.float32),
        y=np.array(y_list, dtype=np.float32),
        dates=np.array(date_list),
        current_close=np.array(current_close_list, dtype=np.float32),
        next_close=np.array(next_close_list, dtype=np.float32),
        feature_names=list(feature_cols),
    )


def fit_transform_sequences(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit the scaler on train only, then transform both train and test.
    We flatten the time dimension before scaling so each feature uses a single
    global train-only mean/std across all timesteps.
    """
    n_train, look_back, n_features = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d).reshape(n_train, look_back, n_features)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(n_test, look_back, n_features)
    return X_train_scaled, X_test_scaled, scaler


def inverse_return_to_price(current_close: np.ndarray, predicted_return: np.ndarray) -> np.ndarray:
    return current_close * (1.0 + predicted_return)
