from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import pandas as pd
import yfinance as yf


DATA_DIR = Path("artifacts") / "data"


@dataclass
class StockDataConfig:
    ticker: str
    period: str = "10y"
    interval: str = "1d"
    auto_adjust: bool = False
    actions: bool = True


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def download_stock_data(config: StockDataConfig, save: bool = True) -> pd.DataFrame:
    """
    Download OHLCV data and corporate actions from Yahoo Finance.

    Notes:
    - `auto_adjust=False` keeps raw OHLC prices and exposes split/dividend columns.
    - We clean missing values conservatively and keep corporate action columns because
      they may be useful later for diagnostics or extra features.
    """
    df = pd.DataFrame()
    for _ in range(3):
        df = yf.download(
            tickers=config.ticker,
            period=config.period,
            interval=config.interval,
            auto_adjust=config.auto_adjust,
            actions=config.actions,
            progress=False,
            threads=False,
        )
        if not df.empty:
            break
        time.sleep(2)

    if df.empty:
        ticker = yf.Ticker(config.ticker)
        df = ticker.history(
            period=config.period,
            interval=config.interval,
            auto_adjust=config.auto_adjust,
            actions=config.actions,
        )

    if df.empty:
        raise ValueError(f"No data returned for ticker '{config.ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index().rename(columns=str.title)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).set_index("Date")

    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    if "Dividends" not in df.columns:
        df["Dividends"] = 0.0
    if "Stock Splits" not in df.columns:
        df["Stock Splits"] = 0.0

    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    df[price_cols] = df[price_cols].ffill().bfill()
    df["Volume"] = df["Volume"].fillna(0.0)
    df["Dividends"] = df["Dividends"].fillna(0.0)
    df["Stock Splits"] = df["Stock Splits"].fillna(0.0)
    df = df.dropna(subset=["Close"])

    if save:
        output_path = ensure_data_dir() / f"{config.ticker.upper()}_{config.period}_{config.interval}.csv"
        df.to_csv(output_path)

    return df


def load_local_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    return df


def get_stock_data(
    ticker: str,
    period: str = "10y",
    interval: str = "1d",
    local_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    if local_path is not None:
        return load_local_data(local_path)
    config = StockDataConfig(ticker=ticker, period=period, interval=interval)
    return download_stock_data(config)
