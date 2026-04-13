from __future__ import annotations

from pathlib import Path

import backtrader as bt
import pandas as pd


RESULTS_DIR = Path("artifacts") / "results"


class PredictionData(bt.feeds.PandasData):
    lines = ("signal",)
    params = (("signal", -1),)


class MLSignalStrategy(bt.Strategy):
    params = dict(stop_loss=0.05)

    def __init__(self):
        self.signal_line = self.datas[0].signal
        self.entry_price = None

    def next(self):
        signal = int(self.signal_line[0])
        close = float(self.data.close[0])

        if not self.position and signal > 0:
            self.buy()
            self.entry_price = close
            return

        if self.position:
            if self.entry_price is not None and close <= self.entry_price * (1 - self.params.stop_loss):
                self.close()
                self.entry_price = None
                return
            if signal <= 0:
                self.close()
                self.entry_price = None


def run_backtest(
    ticker: str,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    stop_loss: float = 0.05,
):
    results_path = RESULTS_DIR / f"{ticker}_walk_forward_predictions.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found at {results_path}. Run train.py first."
        )

    df = pd.read_csv(results_path, parse_dates=["date"]).set_index("date")
    bt_df = pd.DataFrame(
        {
            "open": df["current_close"],
            "high": df[["current_close", "actual_next_close"]].max(axis=1),
            "low": df[["current_close", "actual_next_close"]].min(axis=1),
            "close": df["actual_next_close"],
            "volume": 1_000_000,
            "openinterest": 0,
            "signal": df["predicted_direction"],
        }
    )

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addstrategy(MLSignalStrategy, stop_loss=stop_loss)
    cerebro.adddata(PredictionData(dataname=bt_df))
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    strategy = cerebro.run()[0]
    analyzers = strategy.analyzers
    output = {
        "final_portfolio_value": cerebro.broker.getvalue(),
        "sharpe_ratio": analyzers.sharpe.get_analysis().get("sharperatio"),
        "max_drawdown_pct": analyzers.drawdown.get_analysis().max.drawdown,
        "total_return": analyzers.returns.get_analysis().get("rtot"),
    }
    return output


if __name__ == "__main__":
    print(run_backtest("AAPL"))
