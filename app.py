from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from presets import COMPANY_PRESETS
from backtest import run_backtest
from train import RESULTS_DIR, TrainConfig, train_and_evaluate


st.set_page_config(page_title="Quant Stock Dashboard", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(38, 57, 95, 0.28), transparent 24%),
                radial-gradient(circle at top left, rgba(15, 92, 67, 0.18), transparent 26%),
                linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
            color: #e5eefc;
        }
        header[data-testid="stHeader"] {
            background: #0b1220 !important;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }
        header[data-testid="stHeader"] * {
            color: #e5eefc !important;
        }
        [data-testid="stToolbar"] {
            right: 1rem;
        }
        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        section[data-testid="stSidebar"] * {
            color: #e5eefc !important;
        }
        section[data-testid="stSidebar"] label {
            color: #f8fafc !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(96, 165, 250, 0.18);
            border-radius: 18px;
            padding: 12px 16px;
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #e5eefc !important;
        }
        div[data-baseweb="input"] input {
            color: #0f172a !important;
            background: #f8fafc !important;
            font-weight: 700 !important;
        }
        div[data-baseweb="input"] {
            background: #f8fafc !important;
            border-radius: 14px !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            color: #0f172a !important;
            background: #f8fafc !important;
            font-weight: 700 !important;
            border-radius: 14px !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] span,
        section[data-testid="stSidebar"] div[data-baseweb="select"] div,
        section[data-testid="stSidebar"] div[data-baseweb="select"] p {
            color: #0f172a !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #0f172a !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] input {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
            fill: #0f172a !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            border-radius: 14px !important;
            font-weight: 700 !important;
            background: #2563eb !important;
            color: #eff6ff !important;
            border: 1px solid #3b82f6 !important;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background: #1d4ed8 !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stButton button[kind="secondary"]:disabled,
        section[data-testid="stSidebar"] .stButton button:disabled,
        section[data-testid="stSidebar"] .stButton button[disabled] {
            background: #475569 !important;
            color: #e2e8f0 !important;
            -webkit-text-fill-color: #e2e8f0 !important;
            opacity: 1 !important;
            border: 1px solid #64748b !important;
        }
        section[data-testid="stSidebar"] .stButton button span,
        section[data-testid="stSidebar"] .stButton button p,
        section[data-testid="stSidebar"] .stButton button div {
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(30, 41, 59, 0.92));
            border: 1px solid rgba(96, 165, 250, 0.18);
            border-radius: 24px;
            padding: 28px 32px;
            margin-bottom: 20px;
            box-shadow: 0 18px 60px rgba(2, 6, 23, 0.38);
        }
        .section-card {
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 20px;
            padding: 18px 20px 12px 20px;
            margin-bottom: 18px;
            box-shadow: 0 18px 40px rgba(2, 6, 23, 0.18);
        }
        .caption-chip {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.16);
            color: #93c5fd;
            font-size: 0.84rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .signal-up {
            color: #34d399;
            font-weight: 700;
        }
        .signal-down {
            color: #f87171;
            font-weight: 700;
        }
        .mini-note {
            color: #94a3b8;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.82);
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.14);
            padding: 10px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_results(ticker: str) -> tuple[pd.DataFrame | None, dict | None]:
    results_path = RESULTS_DIR / f"{ticker}_walk_forward_predictions.csv"
    metrics_path = RESULTS_DIR / f"{ticker}_metrics.json"

    df = None
    metrics = None
    if results_path.exists():
        df = pd.read_csv(results_path, parse_dates=["date"])
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as fp:
            metrics = json.load(fp)
    return df, metrics


def prediction_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["actual_next_close"],
            mode="lines",
            name="Actual Next Close",
            line=dict(color="#60a5fa", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["predicted_next_close"],
            mode="lines",
            name="Predicted Next Close",
            line=dict(color="#f59e0b", width=2.2),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted Price",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        height=460,
        legend=dict(orientation="h", y=1.08, x=0.01),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def equity_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["strategy_equity"],
            mode="lines",
            name="Strategy",
            line=dict(color="#22c55e", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["buy_hold_equity"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#93c5fd", width=2.2),
        )
    )
    fig.update_layout(
        title="Equity Curve",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        height=430,
        legend=dict(orientation="h", y=1.08, x=0.01),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["drawdown"],
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color="#f97316", width=2.2),
        )
    )
    fig.update_layout(
        title="Strategy Drawdown",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def probability_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["predicted_prob_up"],
            mode="lines",
            name="Predicted Up Probability",
            line=dict(color="#a78bfa", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["actual_direction"],
            mode="lines",
            name="Actual Direction",
            line=dict(color="#e5e7eb", width=1.5, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Direction Signal Tracking",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        height=360,
        legend=dict(orientation="h", y=1.08, x=0.01),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Probability", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="Actual Direction", secondary_y=True, range=[-0.05, 1.05])
    return fig


def signal_html(direction: int) -> str:
    if direction == 1:
        return '<span class="signal-up">UP</span>'
    return '<span class="signal-down">DOWN</span>'


inject_styles()

with st.sidebar:
    st.markdown("### Model Controls")
    preset_options = {f"{item['name']} ({item['ticker']})": item["ticker"] for item in COMPANY_PRESETS}
    selected_company = st.selectbox("Quick Company", options=list(preset_options.keys()), index=0)
    default_ticker = preset_options[selected_company]
    ticker = st.text_input("Ticker", value=default_ticker).upper()
    mode = st.selectbox(
        "Execution Mode",
        options=["fast_ml", "deep_learning"],
        format_func=lambda x: "Fast ML" if x == "fast_ml" else "Deep Learning",
        index=0,
    )
    look_back = st.slider("Look-back Window", min_value=20, max_value=120, value=60, step=5)
    epochs = st.slider("Epochs", min_value=10, max_value=100, value=50, step=10)
    test_window_size = st.slider("Test Window Size", min_value=21, max_value=126, value=63, step=21)
    run_training = st.button("Run Training", use_container_width=True)
    st.markdown(
        """
        <div class="mini-note">
        Quick companies included: <b>Apple</b>, <b>Alphabet (Google)</b>, <b>Microsoft</b>, <b>TCS</b>, and <b>Infosys</b>.
        <b>Fast ML</b> is recommended for interviews and live demos. It trains much faster and still supports quick ticker switching.
        Use <b>Deep Learning</b> only when you want the slower LSTM-based research version.
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_training:
    with st.spinner("Training models and generating predictions..."):
        config = TrainConfig(
            ticker=ticker,
            look_back=look_back,
            epochs=epochs,
            test_window_size=test_window_size,
            mode=mode,
        )
        metrics = train_and_evaluate(config)
        st.success(f"Training completed for {ticker}.")
        st.session_state["latest_metrics"] = metrics

df, metrics = load_results(ticker)

st.markdown(
    f"""
        <div class="hero-card">
            <div class="caption-chip">Professional Quant Dashboard</div>
            <h1 style="margin: 12px 0 8px 0; font-size: 3.4rem; line-height: 1.02;">Stock Price Prediction and Trading Strategy</h1>
            <p style="margin: 0; color: #9fb0c8; font-size: 1.04rem;">
        Leakage-safe walk-forward validation for next-day return forecasting, directional signals, and strategy evaluation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if df is None:
    st.info("No saved results yet for this ticker. Run training from the sidebar to populate the dashboard.")
else:
    latest = df.iloc[-1]
    reg_metrics = metrics["regression_metrics"] if metrics else {}
    cls_metrics = metrics["classification_metrics"] if metrics else {}
    trade_metrics = metrics["trading_metrics"] if metrics else {}
    company_name = next((item["name"] for item in COMPANY_PRESETS if item["ticker"] == ticker), ticker)

    top_cols = st.columns(5)
    top_cols[0].metric("Company", company_name)
    top_cols[1].metric("Latest Signal", "Up" if latest["predicted_direction"] == 1 else "Down")
    top_cols[2].metric("Up Probability", f"{latest['predicted_prob_up']:.2%}")
    top_cols[3].metric("Predicted Next Close", f"{latest['predicted_next_close']:.2f}")
    top_cols[4].metric("Actual Next Close", f"{latest['actual_next_close']:.2f}")

    st.markdown(
        f"""
        <div class="section-card">
            <div style="display:flex; gap:36px; flex-wrap:wrap;">
                <div><span class="caption-chip">Ticker</span><div style="font-size:1.6rem; margin-top:10px;">{ticker}</div></div>
                <div><span class="caption-chip">Direction</span><div style="font-size:1.8rem; margin-top:10px;">{signal_html(int(latest["predicted_direction"]))}</div></div>
                <div><span class="caption-chip">Mode</span><div style="font-size:1.6rem; margin-top:10px;">{metrics.get("config", {}).get("mode", "unknown") if metrics else "unknown"}</div></div>
                <div><span class="caption-chip">Regression Error</span><div style="font-size:1.6rem; margin-top:10px;">RMSE {reg_metrics.get("rmse", 0):.4f}</div></div>
                <div><span class="caption-chip">Classification Accuracy</span><div style="font-size:1.6rem; margin-top:10px;">{cls_metrics.get("accuracy", 0):.2%}</div></div>
                <div><span class="caption-chip">Sharpe Ratio</span><div style="font-size:1.6rem; margin-top:10px;">{trade_metrics.get("sharpe_ratio", 0):.2f}</div></div>
                <div><span class="caption-chip">Max Drawdown</span><div style="font-size:1.6rem; margin-top:10px;">{trade_metrics.get("max_drawdown", 0):.2%}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_overview, tab_signals, tab_metrics, tab_backtest = st.tabs(
        ["Overview", "Signals", "Metrics", "Backtest"]
    )

    with tab_overview:
        left, right = st.columns([1.55, 1.0])
        with left:
            st.plotly_chart(prediction_chart(df), use_container_width=True)
        with right:
            st.plotly_chart(drawdown_chart(df), use_container_width=True)
        st.plotly_chart(equity_chart(df), use_container_width=True)

    with tab_signals:
        st.plotly_chart(probability_chart(df), use_container_width=True)
        signal_table = df.loc[:, ["date", "predicted_prob_up", "predicted_direction", "predicted_next_close", "actual_next_close"]].copy()
        signal_table["predicted_prob_up"] = signal_table["predicted_prob_up"].map(lambda x: f"{x:.2%}")
        signal_table["predicted_direction"] = signal_table["predicted_direction"].map({1: "Up", 0: "Down"})
        st.dataframe(signal_table.tail(20), use_container_width=True, hide_index=True)

    with tab_metrics:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("#### Regression")
            st.json(reg_metrics)
        with m2:
            st.markdown("#### Classification")
            st.json(cls_metrics)
        with m3:
            st.markdown("#### Trading")
            st.json(trade_metrics)

        st.markdown(
            """
            <div class="section-card">
                <div class="mini-note">
                <b>How to read these metrics:</b><br>
                RMSE and MAE should be lower. Accuracy, precision, and recall should be higher.
                For trading, higher cumulative return and Sharpe are better, while max drawdown should be less negative.
                Compare strategy return against buy-and-hold before trusting the model in practice.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_backtest:
        c1, c2 = st.columns([0.75, 1.25])
        with c1:
            if st.button("Run Backtrader Backtest", use_container_width=True):
                with st.spinner("Running backtest..."):
                    output = run_backtest(ticker)
                st.session_state["backtest_output"] = output
        with c2:
            st.markdown(
                """
                <div class="mini-note">
                This backtest uses the saved directional signal from the classification model.
                It buys on predicted up days, exits on predicted down days, and applies a stop-loss inside Backtrader.
                </div>
                """,
                unsafe_allow_html=True,
            )

        if "backtest_output" in st.session_state:
            st.markdown("#### Backtest Summary")
            st.json(st.session_state["backtest_output"])

    st.markdown(
        """
        <div class="section-card">
            <div class="mini-note">
            <b>Workflow:</b> choose a ticker, adjust hyperparameters in the sidebar, run training, then review prediction quality,
            strategy equity, drawdown, and backtest statistics. For faster experiments, reduce epochs or increase the test window size.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
