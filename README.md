# 📈 Stock Price Prediction Dashboard

A machine learning-powered stock price prediction dashboard built with Python. Predict stock prices, run backtests, and visualize trends using technical indicators and ML models.

---

## 🔗 Live Demo

[Click here to view the live app](https://stock-price-prediction-dashboard-les...)

---

## 🚀 Features

- 📊 Real-time stock data fetching via Yahoo Finance
- 🤖 ML-based price prediction (without TensorFlow — lightweight cloud-friendly models)
- 🔁 Backtesting engine to evaluate model performance
- ⚙️ Configurable presets for different stocks and timeframes
- 📉 Technical feature engineering (moving averages, RSI, etc.)
- ☁️ Cloud-deployable with fast ML mode

---

## 🗂️ Project Structure

```
├── app.py            # Main Streamlit app entry point
├── model.py          # ML model definition and prediction logic
├── train.py          # Model training pipeline
├── features.py       # Feature engineering and technical indicators
├── data.py           # Data fetching (Yahoo Finance) with retry & fallback
├── backtest.py       # Backtesting engine
├── presets.py        # Predefined stock/model configurations
└── requirements.txt  # Python dependencies
```

---

## 🛠️ Installation

```bash
git clone https://github.com/vinothkumarp8090-glitch/stock-price-prediction-dashboard.git
cd stock-price-prediction-dashboard
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 📦 Requirements

- Python 3.8+
- streamlit
- yfinance
- scikit-learn
- pandas
- numpy

> Install all dependencies with: `pip install -r requirements.txt`

---

## 🧠 How It Works

1. **Data** — `data.py` fetches historical OHLCV data from Yahoo Finance with retry and fallback logic
2. **Features** — `features.py` engineers technical indicators (e.g., RSI, MACD, moving averages)
3. **Training** — `train.py` trains a fast ML model (no TensorFlow required)
4. **Prediction** — `model.py` generates future price predictions
5. **Backtest** — `backtest.py` evaluates the model on historical data
6. **Dashboard** — `app.py` ties everything together in an interactive Streamlit UI

---

## 📌 Notes

- This project uses a **cloud-friendly fast ML mode** that avoids heavy TensorFlow dependencies, making it easy to deploy on platforms like Glitch, Render, or Railway.
- Yahoo Finance data fetching includes **retry and fallback** mechanisms for reliability.

---

## 👤 Author

**vinothkumarp8090-glitch**  
GitHub: [@vinothkumarp8090-glitch](https://github.com/vinothkumarp8090-glitch)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
