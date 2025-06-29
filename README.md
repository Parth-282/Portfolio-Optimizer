
# 📊 Portfolio Optimizer & Risk Analyzer

A Python + Streamlit-based dashboard to simulate, optimize, and analyze stock portfolios using Modern Portfolio Theory and advanced risk metrics.

## 🚀 Features
- 🔧 Portfolio Optimization (Max Sharpe & Min Volatility)
- 📈 Custom Weights Input with full performance evaluation
- 📉 Risk Metrics: Sharpe Ratio, Sortino Ratio, Jensen’s Alpha, VaR, Max Drawdown
- 📊 Efficient Frontier Plot, VaR Histogram, Correlation/Covariance Heatmaps
- 💡 Real-time interactive dashboard built with Streamlit

## 📁 Folder Structure
```
portfolio-optimizer/
├── app.py              # Streamlit frontend interface
├── main.py             # Portfolio logic and analytics
├── outputs/            # Generated plots (heatmaps, charts)
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## 💻 How to Run

1. Clone the repo:
```bash
git clone https://github.com/Parth-282/portfolio-optimizer.git
cd portfolio-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## 📸 Screenshots
*(Add screenshots of the Efficient Frontier, metric outputs, and heatmaps here)*

## 🧠 Built With
- `streamlit` – Web app UI
- `pandas`, `numpy` – Data manipulation
- `matplotlib`, `seaborn` – Plotting
- `yfinance` – Stock data API
- `scipy` – Optimization solver
- `pillow` – Image handling

## ✍️ Author
**Parth Bansal**  
[LinkedIn →](https://www.linkedin.com/in/parth-bansal-25b6561a5)

---

> 🎓 This project was built as part of my learning in finance, data analysis, and Python programming. It showcases a full workflow from data acquisition to risk-adjusted performance reporting.

