# CryptoMarket Regime Classifier
**Adaptive Market Intelligence for Crypto Strategies**

Most trading strategies fail not because the logic is wrong, but because they are applied in the wrong **market regime**.

A breakout strategy thrives in trends and bleeds in chop.  
Mean-reversion works in ranges and dies in momentum.

**CryptoMarket Regime Classifier** is a machine learning pipeline that detects and predicts crypto market regimes using **multi-timeframe OHLCV data**, **technical indicators**, and a **two-stage ML approach (HMM → LSTM)**.

It is built as a **foundational intelligence layer** for:
- strategy selection  
- position sizing  
- risk management  

and is intended to power the regime-awareness module in **Dazai**.

---

## High-Level Pipeline
**OHLCV (5m, 15m)**
        ↓
**Feature Engineering (momentum, trend, volatility)**
        ↓
**PCA Reduction**
        ↓
**Hidden Markov Model (Regime Discovery)**
        ↓
**LSTM (Regime Prediction)**
        ↓
**Current Regime (+ future probabilistic output)**

--------

## Key Ideas (Why this is different)

- **Regime-aware, not signal-based**  
  The model does not predict price — it predicts *market conditions*.

- **Unsupervised → Supervised learning**  
  HMM discovers latent regimes first.  
  LSTM then learns temporal structure to predict them.

- **Multi-timeframe context**  
  Combines short-term and slightly higher-timeframe behavior (5m, 15m).

- **Designed for integration**  
  Models and scalers are exported for downstream systems (bots, dashboards, APIs).

---

## Key Features

- Multi-timeframe OHLCV data (5m, 15m) from Binance  
- Technical indicators covering:
  - momentum  
  - volatility  
  - trend  
- Hidden Markov Models (HMM) for unsupervised regime discovery  
- LSTM trained on HMM-labeled sequences  
- **6 discovered regimes**, including:
  - Strong Trend  
  - Weak Trend  
  - Range  
  - Choppy High-Volatility  
  - Volatility Spike  
  - Squeeze  
- Evaluation metrics:
  - Precision / Recall / F1  
  - Confusion Matrix  

---

## Project Structure
--------------------
├── dashboard/        # Visualizations, regime plots  
├── models/           # Trained models & scalers  
├── src/              # Feature engineering + training scripts  
├── main.py           # End-to-end pipeline execution  
├── requirements.txt  # Dependencies  
└── README.md


---

## Workflow Details

### 1. Data Fetching
- Periodically fetches OHLCV data from Binance  
- Currently optimized for **5m data**, with support for higher TF context  

### 2. Feature Engineering
- Computes momentum, trend, and volatility indicators  
- Aligns and scales features for ML stability  

### 3. Regime Discovery (HMM)
- PCA-reduced feature space  
- **6-state HMM selected using lowest BIC**
- Produces regime labels without human bias  

### 4. Regime Prediction (LSTM)
- Sequence model trained on HMM labels  
- Captures temporal transitions between regimes  
- Hyperparameters tuned using Keras Tuner  
- Planned upgrade: **probabilistic regime distributions**

### 5. Model Export & Usage
- Trained LSTM + scalers saved to `/models`  
- Designed for reuse in live systems  

---

## Results (High-Level)

- Strong separation between **trend vs non-trend** regimes  
- Transitional regimes (range ↔ weak trend, spike ↔ chop) are naturally harder — and informative  
- Confusion matrix reflects realistic regime overlap instead of artificial sharp boundaries  

---

## Installation

```bash
git clone https://github.com/akash-kumar5/CryptoMarket_Regime_Classifier.git
cd CryptoMarket_Regime_Classifier
pip install -r requirements.txt
```        

 Usage
--------

Run the full pipeline:
streamlit run dashboard/app.py


Models & scalers will be saved in /models for reuse.

 Notes
--------

*   Data range: ~2 years (to prioritize recent regime behavior and avoid stale market patterns).
    
*   Designed as a **research + foundational tool** for live trading systems.
    
*   Future versions will connect directly into **Dazai** as a core regime intelligence component.
# -----
# Disclaimer

This project is for research and educational purposes only.
It does not constitute financial advice.
