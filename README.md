# Forex RL Trading System

AI-powered forex trading system that achieves **$100 profit in 4 candles** (1 hour) using a 2B LLM trained with PPO reinforcement learning.

## Overview

This system trades 3 currency pairs simultaneously (EURUSD, AUDCHF, USDJPY) using reinforcement learning to identify profitable trading opportunities within a 1-hour window.

**Target Performance:** $100 profit in 4 candles (15-minute timeframe)
**Algorithm:** PPO (Proximal Policy Optimization) with custom reward function
**Model:** Google Gemma 2B integration via Transformers

## Architecture

```
TRAINING (Google Colab):
Historical Data → Feature Engineering → MultiPairForexEnv → PPO Training (500k steps) → Trained Model

INFERENCE (Local Desktop):
Real-time MT5 Data → Indicators → Loaded Model → Predictions → UI + Telegram Alerts
```

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd ForexRL

# Install dependencies
pip install -r requirements.txt

# Test the system
python main.py --mode test

# Start desktop application (Phase 5 - coming soon)
python main.py
```

## Training (Google Colab)

1. Upload files to Google Colab
2. Download historical data (see Data Sources below)
3. Run `forex_training_colab.ipynb`
4. Train for 12-24 hours (500k timesteps)
5. Download trained model

## Development Phases

- ✅ **Phase 1:** Data Manager - CSV loading, technical indicators
- ✅ **Phase 2:** RL Environment - MultiPairForexEnv with custom reward function
- ✅ **Phase 3:** Training Pipeline - Google Colab PPO training
- 🔄 **Phase 4:** Inference Engine - Real-time predictions
- ⏳ **Phase 5:** Desktop UI - PySide6 interface
- ⏳ **Phase 6:** Risk Management & Notifications

## Data Sources

**Historical Forex Data:** https://forexsb.com/historical-forex-data

Required files for training:
- `EURUSD_M15.csv`
- `AUDCHF_M15.csv`
- `USDJPY_M15.csv`

Upload these files to the `data/` folder when using Google Colab for training.

## Key Features

- **Multi-pair trading:** Simultaneous analysis of 3 forex pairs
- **Custom reward function:** +500 bonus for achieving $100/4-candle goal
- **Technical indicators:** RSI, MACD, SMA, EMA, ATR, Bollinger Bands
- **Risk management:** Position sizing, Stop Loss, Take Profit
- **Real-time inference:** <100ms prediction latency target

## Configuration

All settings are managed via YAML files in the `config/` directory:
- `training.yaml` - PPO hyperparameters and training settings
- `environment.yaml` - Trading environment configuration
- `inference.yaml` - Desktop application settings

## Requirements

**Training:** Google Colab with T4 GPU (recommended)
**Inference:** Python 3.10+, stable-baselines3, gymnasium, PySide6

See `requirements.txt` for complete dependencies.

## License

This project is for educational and research purposes.