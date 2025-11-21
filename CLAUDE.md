# CLAUDE.md - Forex Trading System Development Guide

## Project Overview

**Goal**: Build an AI-powered forex trading system that achieves **$100 profit in 4 candles** (15-minute timeframe = 1 hour total) using a 2B LLM trained with PPO reinforcement learning.

**Target Platform**: Desktop Application (Windows/Linux via WSL)
**Development Environment**: VSCode + Claude Code + WSL
**Primary Developer Tool**: Claude Code (Agentic AI Coding Assistant)

## Architecture Summary

```
TRAINING (Google Colab):
Historical Data → Feature Engineering → MultiPairForexEnv → PPO Training (500k steps) → Trained Model

INFERENCE (Local Desktop):
Real-time MT5 Data → Indicators → Loaded Model → Predictions → UI + Telegram Alerts
```

## Key Technical Specifications

### Programming Stack
- **Python 3.10+** (95% of codebase)
- **RL Framework**: `stable-baselines3==2.1.0` with PPO algorithm
- **Environment**: `gymnasium==0.29.1` (custom MultiPairForexEnv)
- **Model**: `google/gemma-2b-it` via `transformers==4.35.0`
- **UI**: `PySide6==6.6.0` (Qt6 for desktop)
- **Data**: `MetaTrader5==5.0.45` (forex broker API)
- **Indicators**: `ta-lib==0.4.28` (technical analysis)
- **Notifications**: `python-telegram-bot==20.6`

### Currency Pairs
- **EUR/USD** (primary)
- **GBP/USD**
- **USD/JPY**

### Target Performance
- **Training**: Episode reward >1000, Win rate >65%
- **Live Trading**: Win rate >60%, Avg profit >$50/trade
- **System**: Latency <100ms, Uptime >99%

## Project Structure (Target)

```
forex-trading-system/
├── src/
│   ├── data_manager.py          # Data ingestion & indicators
│   ├── environment.py           # RL environment (CRITICAL!)
│   ├── inference_engine.py      # Real-time predictions
│   ├── notifications.py         # Telegram alerts
│   ├── risk_manager.py          # Position sizing, SL/TP
│   └── ui/
│       └── main_window.py       # Desktop UI (PySide6)
├── training/
│   ├── train_ppo.py            # Google Colab training script
│   └── forex_training.ipynb    # Jupyter notebook
├── config/
│   ├── environment.yaml        # Env config (reward function!)
│   ├── training.yaml           # PPO hyperparameters
│   └── inference.yaml          # Desktop app settings
├── main.py                     # Entry point
└── requirements.txt            # Dependencies
```

## Implementation Phases

### Phase 1 (Weeks 1-2): Data Pipeline
- Implement `src/data_manager.py` (300+ lines)
- MetaTrader 5 integration
- Technical indicators calculation
- Data preprocessing and normalization

### Phase 2 (Weeks 3-4): RL Environment (MOST CRITICAL!)
- Implement `src/environment.py` (500+ lines)
- MultiPairForexEnv class supporting 3 pairs simultaneously
- Custom reward function encoding "$100 in 4 candles" goal
- Action space: 9 actions (3 per pair: HOLD/BUY/SELL)
- Position management with SL/TP

### Phase 3 (Weeks 5-6): Training in Colab
- Adapt sample Colab notebook for forex
- Switch from A2C to PPO algorithm
- Train for 500k timesteps (12-24 hours)
- Model evaluation and export

### Phase 4 (Week 7): Inference Engine
- Implement `src/inference_engine.py` (300+ lines)
- Load trained model for real-time predictions
- Generate 70% confidence scores for UI

### Phase 5 (Week 8): Desktop UI
- Implement `src/ui/main_window.py` (600+ lines)
- PySide6 interface with 3-column layout
- Real-time updates and confidence bars

### Phase 6 (Week 9): Notifications & Risk Management
- Telegram integration
- Risk management system
- End-to-end testing

### Phase 7-8 (Weeks 10-12): Testing & Deployment
- Backtesting validation
- Paper trading
- Production deployment

## Critical Implementation Notes

### 1. Reward Function (Most Important!)
The reward function in `environment.py` must encode the "$100 in 4 candles" goal:

```python
def _calculate_reward(self):
    reward = 0
    for position in closed_positions:
        if position.profit >= 100 and position.duration <= 4:
            reward += 500  # BIG BONUS!
        elif position.profit < 0:
            reward -= abs(profit) * 3  # Penalty
    return reward
```

### 2. Multi-Pair Training Strategy
**RECOMMENDED**: Train ONE model on ALL 3 pairs simultaneously for better generalization.

### 3. Training/Inference Split
- **Training**: Google Colab (12-24 hours with T4 GPU)
- **Inference**: Local desktop (runs 24/7, <100ms latency)

### 4. Confidence Display
Model naturally outputs action probabilities - display highest confidence as percentage in UI.

## Reference Documents

### Primary Requirements
- **`forex_trading_system_prd_summary.md`**: Complete product requirements, technical specifications, and implementation guidelines

### Training Implementation
- **`TRAINING_GUIDE_COMPREHENSIVE.md`**: Step-by-step training process, Google Colab setup, PPO configuration, and troubleshooting

### Code Reference
- **`sample_colab.txt`**: Working example of stock trading with A2C algorithm - provides foundation for forex adaptation

## Key Adaptations from Sample Code

### What to Reuse from sample_colab.txt:
- Data preprocessing pipeline (indicators calculation)
- Custom environment structure extending gym-anytrading
- Model training loop and evaluation framework
- Technical indicators: RSI, SMA, EMA, ATR, OBV, momentum, volatility

### What to Change:
- **Algorithm**: A2C → PPO (more stable for forex)
- **Environment**: StocksEnv → MultiPairForexEnv (3 pairs simultaneously)
- **Data source**: Alpaca/AlphaVantage → MetaTrader 5
- **Asset type**: Single stock → 3 forex pairs
- **Action space**: 3 actions → 9 actions (3 per pair)
- **Reward function**: Default profit/loss → Custom "$100 in 4 candles"
- **Training duration**: 200k steps → 500k steps
- **Interface**: Colab only → Desktop UI + Colab training

## Success Metrics Tracking

### Training Validation
- Episode reward progression (should reach >1000)
- Win rate on validation set (target >65%)
- Profit factor and Sharpe ratio
- Model convergence via TensorBoard

### Live Performance
- Real win rate (target >60%)
- Average profit per trade (target >$50)
- System latency (<100ms)
- Uptime (>99%)

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv forex_env
source forex_env/bin/activate  # Linux/WSL
# forex_env\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Upload data to Google Colab
# Run training notebook (12-24 hours)
# Download trained model
```

### Local Development
```bash
# Run desktop application
python main.py

# Run inference engine only
python -m src.inference_engine

# Run backtesting
python -m src.backtesting
```

## Git Commit Guidelines

**Keep commit messages concise and professional:**

### Format
- **Maximum 2 lines total**
- **First line:** Brief summary (50 chars or less)
- **Second line:** Optional details if needed
- **NO Claude Code attribution** - keep messages clean and focused

### Examples
```bash
# Good - concise and clear
git commit -m "feat: implement RL environment with custom reward function
Support multi-pair trading with 9-action space"

# Bad - too verbose, includes attribution
git commit -m "feat: Complete Phase 2-3 - RL Environment & Training Setup with multiple features... 🤖 Generated with Claude Code"
```

### Conventional Commits
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code restructuring
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

## Priority Files to Create (Development Order)

1. **`src/data_manager.py`** - Data pipeline and indicators
2. **`src/environment.py`** - MultiPairForexEnv (MOST CRITICAL)
3. **`training/train_ppo.py`** - Google Colab training script
4. **`src/inference_engine.py`** - Real-time predictions
5. **`src/ui/main_window.py`** - Desktop interface

## Code Quality Requirements
- ✅ Type hints everywhere
- ✅ Google-style docstrings
- ✅ Comprehensive error handling
- ✅ Unit tests (80%+ coverage)
- ✅ YAML configs (no hardcoding)
- ✅ Logging with loguru

---

**Next Step**: Begin implementation with Phase 1 (Data Manager) after creating project structure.