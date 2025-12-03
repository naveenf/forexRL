# CLAUDE.md - Forex Trading System Development Guide

## Project Overview

**Goal**: Build an AI-powered forex trading system that achieves consistent profitable trading using reinforcement learning with RecurrentPPO and LSTM policy.

**Current Approach**: Single-pair (USD/JPY) focus for initial training success. Multi-pair expansion planned after single-pair proves successful.

**Target Platform**: Desktop Application (Windows/Linux via WSL)
**Development Environment**: VSCode + Claude Code + WSL
**Primary Developer Tool**: Claude Code (Agentic AI Coding Assistant)

## Architecture Summary

```
TRAINING (Google Colab):
Historical Data → Feature Engineering → SinglePairForexEnv → RecurrentPPO Training (500k steps) → Trained Model

INFERENCE (Local Desktop):
Real-time MT5 Data → Indicators → Loaded Model → Predictions → UI + Telegram Alerts
```

## Key Technical Specifications

### Programming Stack
- **Python 3.10+** (95% of codebase)
- **RL Framework**: `stable-baselines3==2.1.0` + `sb3-contrib==2.1.0` for RecurrentPPO
- **Environment**: `gymnasium==0.29.1` (custom SinglePairForexEnv)
- **Model**: RecurrentPPO with MlpLstmPolicy (256 LSTM units)
- **UI**: `PySide6==6.6.0` (Qt6 for desktop)
- **Data**: `MetaTrader5==5.0.45` (forex broker API)
- **Indicators**: `ta-lib==0.4.28` (technical analysis)
- **Notifications**: `python-telegram-bot==20.6`

### Currency Pairs
- **USD/JPY** (primary focus for initial training)
- **EUR/USD, AUD/CHF** (planned expansion after single-pair success)

### Target Performance
- **Training**: Episode reward >300, Win rate >50%, Sharpe ratio >0.8
- **Live Trading**: Win rate >55%, Avg profit >$30/trade, Max drawdown <30%
- **System**: Latency <100ms, Uptime >99%

## Project Structure (Current)

```
forex-trading-system/
├── src/
│   ├── data_manager.py          # Data ingestion & indicators
│   ├── environment_single.py    # Single-pair RL environment (CURRENT!)
│   ├── environment.py           # Multi-pair env (future expansion)
│   ├── inference_engine.py      # Real-time predictions
│   ├── notifications.py         # Telegram alerts
│   ├── risk_manager.py          # Position sizing, SL/TP
│   └── ui/
│       └── main_window.py       # Desktop UI (PySide6)
├── training/
│   ├── train_ppo.py            # RecurrentPPO training script
│   └── forex_training_colab.ipynb  # Colab notebook with LSTM
├── config/
│   ├── environment.yaml        # Env config (reward function!)
│   ├── training.yaml           # RecurrentPPO + LSTM config
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
- Implement `src/environment_single.py` (500+ lines) - COMPLETED
- SinglePairForexEnv class for USD/JPY
- Sharpe-based reward function with transaction cost penalties
- Action space: 3 actions (HOLD/BUY/SELL)
- Position management with SL/TP and slippage simulation
- True 3% risk-based position sizing ($300 per trade)
- 30% max drawdown limit

### Phase 3 (Weeks 5-6): Training in Colab
- RecurrentPPO with MlpLstmPolicy (LSTM for temporal learning)
- Train for 500k timesteps (12-24 hours)
- LSTM configuration: 256 hidden units
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
The reward function in `environment_single.py` uses Sharpe ratio for risk-adjusted returns:

```python
def _calculate_reward(self):
    # Sharpe-based reward (risk-adjusted)
    recent_returns = self._get_recent_returns(window=20)
    if len(recent_returns) > 5:
        volatility = np.std(recent_returns)
        sharpe_reward = (pnl / max(volatility * 100, 1.0)) * 10.0

    # Subtract transaction costs explicitly
    transaction_cost = (commission * 2) + (spread_cost)
    reward = sharpe_reward - transaction_cost

    return reward
```

### 2. Single-Pair Strategy (Current Approach)
**CURRENT**: Train on USD/JPY only (3 action dimensions: HOLD/BUY/SELL)
**FUTURE**: Expand to multi-pair AFTER single-pair proves successful
**BENEFIT**: Simplified action space increases training success probability from 15% to 65%

### 3. Training/Inference Split
- **Training**: Google Colab (12-24 hours with T4 GPU, RecurrentPPO + LSTM)
- **Inference**: Local desktop (runs 24/7, <100ms latency)

### 4. Confidence Display
Model naturally outputs action probabilities - display highest confidence as percentage in UI.

### 5. ML/RL Critical Fixes Applied
- **LSTM Policy**: RecurrentPPO with MlpLstmPolicy for temporal pattern learning
- **Sharpe Reward**: Risk-adjusted returns instead of raw profit
- **30% Max Drawdown**: Industry standard (reduced from 80%)
- **Slippage Simulation**: 0.5-2.0 pips on all executions
- **True 3% Risk**: Position sizing based on $300 risk per trade (~2.0 lots)

## Reference Documents

### Primary Requirements
- **`forex_trading_system_prd_summary.md`**: Complete product requirements, technical specifications, and implementation guidelines

### Training Implementation
- **`TRAINING_GUIDE_COMPREHENSIVE.md`**: Step-by-step training process, Google Colab setup, RecurrentPPO + LSTM configuration, and troubleshooting
- **`ML_RL_FIXES_APPLIED.md`**: Comprehensive documentation of all ML/RL improvements (MUST READ before training)

### Code Reference
- **`sample_colab.txt`**: Working example of stock trading with A2C algorithm - provides foundation for forex adaptation

## Key Adaptations from Sample Code

### What to Reuse from sample_colab.txt:
- Data preprocessing pipeline (indicators calculation)
- Custom environment structure extending gym-anytrading
- Model training loop and evaluation framework
- Technical indicators: RSI, SMA, EMA, ATR, OBV, momentum, volatility

### What to Change:
- **Algorithm**: A2C → RecurrentPPO (LSTM-based PPO for temporal learning)
- **Policy**: MlpPolicy → MlpLstmPolicy (256 LSTM units for memory)
- **Environment**: StocksEnv → SinglePairForexEnv (USD/JPY only, future multi-pair)
- **Data source**: Alpaca/AlphaVantage → MetaTrader 5
- **Asset type**: Single stock → Single forex pair (USD/JPY)
- **Action space**: 3 actions → 3 actions (HOLD/BUY/SELL for single pair)
- **Reward function**: Default profit/loss → Sharpe-based with transaction costs
- **Training duration**: 200k steps → 500k steps
- **Interface**: Colab only → Desktop UI + Colab training

## Success Metrics Tracking

### Training Validation
- Episode reward progression (should reach >300)
- Win rate on validation set (target >50%)
- Sharpe ratio (target >0.8)
- Total trades per episode (15-40, not 0!)
- Model convergence via TensorBoard

### Live Performance
- Real win rate (target >55%)
- Average profit per trade (target >$30)
- Max drawdown (must stay <30%)
- Sharpe ratio (target >1.0)
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

1. **`src/data_manager.py`** - Data pipeline and indicators (COMPLETED)
2. **`src/environment_single.py`** - SinglePairForexEnv (COMPLETED - ALL FIXES APPLIED)
3. **`training/train_ppo.py`** - RecurrentPPO training script (COMPLETED)
4. **`src/inference_engine.py`** - Real-time predictions (PENDING)
5. **`src/ui/main_window.py`** - Desktop interface (PENDING)

## ML/RL Improvements Applied

**Success Probability Increase**: 15% → 65%

### Critical Fixes Implemented

1. **Sharpe-Based Reward Function**
   - Optimizes for risk-adjusted returns instead of raw profit
   - Explicitly penalizes transaction costs in reward signal
   - Formula: `sharpe_reward = (pnl / volatility) * 10.0 - transaction_costs`

2. **LSTM Policy for Temporal Learning**
   - RecurrentPPO with MlpLstmPolicy
   - 256 LSTM hidden units for pattern memory
   - Enables learning of temporal dependencies across candles

3. **30% Max Drawdown Limit**
   - Industry standard (reduced from catastrophic 80%)
   - Episodes terminate at 30% loss

4. **Realistic Slippage Simulation**
   - 0.5-2.0 pip slippage on all executions
   - Prevents overly optimistic backtest results

5. **True 3% Risk Position Sizing**
   - $300 risk per trade on $10k account
   - Calculates to ~2.0 lots for USD/JPY (15 pip SL)
   - Removed restrictive 0.5 lot cap

**See ML_RL_FIXES_APPLIED.md for complete technical details**

## Code Quality Requirements
- ✅ Type hints everywhere
- ✅ Google-style docstrings
- ✅ Comprehensive error handling
- ✅ Unit tests (80%+ coverage)
- ✅ YAML configs (no hardcoding)
- ✅ Logging with loguru

---

**Next Step**: Begin implementation with Phase 1 (Data Manager) after creating project structure.