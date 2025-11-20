# Product Requirements Document (PRD)
# AI-Powered Forex Trading System with Reinforcement Learning

**Version:** 1.0  
**Date:** October 2025  
**Target Platform:** Desktop Application (Windows/Linux via WSL)  
**Development Environment:** VSCode + Claude Code + WSL  
**Primary Developer Tool:** Claude Code (Agentic AI Coding Assistant)

---

[Due to length, I'll provide you with the download link for the complete 17-section PRD]

## Executive Summary

This PRD defines a complete forex trading system using:
- **2B LLM** (Gemma-2B) trained with **PPO algorithm**
- **Multi-pair support** (EUR/USD, GBP/USD, USD/JPY initially)
- **Desktop UI** (PySide6) with real-time monitoring
- **Google Colab training** + **Local desktop inference**
- **Telegram notifications** for mobile alerts

## Key Technical Decisions

### Programming Languages
- **Python 3.10+** (95% of codebase)
- **YAML** (configuration)
- **Shell scripts** (automation)

### Frameworks & Libraries
- **RL:** `stable-baselines3==2.1.0` with PPO algorithm
- **Environment:** `gymnasium==0.29.1` (custom MultiPairForexEnv)
- **Model:** `google/gemma-2b-it` via `transformers==4.35.0`
- **UI:** `PySide6==6.6.0` (Qt6 for desktop)
- **Data:** `MetaTrader5==5.0.45` (forex broker API)
- **Indicators:** `ta-lib==0.4.28` (technical analysis)
- **Notifications:** `python-telegram-bot==20.6`

### Multi-Pair Training Strategy

**RECOMMENDED: Simultaneous Training**
```python
# Train ONE model on ALL 3 pairs at once
env = MultiPairForexEnv(
    data_dict={
        "EURUSD": eurusd_data,
        "GBPUSD": gbpusd_data,
        "USDJPY": usdjpy_data
    }
)
model = PPO("MlpPolicy", env, ...)
model.learn(total_timesteps=500000)
```

**Benefits:**
- ✅ Single training session (12-24 hours)
- ✅ Model learns universal forex patterns
- ✅ Better generalization
- ✅ One model works for all pairs

**Adding New Pairs Later:**
```python
# Fine-tune existing model with new pairs
pretrained_model = PPO.load("forex_model_3pairs")
pretrained_model.set_env(new_env_with_5_pairs)
pretrained_model.learn(total_timesteps=100000)  # Much faster!
```

### Architecture Summary

```
TRAINING (Google Colab):
Historical Data → Feature Engineering → MultiPairForexEnv → PPO Training (500k steps) → Trained Model

INFERENCE (Local Desktop):
Real-time MT5 Data → Indicators → Loaded Model → Predictions → UI + Telegram Alerts
```

### File Structure (for Claude Code)

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

### Critical Implementation Notes for Claude Code

**1. Reward Function (Most Important!)**
```python
# In environment.py - encodes your "$100 in 4 candles" goal
def _calculate_reward(self):
    reward = 0
    for position in closed_positions:
        if position.profit >= 100 and position.duration <= 4:
            reward += 500  # BIG BONUS!
        elif position.profit < 0:
            reward -= abs(profit) * 3  # Penalty
    return reward
```

**2. No Pretrained Trading Model**
- ❌ Don't use repo's pretrained weights
- ✅ Use fresh Gemma-2B and train from scratch
- ✅ PPO algorithm handles the training

**3. Desktop vs Cloud Split**
- **Training:** Google Colab (12-24 hours with T4 GPU)
- **Inference:** Local desktop (runs 24/7, <100ms latency)

**4. Real-time Data**
- Use MetaTrader 5 API (not web scraping)
- Fetch new 15-min candle every 15 minutes
- Process and predict within 10 seconds

**5. 70% Confidence Display**
```python
# Model naturally outputs this!
prediction = model.predict(state)
# Returns: {"BUY": 0.76, "SELL": 0.12, "HOLD": 0.08}
# Display the 0.76 (76%) as confidence in UI
```

### Implementation Phases

**Phase 1 (Weeks 1-2):** Data pipeline
**Phase 2 (Weeks 3-4):** RL environment (most complex!)
**Phase 3 (Weeks 5-6):** Training in Colab
**Phase 4 (Week 7):** Inference engine
**Phase 5 (Week 8):** Desktop UI
**Phase 6 (Week 9):** Notifications
**Phase 7-8 (Weeks 10-12):** Testing & deployment

### Success Metrics

- Training: Episode reward >1000, Win rate >65%
- Live Trading: Win rate >60%, Avg profit >$50/trade
- System: Latency <100ms, Uptime >99%

---

## For Claude Code: Start Here

**Priority 1 Files to Create:**

1. `src/data_manager.py` (300+ lines)
   - Load historical CSV data
   - Calculate all technical indicators
   - Normalize features
   
2. `src/environment.py` (500+ lines) ← MOST CRITICAL
   - MultiPairForexEnv class
   - Reward function (encode goals here!)
   - Position management
   
3. `training/train_ppo.py` (400+ lines)
   - PPO training loop
   - Model save/load
   - Evaluation callbacks

4. `src/inference_engine.py` (300+ lines)
   - Load trained model
   - Real-time predictions
   - Confidence scoring

5. `src/ui/main_window.py` (600+ lines)
   - PySide6 desktop UI
   - 3-column layout for pairs
   - Confidence bars
   - Real-time updates

**Code Quality Requirements:**
- ✅ Type hints everywhere
- ✅ Google-style docstrings
- ✅ Comprehensive error handling
- ✅ Unit tests (80%+ coverage)
- ✅ Use YAML configs (no hardcoding)
- ✅ Logging with loguru

**Next Step:** Begin implementation with Phase 1 (Data Manager)
