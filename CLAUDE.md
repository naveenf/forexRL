# CLAUDE.md - Forex RL Trading System

## Project Status

**Current Phase**: Single-pair training validation
**Status**: Ready for fresh training after bug fixes (Dec 2024)

## Project Evolution

### Original Plan (Multi-Pair)
Started with 3 pairs (EUR/USD, AUD/CHF, USD/JPY) and 9-action space. Training failed:
- Model chose HOLD exclusively (zero trades)
- Complex action space prevented exploration
- Reward function issues masked by bonuses

### Current Approach (Single-Pair)
Simplified to **USD/JPY only** with 3-action space (HOLD/BUY/SELL).

### Future Plan (Multi-Model Architecture)
Once single-pair proves profitable:
1. Select ~5 pairs for active trading (e.g., USD/JPY, EUR/USD, GBP/USD, AUD/USD, EUR/JPY)
2. Train **separate model per pair** on M15 historical data
3. Inference engine loads all trained models
4. Real-time M15 data feeds each model independently
5. Each model outputs its own BUY/SELL/HOLD action

**Open Decision**: Whether to use completely separate models or share some layers.

## Architecture

```
TRAINING (Google Colab):
Historical M15 Data → Indicators → SinglePairForexEnv → RecurrentPPO (500k steps) → Model

INFERENCE (Local Desktop - Future):
Real-time MT5 Data → [Model per Pair] → Actions → Risk Manager → Execution
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| RL Algorithm | RecurrentPPO (sb3-contrib) |
| Policy | MlpLstmPolicy (256 LSTM units) |
| Environment | gymnasium + custom SinglePairForexEnv |
| Training | Google Colab (T4 GPU) |
| UI (planned) | PySide6 |
| Broker API | MetaTrader5 |

## Current Configuration

### Trading Parameters
| Parameter | Value |
|-----------|-------|
| Pair | USD/JPY |
| Timeframe | M15 |
| Stop Loss | 15 pips |
| Take Profit | 22.5 pips (1:1.5 R:R) |
| Risk per Trade | 3% ($300 on $10k) |
| Min Hold | 8 candles (2 hours) |
| Max Hold | 24 candles (6 hours) |
| Optimal Window | 10-20 candles |

### RL Hyperparameters
| Parameter | Value |
|-----------|-------|
| vf_coef | 1.0 |
| ent_coef | 0.05 |
| learning_rate | 0.0003 |
| n_steps | 2048 |
| batch_size | 64 |
| gamma | 0.99 |

## Recent Fixes (Dec 2024)

| Fix | Description |
|-----|-------------|
| Double Slippage | Was applying spread + slippage (~4.5 pips), now slippage only (~2 pips) |
| Value Function | vf_coef 0.5→1.0 (critic was not learning) |
| Reward Bonuses | Reduced duration bonus 100→30, opening bonus 5→1 |
| Trend Filter | SMA20/SMA50 crossover, -10 penalty for counter-trend |
| Duration | Min 12→8, Max 32→24, Optimal 12-16→10-20 |
| Config Sync | Aligned all values across config files |

## Project Structure

```
ForexRL/
├── src/
│   ├── data_manager.py        # CSV loading + indicators (DONE)
│   ├── environment_single.py  # RL environment (DONE)
│   ├── environment.py         # Multi-pair (future)
│   └── inference_engine.py    # Real-time (TODO)
├── training/
│   └── train_ppo.py          # Training script (DONE)
├── config/
│   ├── training.yaml         # RL hyperparameters
│   └── environment.yaml      # Trading params
└── data/                     # CSV files
```

## Key Files

| File | Purpose |
|------|---------|
| `src/environment_single.py` | RL env with reward function, trend filter |
| `training/train_ppo.py` | RecurrentPPO training loop |
| `config/training.yaml` | PPO hyperparameters |
| `config/environment.yaml` | SL/TP, durations, risk |

## Target Metrics

| Metric | Target |
|--------|--------|
| Win Rate | >50% |
| Profit Factor | >1.5 |
| Max Drawdown | <30% |
| Explained Variance | >0.8 |
| Trades/Episode | 10-25 |

## Development Phases

- [x] Phase 1: Data Manager
- [x] Phase 2: RL Environment
- [x] Phase 3: Training Pipeline
- [ ] Phase 4: Inference Engine
- [ ] Phase 5: Desktop UI
- [ ] Phase 6: Multi-pair expansion

## Commands

```bash
# Training (Colab)
python training/train_ppo.py

# Test environment
python -c "from src.environment_single import SinglePairForexEnv; print('OK')"
```

## Git Practices

- Concise commits (max 2 lines)
- Conventional commits: feat, fix, docs, refactor
- No attribution tags
