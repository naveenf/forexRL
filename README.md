# Forex RL Trading System

AI-powered forex trading using RecurrentPPO with LSTM policy.

## Overview

| Attribute | Value |
|-----------|-------|
| Current Pair | USD/JPY |
| Algorithm | RecurrentPPO + MlpLstmPolicy |
| Timeframe | M15 (15-minute candles) |
| Training | Google Colab (T4 GPU) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test environment
python -c "from src.environment_single import SinglePairForexEnv; print('OK')"

# Run training
python training/train_ppo.py
```

## Project Structure

```
ForexRL/
├── src/
│   ├── data_manager.py        # Data loading + indicators
│   └── environment_single.py  # RL environment
├── training/
│   └── train_ppo.py          # Training script
├── config/
│   ├── training.yaml         # RL hyperparameters
│   └── environment.yaml      # Trading parameters
└── data/                     # CSV data files
```

## Data Source

Historical forex data: https://forexsb.com/historical-forex-data

Required: `USDJPY_M15.csv` in the `data/` folder.

## Configuration

See `CLAUDE.md` for detailed configuration and development notes.

## License

Educational and research purposes.
