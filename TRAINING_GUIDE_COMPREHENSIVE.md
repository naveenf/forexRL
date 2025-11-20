# TRAINING GUIDE: Forex RL Model Training
# Based on Proven Colab Notebook Analysis + Forex Adaptations

**Version:** 1.0  
**Purpose:** Step-by-step guide for training PPO model on forex data  
**Target Environment:** Google Colab with GPU  
**Estimated Duration:** 12-24 hours (500,000 timesteps)  
**GPU Required:** T4 (16GB) or V100 (32GB) via Colab Pro

---

## Table of Contents

1. [Training Overview](#training-overview)
2. [Pre-Training Checklist](#pre-training-checklist)
3. [Step-by-Step Training Process](#step-by-step-training-process)
4. [Complete Training Pseudocode](#complete-training-pseudocode)
5. [Detailed Implementation Code](#detailed-implementation-code)
6. [Configuration Files](#configuration-files)
7. [Monitoring & Debugging](#monitoring--debugging)
8. [Post-Training Validation](#post-training-validation)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Model Export for Desktop](#model-export-for-desktop)

---

## 1. Training Overview

### 1.1 What We're Building

```
INPUT:
├─ 10 years of historical forex data (2015-2025)
├─ 3 currency pairs (EURUSD, GBPUSD, USDJPY)
├─ 15-minute timeframe (~350,000 candles per pair)
└─ Technical indicators (RSI, MACD, BB, EMA, ATR, etc.)

TRAINING PROCESS:
├─ Create custom MultiPairForexEnv (Gymnasium)
├─ Train using PPO algorithm (Stable-Baselines3)
├─ Reward function: "$100 profit in 4 candles"
├─ 500,000 training timesteps
└─ Monitor via TensorBoard

OUTPUT:
├─ Trained model weights (.zip file)
├─ Configuration files (.json)
├─ Data scaler (.pkl for normalization)
├─ Training logs (TensorBoard)
└─ Performance metrics (win rate, profit factor, etc.)
```

### 1.2 Training Philosophy (Key Differences from Stock Trading)

| Aspect | Stock Trading (Original Notebook) | Forex Trading (Your System) |
|--------|-----------------------------------|------------------------------|
| **Assets** | 1 stock (AAPL) | 3 pairs (simultaneously) |
| **Data Source** | Alpaca API | MetaTrader 5 / CSV files |
| **Environment** | StocksEnv (gym-anytrading) | Custom MultiPairForexEnv |
| **Action Space** | 3 actions (HOLD/BUY/SELL) | 9 actions (3 per pair) |
| **Reward Function** | Default profit/loss | Custom: "$100 in 4 candles" |
| **Algorithm** | A2C (simpler) | PPO (more stable) |
| **Training Duration** | 200k steps | 500k steps (more complex) |

---

## 2. Pre-Training Checklist

### 2.1 Data Preparation (Before Opening Colab)

**✅ Task 1: Download Historical Data**

```bash
# On your local machine (WSL)
cd ~/forex-trading-system/data/raw/

# Option A: Download from Dukascopy (recommended)
# Manual download: https://www.dukascopy.com/swiss/english/marketwatch/historical/
# Files: EURUSD_15m_2015_2025.csv, GBPUSD_15m_2015_2025.csv, USDJPY_15m_2015_2025.csv

# Option B: Download from Hugging Face
python scripts/download_from_huggingface.py

# Option C: Export from MetaTrader 5
python scripts/export_from_mt5.py --pair EURUSD --start 2015-01-01 --end 2025-09-30 --timeframe 15m
```

**Expected Data Format:**
```csv
timestamp,open,high,low,close,volume
2015-01-01 00:00:00,1.2050,1.2055,1.2048,1.2052,1250
2015-01-01 00:15:00,1.2052,1.2060,1.2050,1.2058,1430
...
```

**Data Quality Checks:**
```python
import pandas as pd

# Load data
df = pd.read_csv("data/raw/EURUSD_15m_2015_2025.csv")

# Validation checks
assert len(df) >= 300000, "Need at least 300k candles (8+ years)"
assert df['timestamp'].is_monotonic_increasing, "Timestamps must be sorted"
assert df.isnull().sum().sum() == 0, "No missing values allowed"
assert (df['high'] >= df['low']).all(), "High must be >= Low"
assert (df['high'] >= df['close']).all(), "High must be >= Close"
assert (df['low'] <= df['close']).all(), "Low must be <= Close"

print(f"✅ Data validated: {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
```

---

**✅ Task 2: Calculate Technical Indicators**

```python
# Run this locally to pre-process data
python scripts/preprocess_data.py --input data/raw/ --output data/processed/

# This script calculates ALL indicators for ALL pairs
# Output: EURUSD_processed.csv, GBPUSD_processed.csv, USDJPY_processed.csv
```

**Indicator Calculation Script:**
```python
# scripts/preprocess_data.py

import pandas as pd
from finta import TA
import numpy as np

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators (same as Colab notebook)"""
    
    df = df.copy()
    
    # 1. Returns
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Trend Indicators
    df['RSI'] = TA.RSI(df, 14)
    df['SMA_20'] = TA.SMA(df, 20)
    df['SMA_50'] = TA.SMA(df, 50)
    df['EMA_20'] = TA.EMA(df, 20)
    df['EMA_50'] = TA.EMA(df, 50)
    df['MACD'] = TA.MACD(df)['MACD']
    df['MACD_signal'] = TA.MACD(df)['SIGNAL']
    
    # 3. Volatility Indicators
    bb = TA.BBANDS(df)
    df['BB_upper'] = bb['BB_UPPER']
    df['BB_middle'] = bb['BB_MIDDLE']
    df['BB_lower'] = bb['BB_LOWER']
    df['ATR'] = TA.ATR(df, 14)
    
    # 4. Volume Indicator
    df['OBV'] = TA.OBV(df)
    
    # 5. Custom Features (from notebook)
    df['momentum'] = df['return'].rolling(5).mean().shift(1)
    df['volatility'] = df['return'].rolling(20).std().shift(1)
    df['distance'] = (df['close'] - df['close'].rolling(50).mean()).shift(1)
    
    # 6. Linear Regression Prediction (from notebook)
    lags = 5
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['close'].shift(lag)
    
    # Drop rows with NaN (first 50 rows due to indicators)
    df.dropna(inplace=True)
    
    # Calculate linear regression
    lag_cols = [f'lag_{i}' for i in range(1, lags + 1)]
    reg = np.linalg.lstsq(df[lag_cols], df['close'], rcond=None)[0]
    df['prediction'] = np.dot(df[lag_cols], reg)
    
    return df

# Process all pairs
for pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
    print(f"Processing {pair}...")
    df = pd.read_csv(f"data/raw/{pair}_15m_2015_2025.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df_processed = calculate_all_indicators(df)
    
    df_processed.to_csv(f"data/processed/{pair}_processed.csv")
    print(f"✅ {pair}: {len(df_processed)} candles with {len(df_processed.columns)} features")
```

---

**✅ Task 3: Upload Data to Google Drive**

```bash
# Compress processed data
cd ~/forex-trading-system/data/processed/
tar -czf forex_processed_data.tar.gz *.csv

# Upload to Google Drive manually or via CLI
# Using rclone (if configured)
rclone copy forex_processed_data.tar.gz gdrive:forex-trading-data/
```

---

**✅ Task 4: Prepare Configuration File**

```yaml
# config/training_config.yaml
# This will be uploaded to Colab

training:
  pairs:
    - EURUSD
    - GBPUSD
    - USDJPY
  
  data:
    timeframe: "15m"
    train_start: "2015-01-01"
    train_end: "2023-12-31"
    val_start: "2024-01-01"
    val_end: "2024-12-31"
    features:
      - open
      - high
      - low
      - close
      - return
      - RSI
      - SMA_20
      - SMA_50
      - EMA_20
      - EMA_50
      - MACD
      - MACD_signal
      - BB_upper
      - BB_middle
      - BB_lower
      - ATR
      - OBV
      - momentum
      - volatility
      - distance
      - prediction
  
  environment:
    window_size: 100
    initial_balance: 10000.0
    trading_fees: 0.0002  # 0.02% spread
    max_trades_per_pair: 1
  
  reward:
    target_profit: 100      # USD
    target_duration: 4      # candles
    profit_bonus: 500
    quick_profit_bonus: 150
    loss_penalty_multiplier: 3
    overtrading_penalty: 100
    max_drawdown_penalty: 200
  
  ppo:
    algorithm: "PPO"
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
    
  training:
    total_timesteps: 500000
    eval_freq: 10000
    save_freq: 50000
    log_interval: 100
    
  hardware:
    device: "cuda"
    use_fp16: false
```

---

### 2.2 Google Colab Setup Checklist

**✅ Colab Pro Subscription**
- Required for: T4/V100 GPU, longer runtime (24+ hours)
- Cost: ~$10/month
- Sign up: https://colab.research.google.com/signup

**✅ Google Drive Space**
- Minimum: 5GB free (for data + models)
- Check: https://drive.google.com/drive/quota

**✅ Runtime Settings**
```
1. Open Google Colab
2. Runtime → Change runtime type
3. Hardware accelerator: GPU
4. GPU type: T4 or V100 (if available)
5. Save
```

---

## 3. Step-by-Step Training Process

### 3.1 Training Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  TRAINING WORKFLOW                           │
└─────────────────────────────────────────────────────────────┘

STEP 1: Setup (5 minutes)
├─ Mount Google Drive
├─ Install dependencies
├─ Import libraries
└─ Verify GPU availability

STEP 2: Data Loading (10 minutes)
├─ Load processed CSVs from Google Drive
├─ Validate data quality
├─ Split train/validation (80/20)
└─ Verify feature columns

STEP 3: Environment Creation (15 minutes)
├─ Implement MultiPairForexEnv class
├─ Configure reward function
├─ Test environment with random agent
└─ Validate observation/action spaces

STEP 4: Model Setup (10 minutes)
├─ Initialize PPO model
├─ Configure hyperparameters
├─ Setup TensorBoard logging
└─ Create evaluation callbacks

STEP 5: Training (12-24 hours) ← LONG RUNNING
├─ Train for 500,000 timesteps
├─ Save checkpoints every 50k steps
├─ Evaluate every 10k steps
└─ Monitor via TensorBoard

STEP 6: Evaluation (1 hour)
├─ Backtest on validation set
├─ Calculate performance metrics
├─ Analyze failure cases
└─ Generate report

STEP 7: Model Export (15 minutes)
├─ Save final model
├─ Save best model (highest validation score)
├─ Save configuration & scaler
└─ Download to local machine
```

---

## 4. Complete Training Pseudocode

### 4.1 High-Level Pseudocode

```python
# FOREX_TRAINING_PSEUDOCODE.py
# High-level overview of training process

def main_training_pipeline():
    """Complete training pipeline from data to model"""
    
    # === PHASE 1: SETUP ===
    GPU = verify_gpu_availability()
    if not GPU:
        raise Error("GPU required for training!")
    
    DRIVE = mount_google_drive()
    install_dependencies()
    
    # === PHASE 2: DATA LOADING ===
    data = {}
    for pair in ["EURUSD", "GBPUSD", "USDJPY"]:
        df = load_processed_data(pair, source=DRIVE)
        validate_data(df)
        data[pair] = df
    
    # Split data
    train_data = split_data(data, start="2015-01-01", end="2023-12-31")
    val_data = split_data(data, start="2024-01-01", end="2024-12-31")
    
    # === PHASE 3: ENVIRONMENT CREATION ===
    train_env = MultiPairForexEnv(
        data_dict=train_data,
        pairs=["EURUSD", "GBPUSD", "USDJPY"],
        config=load_config("training_config.yaml")
    )
    
    val_env = MultiPairForexEnv(
        data_dict=val_data,
        pairs=["EURUSD", "GBPUSD", "USDJPY"],
        config=load_config("training_config.yaml")
    )
    
    # Test environment
    test_environment(train_env, num_steps=1000)
    
    # === PHASE 4: MODEL INITIALIZATION ===
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path="./models/best/",
        eval_freq=10000,
        deterministic=True
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/checkpoints/"
    )
    
    # === PHASE 5: TRAINING ===
    print("🚀 Starting training for 500,000 timesteps...")
    print(f"📊 Estimated duration: 12-24 hours on {GPU}")
    
    model.learn(
        total_timesteps=500000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # === PHASE 6: EVALUATION ===
    print("📈 Evaluating final model...")
    
    metrics = evaluate_model(
        model=model,
        env=val_env,
        n_episodes=100
    )
    
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # === PHASE 7: MODEL EXPORT ===
    model.save("./models/forex_ppo_final")
    save_config(model, "./models/forex_ppo_final_config.json")
    save_scaler(train_env.scaler, "./models/scaler.pkl")
    
    # Download to local
    download_to_local([
        "./models/forex_ppo_final.zip",
        "./models/best/best_model.zip",
        "./models/forex_ppo_final_config.json",
        "./models/scaler.pkl"
    ])
    
    print("✅ Training complete! Model ready for desktop deployment.")

# Run training
if __name__ == "__main__":
    main_training_pipeline()
```

---

## 5. Detailed Implementation Code

### 5.1 Google Colab Notebook (Complete)

```python
# ============================================================
# FOREX PPO TRAINING - GOOGLE COLAB NOTEBOOK
# Based on proven stock trading notebook + forex adaptations
# ============================================================

# ============================================================
# CELL 1: Setup & Installation (Run First)
# ============================================================

# Install dependencies
!pip install -q git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support
!pip install -q git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support
!pip install -q gymnasium shimmy>=0.2.1 finta quantstats pyyaml

print("✅ Dependencies installed")

# ============================================================
# CELL 2: Mount Google Drive
# ============================================================

from google.colab import drive
drive.mount('/content/drive/')

print("✅ Google Drive mounted")

# ============================================================
# CELL 3: Import Libraries
# ============================================================

# Core libraries
import os
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Gym/Gymnasium
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Technical indicators
from finta import TA

# Performance metrics
import quantstats as qs

# PyTorch (check GPU)
import torch

print("✅ Libraries imported")

# Verify GPU
print(f"🔥 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔥 GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================
# CELL 4: Load Configuration
# ============================================================

# Load training config from Google Drive
config_path = "/content/drive/MyDrive/forex-training/training_config.yaml"

with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

print("✅ Configuration loaded:")
print(f"  Pairs: {CONFIG['training']['pairs']}")
print(f"  Total timesteps: {CONFIG['training']['training']['total_timesteps']}")
print(f"  Window size: {CONFIG['training']['environment']['window_size']}")

# ============================================================
# CELL 5: Load Processed Data
# ============================================================

def load_forex_data(pair: str, data_path: str) -> pd.DataFrame:
    """Load processed forex data for a pair"""
    file_path = f"{data_path}/{pair}_processed.csv"
    
    print(f"Loading {pair} data from {file_path}...")
    
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"  ✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Validate
    assert len(df) > 100000, f"Need at least 100k candles for {pair}"
    assert df.isnull().sum().sum() == 0, f"Found NaN values in {pair}"
    
    return df

# Load all pairs
data_path = "/content/drive/MyDrive/forex-training/processed_data"
pairs = CONFIG['training']['pairs']

all_data = {}
for pair in pairs:
    all_data[pair] = load_forex_data(pair, data_path)

print(f"\n✅ All data loaded: {sum(len(df) for df in all_data.values())} total candles")

# ============================================================
# CELL 6: Split Train/Validation Data
# ============================================================

def split_data(data: Dict[str, pd.DataFrame], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Split data by date range"""
    split_data = {}
    
    for pair, df in data.items():
        mask = (df.index >= start) & (df.index <= end)
        split_data[pair] = df[mask].copy()
        print(f"{pair}: {len(split_data[pair])} candles ({start} to {end})")
    
    return split_data

# Split data
train_data = split_data(
    all_data,
    CONFIG['training']['data']['train_start'],
    CONFIG['training']['data']['train_end']
)

val_data = split_data(
    all_data,
    CONFIG['training']['data']['val_start'],
    CONFIG['training']['data']['val_end']
)

print(f"\n✅ Data split complete")
print(f"  Train: {sum(len(df) for df in train_data.values())} candles")
print(f"  Validation: {sum(len(df) for df in val_data.values())} candles")

# ============================================================
# CELL 7: Define MultiPairForexEnv (CRITICAL!)
# ============================================================

class MultiPairForexEnv(gym.Env):
    """
    Multi-pair forex trading environment for RL training
    
    Based on gym-anytrading StocksEnv but adapted for:
    - Multiple currency pairs simultaneously
    - Custom reward function for "$100 in 4 candles"
    - Forex-specific position management
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        pairs: List[str],
        config: Dict
    ):
        super().__init__()
        
        self.pairs = pairs
        self.data = data_dict
        self.config = config
        
        # Extract config
        self.window_size = config['training']['environment']['window_size']
        self.initial_balance = config['training']['environment']['initial_balance']
        self.trading_fees = config['training']['environment']['trading_fees']
        self.reward_config = config['training']['reward']
        
        # Feature columns (all indicators)
        self.feature_cols = config['training']['data']['features']
        
        # Action space: 3 actions per pair (HOLD=0, BUY=1, SELL=2)
        num_actions = len(pairs) * 3
        self.action_space = spaces.Discrete(num_actions)
        
        # Observation space: window_size × features × pairs
        num_features = len(self.feature_cols)
        obs_size = self.window_size * num_features * len(pairs)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {}
        self.closed_positions = []
        self.trade_history = []
        
        # For each pair, get min/max steps
        self.min_steps = min(len(df) for df in data_dict.values())
        
        print(f"Environment initialized:")
        print(f"  Pairs: {len(pairs)}")
        print(f"  Action space: {self.action_space.n} actions")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"  Max steps per episode: {self.min_steps - self.window_size}")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.positions = {}
        self.closed_positions = []
        self.trade_history = []
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: int):
        """Execute one step"""
        # Decode action (which pair, what action)
        pair_idx = action // 3
        action_type = action % 3  # 0=HOLD, 1=BUY, 2=SELL
        
        pair = self.pairs[pair_idx]
        
        # Execute trade
        self._execute_action(pair, action_type)
        
        # Update existing positions
        self._update_positions()
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        terminated = self._is_done()
        truncated = False
        
        # Get new observation
        obs = self._get_observation()
        
        # Info
        info = {
            'balance': self.balance,
            'equity': self._calculate_equity(),
            'positions': len(self.positions),
            'total_profit': sum(p.profit for p in self.closed_positions),
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current market state observation"""
        obs_list = []
        
        for pair in self.pairs:
            # Get window of data
            start_idx = self.current_step - self.window_size
            end_idx = self.current_step
            
            window_data = self.data[pair].iloc[start_idx:end_idx]
            
            # Extract features
            features = window_data[self.feature_cols].values
            
            # Normalize (simple min-max for now)
            # TODO: Use sklearn StandardScaler fitted on training data
            features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            # Flatten
            obs_list.append(features_normalized.flatten())
        
        # Concatenate all pairs
        observation = np.concatenate(obs_list).astype(np.float32)
        
        return observation
    
    def _execute_action(self, pair: str, action_type: int):
        """Execute trading action"""
        current_price = self._get_current_price(pair)
        current_atr = self._get_current_atr(pair)
        
        if action_type == 0:  # HOLD
            return
        
        # Only open new position if none exists for this pair
        if pair not in self.positions:
            
            if action_type == 1:  # BUY
                position = {
                    'pair': pair,
                    'type': 'BUY',
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'stop_loss': current_price - (2 * current_atr),
                    'take_profit': current_price + (3 * current_atr),
                    'size': 0.01  # 0.01 lot = 1000 units
                }
                self.positions[pair] = position
                self.trade_history.append(('OPEN_BUY', pair, self.current_step, current_price))
            
            elif action_type == 2:  # SELL
                position = {
                    'pair': pair,
                    'type': 'SELL',
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'stop_loss': current_price + (2 * current_atr),
                    'take_profit': current_price - (3 * current_atr),
                    'size': 0.01
                }
                self.positions[pair] = position
                self.trade_history.append(('OPEN_SELL', pair, self.current_step, current_price))
    
    def _update_positions(self):
        """Update existing positions (check SL/TP)"""
        to_close = []
        
        for pair, position in self.positions.items():
            current_price = self._get_current_price(pair)
            
            # Check stop loss
            if position['type'] == 'BUY' and current_price <= position['stop_loss']:
                to_close.append((pair, 'STOP_LOSS'))
            elif position['type'] == 'SELL' and current_price >= position['stop_loss']:
                to_close.append((pair, 'STOP_LOSS'))
            
            # Check take profit
            elif position['type'] == 'BUY' and current_price >= position['take_profit']:
                to_close.append((pair, 'TAKE_PROFIT'))
            elif position['type'] == 'SELL' and current_price <= position['take_profit']:
                to_close.append((pair, 'TAKE_PROFIT'))
        
        # Close positions
        for pair, reason in to_close:
            self._close_position(pair, reason)
    
    def _close_position(self, pair: str, reason: str):
        """Close a position"""
        position = self.positions[pair]
        current_price = self._get_current_price(pair)
        
        # Calculate profit
        if position['type'] == 'BUY':
            profit_pips = (current_price - position['entry_price']) * 10000
        else:  # SELL
            profit_pips = (position['entry_price'] - current_price) * 10000
        
        # Convert pips to USD (approximate)
        profit_usd = profit_pips * position['size'] * 10
        
        # Apply trading fees
        profit_usd -= (self.trading_fees * position['entry_price'] * position['size'] * 100000)
        
        # Update balance
        self.balance += profit_usd
        
        # Store closed position
        closed_pos = {
            'pair': pair,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'entry_step': position['entry_step'],
            'exit_step': self.current_step,
            'duration_candles': self.current_step - position['entry_step'],
            'profit_pips': profit_pips,
            'profit_usd': profit_usd,
            'reason': reason
        }
        
        self.closed_positions.append(closed_pos)
        self.trade_history.append(('CLOSE', pair, self.current_step, current_price, reason))
        
        # Remove from open positions
        del self.positions[pair]
    
    def _calculate_reward(self) -> float:
        """
        CRITICAL: Reward function encodes trading goals
        
        Goal: $100 profit from $1000 in 4 candles (15-min = 1 hour total)
        """
        reward = 0.0
        
        # Only calculate rewards for recently closed positions
        for position in self.closed_positions[-5:]:  # Last 5 closed
            profit = position['profit_usd']
            duration = position['duration_candles']
            
            # TIER 1: Perfect execution (TARGET!)
            if profit >= self.reward_config['target_profit'] and duration <= self.reward_config['target_duration']:
                reward += self.reward_config['profit_bonus']  # +500
            
            # TIER 2: Good profit, slightly slower
            elif profit >= self.reward_config['target_profit'] and duration <= 6:
                reward += self.reward_config['profit_bonus'] * 0.6  # +300
            
            # TIER 3: Quick smaller profit
            elif profit >= 50 and duration <= self.reward_config['target_duration']:
                reward += self.reward_config['quick_profit_bonus']  # +150
            
            # TIER 4: Any profit
            elif profit > 0:
                reward += profit * 2
            
            # TIER 5: Small loss
            elif profit >= -20:
                reward -= abs(profit) * 2
            
            # TIER 6: Large loss
            else:
                reward -= abs(profit) * self.reward_config['loss_penalty_multiplier']  # ×3
            
            # PENALTY: Holding too long
            if duration > 6:
                reward -= 50 * (duration - 6)
        
        # PENALTY: Excessive drawdown
        current_equity = self._calculate_equity()
        drawdown = max(0, (self.initial_balance - current_equity) / self.initial_balance)
        
        if drawdown > 0.05:  # > 5%
            reward -= self.reward_config['max_drawdown_penalty'] * drawdown  # -200 × dd
        
        # PENALTY: Over-trading
        recent_trades = len([t for t in self.trade_history[-20:] if t[0].startswith('OPEN')])
        if recent_trades > 5:
            reward -= self.reward_config['overtrading_penalty'] * (recent_trades - 5)
        
        return reward
    
    def _calculate_equity(self) -> float:
        """Calculate current equity (balance + unrealized P&L)"""
        equity = self.balance
        
        for pair, position in self.positions.items():
            current_price = self._get_current_price(pair)
            
            if position['type'] == 'BUY':
                unrealized_pips = (current_price - position['entry_price']) * 10000
            else:
                unrealized_pips = (position['entry_price'] - current_price) * 10000
            
            unrealized_usd = unrealized_pips * position['size'] * 10
            equity += unrealized_usd
        
        return equity
    
    def _get_current_price(self, pair: str) -> float:
        """Get current close price for pair"""
        return self.data[pair].iloc[self.current_step]['close']
    
    def _get_current_atr(self, pair: str) -> float:
        """Get current ATR for pair"""
        return self.data[pair].iloc[self.current_step]['ATR']
    
    def _is_done(self) -> bool:
        """Check if episode should end"""
        # End if reached end of data
        if self.current_step >= self.min_steps - 1:
            return True
        
        # End if lost 50% of capital
        if self.balance <= self.initial_balance * 0.5:
            return True
        
        return False
    
    def render(self):
        """Print current state (optional)"""
        print(f"Step: {self.current_step} | Balance: ${self.balance:.2f} | "
              f"Equity: ${self._calculate_equity():.2f} | Positions: {len(self.positions)}")

print("✅ MultiPairForexEnv defined")

# ============================================================
# CELL 8: Create Training Environment
# ============================================================

# Wrap environment in Monitor for logging
train_env = MultiPairForexEnv(
    data_dict=train_data,
    pairs=CONFIG['training']['pairs'],
    config=CONFIG
)

train_env = Monitor(train_env)

# Wrap in DummyVecEnv (required by Stable-Baselines3)
train_env = DummyVecEnv([lambda: train_env])

print("✅ Training environment created")

# ============================================================
# CELL 9: Create Validation Environment
# ============================================================

val_env = MultiPairForexEnv(
    data_dict=val_data,
    pairs=CONFIG['training']['pairs'],
    config=CONFIG
)

val_env = Monitor(val_env)

print("✅ Validation environment created")

# ============================================================
# CELL 10: Test Environment (Quick Check)
# ============================================================

print("Testing environment with random agent...")

obs = train_env.reset()
total_reward = 0

for step in range(1000):
    action = train_env.action_space.sample()  # Random action
    obs, reward, done, info = train_env.step(action)
    total_reward += reward[0]
    
    if done[0]:
        break

print(f"✅ Test complete: {step} steps, Total reward: {total_reward:.2f}")

# ============================================================
# CELL 11: Initialize PPO Model
# ============================================================

ppo_config = CONFIG['training']['ppo']

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=ppo_config['learning_rate'],
    n_steps=ppo_config['n_steps'],
    batch_size=ppo_config['batch_size'],
    n_epochs=ppo_config['n_epochs'],
    gamma=ppo_config['gamma'],
    gae_lambda=ppo_config['gae_lambda'],
    clip_range=ppo_config['clip_range'],
    ent_coef=ppo_config['ent_coef'],
    vf_coef=ppo_config['vf_coef'],
    max_grad_norm=ppo_config['max_grad_norm'],
    verbose=1,
    tensorboard_log="/content/drive/MyDrive/forex-training/logs/tensorboard/",
    device="cuda"
)

print("✅ PPO model initialized")
print(f"   Device: {model.device}")
print(f"   Policy: {model.policy}")

# ============================================================
# CELL 12: Setup Callbacks
# ============================================================

# Evaluation callback (save best model)
eval_callback = EvalCallback(
    val_env,
    best_model_save_path="/content/drive/MyDrive/forex-training/models/best/",
    log_path="/content/drive/MyDrive/forex-training/logs/eval/",
    eval_freq=CONFIG['training']['training']['eval_freq'],
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1
)

# Checkpoint callback (save periodically)
checkpoint_callback = CheckpointCallback(
    save_freq=CONFIG['training']['training']['save_freq'],
    save_path="/content/drive/MyDrive/forex-training/models/checkpoints/",
    name_prefix="forex_ppo",
    save_replay_buffer=False,
    save_vecnormalize=False,
    verbose=1
)

print("✅ Callbacks configured")

# ============================================================
# CELL 13: START TRAINING (LONG RUNNING - 12-24 HOURS)
# ============================================================

print("=" * 60)
print("🚀 STARTING PPO TRAINING")
print("=" * 60)
print(f"Total timesteps: {CONFIG['training']['training']['total_timesteps']}")
print(f"Estimated duration: 12-24 hours on {torch.cuda.get_device_name(0)}")
print(f"Training data: {sum(len(df) for df in train_data.values())} candles")
print(f"Validation data: {sum(len(df) for df in val_data.values())} candles")
print("")
print("💡 Monitor training progress:")
print("   1. Check TensorBoard: logs/tensorboard/")
print("   2. Watch eval callbacks every 10k steps")
print("   3. Checkpoints saved every 50k steps")
print("")
print("⚠️  DO NOT close this tab! Training will stop if tab closes.")
print("=" * 60)

# Start training
model.learn(
    total_timesteps=CONFIG['training']['training']['total_timesteps'],
    callback=[eval_callback, checkpoint_callback],
    log_interval=CONFIG['training']['training']['log_interval'],
    progress_bar=True
)

print("")
print("=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)

# ============================================================
# CELL 14: Save Final Model
# ============================================================

model.save("/content/drive/MyDrive/forex-training/models/forex_ppo_final")

print("✅ Final model saved to Google Drive")

# ============================================================
# CELL 15: Evaluate Final Model
# ============================================================

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(
    model,
    val_env,
    n_eval_episodes=50,
    deterministic=True,
    render=False
)

print(f"📊 Final Model Performance (Validation Set):")
print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

# ============================================================
# CELL 16: Detailed Backtest Analysis
# ============================================================

def detailed_backtest(model, env, n_episodes=10):
    """Run detailed backtest and collect metrics"""
    
    all_trades = []
    episode_rewards = []
    episode_balances = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        # Collect episode data
        episode_rewards.append(episode_reward)
        episode_balances.append(info[0]['balance'])
        
        # Collect trades from this episode
        closed_positions = env.envs[0].env.closed_positions
        all_trades.extend(closed_positions)
    
    # Calculate metrics
    trades_df = pd.DataFrame(all_trades)
    
    if len(trades_df) > 0:
        wins = trades_df[trades_df['profit_usd'] > 0]
        losses = trades_df[trades_df['profit_usd'] <= 0]
        
        metrics = {
            'total_trades': len(trades_df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_win': wins['profit_usd'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['profit_usd'].mean() if len(losses) > 0 else 0,
            'profit_factor': abs(wins['profit_usd'].sum() / losses['profit_usd'].sum()) if len(losses) > 0 and losses['profit_usd'].sum() != 0 else 0,
            'total_profit': trades_df['profit_usd'].sum(),
            'avg_duration': trades_df['duration_candles'].mean(),
            'avg_reward_per_episode': np.mean(episode_rewards),
            'final_balance_avg': np.mean(episode_balances)
        }
    else:
        metrics = {'error': 'No trades executed'}
    
    return metrics, trades_df

print("Running detailed backtest...")
metrics, trades_df = detailed_backtest(model, val_env, n_episodes=20)

print("\n" + "=" * 60)
print("📈 BACKTEST RESULTS (20 Episodes on Validation Data)")
print("=" * 60)
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"   {key}: {value:.2f}")
    else:
        print(f"   {key}: {value}")
print("=" * 60)

# ============================================================
# CELL 17: Visualize Trading Performance
# ============================================================

if len(trades_df) > 0:
    # Plot profit distribution
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(trades_df['profit_usd'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', label='Break-even')
    plt.xlabel('Profit (USD)')
    plt.ylabel('Frequency')
    plt.title('Trade Profit Distribution')
    plt.legend()
    
    # Plot duration vs profit
    plt.subplot(1, 2, 2)
    colors = ['green' if p > 0 else 'red' for p in trades_df['profit_usd']]
    plt.scatter(trades_df['duration_candles'], trades_df['profit_usd'], c=colors, alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axvline(x=4, color='blue', linestyle='--', label='Target Duration (4 candles)')
    plt.xlabel('Duration (candles)')
    plt.ylabel('Profit (USD)')
    plt.title('Trade Duration vs Profit')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/forex-training/results/backtest_analysis.png', dpi=150)
    plt.show()
    
    print("✅ Visualizations saved")

# ============================================================
# CELL 18: Export Model for Desktop Use
# ============================================================

import zipfile
import shutil

print("Preparing model for desktop deployment...")

# Create export directory
export_dir = "/content/drive/MyDrive/forex-training/export"
os.makedirs(export_dir, exist_ok=True)

# Copy files
shutil.copy(
    "/content/drive/MyDrive/forex-training/models/forex_ppo_final.zip",
    f"{export_dir}/forex_ppo_final.zip"
)

shutil.copy(
    "/content/drive/MyDrive/forex-training/models/best/best_model.zip",
    f"{export_dir}/best_model.zip"
)

# Save config
import json
with open(f"{export_dir}/model_config.json", 'w') as f:
    json.dump(CONFIG, f, indent=2)

# Save metrics
with open(f"{export_dir}/training_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Model exported to: {export_dir}")
print("\nDownload these files to your desktop:")
print(f"  1. forex_ppo_final.zip (final model)")
print(f"  2. best_model.zip (best validation performance)")
print(f"  3. model_config.json (configuration)")
print(f"  4. training_metrics.json (performance metrics)")

# ============================================================
# CELL 19: Generate Training Report
# ============================================================

report = f"""
╔══════════════════════════════════════════════════════════╗
║          FOREX PPO TRAINING - FINAL REPORT              ║
╚══════════════════════════════════════════════════════════╝

📅 Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 TRAINING CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pairs:              {', '.join(CONFIG['training']['pairs'])}
Timeframe:          {CONFIG['training']['data']['timeframe']}
Training Period:    {CONFIG['training']['data']['train_start']} to {CONFIG['training']['data']['train_end']}
Validation Period:  {CONFIG['training']['data']['val_start']} to {CONFIG['training']['data']['val_end']}

Total Timesteps:    {CONFIG['training']['training']['total_timesteps']:,}
Learning Rate:      {CONFIG['training']['ppo']['learning_rate']}
Batch Size:         {CONFIG['training']['ppo']['batch_size']}
Epochs per Update:  {CONFIG['training']['ppo']['n_epochs']}

📈 PERFORMANCE METRICS (Validation Set)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades:       {metrics.get('total_trades', 'N/A')}
Win Rate:           {metrics.get('win_rate', 0)*100:.2f}%
Wins:               {metrics.get('wins', 'N/A')}
Losses:             {metrics.get('losses', 'N/A')}

Average Win:        ${metrics.get('avg_win', 0):.2f}
Average Loss:       ${metrics.get('avg_loss', 0):.2f}
Profit Factor:      {metrics.get('profit_factor', 0):.2f}

Total Profit:       ${metrics.get('total_profit', 0):.2f}
Avg Duration:       {metrics.get('avg_duration', 0):.1f} candles

Mean Episode Reward: {metrics.get('avg_reward_per_episode', 0):.2f}
Final Balance (avg):${metrics.get('final_balance_avg', 0):.2f}

🎯 GOAL ACHIEVEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target:             $100 profit in 4 candles
"""

# Calculate how many trades hit target
if len(trades_df) > 0:
    target_hits = len(trades_df[
        (trades_df['profit_usd'] >= 100) & 
        (trades_df['duration_candles'] <= 4)
    ])
    target_rate = target_hits / len(trades_df) * 100
    report += f"Target Hits:        {target_hits} trades ({target_rate:.1f}%)\n"

report += f"""
✅ NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Download model files from: {export_dir}
2. Copy to desktop: ~/forex-trading-system/models/
3. Run inference engine to test on real-time data
4. Deploy to production with paper trading first
5. Monitor performance and retrain monthly

╚══════════════════════════════════════════════════════════╝
"""

print(report)

# Save report
with open(f"{export_dir}/TRAINING_REPORT.txt", 'w') as f:
    f.write(report)

print(f"✅ Report saved to: {export_dir}/TRAINING_REPORT.txt")

# ============================================================
# END OF TRAINING NOTEBOOK
# ============================================================
```

---

## 6. Configuration Files

### 6.1 training_config.yaml (Complete)

```yaml
# config/training_config.yaml
# Complete training configuration for Google Colab

training:
  # Currency pairs to train on
  pairs:
    - EURUSD
    - GBPUSD
    - USDJPY
  
  # Data configuration
  data:
    timeframe: "15m"
    
    # Training period (80% of data)
    train_start: "2015-01-01"
    train_end: "2023-12-31"
    
    # Validation period (20% of data)
    val_start: "2024-01-01"
    val_end: "2024-12-31"
    
    # Feature columns (indicators calculated in preprocessing)
    features:
      # Price features
      - open
      - high
      - low
      - close
      - return
      
      # Trend indicators
      - RSI
      - SMA_20
      - SMA_50
      - EMA_20
      - EMA_50
      - MACD
      - MACD_signal
      
      # Volatility indicators
      - BB_upper
      - BB_middle
      - BB_lower
      - ATR
      
      # Volume
      - OBV
      
      # Custom features
      - momentum
      - volatility
      - distance
      - prediction
  
  # Environment configuration
  environment:
    window_size: 100          # Historical candles to observe
    initial_balance: 10000.0  # Starting capital (USD)
    trading_fees: 0.0002      # 0.02% spread per trade
    max_trades_per_pair: 1    # Max open positions per pair
  
  # Reward function configuration (CRITICAL!)
  reward:
    target_profit: 100        # Target profit in USD
    target_duration: 4        # Target duration in candles
    profit_bonus: 500         # Bonus for hitting target
    quick_profit_bonus: 150   # Bonus for quick smaller profits
    loss_penalty_multiplier: 3  # Multiplier for loss penalties
    overtrading_penalty: 100  # Penalty per excess trade
    max_drawdown_penalty: 200  # Penalty for drawdown
  
  # PPO algorithm configuration
  ppo:
    algorithm: "PPO"
    learning_rate: 0.0003     # Adam optimizer learning rate
    n_steps: 2048             # Steps per rollout
    batch_size: 64            # Mini-batch size for training
    n_epochs: 10              # Epochs per policy update
    gamma: 0.99               # Discount factor
    gae_lambda: 0.95          # GAE parameter
    clip_range: 0.2           # PPO clipping parameter
    ent_coef: 0.01            # Entropy coefficient (exploration)
    vf_coef: 0.5              # Value function coefficient
    max_grad_norm: 0.5        # Gradient clipping
  
  # Training loop configuration
  training:
    total_timesteps: 500000   # Total training steps
    eval_freq: 10000          # Evaluate every N steps
    save_freq: 50000          # Save checkpoint every N steps
    log_interval: 100         # Log metrics every N steps
  
  # Hardware configuration
  hardware:
    device: "cuda"            # GPU required
    use_fp16: false           # Mixed precision training (experimental)
```

---

## 7. Monitoring & Debugging

### 7.1 TensorBoard Monitoring

```bash
# In a separate Colab cell, run TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/forex-training/logs/tensorboard/
```

**Key Metrics to Monitor:**

| Metric | What It Means | Target |
|--------|---------------|--------|
| `ep_rew_mean` | Average episode reward | Should increase over time |
| `ep_len_mean` | Average episode length | Should stabilize |
| `policy_loss` | Policy network loss | Should decrease |
| `value_loss` | Value network loss | Should decrease |
| `explained_variance` | How well value function predicts returns | Should be > 0.8 |
| `learning_rate` | Current learning rate | Constant (unless scheduler used) |
| `entropy_loss` | Exploration incentive | Should decrease slowly |

**Good Training Progress:**
```
Step 10000:  ep_rew_mean=-500,  explained_variance=0.2
Step 50000:  ep_rew_mean=-100,  explained_variance=0.5
Step 100000: ep_rew_mean=200,   explained_variance=0.7
Step 200000: ep_rew_mean=600,   explained_variance=0.85
Step 500000: ep_rew_mean=1200,  explained_variance=0.92  ✅ GOOD!
```

---

### 7.2 Real-Time Progress Tracking

```python
# Add this to training cell for detailed progress
import time
from IPython.display import clear_output

class TrainingProgressCallback:
    def __init__(self):
        self.start_time = time.time()
        self.last_log_time = time.time()
    
    def __call__(self, locals, globals):
        # Log every 1000 steps
        if locals['self'].num_timesteps % 1000 == 0:
            elapsed = time.time() - self.start_time
            steps = locals['self'].num_timesteps
            total_steps = 500000
            
            progress = steps / total_steps * 100
            eta = (elapsed / steps) * (total_steps - steps)
            
            clear_output(wait=True)
            print(f"Progress: {progress:.1f}% | Step: {steps}/{total_steps}")
            print(f"Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")
            print(f"Recent reward: {locals.get('reward', 0):.2f}")

# Use in training
model.learn(
    total_timesteps=500000,
    callback=[eval_callback, checkpoint_callback, TrainingProgressCallback()]
)
```

---

## 8. Post-Training Validation

### 8.1 Validation Checklist

**✅ Task 1: Load Best Model**
```python
# Load the best model (highest validation score)
best_model = PPO.load("/content/drive/MyDrive/forex-training/models/best/best_model")

# Compare to final model
final_model = PPO.load("/content/drive/MyDrive/forex-training/models/forex_ppo_final")
```

**✅ Task 2: Comprehensive Backtest**
```python
# Run 100 episodes on validation data
metrics, trades = detailed_backtest(best_model, val_env, n_episodes=100)

# Check if meets minimum standards
assert metrics['win_rate'] >= 0.60, "Win rate too low!"
assert metrics['profit_factor'] >= 1.5, "Profit factor too low!"
assert metrics['total_profit'] > 0, "Model is losing money!"
```

**✅ Task 3: Analyze Failure Cases**
```python
# Get worst trades
worst_trades = trades.nsmallest(20, 'profit_usd')

print("Worst trades:")
print(worst_trades[['pair', 'type', 'duration_candles', 'profit_usd', 'reason']])

# Look for patterns
print("\nFailure analysis:")
print(f"Most losses on pair: {worst_trades['pair'].mode()[0]}")
print(f"Avg duration of losses: {worst_trades['duration_candles'].mean():.1f} candles")
```

**✅ Task 4: Test on Unseen Data (If Available)**
```python
# If you have 2025 data, test on completely unseen period
test_data = load_data(pairs, start="2025-01-01", end="2025-09-30")
test_env = MultiPairForexEnv(test_data, pairs, CONFIG)

test_metrics, test_trades = detailed_backtest(best_model, test_env, n_episodes=50)

print("Performance on completely unseen data:")
print(test_metrics)
```

---

## 9. Troubleshooting Guide

### 9.1 Common Training Issues

**Issue 1: Model Not Learning (Reward Stays Negative)**

**Symptoms:**
- `ep_rew_mean` stays around -500 to -1000
- `explained_variance` < 0.3

**Solutions:**
```python
# 1. Reduce reward penalties (too harsh)
CONFIG['training']['reward']['loss_penalty_multiplier'] = 2  # Was 3

# 2. Increase profit bonuses (more incentive)
CONFIG['training']['reward']['profit_bonus'] = 1000  # Was 500

# 3. Simplify reward function (remove overtrading penalty initially)
# Comment out overtrading penalty in _calculate_reward()

# 4. Increase learning rate
CONFIG['training']['ppo']['learning_rate'] = 0.001  # Was 0.0003
```

---

**Issue 2: Model Only Learns to HOLD (Risk Averse)**

**Symptoms:**
- Very few trades executed
- All actions are HOLD

**Solutions:**
```python
# 1. Add penalty for excessive holding
def _calculate_reward(self):
    reward = ...
    
    # NEW: Penalty for not trading
    if len(self.closed_positions) == 0 and self.current_step > 1000:
        reward -= 10  # Small penalty for inaction
    
    return reward

# 2. Increase entropy coefficient (more exploration)
CONFIG['training']['ppo']['ent_coef'] = 0.05  # Was 0.01

# 3. Add reward for simply taking trades (not just profit)
if len(self.closed_positions) > 0:
    reward += 10  # Small bonus for any trade
```

---

**Issue 3: Training Too Slow**

**Symptoms:**
- < 1000 steps/minute
- ETA > 48 hours

**Solutions:**
```python
# 1. Reduce window_size
CONFIG['training']['environment']['window_size'] = 50  # Was 100

# 2. Reduce n_steps
CONFIG['training']['ppo']['n_steps'] = 1024  # Was 2048

# 3. Use GPU acceleration
assert torch.cuda.is_available(), "GPU required!"
model = PPO(..., device="cuda")

# 4. Reduce total_timesteps (compromise on quality)
CONFIG['training']['training']['total_timesteps'] = 300000  # Was 500000
```

---

**Issue 4: Colab Disconnects Mid-Training**

**Solutions:**
```python
# 1. Enable Colab Pro (longer runtime)

# 2. Add auto-save every 10k steps
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # More frequent saves
    save_path="./models/checkpoints/"
)

# 3. Resume from checkpoint if disconnected
if os.path.exists("./models/checkpoints/forex_ppo_450000_steps.zip"):
    print("Resuming from checkpoint...")
    model = PPO.load("./models/checkpoints/forex_ppo_450000_steps")
    model.set_env(train_env)
    remaining_steps = 500000 - 450000
    model.learn(total_timesteps=remaining_steps, reset_num_timesteps=False)
```

---

## 10. Model Export for Desktop

### 10.1 Files to Download

After training completes, download these files to your desktop:

```bash
# On your local machine (WSL)
cd ~/forex-trading-system/models/

# Create directory for trained model
mkdir -p trained_model_2025_10

# Download from Google Drive (manually or via rclone)
# These files should be in: /content/drive/MyDrive/forex-training/export/

# Required files:
# 1. forex_ppo_final.zip (or best_model.zip)
# 2. model_config.json
# 3. training_metrics.json
# 4. scaler.pkl (if saved)
```

### 10.2 Loading Model on Desktop

```python
# On your desktop machine
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("models/trained_model_2025_10/forex_ppo_final")

# Verify model loaded correctly
print(f"Model device: {model.device}")
print(f"Policy: {model.policy}")

# Test inference
import numpy as np

dummy_obs = np.random.randn(1, 3 * 21 * 100).astype(np.float32)  # Dummy observation
action, _states = model.predict(dummy_obs)

print(f"✅ Model inference working! Action: {action}")
```

### 10.3 Integration with Desktop App

```python
# src/inference_engine.py

class ForexInferenceEngine:
    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)
        self.model.policy.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def predict(self, observation: np.ndarray) -> Dict:
        """Generate prediction from trained model"""
        action, _states = self.model.predict(observation, deterministic=True)
        
        # Get action probabilities
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            if torch.cuda.is_available():
                obs_tensor = obs_tensor.cuda()
            
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = torch.softmax(distribution.distribution.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
        
        # Decode action
        pair_idx = action // 3
        action_type = action % 3
        
        return {
            "pair_idx": pair_idx,
            "action": ["HOLD", "BUY", "SELL"][action_type],
            "confidence": float(probs[action]),
            "all_probs": probs.tolist()
        }
```

---

## TRAINING CHECKLIST (Summary)

```
PRE-TRAINING (Local Machine)
☐ Download 10 years of historical data (EURUSD, GBPUSD, USDJPY)
☐ Run preprocessing script to calculate indicators
☐ Validate data quality (no NaN, correct date ranges)
☐ Upload processed data to Google Drive
☐ Upload training_config.yaml to Google Drive
☐ Get Colab Pro subscription

TRAINING (Google Colab)
☐ Open Colab notebook
☐ Set runtime to GPU (T4 or V100)
☐ Run Cell 1-3: Setup, mount Drive, import libraries
☐ Run Cell 4-6: Load config and data
☐ Run Cell 7-9: Define environment, create train/val envs
☐ Run Cell 10: Test environment with random agent
☐ Run Cell 11-12: Initialize PPO model and callbacks
☐ Run Cell 13: START TRAINING (12-24 hours)
☐ Monitor TensorBoard during training
☐ Run Cell 14-18: Save model, evaluate, analyze
☐ Run Cell 19: Generate report

POST-TRAINING (Local Machine)
☐ Download model files from Google Drive
☐ Copy to ~/forex-trading-system/models/
☐ Test model loading on desktop
☐ Integrate with inference engine
☐ Run paper trading validation
☐ Deploy to production

MONTHLY MAINTENANCE
☐ Export last month's live trading data
☐ Append to historical dataset
☐ Re-upload to Google Drive
☐ Retrain model (can use fewer timesteps: 100k-200k)
☐ Download updated model
☐ Replace on desktop
```

---

## END OF TRAINING GUIDE

This guide is now complete and ready to use with Claude Code for implementation!
