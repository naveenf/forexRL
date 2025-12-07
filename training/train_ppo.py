"""
PPO Training Script for Forex Trading System

This script implements the complete training pipeline for the forex trading
reinforcement learning system using PPO algorithm with Gemma 2B model.

Optimized for Google Colab with GPU acceleration.
Target: $100 profit in 4 candles (15-minute timeframe)

Usage in Google Colab:
1. Upload this file and dependencies
2. Install requirements: !pip install -r requirements_colab.txt
3. Run: python train_ppo.py
"""

import os
import sys
import time
import yaml
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# Import RL frameworks
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO  # LSTM-based PPO for temporal learning
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Import custom modules (ensure these are uploaded to Colab)
try:
    from data_manager import DataManager
    from environment_single import SinglePairForexEnv  # CHANGED: Use single-pair environment
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Make sure to upload data_manager.py and environment_single.py to Colab")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class EnhancedForexCallback(BaseCallback):
    """
    Enhanced callback for comprehensive forex trading analytics.

    Tracks:
    - Detailed trade statistics and performance metrics
    - Risk-adjusted returns and drawdown analysis
    - Per-pair performance analysis
    - Real-time profitability indicators
    """

    def __init__(self, config: Dict[str, Any], verbose: int = 1):
        super().__init__(verbose)
        self.config = config

        # Trade tracking
        self.all_trades = []
        self.trades_by_pair = {'USDJPY': []}  # CHANGED: Single pair only
        self.successful_goal_trades = []

        # Episode tracking
        self.total_episodes = 0
        self.episode_rewards = []
        self.episode_balances = []
        self.episode_trade_counts = []

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_consecutive_losses = 0
        self.current_consecutive_losses = 0
        self.max_drawdown = 0.0
        self.peak_balance = 10000.0  # Starting balance

        # Risk metrics
        self.risk_metrics = []
        self.hourly_performance = {}

        # Analytics for export
        self.chunk_analytics = {}

        # Action distribution tracking (Fix #2 for "0 trades" issue)
        self.action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL

    def _on_step(self) -> bool:
        # Track action distribution (Fix #2 for "0 trades" issue)
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, (np.ndarray, list)):
                for action in np.atleast_1d(actions):
                    self.action_counts[int(action)] += 1

        # Extract info from the environment
        # FIX: self.locals is a dict, use 'in' instead of hasattr()
        if 'infos' in self.locals and self.locals['infos']:
            for info in self.locals['infos']:
                if isinstance(info, dict):
                    # Track closed position (CHANGED for single-pair)
                    if 'closed_position' in info and info['closed_position']:
                        pos_info = info['closed_position']
                        trade_data = {
                            'step': self.num_timesteps,
                            'pair': 'USDJPY',  # Single pair only
                            'pnl': pos_info.get('pnl', 0),
                            'duration': pos_info.get('duration', 0),
                            'reason': pos_info.get('reason', 'UNKNOWN'),
                            'timestamp': pd.Timestamp.now(),
                            'is_winning': pos_info.get('pnl', 0) > 0
                        }

                        # Add to trade records
                        self.all_trades.append(trade_data)
                        self.trades_by_pair['USDJPY'].append(trade_data)

                        # Update performance counters
                        self.total_trades += 1
                        pnl = trade_data['pnl']

                        if pnl > 0:
                            self.winning_trades += 1
                            self.total_profit += pnl
                            self.current_consecutive_losses = 0
                        else:
                            self.losing_trades += 1
                            self.total_loss += abs(pnl)
                            self.current_consecutive_losses += 1
                            self.max_consecutive_losses = max(
                                self.max_consecutive_losses,
                                self.current_consecutive_losses
                            )

                        # Track goal achievements (updated criteria)
                        if (pnl >= 37.50 and trade_data['duration'] <= 12):
                            self.successful_goal_trades.append(trade_data)
                            logger.info(f"🎯 PRIMARY GOAL! ${pnl:.2f} "
                                      f"in {trade_data['duration']} candles - "
                                      f"Pair: USDJPY")

                    # Track episode completion and balance
                    if 'episode' in info and info['episode']:
                        ep_info = info['episode']
                        self.episode_rewards.append(ep_info['r'])
                        self.total_episodes += 1

                        # Track balance for drawdown calculation (reset peak per episode)
                        current_balance = info.get('balance', 10000.0)
                        self.episode_balances.append(current_balance)

                        # Reset peak balance for each new episode to track per-episode drawdown
                        if len(self.episode_balances) == 1 or self.total_episodes == 0:
                            self.peak_balance = 10000.0  # Reset to initial balance

                        self.peak_balance = max(self.peak_balance, current_balance)

                        # Calculate drawdown within this episode
                        if current_balance < self.peak_balance:
                            drawdown = (self.peak_balance - current_balance) / self.peak_balance
                            self.max_drawdown = max(self.max_drawdown, drawdown)

                        # Count trades in this episode
                        episode_trades = len([t for t in self.all_trades
                                            if t['step'] >= self.num_timesteps - 1000])
                        self.episode_trade_counts.append(episode_trades)

                        # Enhanced logging every 5 episodes
                        if self.total_episodes % 5 == 0:
                            self._log_detailed_progress()

        return True

    def _log_detailed_progress(self):
        """Log detailed training progress with analytics."""
        recent_rewards = self.episode_rewards[-5:]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0

        # Calculate performance metrics
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        profit_factor = self.total_profit / max(abs(self.total_loss), 1)
        avg_profit = self.total_profit / max(self.winning_trades, 1)
        avg_loss = abs(self.total_loss) / max(self.losing_trades, 1)

        # Calculate mean duration for console display
        mean_duration = np.mean([trade['duration'] for trade in self.all_trades]) if self.all_trades else 0.0

        logger.info(f"📊 Episode {self.total_episodes} Analytics:")
        logger.info(f"  💰 Avg Reward: {avg_reward:.1f}")
        logger.info(f"  📈 Total Trades: {self.total_trades} (W:{self.winning_trades}, L:{self.losing_trades})")
        logger.info(f"  🎯 Win Rate: {win_rate:.1f}%")
        logger.info(f"  ⏱️  Mean Duration: {mean_duration:.1f} candles")
        logger.info(f"  💵 Profit Factor: {profit_factor:.2f}")
        logger.info(f"  ⚡ Avg Win: ${avg_profit:.2f}, Avg Loss: ${avg_loss:.2f}")
        logger.info(f"  📉 Max Drawdown: {self.max_drawdown*100:.2f}%")

        # Log action distribution (Fix #2 for "0 trades" issue)
        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            hold_pct = (self.action_counts[0] / total_actions) * 100
            buy_pct = (self.action_counts[1] / total_actions) * 100
            sell_pct = (self.action_counts[2] / total_actions) * 100
            logger.info(f"  🎬 Actions: HOLD {hold_pct:.1f}%, BUY {buy_pct:.1f}%, SELL {sell_pct:.1f}%")
        logger.info(f"  🔄 Max Consecutive Losses: {self.max_consecutive_losses}")
        logger.info(f"  🌟 Goal Trades: {len(self.successful_goal_trades)}")

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout with comprehensive TensorBoard logging."""
        # Basic performance metrics
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        profit_factor = self.total_profit / max(abs(self.total_loss), 1)

        # Recent performance (last rollout)
        recent_trades = [t for t in self.all_trades
                        if self.num_timesteps - t['step'] <= 2048]

        recent_wins = len([t for t in recent_trades if t['is_winning']])
        recent_win_rate = (recent_wins / max(len(recent_trades), 1)) * 100

        # Goal trade analysis
        recent_goals = [trade for trade in self.successful_goal_trades
                       if self.num_timesteps - trade['step'] <= 2048]

        # Calculate mean trade duration
        mean_duration = 0.0
        if self.all_trades:
            mean_duration = np.mean([trade['duration'] for trade in self.all_trades])

        # Log comprehensive metrics to TensorBoard
        self.logger.record("forex/total_trades", self.total_trades)
        self.logger.record("forex/win_rate", win_rate)
        self.logger.record("forex/profit_factor", profit_factor)
        self.logger.record("forex/max_drawdown", self.max_drawdown)
        self.logger.record("forex/max_consecutive_losses", self.max_consecutive_losses)
        self.logger.record("forex/mean_trade_duration", mean_duration)

        # Recent performance
        self.logger.record("forex/recent_trades_count", len(recent_trades))
        self.logger.record("forex/recent_win_rate", recent_win_rate)

        # Goal achievements
        self.logger.record("forex/total_goal_trades", len(self.successful_goal_trades))
        self.logger.record("forex/recent_goal_trades", len(recent_goals))

        if recent_goals:
            avg_goal_pnl = np.mean([trade['pnl'] for trade in recent_goals])
            avg_goal_duration = np.mean([trade['duration'] for trade in recent_goals])
            self.logger.record("forex/avg_goal_trade_pnl", avg_goal_pnl)
            self.logger.record("forex/avg_goal_trade_duration", avg_goal_duration)

        # Per-pair analysis
        for pair in self.trades_by_pair:
            pair_trades = self.trades_by_pair[pair]
            if pair_trades:
                pair_wins = len([t for t in pair_trades if t['is_winning']])
                pair_win_rate = (pair_wins / len(pair_trades)) * 100
                self.logger.record(f"forex/{pair.lower()}_win_rate", pair_win_rate)
                self.logger.record(f"forex/{pair.lower()}_trades", len(pair_trades))

    def export_analytics(self, chunk_num: int) -> Dict:
        """Export comprehensive analytics for this training chunk."""
        analytics = {
            'chunk_number': chunk_num,
            'total_timesteps': self.num_timesteps,
            'total_episodes': self.total_episodes,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / max(self.total_trades, 1)) * 100,
            'profit_factor': self.total_profit / max(abs(self.total_loss), 1),
            'total_profit': self.total_profit,
            'total_loss': abs(self.total_loss),
            'max_drawdown': self.max_drawdown,
            'max_consecutive_losses': self.max_consecutive_losses,
            'goal_trades_count': len(self.successful_goal_trades),
            'trades_by_pair': {
                pair: {
                    'count': len(trades),
                    'wins': len([t for t in trades if t['is_winning']]),
                    'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0
                } for pair, trades in self.trades_by_pair.items()
            },
            'avg_trade_duration': np.mean([t['duration'] for t in self.all_trades]) if self.all_trades else 0,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'peak_balance': self.peak_balance,
            'current_balance': self.episode_balances[-1] if self.episode_balances else 1000.0
        }

        # Save analytics
        self.chunk_analytics[chunk_num] = analytics
        return analytics

    def export_detailed_trades(self, chunk_num: int) -> pd.DataFrame:
        """Export detailed trade data as DataFrame."""
        if not self.all_trades:
            return pd.DataFrame()

        trades_df = pd.DataFrame(self.all_trades)
        trades_df['chunk_number'] = chunk_num
        return trades_df


class ForexTrainer:
    """
    Complete PPO training pipeline for forex trading system.
    """

    def __init__(self, config_path: str = "config/training.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.setup_logging()

        # Initialize components
        self.data_manager = None
        self.train_data = None
        self.val_data = None
        self.env = None
        self.model = None

        logger.info("ForexTrainer initialized")
        logger.info(f"Target: ${self.config['evaluation']['forex_metrics']['target_profit_per_trade']} "
                   f"in {self.config['evaluation']['forex_metrics']['target_duration_candles']} candles")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            # Return default configuration
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found."""
        return {
            'algorithm': {'total_timesteps': 500000, 'device': 'cuda'},
            'model': {'name': 'google/gemma-2b-it', 'policy': 'MlpPolicy'},
            'ppo_params': {
                'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64,
                'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
                'clip_range': 0.2, 'ent_coef': 0.01, 'vf_coef': 0.5
            },
            'training': {
                'validation_split': 0.2, 'save_frequency': 25000,
                'eval_frequency': 10000, 'tensorboard_log': './logs/tensorboard'
            },
            'environment': {
                'initial_balance': 10000.0, 'pairs': ['EURUSD', 'AUDCHF', 'USDJPY']
            },
            'evaluation': {
                'episodes': 50,
                'forex_metrics': {'target_profit_per_trade': 100.0, 'target_duration_candles': 4}
            }
        }

    def setup_directories(self):
        """Create necessary directories for training."""
        dirs_to_create = [
            self.config['training'].get('model_save_path', './models/checkpoints'),
            self.config['training'].get('tensorboard_log', './logs/tensorboard'),
            self.config['training'].get('eval_log_path', './logs/evaluations'),
            './models/final',
            './logs'
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        logger.info("Training directories created")

    def setup_logging(self):
        """Configure TensorBoard and file logging."""
        # Configure stable-baselines3 logger
        log_path = self.config['training'].get('tensorboard_log', './logs/tensorboard')
        self.sb3_logger = configure(log_path, ["stdout", "tensorboard"])

        logger.info(f"TensorBoard logging configured: {log_path}")
        logger.info("Run in Colab cell: %load_ext tensorboard")
        logger.info(f"Then: %tensorboard --logdir {log_path}")

    def prepare_data(self, csv_files: Optional[Dict[str, str]] = None):
        """Load and prepare training/validation data."""
        logger.info("Preparing forex data...")

        # Initialize data manager
        self.data_manager = DataManager()

        # Load multi-pair data (CSV files have priority if provided)
        all_data = self.data_manager.get_multi_pair_data(
            days=self.config['data'].get('history_days', 365),
            csv_files=csv_files
        )

        # Split data for training/validation
        validation_split = self.config['training'].get('validation_split', 0.2)
        split_point = int(len(next(iter(all_data.values()))) * (1 - validation_split))

        self.train_data = {}
        self.val_data = {}

        for pair, data in all_data.items():
            self.train_data[pair] = data.iloc[:split_point].copy()
            self.val_data[pair] = data.iloc[split_point:].copy()

        logger.info(f"Data prepared - Train: {len(self.train_data[list(all_data.keys())[0]])} "
                   f"samples, Val: {len(self.val_data[list(all_data.keys())[0]])} samples")

        # Log data quality
        total_features = len(next(iter(all_data.values())).columns)
        logger.info(f"Features per pair: {total_features}")
        logger.info(f"Currency pairs: {list(all_data.keys())}")

    def create_environment(self, data: Dict[str, pd.DataFrame], is_training: bool = True) -> gym.Env:
        """Create forex trading environment."""
        env_config = self.config.get('environment', {})

        # CHANGED: Use single-pair environment with USD/JPY data only
        usdjpy_data = data.get('USDJPY', list(data.values())[0])  # Get USDJPY or first available

        env = SinglePairForexEnv(
            data=usdjpy_data,
            initial_balance=env_config.get('initial_balance', 10000.0),
            lot_size=env_config.get('lot_size', 0.01),
            spread_pips=env_config.get('spread_pips', {}).get('USDJPY', 1.8),
            sl_tp_pips=tuple(env_config.get('sl_tp_pips', [15.0, 22.5])),
            max_episode_steps=env_config.get('max_episode_steps', 1500),
            commission_per_lot=env_config.get('commission_per_lot', 0.5),
            risk_per_trade_pct=env_config.get('risk_per_trade_pct', 0.03),
            max_position_duration=env_config.get('max_position_duration', 24)
        )

        # Wrap environment for monitoring
        env = Monitor(env)

        # Vectorize environment (required for stable-baselines3)
        env = DummyVecEnv([lambda: env])

        # Add VecNormalize for observation and reward normalization
        if is_training:
            env = VecNormalize(
                env,
                norm_obs=True,       # Normalize observations (prices ~145 vs indicators 0-1)
                norm_reward=True,    # ENABLED: Normalize rewards for stable learning
                clip_obs=10.0,       # Clip normalized obs to [-10, 10]
                clip_reward=10.0,    # Clip rewards to [-10, 10]
                gamma=0.99           # Discount factor
            )
            logger.info("VecNormalize applied for training (observation and reward normalization enabled)")

        logger.info(f"Single-pair environment created (USD/JPY) - Training: {is_training}")
        return env

    def create_model(self) -> RecurrentPPO:
        """Create and configure RecurrentPPO model with LSTM policy."""
        logger.info("Creating RecurrentPPO model with LSTM policy...")

        # Create training environment
        self.env = self.create_environment(self.train_data, is_training=True)

        # PPO parameters
        ppo_params = self.config.get('ppo_params', {})

        # Adjust for LSTM (memory intensive, use smaller batches)
        ppo_params['n_steps'] = 2048  # Reduced from 4096 for LSTM
        ppo_params['batch_size'] = 64  # Reduced from 128 for LSTM

        # Ensure learning_rate is a float
        lr = ppo_params.get('learning_rate', 1e-4)
        if isinstance(lr, str):
            try:
                lr = float(lr)
            except ValueError:
                logger.warning(f"Invalid learning rate: {lr}, using default 1e-4")
                lr = 1e-4
        ppo_params['learning_rate'] = lr

        # Model configuration
        model_config = self.config.get('model', {})

        # Policy kwargs for LSTM
        policy_kwargs = {
            'net_arch': [512, 256, 128],
            'activation_fn': torch.nn.Tanh,
            'lstm_hidden_size': 256  # LSTM hidden units
        }

        logger.info("🧠 Using RecurrentPPO with LSTM for temporal learning")
        logger.info(f"Net architecture: {policy_kwargs['net_arch']}")
        logger.info(f"LSTM hidden size: {policy_kwargs['lstm_hidden_size']}")
        logger.info(f"n_steps: {ppo_params['n_steps']}, batch_size: {ppo_params['batch_size']}")

        # Create RecurrentPPO model
        self.model = RecurrentPPO(
            policy="MlpLstmPolicy",  # LSTM-based policy for time-series
            env=self.env,
            learning_rate=ppo_params.get('learning_rate', 1e-4),
            n_steps=ppo_params.get('n_steps', 2048),
            batch_size=ppo_params.get('batch_size', 64),
            n_epochs=ppo_params.get('n_epochs', 10),
            gamma=ppo_params.get('gamma', 0.99),
            gae_lambda=ppo_params.get('gae_lambda', 0.95),
            clip_range=ppo_params.get('clip_range', 0.2),
            clip_range_vf=ppo_params.get('clip_range_vf'),
            ent_coef=ppo_params.get('ent_coef', 0.15),  # CRITICAL: Use 0.15 to force exploration
            vf_coef=ppo_params.get('vf_coef', 0.5),
            max_grad_norm=ppo_params.get('max_grad_norm', 0.5),
            target_kl=ppo_params.get('target_kl'),
            policy_kwargs=policy_kwargs,
            device=self.config['algorithm'].get('device', 'cuda'),
            verbose=self.config['algorithm'].get('verbose', 1)
        )

        # Set custom logger
        self.model.set_logger(self.sb3_logger)

        logger.info("✅ RecurrentPPO model with LSTM created successfully")
        logger.info(f"Policy: MlpLstmPolicy (temporal learning enabled)")
        logger.info(f"Learning rate: {ppo_params.get('learning_rate', 1e-4)}")

        return self.model

    def create_callbacks(self) -> CallbackList:
        """Create training callbacks."""
        callbacks = []

        # Evaluation callback
        if self.val_data:
            # IMPORTANT: eval_env must be wrapped the SAME way as training env
            # But we pass is_training=True to ensure VecNormalize is applied
            eval_env = self.create_environment(self.val_data, is_training=True)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.config['training'].get('model_save_path', './models/checkpoints'),
                log_path=self.config['training'].get('eval_log_path', './logs/evaluations'),
                eval_freq=self.config['training'].get('eval_frequency', 10000),
                n_eval_episodes=self.config['evaluation'].get('episodes', 50),
                deterministic=self.config['evaluation'].get('deterministic', True),
                render=False,
                verbose=1
            )
            callbacks.append(eval_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['training'].get('save_frequency', 25000),
            save_path=self.config['training'].get('model_save_path', './models/checkpoints'),
            name_prefix='forex_ppo',
            verbose=1
        )
        callbacks.append(checkpoint_callback)

        # Enhanced forex analytics callback
        self.forex_callback = EnhancedForexCallback(self.config, verbose=1)
        callbacks.append(self.forex_callback)

        logger.info(f"Callbacks created: {len(callbacks)} callbacks")
        return CallbackList(callbacks)

    def train(self, csv_files: Optional[Dict[str, str]] = None, resume_from: Optional[str] = None):
        """Execute complete training pipeline with checkpointing support."""
        logger.info("🚀 Starting PPO training for Forex Trading System")
        start_time = time.time()

        try:
            # Prepare data
            self.prepare_data(csv_files)

            # Create or load model
            if resume_from and Path(resume_from).exists():
                self.load_checkpoint(resume_from)
                logger.info(f"🔄 Resumed from checkpoint: {resume_from}")
            else:
                self.create_model()
                logger.info("🆕 Created new model")

            # Create callbacks with checkpointing
            callbacks = self.create_callbacks()

            # Training in 200k step chunks
            chunk_size = 200000
            total_timesteps = self.config['algorithm'].get('total_timesteps', 500000)
            chunks_needed = (total_timesteps + chunk_size - 1) // chunk_size  # Ceiling division

            current_timesteps = getattr(self.model, 'num_timesteps', 0)
            chunk_num = (current_timesteps // chunk_size) + 1

            logger.info(f"📊 Training plan: {chunks_needed} chunks of {chunk_size:,} steps each")
            logger.info(f"🎯 Starting from chunk {chunk_num} (timestep {current_timesteps:,})")

            # Train in chunks
            while current_timesteps < total_timesteps:
                remaining_steps = min(chunk_size, total_timesteps - current_timesteps)

                logger.info(f"🏃 Training chunk {chunk_num}/{chunks_needed}: {remaining_steps:,} steps")

                # Train for this chunk
                self.model.learn(
                    total_timesteps=remaining_steps,
                    callback=callbacks,
                    log_interval=self.config['training'].get('log_frequency', 100),
                    progress_bar=True,
                    reset_num_timesteps=False
                )

                # Save checkpoint after each chunk
                checkpoint_path = self.save_checkpoint(chunk_num)
                current_timesteps = self.model.num_timesteps

                logger.info(f"✅ Chunk {chunk_num} completed - Total: {current_timesteps:,} steps")
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

                chunk_num += 1

            # Training completed
            training_time = time.time() - start_time
            logger.info(f"✅ Full training completed in {training_time/3600:.2f} hours")

            # Save final model
            self.save_final_model()

            # Run final evaluation
            self.evaluate_final_model()

        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            # Save emergency checkpoint
            try:
                emergency_path = self.save_checkpoint(f"emergency_{int(time.time())}")
                logger.info(f"💾 Emergency checkpoint saved: {emergency_path}")
            except:
                pass
            raise

    def save_final_model(self):
        """Save the final trained model."""
        final_model_path = self.config['export'].get('final_model_path', './models/final/forex_ppo_model')

        # Create directory
        Path(final_model_path).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(final_model_path)

        # Save configuration
        if self.config['export'].get('include_config', True):
            config_path = f"{final_model_path}_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"💾 Final model saved: {final_model_path}")
        logger.info(f"📥 Download this file from Colab to use locally")

    def evaluate_final_model(self):
        """Evaluate the final trained model."""
        logger.info("📊 Evaluating final model...")

        # Create evaluation environment
        eval_env = self.create_environment(self.val_data, is_training=False)

        # Evaluation parameters
        n_episodes = self.config['evaluation'].get('episodes', 50)
        deterministic = self.config['evaluation'].get('deterministic', True)

        # Run evaluation
        episode_rewards = []
        successful_goal_trades = 0
        total_trades = 0

        for episode in range(n_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            episode_goal_trades = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward

                # Check for goal trades
                if isinstance(info, dict) and 'closed_positions' in info:
                    for pos_info in info['closed_positions']:
                        total_trades += 1
                        if (pos_info.get('pnl', 0) >= 100.0 and
                            pos_info.get('duration', float('inf')) <= 4):
                            successful_goal_trades += 1
                            episode_goal_trades += 1

            episode_rewards.append(episode_reward)

            if episode % 10 == 0:
                logger.info(f"Eval Episode {episode}: Reward={episode_reward:.1f}, "
                           f"Goal Trades={episode_goal_trades}")

        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        goal_trade_rate = successful_goal_trades / max(total_trades, 1) * 100
        positive_episodes = sum(1 for r in episode_rewards if r > 0)
        win_rate = positive_episodes / n_episodes * 100

        # Log results
        logger.info("🏆 FINAL EVALUATION RESULTS:")
        logger.info(f"   Average Episode Reward: {avg_reward:.2f}")
        logger.info(f"   Win Rate: {win_rate:.1f}%")
        logger.info(f"   Goal Trade Rate: {goal_trade_rate:.1f}% ({successful_goal_trades}/{total_trades})")
        logger.info(f"   Total Episodes: {n_episodes}")

        # Check if targets achieved
        target_reward = self.config['evaluation']['target_metrics'].get('episode_reward', 1000.0)
        target_win_rate = self.config['evaluation']['target_metrics'].get('win_rate', 0.65) * 100

        if avg_reward >= target_reward and win_rate >= target_win_rate:
            logger.info("🎉 TRAINING SUCCESS! Targets achieved!")
        else:
            logger.info(f"📈 Targets: Reward≥{target_reward}, WinRate≥{target_win_rate:.1f}%")

    def save_checkpoint(self, chunk_num: int) -> str:
        """Save training checkpoint with comprehensive analytics."""
        checkpoint_dir = Path("./models/checkpoints")
        analytics_dir = Path("./analytics")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        analytics_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"forex_ppo_chunk_{chunk_num}_{timestamp}"

        # Save model
        self.model.save(checkpoint_path)

        # Export and save analytics
        if hasattr(self, 'forex_callback'):
            # Export analytics summary
            analytics = self.forex_callback.export_analytics(chunk_num)
            analytics_path = analytics_dir / f"analytics_chunk_{chunk_num}_{timestamp}.json"
            with open(analytics_path, 'w') as f:
                import json
                json.dump(analytics, f, indent=2, default=str)

            # Export detailed trades CSV
            trades_df = self.forex_callback.export_detailed_trades(chunk_num)
            if not trades_df.empty:
                trades_path = analytics_dir / f"trades_chunk_{chunk_num}_{timestamp}.csv"
                trades_df.to_csv(trades_path, index=False)
                logger.info(f"📊 Analytics exported: {len(trades_df)} trades")

        # Save metadata
        metadata = {
            'chunk_number': chunk_num,
            'total_timesteps': self.model.num_timesteps,
            'timestamp': timestamp,
            'training_started': datetime.now().isoformat(),
            'config': self.config
        }

        if hasattr(self, 'forex_callback'):
            metadata.update({
                'total_trades': self.forex_callback.total_trades,
                'win_rate': (self.forex_callback.winning_trades / max(self.forex_callback.total_trades, 1)) * 100,
                'profit_factor': self.forex_callback.total_profit / max(abs(self.forex_callback.total_loss), 1),
                'max_drawdown': self.forex_callback.max_drawdown,
                'goal_trades': len(self.forex_callback.successful_goal_trades)
            })

        metadata_path = f"{checkpoint_path}_metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        logger.info(f"📈 Performance: {metadata.get('total_trades', 0)} trades, "
                   f"{metadata.get('win_rate', 0):.1f}% win rate")
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint and restore state."""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Prepare data if not already done
        if self.train_data is None:
            self.prepare_data()

        # Create environment
        self.env = self.create_environment(self.train_data, is_training=True)

        # Load model
        self.model = PPO.load(checkpoint_path, env=self.env)

        # Load metadata if available
        metadata_path = f"{checkpoint_path}_metadata.yaml"
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            logger.info(f"📊 Loaded checkpoint from chunk {metadata.get('chunk_number', '?')}")
            logger.info(f"🕐 Previously trained: {metadata.get('total_timesteps', 0):,} steps")

        logger.info("✅ Checkpoint loaded successfully")

    def load_and_continue_training(self, model_path: str, additional_steps: int = 100000):
        """Load existing model and continue training."""
        logger.info(f"Loading model from {model_path}")

        # Prepare data if not already done
        if self.train_data is None:
            self.prepare_data()

        # Create environment
        env = self.create_environment(self.train_data, is_training=True)

        # Load model
        self.model = PPO.load(model_path, env=env)
        logger.info("Model loaded successfully")

        # Continue training
        callbacks = self.create_callbacks()
        self.model.learn(
            total_timesteps=additional_steps,
            callback=callbacks,
            reset_num_timesteps=False
        )

        # Save updated model
        self.save_final_model()


def find_latest_checkpoint() -> Optional[str]:
    """Find the latest checkpoint file for resuming training."""
    checkpoint_dir = Path("./models/checkpoints")
    if not checkpoint_dir.exists():
        return None

    # Find all checkpoint ZIP files (exclude .yaml metadata files)
    checkpoint_files = [
        f for f in checkpoint_dir.glob("forex_ppo_chunk_*.zip")
        if 'emergency' not in f.name  # Prefer regular checkpoints over emergency
    ]

    # If no regular checkpoints, check for emergency checkpoint
    if not checkpoint_files:
        checkpoint_files = list(checkpoint_dir.glob("forex_ppo_chunk_emergency_*.zip"))

    if not checkpoint_files:
        return None

    # Sort by chunk number and timestamp to get the latest
    def get_sort_key(path):
        name = path.stem
        parts = name.split('_')
        try:
            # Handle both regular (forex_ppo_chunk_X_YYYYMMDD_HHMMSS)
            # and emergency (forex_ppo_chunk_emergency_TIMESTAMP_YYYYMMDD_HHMMSS) formats
            if 'emergency' in name:
                return (999, parts[-2] + '_' + parts[-1])  # Emergency gets high priority
            else:
                chunk_num = int(parts[3])  # forex_ppo_chunk_X_timestamp
                timestamp = parts[4] + '_' + parts[5]  # YYYYMMDD_HHMMSS
                return (chunk_num, timestamp)
        except (IndexError, ValueError):
            return (0, "")

    latest_checkpoint = max(checkpoint_files, key=get_sort_key)
    return str(latest_checkpoint)


def main(csv_files: Optional[Dict[str, str]] = None, resume_from: Optional[str] = None):
    """Main training function for Google Colab with checkpoint support."""
    print("🚀 Forex Trading System - PPO Training")
    print("=" * 50)

    # Check for GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU Available: {gpu_name}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected - training will be very slow")

    # Check for existing checkpoints if not resuming from specific file
    if not resume_from:
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint:
            print(f"🔍 Found existing checkpoint: {latest_checkpoint}")
            response = input("Resume from this checkpoint? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                resume_from = latest_checkpoint

    # Show training mode
    if resume_from:
        print(f"🔄 RESUMING training from: {resume_from}")
    else:
        print("🆕 STARTING new training session")

    # Show data source
    if csv_files:
        print("📊 Using custom CSV data files:")
        for pair, file_path in csv_files.items():
            print(f"   {pair}: {file_path}")
    else:
        print("📊 Using generated sample data (fallback)")

    print("\n⚠️ Training will save checkpoints every 200k steps")
    print("💾 Download checkpoint files after each chunk completes!")

    # Initialize trainer
    try:
        trainer = ForexTrainer()

        # Start training (new or resumed)
        trainer.train(csv_files, resume_from=resume_from)

        print("\n🎉 Training completed successfully!")
        print("📥 Download the model files from:")
        print("   - ./models/final/forex_ppo_model.zip")
        print("   - ./models/final/forex_ppo_model_config.yaml")
        print("   - ./models/checkpoints/ (all checkpoint files)")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        print("💾 Emergency checkpoint should be available in ./models/checkpoints/")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        print("💾 Emergency checkpoint should be available in ./models/checkpoints/")
        import traceback
        traceback.print_exc()


def resume_training_from_file(checkpoint_file: str, csv_files: Optional[Dict[str, str]] = None):
    """Helper function to resume training from a specific checkpoint file."""
    print(f"🔄 Resuming training from: {checkpoint_file}")
    return main(csv_files, resume_from=checkpoint_file)


if __name__ == "__main__":
    main()