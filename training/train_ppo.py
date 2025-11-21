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
    from environment import MultiPairForexEnv
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Make sure to upload data_manager.py and environment.py to Colab")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ForexTrainingCallback(BaseCallback):
    """
    Custom callback for forex-specific training monitoring.

    Tracks:
    - Successful trades ($100 in ≤4 candles)
    - Win rate and profit metrics
    - Session performance
    """

    def __init__(self, config: Dict[str, Any], verbose: int = 1):
        super().__init__(verbose)
        self.config = config
        self.successful_goal_trades = []
        self.total_episodes = 0
        self.episode_rewards = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        # Extract info from the environment
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if isinstance(info, dict):
                    # Track closed positions that achieved the goal
                    if 'closed_positions' in info:
                        for pos_info in info['closed_positions']:
                            if (pos_info.get('pnl', 0) >= 100.0 and
                                pos_info.get('duration', float('inf')) <= 4):
                                self.successful_goal_trades.append({
                                    'step': self.num_timesteps,
                                    'pair': pos_info.get('pair'),
                                    'pnl': pos_info['pnl'],
                                    'duration': pos_info['duration']
                                })
                                logger.info(f"🎯 GOAL ACHIEVED! ${pos_info['pnl']:.2f} "
                                          f"in {pos_info['duration']} candles - "
                                          f"Pair: {pos_info.get('pair')}")

                    # Track episode completion
                    if 'episode' in info and info['episode']:
                        ep_info = info['episode']
                        self.episode_rewards.append(ep_info['r'])
                        self.total_episodes += 1

                        # Log episode summary every 10 episodes
                        if self.total_episodes % 10 == 0:
                            recent_rewards = self.episode_rewards[-10:]
                            avg_reward = np.mean(recent_rewards)
                            goal_trades_count = len(self.successful_goal_trades)

                            logger.info(f"Episode {self.total_episodes}: "
                                      f"Avg Reward: {avg_reward:.1f}, "
                                      f"Goal Trades: {goal_trades_count}")

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if len(self.successful_goal_trades) > 0:
            recent_goals = [trade for trade in self.successful_goal_trades
                           if self.num_timesteps - trade['step'] <= 2048]  # Last rollout

            if recent_goals:
                avg_pnl = np.mean([trade['pnl'] for trade in recent_goals])
                avg_duration = np.mean([trade['duration'] for trade in recent_goals])

                # Log to TensorBoard
                self.logger.record("forex/goal_trades_per_rollout", len(recent_goals))
                self.logger.record("forex/avg_goal_trade_pnl", avg_pnl)
                self.logger.record("forex/avg_goal_trade_duration", avg_duration)
                self.logger.record("forex/total_goal_trades", len(self.successful_goal_trades))


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
                'initial_balance': 10000.0, 'pairs': ['EURUSD', 'GBPUSD', 'USDJPY']
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

        # Create base environment
        env = MultiPairForexEnv(
            data=data,
            pairs=env_config.get('pairs', ['EURUSD', 'GBPUSD', 'USDJPY']),
            initial_balance=env_config.get('initial_balance', 10000.0),
            max_positions_per_pair=env_config.get('max_positions_per_pair', 1),
            lot_size=env_config.get('lot_size', 0.1),
            spread_pips=env_config.get('spread_pips', {'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.8}),
            sl_tp_pips=tuple(env_config.get('sl_tp_pips', [20.0, 40.0])),
            max_episode_steps=env_config.get('max_episode_steps', 1000),
            commission_per_lot=env_config.get('commission_per_lot', 3.0)
        )

        # Wrap environment for monitoring
        env = Monitor(env)

        logger.info(f"Environment created - Training: {is_training}")
        return env

    def create_model(self) -> PPO:
        """Create and configure PPO model."""
        logger.info("Creating PPO model...")

        # Create training environment
        self.env = self.create_environment(self.train_data, is_training=True)

        # PPO parameters
        ppo_params = self.config.get('ppo_params', {})

        # Model configuration
        model_config = self.config.get('model', {})
        policy_kwargs = model_config.get('policy_kwargs', {
            'net_arch': [512, 256, 128],
            'activation_fn': torch.nn.Tanh
        })

        # Create PPO model
        self.model = PPO(
            policy=model_config.get('policy', 'MlpPolicy'),
            env=self.env,
            learning_rate=ppo_params.get('learning_rate', 3e-4),
            n_steps=ppo_params.get('n_steps', 2048),
            batch_size=ppo_params.get('batch_size', 64),
            n_epochs=ppo_params.get('n_epochs', 10),
            gamma=ppo_params.get('gamma', 0.99),
            gae_lambda=ppo_params.get('gae_lambda', 0.95),
            clip_range=ppo_params.get('clip_range', 0.2),
            clip_range_vf=ppo_params.get('clip_range_vf'),
            ent_coef=ppo_params.get('ent_coef', 0.01),
            vf_coef=ppo_params.get('vf_coef', 0.5),
            max_grad_norm=ppo_params.get('max_grad_norm', 0.5),
            target_kl=ppo_params.get('target_kl'),
            policy_kwargs=policy_kwargs,
            device=self.config['algorithm'].get('device', 'cuda'),
            verbose=self.config['algorithm'].get('verbose', 1)
        )

        # Set custom logger
        self.model.set_logger(self.sb3_logger)

        logger.info("PPO model created successfully")
        logger.info(f"Policy network: {policy_kwargs}")
        logger.info(f"Learning rate: {ppo_params.get('learning_rate', 3e-4)}")

        return self.model

    def create_callbacks(self) -> CallbackList:
        """Create training callbacks."""
        callbacks = []

        # Evaluation callback
        if self.val_data:
            eval_env = self.create_environment(self.val_data, is_training=False)
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

        # Custom forex callback
        forex_callback = ForexTrainingCallback(self.config, verbose=1)
        callbacks.append(forex_callback)

        logger.info(f"Callbacks created: {len(callbacks)} callbacks")
        return CallbackList(callbacks)

    def train(self, csv_files: Optional[Dict[str, str]] = None):
        """Execute complete training pipeline."""
        logger.info("🚀 Starting PPO training for Forex Trading System")
        start_time = time.time()

        try:
            # Prepare data
            self.prepare_data(csv_files)

            # Create model
            self.create_model()

            # Create callbacks
            callbacks = self.create_callbacks()

            # Get training parameters
            total_timesteps = self.config['algorithm'].get('total_timesteps', 500000)
            log_interval = self.config['training'].get('log_frequency', 100)

            logger.info(f"🎯 Training target: ${self.config['evaluation']['forex_metrics']['target_profit_per_trade']} "
                       f"in {self.config['evaluation']['forex_metrics']['target_duration_candles']} candles")
            logger.info(f"📊 Total timesteps: {total_timesteps:,}")
            logger.info(f"🔄 Expected time: 12-24 hours on GPU")

            # Start training
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=log_interval,
                progress_bar=True
            )

            # Training completed
            training_time = time.time() - start_time
            logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")

            # Save final model
            self.save_final_model()

            # Run final evaluation
            self.evaluate_final_model()

        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
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


def main(csv_files: Optional[Dict[str, str]] = None):
    """Main training function for Google Colab."""
    print("🚀 Forex Trading System - PPO Training")
    print("=" * 50)

    # Check for GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU Available: {gpu_name}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected - training will be very slow")

    # Show data source
    if csv_files:
        print("📊 Using custom CSV data files:")
        for pair, file_path in csv_files.items():
            print(f"   {pair}: {file_path}")
    else:
        print("📊 Using generated sample data (fallback)")

    # Initialize trainer
    try:
        trainer = ForexTrainer()

        # Start training
        trainer.train(csv_files)

        print("\n🎉 Training completed successfully!")
        print("📥 Download the model files from:")
        print("   - ./models/final/forex_ppo_model.zip")
        print("   - ./models/final/forex_ppo_model_config.yaml")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()