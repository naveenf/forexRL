"""
Multi-Pair Forex Trading Environment for Reinforcement Learning

This module implements a custom gymnasium environment for forex trading
that supports simultaneous trading across multiple currency pairs.

Key Features:
- Multi-pair trading (EUR/USD, AUD/CHF, USD/JPY)
- Custom reward function targeting $100 profit in 4 candles
- Position management with Stop Loss / Take Profit
- Realistic spread simulation and transaction costs
- Portfolio state tracking across multiple positions

Action Space:
- 9 discrete actions: 3 per currency pair
- Actions per pair: HOLD (0), BUY (1), SELL (2)
- Example: [0, 1, 2] = HOLD EURUSD, BUY AUDCHF, SELL USDJPY

Observation Space:
- Market data: OHLCV + technical indicators per pair
- Portfolio state: positions, balance, unrealized P&L
- Time features: hour, day of week, market session
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Trading action types for each currency pair."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Position:
    """Represents a forex trading position."""
    pair: str
    action: ActionType  # BUY or SELL
    entry_price: float
    entry_time: int  # Step number when position was opened
    size: float  # Position size (lot size)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0

    @property
    def duration(self) -> int:
        """Get position duration in steps."""
        return self.current_step - self.entry_time if hasattr(self, 'current_step') else 0

    def update_pnl(self, current_price: float, current_step: int):
        """Update unrealized P&L for this position."""
        self.current_step = current_step

        if self.action == ActionType.BUY:
            price_diff = current_price - self.entry_price
        else:  # SELL
            price_diff = self.entry_price - current_price

        # Calculate P&L (assuming 1 standard lot = $10 per pip for majors)
        if 'JPY' in self.pair:
            pip_value = 10.0  # For JPY pairs, 1 pip = 0.01
            pip_diff = price_diff * 100
        else:
            pip_value = 10.0  # For major pairs, 1 pip = 0.0001
            pip_diff = price_diff * 10000

        self.unrealized_pnl = pip_diff * pip_value * self.size

    def should_close_sl_tp(self, current_price: float) -> bool:
        """Check if position should be closed due to SL/TP."""
        if self.stop_loss and self.action == ActionType.BUY and current_price <= self.stop_loss:
            return True
        if self.stop_loss and self.action == ActionType.SELL and current_price >= self.stop_loss:
            return True
        if self.take_profit and self.action == ActionType.BUY and current_price >= self.take_profit:
            return True
        if self.take_profit and self.action == ActionType.SELL and current_price <= self.take_profit:
            return True
        return False


class MultiPairForexEnv(gym.Env):
    """
    Multi-pair forex trading environment for reinforcement learning.

    This environment simulates trading across multiple currency pairs
    with a custom reward function that incentivizes achieving $100 profit
    within 4 candles (1 hour in 15-minute timeframe).
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 pairs: List[str] = None,
                 initial_balance: float = 10000.0,  # Realistic retail account
                 max_positions_per_pair: int = 1,
                 lot_size: float = 0.01,  # Will be dynamically calculated
                 spread_pips: Dict[str, float] = None,
                 sl_tp_pips: Tuple[float, float] = (15.0, 22.5),  # 1:1.5 risk/reward
                 max_episode_steps: int = 1500,  # Longer episodes for patience
                 commission_per_lot: float = 1.0,  # CHANGED from 3.0
                 risk_per_trade_pct: float = 0.03,  # CHANGED from 0.05
                 dynamic_position_sizing: bool = True,
                 max_position_duration: int = 20,  # CHANGED from 24
                 reward_config_path: Optional[str] = None):  # NEW
        """
        Initialize the multi-pair forex environment.

        Args:
            data: Dictionary of DataFrames with OHLCV + indicators for each pair
            pairs: List of currency pairs to trade (default: all pairs in data)
            initial_balance: Starting account balance in USD (default: $1,000)
            max_positions_per_pair: Maximum concurrent positions per pair
            lot_size: Base lot size (will be dynamically calculated if dynamic_position_sizing=True)
            spread_pips: Dictionary of spread in pips for each pair
            sl_tp_pips: Tuple of (stop_loss_pips, take_profit_pips) - default 1:1.5 ratio
            max_episode_steps: Maximum steps per episode (default: 1500 for longer episodes)
            commission_per_lot: Commission cost per lot traded (default: $1)
            risk_per_trade_pct: Percentage of account to risk per trade (default: 3%)
            dynamic_position_sizing: Enable dynamic position sizing based on account risk
            max_position_duration: Maximum position duration in candles (default: 20 = 5 hours)
            reward_config_path: Path to YAML config file for reward function (default: config/environment.yaml)
        """
        super().__init__()

        # Environment configuration
        self.data = data
        self.pairs = pairs or list(data.keys())
        self.initial_balance = initial_balance
        self.max_positions_per_pair = max_positions_per_pair
        self.base_lot_size = lot_size
        self.max_episode_steps = max_episode_steps
        self.commission_per_lot = commission_per_lot

        # Risk management parameters
        self.risk_per_trade_pct = risk_per_trade_pct
        self.dynamic_position_sizing = dynamic_position_sizing
        self.max_position_duration = max_position_duration
        self.position_durations = {'profitable': [], 'losing': []}  # Track for optimization

        # Validate pairs
        for pair in self.pairs:
            if pair not in self.data:
                raise ValueError(f"Data not available for pair: {pair}")

        # Default spreads (in pips)
        self.spread_pips = spread_pips or {
            'EURUSD': 1.5,
            'AUDCHF': 2.5,
            'USDJPY': 1.8
        }

        # Stop Loss / Take Profit configuration
        self.sl_pips, self.tp_pips = sl_tp_pips

        logger.info(f"Environment initialized for pairs: {self.pairs}")
        logger.info(f"Initial balance: ${self.initial_balance:,.2f}")

        # Define action space: 3 actions per pair (HOLD, BUY, SELL)
        # Total actions = len(pairs) * 3
        self.action_space = spaces.MultiDiscrete([3] * len(self.pairs))

        # Define observation space
        self._setup_observation_space()

        # Initialize state variables
        self.reset()

        # Load reward configuration
        self.reward_config = self._load_reward_config(reward_config_path)

    def _setup_observation_space(self):
        """Setup the observation space based on available data features."""
        # Get sample data to determine observation dimension
        sample_pair = self.pairs[0]
        sample_data = self.data[sample_pair]

        # Count features per pair (exclude Symbol column if present)
        features_per_pair = len([col for col in sample_data.columns if col != 'Symbol'])

        # Total market features for all pairs
        market_features = features_per_pair * len(self.pairs)

        # Portfolio features per pair
        portfolio_features_per_pair = 5  # position_size, unrealized_pnl, position_duration, position_type, can_trade
        portfolio_features = portfolio_features_per_pair * len(self.pairs)

        # Global portfolio features
        global_features = 4  # balance, total_unrealized_pnl, total_positions, equity

        # Time features
        time_features = 3  # hour, day_of_week, market_session

        # Total observation dimension
        total_features = market_features + portfolio_features + global_features + time_features

        # All features are continuous values
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        logger.info(f"Observation space: {total_features} features")
        logger.info(f"  - Market features: {market_features} ({features_per_pair} per pair)")
        logger.info(f"  - Portfolio features: {portfolio_features}")
        logger.info(f"  - Global features: {global_features}")
        logger.info(f"  - Time features: {time_features}")

    def _load_reward_config(self, config_path: Optional[str] = None) -> Dict:
        """Load reward function configuration from YAML."""
        if config_path is None:
            config_path = "config/environment.yaml"

        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('reward_function', {})

        return self._get_default_reward_config()

    def _get_default_reward_config(self) -> Dict:
        """Default reward configuration."""
        return {
            'primary_goal': {'profit_threshold': 37.50, 'duration_max': 12, 'bonus': 300.0},
            'excellent_trade': {'profit_threshold': 30.0, 'duration_max': 10, 'bonus': 200.0},
            'good_trade': {'profit_threshold': 22.50, 'duration_max': 8, 'bonus': 120.0},
            'quick_profit': {'profit_threshold': 15.0, 'duration_max': 6, 'bonus': 60.0},
            'decent_profit': {'profit_threshold': 10.0, 'base_bonus': 30.0, 'multiplier': 0.5},
            'small_profit': {'profit_threshold': 5.0, 'base_bonus': 15.0, 'multiplier': 0.3},
            'loss_penalties': {
                'breakeven': {'base_reward': 5.0},
                'tiny_loss': {'threshold': -10.0, 'base_reward': 2.0, 'multiplier': -0.2},
                'small_loss': {'threshold': -20.0, 'multiplier': -0.5},
                'normal_loss': {'threshold': -30.0, 'multiplier': -1.0},
                'full_sl': {'threshold': -40.0, 'multiplier': -1.5},
                'large_loss': {'threshold': -60.0, 'multiplier': -2.0, 'extra_penalty': 20.0},
                'catastrophic': {'multiplier': -2.5, 'extra_penalty': 50.0}
            },
            'activity': {
                'enabled': True,
                'base_trade_reward': 3.0,
                'favorable_session_bonus': 2.0,
                'position_discipline_bonus': 2.0,
                'overtrading_penalty_per_trade': 2.0,
                'overtrading_threshold': 5
            }
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for environment
            options: Additional options for reset

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)

        # Reset environment state
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions: Dict[str, List[Position]] = {pair: [] for pair in self.pairs}
        self.closed_positions: List[Position] = []
        self.total_commission_paid = 0.0
        self.trade_summary_count = 0  # Track trades for summary logging
        self.last_trade_info = {'trades_opened': [], 'trades_closed': [], 'failed_opens': []}  # Track for reward calculation

        # Episode statistics
        self.episode_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_balance': self.initial_balance,
            'min_balance': self.initial_balance,
            'max_drawdown': 0.0
        }

        # Ensure we have enough data for the episode
        min_data_length = min(len(self.data[pair]) for pair in self.pairs)
        if min_data_length < self.max_episode_steps + 100:
            logger.warning(f"Limited data available: {min_data_length} candles")

        # Random starting point (leave room for full episode)
        max_start = max(100, min_data_length - self.max_episode_steps - 1)
        self.start_step = self.np_random.integers(100, max_start) if max_start > 100 else 100

        # Get initial observation
        observation = self._get_observation()

        info = {
            'balance': self.balance,
            'positions': len(self._get_all_positions()),
            'start_step': self.start_step
        }

        logger.info(f"Environment reset - Balance: ${self.balance:,.2f}, Start step: {self.start_step}")
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Array of actions for each currency pair

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if len(action) != len(self.pairs):
            raise ValueError(f"Action length {len(action)} != number of pairs {len(self.pairs)}")

        # Execute trading actions
        trade_info = self._execute_actions(action)
        self.last_trade_info = trade_info  # Store for reward calculation

        # Update existing positions
        self._update_positions()

        # Check for position closures (SL/TP)
        closed_positions_info = self._check_position_closures()

        # Calculate reward
        reward = self._calculate_reward()

        # Update step
        self.current_step += 1

        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps

        # Get new observation
        observation = self._get_observation()

        # Update episode statistics
        self._update_episode_stats()

        # Prepare info dictionary
        info = {
            'balance': self.balance,
            'equity': self._get_equity(),
            'total_positions': len(self._get_all_positions()),
            'total_unrealized_pnl': self._get_total_unrealized_pnl(),
            'trade_info': trade_info,
            'closed_positions': closed_positions_info,
            'commission_paid': self.total_commission_paid,
            'step': self.current_step
        }

        return observation, reward, terminated, truncated, info

    def _execute_actions(self, actions: np.ndarray) -> Dict:
        """
        Execute trading actions for each currency pair.

        Args:
            actions: Array of actions for each pair

        Returns:
            Dictionary with trade execution information
        """
        trade_info = {'trades_opened': [], 'trades_closed': [], 'failed_opens': []}

        for i, (pair, action_int) in enumerate(zip(self.pairs, actions)):
            current_price = self._get_current_price(pair)
            action = ActionType(action_int)  # Convert int to ActionType enum

            if action == ActionType.HOLD:
                continue

            # Check if we can open a new position
            current_positions = self.positions[pair]
            if len(current_positions) >= self.max_positions_per_pair:
                # Close existing positions before opening new ones
                for position in current_positions:
                    self._close_position(position, current_price)
                    trade_info['trades_closed'].append({
                        'pair': pair,
                        'action': 'CLOSE',
                        'price': current_price,
                        'pnl': position.unrealized_pnl
                    })
                self.positions[pair] = []

            # Open new position
            if action in [ActionType.BUY, ActionType.SELL]:
                logger.debug(f"Attempting to open {action.name} position for {pair} at ${current_price}")
                position = self._open_position(pair, action, current_price)
                if position:
                    logger.info(f"✅ Opened {action.name} position: {pair} {position.size} lots at ${current_price}")
                    trade_info['trades_opened'].append({
                        'pair': pair,
                        'action': 'BUY' if action == ActionType.BUY else 'SELL',
                        'price': current_price,
                        'size': position.size,
                        'sl': position.stop_loss,
                        'tp': position.take_profit
                    })
                else:
                    logger.debug(f"❌ Failed to open {action.name} position for {pair}")
                    trade_info['failed_opens'].append({
                        'pair': pair,
                        'action': action.name
                    })

        return trade_info

    def _calculate_position_size(self, pair: str, stop_loss_pips: float) -> float:
        """
        Calculate position size based on account risk percentage.

        Args:
            pair: Currency pair
            stop_loss_pips: Stop loss distance in pips

        Returns:
            Calculated lot size based on risk management
        """
        if not self.dynamic_position_sizing:
            return self.base_lot_size

        # Calculate account risk in USD
        account_risk_usd = self.balance * self.risk_per_trade_pct

        # Get pip value for this pair
        pip_value = self._get_pip_value(pair)

        # Calculate maximum lot size based on risk
        if stop_loss_pips > 0:
            max_lot_size = account_risk_usd / (stop_loss_pips * pip_value)

            # Cap the lot size to reasonable limits for retail trading
            max_lot_size = min(max_lot_size, 0.5)  # Maximum 0.5 standard lot (conservative)
            max_lot_size = max(max_lot_size, 0.01)  # Minimum 0.01 lots (micro lot)

            # Additional safety check for unrealistic calculations
            if max_lot_size > 0.5:
                logger.warning(f"Position size calculation unrealistic for {pair}: {max_lot_size:.3f} lots, capping to 0.5")
                max_lot_size = 0.5

            logger.debug(f"Calculated lot size for {pair}: {max_lot_size:.3f} "
                        f"(Risk: ${account_risk_usd:.2f}, SL: {stop_loss_pips} pips, PipValue: ${pip_value})")

            return round(max_lot_size, 3)
        else:
            return self.base_lot_size

    def _get_pip_value(self, pair: str) -> float:
        """Get pip value in USD for position sizing calculations."""
        if 'JPY' in pair:
            return 10.0  # For JPY pairs, 1 pip = 0.01, worth ~$10 per standard lot
        else:
            return 10.0  # For major pairs, 1 pip = 0.0001, worth ~$10 per standard lot

    def _open_position(self, pair: str, action: ActionType, current_price: float) -> Optional[Position]:
        """
        Open a new trading position.

        Args:
            pair: Currency pair
            action: BUY or SELL
            current_price: Current market price

        Returns:
            Position object if successfully opened, None otherwise
        """
        # Calculate spread-adjusted entry price
        spread = self._get_spread(pair)
        if action == ActionType.BUY:
            entry_price = current_price + spread / 2
        else:  # SELL
            entry_price = current_price - spread / 2

        # Calculate Stop Loss and Take Profit
        sl_distance = self._pips_to_price(pair, self.sl_pips)
        tp_distance = self._pips_to_price(pair, self.tp_pips)

        if action == ActionType.BUY:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SELL
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Calculate position size based on risk management
        position_size = self._calculate_position_size(pair, self.sl_pips)

        # Simplified margin check - just ensure balance > 0
        if self.balance <= 100:  # Only block if nearly broke
            logger.warning(f"Insufficient balance: ${self.balance:.2f}")
            return None

        # No additional risk checks during training - let agent learn from outcomes

        # Create position
        position = Position(
            pair=pair,
            action=action,
            entry_price=entry_price,
            entry_time=self.current_step,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Add to positions
        self.positions[pair].append(position)

        # Deduct commission
        commission = self.commission_per_lot * position_size
        self.balance -= commission
        self.total_commission_paid += commission

        # Increment trade counter for summary logging
        self.trade_summary_count += 1

        # Only log every 100 trades to reduce verbosity
        if self.trade_summary_count % 100 == 0:
            total_trades = self.episode_stats['total_trades']
            win_rate = (self.episode_stats['winning_trades'] / max(total_trades, 1)) * 100
            logger.info(f"Trade Summary - Completed {self.trade_summary_count} trades | Win Rate: {win_rate:.1f}% | Balance: ${self.balance:,.2f}")
        return position

    def _close_position(self, position: Position, current_price: float) -> float:
        """
        Close a trading position.

        Args:
            position: Position to close
            current_price: Current market price

        Returns:
            Realized P&L
        """
        # Update final P&L
        position.update_pnl(current_price, self.current_step)

        # Calculate spread-adjusted exit price
        spread = self._get_spread(position.pair)
        if position.action == ActionType.BUY:
            exit_price = current_price - spread / 2
        else:  # SELL
            exit_price = current_price + spread / 2

        # Recalculate P&L with exit price
        position.update_pnl(exit_price, self.current_step)

        # Realize P&L
        realized_pnl = position.unrealized_pnl
        self.balance += realized_pnl

        # Deduct commission for closing
        commission = self.commission_per_lot * position.size
        self.balance -= commission
        self.total_commission_paid += commission

        # Add to closed positions for reward calculation
        position.close_time = self.current_step  # Track when it was closed
        self.closed_positions.append(position)

        # Update episode statistics
        self.episode_stats['total_trades'] += 1
        if realized_pnl > 0:
            self.episode_stats['winning_trades'] += 1
        else:
            self.episode_stats['losing_trades'] += 1

        # Silent individual trade closures - summary logging only
        return realized_pnl

    def _update_positions(self):
        """Update unrealized P&L for all open positions."""
        for pair in self.pairs:
            current_price = self._get_current_price(pair)
            for position in self.positions[pair]:
                position.update_pnl(current_price, self.current_step)

    def _check_position_closures(self) -> List[Dict]:
        """
        Check and close positions that hit SL/TP or duration limit.

        Returns:
            List of closed position information
        """
        closed_info = []

        for pair in self.pairs:
            current_price = self._get_current_price(pair)
            positions_to_close = []

            for position in self.positions[pair]:
                should_close = False
                close_reason = ""

                # Check SL/TP
                if position.should_close_sl_tp(current_price):
                    should_close = True
                    if position.action == ActionType.BUY:
                        if current_price <= position.stop_loss:
                            close_reason = "Stop Loss"
                        elif current_price >= position.take_profit:
                            close_reason = "Take Profit"
                    else:  # SELL
                        if current_price >= position.stop_loss:
                            close_reason = "Stop Loss"
                        elif current_price <= position.take_profit:
                            close_reason = "Take Profit"

                # Check maximum duration (configurable, default 24 candles = 6 hours)
                if position.duration >= self.max_position_duration:
                    should_close = True
                    close_reason = "Max Duration"

                if should_close:
                    positions_to_close.append(position)
                    pnl = self._close_position(position, current_price)

                    # Track duration patterns for optimization
                    if pnl > 0:
                        self.position_durations['profitable'].append(position.duration)
                    else:
                        self.position_durations['losing'].append(position.duration)

                    closed_info.append({
                        'pair': pair,
                        'reason': close_reason,
                        'pnl': pnl,
                        'duration': position.duration
                    })

            # Remove closed positions
            for position in positions_to_close:
                self.positions[pair].remove(position)

        return closed_info

    def _calculate_reward(self) -> float:
        """
        Improved reward function with immediate feedback - V5.

        Key Features:
        - Immediate +2.0 reward when position opens successfully
        - Large reward/penalty when position closes (uses sophisticated _calculate_trade_outcome_reward)
        - Penalty for failed position opening attempts (-5.0)
        - Small penalty for pure inaction (-0.5 per step)

        This solves PPO's credit assignment problem by providing immediate feedback.
        """
        reward = 0.0

        # 1. IMMEDIATE FEEDBACK: Reward for opening positions THIS STEP
        recent_opened = [pos for pos in self._get_all_positions()
                        if pos.entry_time == self.current_step]

        if len(recent_opened) > 0:
            reward += 2.0 * len(recent_opened)  # Small bonus for each position opened
            logger.debug(f"Step {self.current_step}: Opened {len(recent_opened)} positions → +{2.0 * len(recent_opened):.1f} reward")

        # 2. OUTCOME FEEDBACK: Large reward/penalty when positions close
        # Only reward positions that closed JUST NOW (this step or last step)
        # Track via close_time which should be set when position closes
        recent_closed = [pos for pos in self.closed_positions
                        if hasattr(pos, 'close_time') and pos.close_time >= self.current_step - 1]

        for position in recent_closed:
            profit = position.unrealized_pnl
            duration = position.close_time - position.entry_time if hasattr(position, 'close_time') else 1
            # Use the sophisticated reward function that was previously unused!
            outcome_reward = self._calculate_trade_outcome_reward(profit, duration)
            reward += outcome_reward
            logger.debug(f"Step {self.current_step}: Closed position ${profit:.2f} in {duration} candles → {outcome_reward:+.1f} reward")

        # 3. PENALTY FOR FAILED POSITION OPENINGS
        # Check if any BUY/SELL actions failed to open positions
        failed_opens = self.last_trade_info.get('failed_opens', [])
        if len(failed_opens) > 0:
            penalty = -5.0 * len(failed_opens)
            reward += penalty
            logger.debug(f"Step {self.current_step}: {len(failed_opens)} failed opens → {penalty:.1f} reward")

        # 4. SMALL PENALTY FOR PURE INACTION
        # Discourage holding with no positions AND no recent trades
        if len(self._get_all_positions()) == 0 and len(recent_closed) == 0:
            reward -= 0.5  # Smaller, consistent penalty

        return reward

    def _calculate_trade_outcome_reward(self, profit: float, duration: int) -> float:
        """Calculate reward based on trade profit/loss with gradual scaling."""
        cfg = self.reward_config

        # Exceptional trades
        primary = cfg.get('primary_goal', {})
        if profit >= primary.get('profit_threshold', 37.50):
            if duration <= primary.get('duration_max', 12):
                logger.info(f"🎯 PRIMARY GOAL! ${profit:.2f} in {duration} candles")
                return primary.get('bonus', 300.0)

        excellent = cfg.get('excellent_trade', {})
        if profit >= excellent.get('profit_threshold', 30.0):
            if duration <= excellent.get('duration_max', 10):
                return excellent.get('bonus', 200.0)

        good = cfg.get('good_trade', {})
        if profit >= good.get('profit_threshold', 22.50):
            if duration <= good.get('duration_max', 8):
                return good.get('bonus', 120.0)

        # Profitable trades
        quick = cfg.get('quick_profit', {})
        if profit >= quick.get('profit_threshold', 15.0):
            if duration <= quick.get('duration_max', 6):
                return quick.get('bonus', 60.0)

        decent = cfg.get('decent_profit', {})
        if profit >= decent.get('profit_threshold', 10.0):
            return decent.get('base_bonus', 30.0) + (profit * decent.get('multiplier', 0.5))

        small = cfg.get('small_profit', {})
        if profit >= small.get('profit_threshold', 5.0):
            return small.get('base_bonus', 15.0) + (profit * small.get('multiplier', 0.3))

        # GRADUAL LOSS SCALING (NO CLIFF!)
        loss_cfg = cfg.get('loss_penalties', {})

        if profit >= 0.0:
            return loss_cfg.get('breakeven', {}).get('base_reward', 5.0)

        tiny = loss_cfg.get('tiny_loss', {})
        if profit >= tiny.get('threshold', -10.0):
            base = tiny.get('base_reward', 2.0)
            mult = tiny.get('multiplier', -0.2)
            return base + (abs(profit) * mult)

        small_loss = loss_cfg.get('small_loss', {})
        if profit >= small_loss.get('threshold', -20.0):
            return abs(profit) * small_loss.get('multiplier', -0.5)

        normal = loss_cfg.get('normal_loss', {})
        if profit >= normal.get('threshold', -30.0):
            return abs(profit) * normal.get('multiplier', -1.0)

        full_sl = loss_cfg.get('full_sl', {})
        if profit >= full_sl.get('threshold', -40.0):
            return abs(profit) * full_sl.get('multiplier', -1.5)

        large = loss_cfg.get('large_loss', {})
        if profit >= large.get('threshold', -60.0):
            return (abs(profit) * large.get('multiplier', -2.0)) - large.get('extra_penalty', 20.0)

        catastrophic = loss_cfg.get('catastrophic', {})
        return (abs(profit) * catastrophic.get('multiplier', -2.5)) - catastrophic.get('extra_penalty', 50.0)

    def _calculate_activity_reward(self, position: Position) -> float:
        """Calculate reward for trading activity to encourage moderate trading."""
        cfg = self.reward_config.get('activity', {})

        if not cfg.get('enabled', True):
            return 0.0

        activity_reward = cfg.get('base_trade_reward', 3.0)

        if self._is_favorable_session():
            activity_reward += cfg.get('favorable_session_bonus', 2.0)

        if len(self._get_all_positions()) <= 2:
            activity_reward += cfg.get('position_discipline_bonus', 2.0)

        threshold = cfg.get('overtrading_threshold', 5)
        recent_trades = len([p for p in self.closed_positions
                            if p.entry_time >= self.current_step - 20])

        if recent_trades > threshold:
            penalty_per_trade = cfg.get('overtrading_penalty_per_trade', 2.0)
            activity_reward -= (recent_trades - threshold) * penalty_per_trade

        return activity_reward

    def _calculate_account_performance_reward(self) -> float:
        """Calculate reward based on overall account performance."""
        reward = 0.0
        account_return = (self.balance - self.initial_balance) / self.initial_balance

        if account_return > 0.05:
            reward += 50.0
        elif account_return > 0.02:
            reward += 20.0

        if account_return < -0.05:
            reward -= 50.0
        if account_return < -0.10:
            reward -= 150.0

        return reward

    def _calculate_position_management_reward(self) -> float:
        """Reward for good position management - penalize excessive positions only."""
        reward = 0.0
        total_positions = len(self._get_all_positions())

        # ONLY penalize overtrading (>3 positions)
        # Do NOT reward holding or low position counts
        if total_positions > 3:
            reward -= (total_positions - 3) * 10.0

        return reward

    def _calculate_unrealized_pnl_reward(self) -> float:
        """Small influence from unrealized P&L."""
        reward = 0.0
        total_unrealized = self._get_total_unrealized_pnl()

        if total_unrealized > 20.0:
            reward += 10.0
        elif total_unrealized < -30.0:
            reward -= 15.0

        return reward

    def _is_favorable_session(self) -> bool:
        """Check if current time is in favorable trading session."""
        data_index = self.start_step + self.current_step
        if data_index >= len(self.data[self.pairs[0]]):
            data_index = len(self.data[self.pairs[0]]) - 1

        timestamp = self.data[self.pairs[0]].index[data_index]
        hour = timestamp.hour

        # European session (8-16 GMT) and early American overlap
        return (8 <= hour < 17)

    def _get_observation(self) -> np.ndarray:
        """
        Get current environment observation.

        Returns:
            Observation array with market data, portfolio state, and time features
        """
        observation_parts = []

        # 1. Market data for each pair
        for pair in self.pairs:
            pair_data = self._get_current_market_data(pair)
            observation_parts.extend(pair_data)

        # 2. Portfolio state for each pair
        for pair in self.pairs:
            portfolio_state = self._get_pair_portfolio_state(pair)
            observation_parts.extend(portfolio_state)

        # 3. Global portfolio state
        global_state = [
            self.balance / self.initial_balance,  # Normalized balance
            self._get_total_unrealized_pnl() / 1000.0,  # Normalized total unrealized P&L
            len(self._get_all_positions()) / (len(self.pairs) * 2),  # Normalized position count
            self._get_equity() / self.initial_balance  # Normalized equity
        ]
        observation_parts.extend(global_state)

        # 4. Time features
        time_features = self._get_time_features()
        observation_parts.extend(time_features)

        # Convert to numpy array and ensure float32
        observation = np.array(observation_parts, dtype=np.float32)

        # Handle NaN values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return observation

    def _get_current_market_data(self, pair: str) -> List[float]:
        """Get current market data for a specific pair."""
        data_index = self.start_step + self.current_step
        if data_index >= len(self.data[pair]):
            data_index = len(self.data[pair]) - 1

        row = self.data[pair].iloc[data_index]

        # Extract all numeric columns except Symbol
        market_data = []
        for col in self.data[pair].columns:
            if col != 'Symbol' and pd.api.types.is_numeric_dtype(self.data[pair][col]):
                value = row[col] if pd.notna(row[col]) else 0.0
                market_data.append(float(value))

        return market_data

    def _get_pair_portfolio_state(self, pair: str) -> List[float]:
        """Get portfolio state for a specific pair."""
        positions = self.positions[pair]

        if not positions:
            return [0.0, 0.0, 0.0, 0.0, 1.0]  # No position, can trade

        # Aggregate position information
        total_size = sum(pos.size for pos in positions)
        total_unrealized = sum(pos.unrealized_pnl for pos in positions)
        avg_duration = np.mean([pos.duration for pos in positions])

        # Position type: +1 for net long, -1 for net short, 0 for neutral
        long_size = sum(pos.size for pos in positions if pos.action == ActionType.BUY)
        short_size = sum(pos.size for pos in positions if pos.action == ActionType.SELL)
        net_position = (long_size - short_size) / max(total_size, 0.1)

        # Can trade: 1 if can open new position, 0 if at limit
        can_trade = 1.0 if len(positions) < self.max_positions_per_pair else 0.0

        return [
            total_size / 1.0,  # Normalized position size
            total_unrealized / 100.0,  # Normalized unrealized P&L
            avg_duration / 24.0,  # Normalized duration (in hours)
            net_position,  # Net position direction
            can_trade  # Can trade flag
        ]

    def _get_time_features(self) -> List[float]:
        """Get time-based features."""
        data_index = self.start_step + self.current_step
        if data_index >= len(self.data[self.pairs[0]]):
            data_index = len(self.data[self.pairs[0]]) - 1

        timestamp = self.data[self.pairs[0]].index[data_index]

        # Normalize time features
        hour = timestamp.hour / 24.0  # 0-1
        day_of_week = timestamp.dayofweek / 6.0  # 0-1 (Monday=0, Sunday=6)

        # Market session (simplified): 0=Asian, 0.33=European, 0.66=American, 1=Overlap
        if 0 <= hour * 24 < 8:  # Asian session
            market_session = 0.0
        elif 8 <= hour * 24 < 16:  # European session
            market_session = 0.33
        elif 16 <= hour * 24 < 24:  # American session
            market_session = 0.66
        else:
            market_session = 1.0  # Overlap

        return [hour, day_of_week, market_session]

    def _get_current_price(self, pair: str) -> float:
        """Get current Close price for a pair."""
        data_index = self.start_step + self.current_step
        if data_index >= len(self.data[pair]):
            data_index = len(self.data[pair]) - 1
        return float(self.data[pair].iloc[data_index]['Close'])

    def _get_spread(self, pair: str) -> float:
        """Get spread for a pair in price units."""
        return self._pips_to_price(pair, self.spread_pips.get(pair, 2.0))

    def _pips_to_price(self, pair: str, pips: float) -> float:
        """Convert pips to price units."""
        if 'JPY' in pair:
            return pips * 0.01  # JPY pairs: 1 pip = 0.01
        else:
            return pips * 0.0001  # Major pairs: 1 pip = 0.0001

    def _calculate_required_margin(self, pair: str, lot_size: float) -> float:
        """Calculate required margin for a position (simplified)."""
        current_price = self._get_current_price(pair)
        leverage = 100  # 1:100 leverage

        if 'JPY' in pair:
            notional_value = lot_size * 100000 * current_price / 100  # JPY adjustment
        else:
            notional_value = lot_size * 100000 * current_price

        return notional_value / leverage

    def _get_all_positions(self) -> List[Position]:
        """Get all open positions across all pairs."""
        all_positions = []
        for positions in self.positions.values():
            all_positions.extend(positions)
        return all_positions

    def _get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self._get_all_positions())

    def _get_equity(self) -> float:
        """Get current equity (balance + unrealized P&L)."""
        return self.balance + self._get_total_unrealized_pnl()

    def _get_total_used_margin(self) -> float:
        """Calculate total margin currently used by all open positions."""
        total_margin = 0.0
        for pair, positions in self.positions.items():
            for position in positions:
                total_margin += self._calculate_required_margin(pair, position.size)
        return total_margin

    def _get_available_margin(self) -> float:
        """Get available margin for new positions."""
        max_margin = self.balance * 0.8  # Use max 80% of balance for margin (allows more trading)
        used_margin = self._get_total_used_margin()
        available = max(0, max_margin - used_margin)
        logger.debug(f"Margin check: max=${max_margin:.2f}, used=${used_margin:.2f}, available=${available:.2f}")
        return available

    def get_duration_insights(self) -> Dict:
        """Get insights from collected duration data for optimization."""
        insights = {}

        profitable_durations = self.position_durations['profitable']
        losing_durations = self.position_durations['losing']

        if profitable_durations:
            insights['profitable'] = {
                'count': len(profitable_durations),
                'mean': np.mean(profitable_durations),
                'median': np.median(profitable_durations),
                'p95': np.percentile(profitable_durations, 95),
                'max': max(profitable_durations)
            }

        if losing_durations:
            insights['losing'] = {
                'count': len(losing_durations),
                'mean': np.mean(losing_durations),
                'median': np.median(losing_durations),
                'p95': np.percentile(losing_durations, 95),
                'max': max(losing_durations)
            }

        # Suggest optimal duration
        if profitable_durations and losing_durations:
            prof_mean = np.mean(profitable_durations)
            prof_p95 = np.percentile(profitable_durations, 95)

            # Conservative approach: capture 95% of profitable trades
            optimal_duration = max(8, min(int(prof_p95), 20))  # Between 8-20 candles

            insights['suggestion'] = {
                'optimal_duration': optimal_duration,
                'current_duration': self.max_position_duration,
                'reasoning': f"Captures 95% of profitable trades ({prof_p95:.1f} candles), capped at 20"
            }

        return insights

    def _is_terminated(self) -> bool:
        """Check if episode should be terminated."""
        # Terminate if balance drops too low (margin call)
        if self.balance < self.initial_balance * 0.5:
            logger.info("Episode terminated - Margin call")
            return True

        # Terminate if achieved goal multiple times (success condition)
        successful_trades = sum(1 for pos in self.closed_positions
                              if pos.unrealized_pnl >= 100.0 and pos.duration <= 4)
        if successful_trades >= 3:
            logger.info("Episode terminated - Multiple successful trades")
            return True

        return False

    def _update_episode_stats(self):
        """Update episode statistics."""
        current_equity = self._get_equity()
        self.episode_stats['max_balance'] = max(self.episode_stats['max_balance'], current_equity)
        self.episode_stats['min_balance'] = min(self.episode_stats['min_balance'], current_equity)

        # Calculate drawdown
        drawdown = (self.episode_stats['max_balance'] - current_equity) / self.episode_stats['max_balance']
        self.episode_stats['max_drawdown'] = max(self.episode_stats['max_drawdown'], drawdown)

    def render(self, mode: str = "human"):
        """Render the environment state."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Equity: ${self._get_equity():.2f}")
            print(f"Total Positions: {len(self._get_all_positions())}")
            print(f"Unrealized P&L: ${self._get_total_unrealized_pnl():.2f}")
            print("-" * 50)

    def close(self):
        """Clean up environment resources."""
        pass


if __name__ == "__main__":
    # Test the environment
    print("Testing MultiPairForexEnv...")

    # This would be run with actual DataManager data
    # For now, we'll import and test basic functionality
    try:
        from data_manager import DataManager

        # Initialize data manager and get sample data
        dm = DataManager()
        data = dm.get_multi_pair_data()

        # Create environment
        env = MultiPairForexEnv(data)
        print(f"✅ Environment created successfully")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space.shape}")

        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful - Observation shape: {obs.shape}")

        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Balance=${info['balance']:.2f}")

            if terminated or truncated:
                break

        print("✅ Environment test completed successfully!")

    except ImportError:
        print("❌ DataManager not available - Please run from main project directory")
    except Exception as e:
        print(f"❌ Error testing environment: {e}")