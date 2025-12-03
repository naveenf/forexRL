"""
Single-Pair Forex Trading Environment for Reinforcement Learning

Simplified version focused on USD/JPY pair only.
Designed to encourage trading and solve the "do nothing" problem.

Action Space:
- 3 discrete actions: HOLD (0), BUY (1), SELL (2)

Observation Space:
- Market data: OHLCV + technical indicators
- Portfolio state: position info, balance, P&L
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Trading action types."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Position:
    """Represents a forex trading position."""
    action: ActionType  # BUY or SELL
    entry_price: float
    entry_time: int
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    close_time: int = 0

    @property
    def duration(self) -> int:
        """Get position duration in steps."""
        if self.close_time > 0:
            return self.close_time - self.entry_time
        return 0

    def update_pnl(self, current_price: float):
        """Update unrealized P&L for this position."""
        if self.action == ActionType.BUY:
            price_diff = current_price - self.entry_price
        else:  # SELL
            price_diff = self.entry_price - current_price

        # Calculate P&L (for JPY pairs, 1 pip = 0.01)
        pip_value = 10.0
        pip_diff = price_diff * 100
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


class SinglePairForexEnv(gym.Env):
    """
    Single-pair forex trading environment for reinforcement learning.

    Focused on USD/JPY pair with simplified reward function.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 lot_size: float = 0.01,
                 spread_pips: float = 1.8,
                 sl_tp_pips: Tuple[float, float] = (15.0, 22.5),
                 max_episode_steps: int = 1500,
                 commission_per_lot: float = 0.5,
                 risk_per_trade_pct: float = 0.03,
                 max_position_duration: int = 24):
        """
        Initialize the single-pair forex environment.

        Args:
            data: DataFrame with OHLCV + indicators for USD/JPY
            initial_balance: Starting account balance (default: $10,000)
            lot_size: Base lot size (default: 0.01)
            spread_pips: Spread in pips (default: 1.8)
            sl_tp_pips: (stop_loss_pips, take_profit_pips) - default 1:1.5 ratio
            max_episode_steps: Maximum steps per episode (default: 1500)
            commission_per_lot: Commission cost per lot (default: $0.50)
            risk_per_trade_pct: Risk per trade as % of account (default: 3%)
            max_position_duration: Max position duration in candles (default: 24)
        """
        super().__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.base_lot_size = lot_size
        self.spread_pips = spread_pips
        self.max_episode_steps = max_episode_steps
        self.commission_per_lot = commission_per_lot
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_duration = max_position_duration

        # SL/TP configuration
        self.sl_pips, self.tp_pips = sl_tp_pips

        logger.info(f"Single-pair environment initialized for USD/JPY")
        logger.info(f"Initial balance: ${self.initial_balance:,.2f}")

        # Action space: 3 actions (HOLD, BUY, SELL)
        self.action_space = spaces.Discrete(3)

        # Setup observation space
        self._setup_observation_space()

        # Initialize state
        self.reset()

    def _setup_observation_space(self):
        """Setup the observation space based on available data features."""
        # Count features (exclude Symbol column if present)
        features_per_candle = len([col for col in self.data.columns if col != 'Symbol'])

        # Portfolio features: position_size, unrealized_pnl, position_duration, position_type, has_position
        portfolio_features = 5

        # Global features: balance, equity
        global_features = 2

        # Time features: hour, day_of_week, market_session
        time_features = 3

        total_features = features_per_candle + portfolio_features + global_features + time_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        logger.info(f"Observation space: {total_features} features")
        logger.info(f"  - Market features: {features_per_candle}")
        logger.info(f"  - Portfolio features: {portfolio_features}")
        logger.info(f"  - Global features: {global_features}")
        logger.info(f"  - Time features: {time_features}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset state
        self.current_step = 0
        self.balance = self.initial_balance
        self.position: Optional[Position] = None
        self.closed_positions: List[Position] = []
        self.total_commission_paid = 0.0

        # Episode statistics
        self.episode_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_balance': self.initial_balance,
            'min_balance': self.initial_balance,
        }

        # Random starting point (200 candles minimum for indicator warm-up)
        min_data_length = len(self.data)
        max_start = max(200, min_data_length - self.max_episode_steps - 1)
        self.start_step = self.np_random.integers(200, max_start) if max_start > 200 else 200

        observation = self._get_observation()

        info = {
            'balance': self.balance,
            'has_position': self.position is not None,
            'start_step': self.start_step
        }

        logger.info(f"Environment reset - Balance: ${self.balance:,.2f}, Start: {self.start_step}")
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Execute action (may close existing position when opening new one)
        trade_info = self._execute_action(ActionType(action))

        # FIX: Capture position closed via trade replacement from trade_info
        # This happens when BUY/SELL closes an existing position before opening new one
        closed_position_info = trade_info.get('closed_position', None)

        # Update existing position
        self._update_position()

        # Check for SL/TP/duration closures (only if no position was just closed)
        if closed_position_info is None:
            closed_position_info = self._check_position_closure()

        # Calculate reward
        reward = self._calculate_reward(trade_info, closed_position_info)

        # Update step
        self.current_step += 1

        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps

        # Force close any open position at episode end
        if (terminated or truncated) and self.position is not None:
            current_price = self._get_current_price()
            position_size = self.position.size
            duration = self.current_step - self.position.entry_time
            pnl = self._close_position(current_price)

            # Create closed_position_info for proper tracking
            closed_position_info = {
                'reason': 'Episode End',
                'pnl': pnl,
                'duration': duration,
                'size': position_size
            }
            logger.info(f"🏁 Forced position close at episode end: ${pnl:.2f} P&L")

        # Get new observation
        observation = self._get_observation()

        # Update stats
        self._update_episode_stats()

        # Info
        info = {
            'balance': self.balance,
            'equity': self._get_equity(),
            'has_position': self.position is not None,
            'unrealized_pnl': self.position.unrealized_pnl if self.position else 0.0,
            'trade_info': trade_info,
            'closed_position': closed_position_info,
            'commission_paid': self.total_commission_paid,
            'step': self.current_step
        }

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: ActionType) -> Dict:
        """Execute trading action."""
        trade_info = {'action': action.name, 'success': False}

        if action == ActionType.HOLD:
            return trade_info

        current_price = self._get_current_price()

        # Close existing position if opening new one
        if self.position is not None:
            # Save position info before closing
            position_size = self.position.size
            duration = self.current_step - self.position.entry_time
            pnl = self._close_position(current_price)
            # FIX: Include all fields the callback expects
            trade_info['closed_position'] = {
                'pnl': pnl,
                'duration': duration,
                'size': position_size,
                'reason': 'New Position'
            }

        # Open new position
        if action in [ActionType.BUY, ActionType.SELL]:
            position = self._open_position(action, current_price)
            if position:
                trade_info['success'] = True
                trade_info['entry_price'] = position.entry_price
                trade_info['size'] = position.size
                logger.info(f"✅ Opened {action.name} position: {position.size} lots at ${current_price:.3f}")

        return trade_info

    def _apply_slippage(self, price: float, action: ActionType) -> float:
        """
        Apply realistic slippage to execution price.

        Slippage simulates real-world execution where you don't always get
        the exact price you see. Typically 0.5-2.0 pips in normal conditions.

        Args:
            price: Base execution price
            action: BUY or SELL

        Returns:
            Price with slippage applied
        """
        # Random slippage between 0.5-2.0 pips (realistic for normal market conditions)
        slippage_pips = np.random.uniform(0.5, 2.0)
        slippage = slippage_pips * 0.01  # Convert pips to price for JPY pairs

        # Slippage always increases cost:
        # - BUY: pay slightly more than expected
        # - SELL: receive slightly less than expected
        if action == ActionType.BUY:
            return price + slippage
        else:  # SELL
            return price - slippage

    def _calculate_position_size(self, stop_loss_pips: float) -> float:
        """
        Calculate position size to risk exactly 3% of account per trade.

        Formula: Lot Size = Risk Amount / (SL in pips × Pip Value)
        Example: $300 / (15 pips × $10/pip) = 2.0 lots

        This ensures we risk exactly $300 on a $10,000 account (3% risk).
        """
        # Risk per trade in USD
        account_risk = self.balance * self.risk_per_trade_pct  # $10,000 × 0.03 = $300

        # Pip value for JPY pairs ($10 per pip per standard lot)
        pip_value = 10.0

        # Calculate lot size
        if stop_loss_pips > 0:
            lot_size = account_risk / (stop_loss_pips * pip_value)
            # Example: $300 / (15 pips × $10) = $300 / $150 = 2.0 lots

            # Safety bounds (allow up to 5 lots for proper risk scaling)
            lot_size = min(lot_size, 5.0)  # Cap at 5 lots (was 0.5 - FIXED!)
            lot_size = max(lot_size, 0.01)  # Min 0.01 lots (micro lot)

            logger.debug(f"Position sizing: Balance=${self.balance:.2f}, "
                        f"Risk=${account_risk:.2f}, SL={stop_loss_pips}pips, "
                        f"Calculated size={lot_size:.3f} lots")

            return round(lot_size, 3)

        return self.base_lot_size

    def _open_position(self, action: ActionType, current_price: float) -> Optional[Position]:
        """Open a new trading position."""
        # Spread-adjusted entry price
        spread = self.spread_pips * 0.01  # Convert pips to price for JPY
        if action == ActionType.BUY:
            entry_price = current_price + spread / 2
        else:
            entry_price = current_price - spread / 2

        # Apply slippage to entry price (realistic execution)
        entry_price = self._apply_slippage(entry_price, action)

        # Calculate SL and TP
        sl_distance = self.sl_pips * 0.01
        tp_distance = self.tp_pips * 0.01

        if action == ActionType.BUY:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Calculate position size
        position_size = self._calculate_position_size(self.sl_pips)

        # Simple balance check
        if self.balance <= 100:
            logger.warning(f"Insufficient balance: ${self.balance:.2f}")
            return None

        # Create position
        position = Position(
            action=action,
            entry_price=entry_price,
            entry_time=self.current_step,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.position = position

        # Deduct commission
        commission = self.commission_per_lot * position_size
        self.balance -= commission
        self.total_commission_paid += commission

        return position

    def _close_position(self, current_price: float) -> float:
        """Close the current position."""
        if self.position is None:
            return 0.0

        # Update P&L with spread-adjusted exit price
        spread = self.spread_pips * 0.01
        if self.position.action == ActionType.BUY:
            exit_price = current_price - spread / 2
        else:
            exit_price = current_price + spread / 2

        # Apply slippage to exit price (realistic execution)
        exit_price = self._apply_slippage(exit_price, self.position.action)

        self.position.update_pnl(exit_price)
        realized_pnl = self.position.unrealized_pnl

        # Apply P&L to balance
        self.balance += realized_pnl

        # Deduct commission
        commission = self.commission_per_lot * self.position.size
        self.balance -= commission
        self.total_commission_paid += commission

        # Track closing
        self.position.close_time = self.current_step
        self.closed_positions.append(self.position)

        # Update stats
        self.episode_stats['total_trades'] += 1
        if realized_pnl > 0:
            self.episode_stats['winning_trades'] += 1
        else:
            self.episode_stats['losing_trades'] += 1

        logger.info(f"💰 Closed position: ${realized_pnl:.2f} P&L in {self.position.duration} candles")

        self.position = None
        return realized_pnl

    def _update_position(self):
        """Update unrealized P&L for open position."""
        if self.position:
            current_price = self._get_current_price()
            self.position.update_pnl(current_price)

    def _check_position_closure(self) -> Optional[Dict]:
        """Check if position should be closed due to SL/TP or duration."""
        if self.position is None:
            return None

        current_price = self._get_current_price()
        should_close = False
        close_reason = ""

        # Check SL/TP
        if self.position.should_close_sl_tp(current_price):
            should_close = True
            if self.position.action == ActionType.BUY:
                if current_price <= self.position.stop_loss:
                    close_reason = "Stop Loss"
                elif current_price >= self.position.take_profit:
                    close_reason = "Take Profit"
            else:
                if current_price >= self.position.stop_loss:
                    close_reason = "Stop Loss"
                elif current_price <= self.position.take_profit:
                    close_reason = "Take Profit"

        # Check duration
        duration = self.current_step - self.position.entry_time
        if duration >= self.max_position_duration:
            should_close = True
            close_reason = "Max Duration"

        if should_close:
            position_size = self.position.size  # Save before closing
            pnl = self._close_position(current_price)
            return {
                'reason': close_reason,
                'pnl': pnl,
                'duration': duration,
                'size': position_size
            }

        return None

    def _get_recent_returns(self, window: int = 20) -> List[float]:
        """
        Calculate recent price returns for volatility measurement.

        Args:
            window: Number of candles to look back

        Returns:
            List of returns (price changes as percentages)
        """
        if self.current_step < window:
            return []

        returns = []
        for i in range(window):
            idx = self.start_step + self.current_step - i - 1
            if idx > 0 and idx < len(self.data):
                curr_price = self.data.iloc[idx]['Close']
                prev_price = self.data.iloc[idx - 1]['Close']
                returns.append((curr_price - prev_price) / prev_price)

        return returns

    def _calculate_reward(self, trade_info: Dict, closed_position_info: Optional[Dict]) -> float:
        """
        Sharpe-based reward function with risk adjustment.

        Formula: R_t = Opening_Signal + Sharpe_Reward - Transaction_Costs - Inactivity_Penalty - Holding_Penalty

        Key improvements:
        1. Risk-adjusted returns (Sharpe ratio-like)
        2. Transaction costs explicitly penalized in reward
        3. Holding time penalty to prevent infinite positions
        """
        reward = 0.0

        # 1. OPENING POSITION SIGNAL (small bonus to offset initial transaction cost)
        if trade_info['action'] in ['BUY', 'SELL'] and trade_info['success']:
            reward += 5.0  # Reduced from 10.0 to be more conservative
            logger.debug(f"Step {self.current_step}: Opened position → +5.0 reward")

        # 2. RISK-ADJUSTED REWARD FOR CLOSED POSITIONS
        if closed_position_info is not None:
            pnl = closed_position_info['pnl']
            duration = closed_position_info['duration']

            # Calculate volatility from recent returns
            recent_returns = self._get_recent_returns(window=20)

            if len(recent_returns) > 5:
                volatility = np.std(recent_returns)
                # Sharpe-like reward: profit normalized by volatility
                # Multiply by 100 to scale volatility appropriately for forex
                sharpe_reward = (pnl / max(volatility * 100, 1.0)) * 10.0
            else:
                # Fallback if insufficient history (early in episode)
                sharpe_reward = pnl * 0.5

            # Duration bonuses for quick profitable trades
            if pnl > 0:
                if duration <= 12:  # Quick profit (<3 hours)
                    sharpe_reward += 50.0
                elif duration <= 20:  # Moderate speed
                    sharpe_reward += 20.0

            # CRITICAL: Transaction cost penalty (spread + commission)
            position_size = closed_position_info.get('size', 1.0)
            transaction_cost = (self.commission_per_lot * position_size * 2) + \
                              (self.spread_pips * 0.01 * 10.0 * position_size)

            reward += sharpe_reward - transaction_cost

            logger.debug(f"Step {self.current_step}: Closed ${pnl:.2f}, "
                        f"Sharpe reward: {sharpe_reward:.2f}, "
                        f"Transaction cost: {transaction_cost:.2f}, "
                        f"Net: {sharpe_reward - transaction_cost:+.2f}")

        # 3. INACTIVITY/HOLDING PENALTIES
        if trade_info['action'] == 'HOLD':
            if self.position is None:
                # No position: AGGRESSIVE penalty to force exploration
                reward -= 5.0  # Increased from 2.0 - must be worse than trying
            else:
                # Has position: progressive penalty for holding too long
                duration = self.current_step - self.position.entry_time
                # Penalty ramps up: -0.1 at 24 candles, -0.5 at 120 candles
                holding_penalty = -0.5 * (duration / 24.0)
                reward += max(holding_penalty, -3.0)  # Cap at -3.0 per step

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current environment observation."""
        observation_parts = []

        # 1. Market data
        market_data = self._get_current_market_data()
        observation_parts.extend(market_data)

        # 2. Portfolio state
        if self.position:
            duration = self.current_step - self.position.entry_time
            portfolio_state = [
                self.position.size / 1.0,  # Normalized position size
                self.position.unrealized_pnl / 100.0,  # Normalized P&L
                duration / 24.0,  # Normalized duration
                1.0 if self.position.action == ActionType.BUY else -1.0,  # Position direction
                1.0  # Has position
            ]
        else:
            portfolio_state = [0.0, 0.0, 0.0, 0.0, 0.0]  # No position

        observation_parts.extend(portfolio_state)

        # 3. Global state
        global_state = [
            self.balance / self.initial_balance,  # Normalized balance
            self._get_equity() / self.initial_balance  # Normalized equity
        ]
        observation_parts.extend(global_state)

        # 4. Time features
        time_features = self._get_time_features()
        observation_parts.extend(time_features)

        # Convert to array
        observation = np.array(observation_parts, dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return observation

    def _get_current_market_data(self) -> List[float]:
        """Get current market data."""
        data_index = self.start_step + self.current_step
        if data_index >= len(self.data):
            data_index = len(self.data) - 1

        row = self.data.iloc[data_index]

        market_data = []
        for col in self.data.columns:
            if col != 'Symbol' and pd.api.types.is_numeric_dtype(self.data[col]):
                value = row[col] if pd.notna(row[col]) else 0.0
                market_data.append(float(value))

        return market_data

    def _get_time_features(self) -> List[float]:
        """Get time-based features."""
        data_index = self.start_step + self.current_step
        if data_index >= len(self.data):
            data_index = len(self.data) - 1

        timestamp = self.data.index[data_index]

        hour = timestamp.hour / 24.0
        day_of_week = timestamp.dayofweek / 6.0

        # Market session
        if 0 <= timestamp.hour < 8:
            market_session = 0.0  # Asian
        elif 8 <= timestamp.hour < 16:
            market_session = 0.5  # European
        else:
            market_session = 1.0  # American

        return [hour, day_of_week, market_session]

    def _get_current_price(self) -> float:
        """Get current Close price."""
        data_index = self.start_step + self.current_step
        if data_index >= len(self.data):
            data_index = len(self.data) - 1
        return float(self.data.iloc[data_index]['Close'])

    def _get_equity(self) -> float:
        """Get current equity (balance + unrealized P&L)."""
        unrealized = self.position.unrealized_pnl if self.position else 0.0
        return self.balance + unrealized

    def _is_terminated(self) -> bool:
        """Check if episode should be terminated."""
        # Terminate if balance drops too low (30% max drawdown - industry standard)
        if self.balance < self.initial_balance * 0.7:  # 30% loss limit (was 80%)
            logger.info(f"Episode terminated - Max drawdown reached (Balance: ${self.balance:.2f})")
            return True

        # Terminate if balance goes negative (safety check)
        if self.balance < 0:
            logger.warning(f"Episode terminated - Negative balance (${self.balance:.2f})")
            return True

        return False

    def _update_episode_stats(self):
        """Update episode statistics."""
        current_equity = self._get_equity()
        self.episode_stats['max_balance'] = max(self.episode_stats['max_balance'], current_equity)
        self.episode_stats['min_balance'] = min(self.episode_stats['min_balance'], current_equity)

    def render(self, mode: str = "human"):
        """Render environment state."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Equity: ${self._get_equity():.2f}")
            print(f"Has Position: {self.position is not None}")
            if self.position:
                print(f"Unrealized P&L: ${self.position.unrealized_pnl:.2f}")
            print("-" * 50)

    def close(self):
        """Clean up environment resources."""
        pass


if __name__ == "__main__":
    print("Testing SinglePairForexEnv...")

    try:
        from data_manager import DataManager

        dm = DataManager()
        data = dm.get_multi_pair_data()
        usdjpy_data = data['USDJPY']

        env = SinglePairForexEnv(usdjpy_data)
        print(f"✅ Environment created successfully")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space.shape}")

        obs, info = env.reset()
        print(f"✅ Reset successful - Observation shape: {obs.shape}")

        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Balance=${info['balance']:.2f}")

            if terminated or truncated:
                break

        print("✅ Environment test completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
