"""
Data Manager for Forex Trading System

This module handles:
- MetaTrader 5 integration for real-time and historical data
- Technical indicators calculation
- Data preprocessing and normalization
- Sample data generation for testing
- Data validation and visualization

Key Features:
- Multi-pair forex data (EUR/USD, GBP/USD, USD/JPY)
- 15+ technical indicators (RSI, MACD, SMA, EMA, ATR, etc.)
- Real-time and historical data modes
- Data quality validation
- Interactive plotting capabilities
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logging.warning("MetaTrader5 not available - using sample data only")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    talib = None
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available - using basic indicators only")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Comprehensive data manager for forex trading system.

    Handles data acquisition, preprocessing, indicator calculation,
    and validation for EUR/USD, GBP/USD, and USD/JPY pairs.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataManager with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.mt5_initialized = False
        self.currency_pairs = ['EURUSD', 'AUDCHF', 'USDJPY']
        self.timeframe = 15 if not MT5_AVAILABLE else mt5.TIMEFRAME_M15  # 15-minute candles

        # Technical indicators configuration
        self.indicators_config = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'sma_periods': [20, 50, 200],
            'ema_periods': [12, 26],
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2,
            'stoch_k': 14,
            'stoch_d': 3,
            'cci_period': 20,
            'williams_period': 14,
            'momentum_period': 10,
            'roc_period': 12
        }

        logger.info("DataManager initialized successfully")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

        # Default configuration
        return {
            'data': {
                'currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'timeframe': '15M',
                'history_days': 365,
                'real_time': False
            },
            'indicators': {
                'enable_all': True,
                'custom_params': {}
            },
            'preprocessing': {
                'normalize': True,
                'handle_missing': True,
                'outlier_detection': True
            }
        }

    def initialize_mt5(self) -> bool:
        """
        Initialize MetaTrader 5 connection.

        Returns:
            bool: True if successful, False otherwise
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available - cannot initialize")
            return False

        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False

            self.mt5_initialized = True
            logger.info("MT5 initialized successfully")

            # Get account information
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Connected to MT5 account: {account_info.login}")

            return True

        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False

    def get_historical_data(self,
                          symbol: str,
                          days: int = 365,
                          timeframe: int = None) -> pd.DataFrame:
        """
        Fetch historical data from MetaTrader 5.

        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            days: Number of days of historical data
            timeframe: MT5 timeframe constant

        Returns:
            DataFrame with OHLCV data
        """
        if not self.mt5_initialized:
            if not self.initialize_mt5():
                raise RuntimeError("Cannot initialize MT5 connection")

        if timeframe is None:
            timeframe = self.timeframe

        # Calculate date range
        utc_to = datetime.now()
        utc_from = utc_to - timedelta(days=days)

        try:
            # Get historical rates
            rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)

            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Rename columns to standard format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            })

            # Add symbol column
            df['Symbol'] = symbol

            logger.info(f"Fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def load_csv_data(self, csv_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load forex data from CSV files provided by user.

        Expected CSV format:
        - Columns: Date, Time, Open, High, Low, Close, Volume
        - OR: DateTime, Open, High, Low, Close, Volume
        - Date format: YYYY-MM-DD or DD/MM/YYYY
        - Time format: HH:MM:SS (if separate) or combined DateTime

        Args:
            csv_files: Dictionary mapping pair names to file paths
                      e.g., {'EURUSD': 'data/EURUSD_15M.csv', 'GBPUSD': 'data/GBPUSD_15M.csv'}

        Returns:
            Dictionary with loaded and processed data for each currency pair
        """
        logger.info(f"Loading CSV data for {len(csv_files)} currency pairs...")

        loaded_data = {}

        for pair, file_path in csv_files.items():
            try:
                logger.info(f"Loading {pair} from {file_path}")

                # Try to read CSV with different common formats
                df = self._read_csv_flexible(file_path)

                if df is None:
                    logger.error(f"Failed to load {pair} from {file_path}")
                    continue

                # Validate and process the data
                df = self._process_csv_data(df, pair)

                if len(df) > 0:
                    loaded_data[pair] = df
                    logger.info(f"✅ {pair}: {len(df)} candles loaded ({df.index.min()} to {df.index.max()})")
                else:
                    logger.warning(f"❌ {pair}: No valid data after processing")

            except Exception as e:
                logger.error(f"Error loading {pair}: {e}")
                continue

        if loaded_data:
            logger.info(f"Successfully loaded data for {list(loaded_data.keys())}")
        else:
            logger.warning("No data loaded from CSV files - falling back to sample data")
            return self.load_sample_data()

        return loaded_data

    def _read_csv_flexible(self, file_path: str) -> Optional[pd.DataFrame]:
        """Try different CSV reading approaches."""
        read_attempts = [
            # Common separators and date formats
            {'sep': ',', 'parse_dates': [0], 'index_col': 0},
            {'sep': ';', 'parse_dates': [0], 'index_col': 0},
            {'sep': '\t', 'parse_dates': [0], 'index_col': 0},
            {'sep': ',', 'parse_dates': [[0, 1]], 'index_col': 0},  # Separate date/time columns
            {'sep': ','}  # No date parsing, will handle manually
        ]

        for attempt in read_attempts:
            try:
                df = pd.read_csv(file_path, **attempt)
                if len(df) > 100:  # Minimum viable dataset
                    return df
            except:
                continue

        return None

    def _process_csv_data(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Process and standardize CSV data format."""
        # Create a copy
        df = df.copy()

        # Standardize column names (case-insensitive)
        df.columns = df.columns.str.strip().str.lower()

        # Map common column name variations
        column_mapping = {
            'datetime': 'datetime',
            'date': 'date',
            'time': 'time',
            'timestamp': 'datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'vol': 'Volume',
            'tick_volume': 'Volume'
        }

        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Handle datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
                df.set_index('datetime', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                # Try to use first column as datetime
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df.set_index(df.columns[0], inplace=True)

        # Ensure required OHLC columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return pd.DataFrame()

        # Add Volume if missing (use tick volume approximation)
        if 'Volume' not in df.columns:
            df['Volume'] = np.random.lognormal(8, 0.5, len(df)).astype(int)
            logger.info(f"Added synthetic volume data for {pair}")

        # Add Symbol column
        df['Symbol'] = pair

        # Sort by datetime
        df.sort_index(inplace=True)

        # Remove invalid data
        df = df.dropna(subset=required_cols)

        # Validate OHLC relationships
        invalid_rows = (
            (df['High'] < df[['Open', 'Close']].max(axis=1)) |
            (df['Low'] > df[['Open', 'Close']].min(axis=1)) |
            (df['High'] < df['Low'])
        )

        if invalid_rows.sum() > 0:
            logger.warning(f"Removing {invalid_rows.sum()} rows with invalid OHLC data")
            df = df[~invalid_rows]

        # Convert to numeric
        for col in required_cols + ['Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def load_sample_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate sample forex data for testing when real data is not available.

        Returns:
            Dictionary with sample data for each currency pair
        """
        logger.info("Generating sample forex data for testing...")

        # Generate 1000 candles (about 10 days of 15-min data)
        n_candles = 1000
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=10),
            periods=n_candles,
            freq='15T'
        )

        sample_data = {}

        # Realistic forex price parameters
        pair_configs = {
            'EURUSD': {'base': 1.0800, 'volatility': 0.001, 'trend': 0.00001},
            'GBPUSD': {'base': 1.2700, 'volatility': 0.0012, 'trend': -0.00002},
            'USDJPY': {'base': 150.00, 'volatility': 0.15, 'trend': 0.001}
        }

        for pair, config in pair_configs.items():
            np.random.seed(42 + hash(pair) % 1000)  # Deterministic but different for each pair

            # Generate realistic price movement
            returns = np.random.normal(
                config['trend'],
                config['volatility'],
                n_candles
            )

            # Add some trend and mean reversion
            returns += 0.0001 * np.sin(np.arange(n_candles) * 0.01)

            # Generate OHLC from returns
            close_prices = config['base'] * np.cumprod(1 + returns)

            # Generate High, Low, Open
            high_low_range = np.random.uniform(0.0002, 0.0008, n_candles)
            high_ratio = np.random.uniform(0.3, 0.7, n_candles)

            highs = close_prices + high_low_range * high_ratio
            lows = close_prices - high_low_range * (1 - high_ratio)
            opens = np.roll(close_prices, 1)
            opens[0] = close_prices[0]

            # Generate volume (realistic for forex)
            volumes = np.random.lognormal(8, 0.5, n_candles).astype(int)

            # Create DataFrame
            df = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': close_prices,
                'Volume': volumes,
                'Symbol': pair
            }, index=dates)

            # Ensure OHLC logic (High >= max(O,C), Low <= min(O,C))
            df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
            df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))

            # Round to realistic pip precision
            if 'JPY' in pair:
                decimal_places = 3
            else:
                decimal_places = 5

            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].round(decimal_places)

            sample_data[pair] = df
            logger.info(f"Generated {len(df)} sample candles for {pair}")

        return sample_data

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for forex data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()

        # Use talib if available, otherwise basic calculations
        if TALIB_AVAILABLE:
            return self._calculate_talib_indicators(df)
        else:
            return self._calculate_basic_indicators(df)

    def _calculate_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using TA-Lib."""
        # Price arrays for talib
        open_prices = df['Open'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        close_prices = df['Close'].values
        volume = df['Volume'].values

        try:
            # 1. Moving Averages
            for period in self.indicators_config['sma_periods']:
                df[f'SMA_{period}'] = talib.SMA(close_prices, timeperiod=period)

            for period in self.indicators_config['ema_periods']:
                df[f'EMA_{period}'] = talib.EMA(close_prices, timeperiod=period)

            # 2. RSI (Relative Strength Index)
            df['RSI'] = talib.RSI(close_prices, timeperiod=self.indicators_config['rsi_period'])

            # 3. MACD
            macd, signal, hist = talib.MACD(
                close_prices,
                fastperiod=self.indicators_config['macd_fast'],
                slowperiod=self.indicators_config['macd_slow'],
                signalperiod=self.indicators_config['macd_signal']
            )
            df['MACD'] = macd
            df['MACD_Signal'] = signal
            df['MACD_Hist'] = hist

            # 4. Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                close_prices,
                timeperiod=self.indicators_config['bb_period'],
                nbdevup=self.indicators_config['bb_std'],
                nbdevdn=self.indicators_config['bb_std']
            )
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
            df['BB_Width'] = (upper - lower) / middle
            df['BB_Position'] = (close_prices - lower) / (upper - lower)

            # 5. ATR (Average True Range)
            df['ATR'] = talib.ATR(
                high_prices,
                low_prices,
                close_prices,
                timeperiod=self.indicators_config['atr_period']
            )

            # 6. Stochastic Oscillator
            slowk, slowd = talib.STOCH(
                high_prices,
                low_prices,
                close_prices,
                fastk_period=self.indicators_config['stoch_k'],
                slowk_period=self.indicators_config['stoch_d'],
                slowd_period=self.indicators_config['stoch_d']
            )
            df['Stoch_K'] = slowk
            df['Stoch_D'] = slowd

            # 7. CCI (Commodity Channel Index)
            df['CCI'] = talib.CCI(
                high_prices,
                low_prices,
                close_prices,
                timeperiod=self.indicators_config['cci_period']
            )

            # 8. Williams %R
            df['Williams_R'] = talib.WILLR(
                high_prices,
                low_prices,
                close_prices,
                timeperiod=self.indicators_config['williams_period']
            )

            # 9. Momentum and ROC
            df['Momentum'] = talib.MOM(
                close_prices,
                timeperiod=self.indicators_config['momentum_period']
            )
            df['ROC'] = talib.ROC(
                close_prices,
                timeperiod=self.indicators_config['roc_period']
            )

            # 10. OBV (On Balance Volume)
            df['OBV'] = talib.OBV(close_prices, volume)

            # 11. Custom Forex Indicators
            # Spread simulation (approximate)
            df['Spread'] = np.random.uniform(0.00008, 0.00020, len(df))

            # Volatility measures
            df['Price_Change'] = df['Close'].pct_change()
            df['Volatility_20'] = df['Price_Change'].rolling(20).std()
            df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']

            # Support/Resistance levels (simplified)
            df['Resistance'] = df['High'].rolling(20).max()
            df['Support'] = df['Low'].rolling(20).min()
            df['Price_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])

            logger.info(f"Calculated {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']])} technical indicators with TA-Lib")

        except Exception as e:
            logger.error(f"Error calculating TA-Lib indicators: {e}")

        return df

    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic indicators without TA-Lib dependency."""
        df = df.copy()

        try:
            # 1. Simple Moving Averages
            for period in self.indicators_config['sma_periods']:
                df[f'SMA_{period}'] = df['Close'].rolling(period).mean()

            # 2. Exponential Moving Averages
            for period in self.indicators_config['ema_periods']:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

            # 3. Basic RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.indicators_config['rsi_period']).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.indicators_config['rsi_period']).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # 4. MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=self.indicators_config['macd_signal']).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

            # 5. Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(self.indicators_config['bb_period']).mean()
            bb_std = df['Close'].rolling(self.indicators_config['bb_period']).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * self.indicators_config['bb_std'])
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * self.indicators_config['bb_std'])
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

            # 6. Simple ATR (True Range approximation)
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(self.indicators_config['atr_period']).mean()

            # 7. Stochastic Oscillator (simplified)
            low_min = df['Low'].rolling(self.indicators_config['stoch_k']).min()
            high_max = df['High'].rolling(self.indicators_config['stoch_k']).max()
            df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df['Stoch_D'] = df['Stoch_K'].rolling(self.indicators_config['stoch_d']).mean()

            # 8. Simple momentum indicators
            df['Momentum'] = df['Close'] - df['Close'].shift(self.indicators_config['momentum_period'])
            df['ROC'] = df['Close'].pct_change(self.indicators_config['roc_period']) * 100

            # 9. Price-based indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['Volatility_20'] = df['Price_Change'].rolling(20).std()
            df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']

            # 10. Support/Resistance levels
            df['Resistance'] = df['High'].rolling(20).max()
            df['Support'] = df['Low'].rolling(20).min()
            df['Price_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])

            # 11. Volume-based (simplified OBV)
            df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()

            logger.info(f"Calculated {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']])} basic indicators")

        except Exception as e:
            logger.error(f"Error calculating basic indicators: {e}")

        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess and normalize data for ML model.

        Args:
            df: DataFrame with raw data and indicators

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # 1. Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # Forward fill first, then backward fill for any remaining NaN
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')

        # 2. Remove outliers using IQR method
        for col in numeric_columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:  # Don't remove outliers from price data
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)

        # 3. Normalize indicators (keep prices in original scale for interpretability)
        indicators_to_normalize = [col for col in numeric_columns
                                 if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

        for col in indicators_to_normalize:
            if df[col].std() != 0:  # Avoid division by zero
                df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()

        # 4. Add lagged features
        for lag in [1, 2, 3]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
            df[f'MACD_lag_{lag}'] = df['MACD'].shift(lag)

        # 5. Add time-based features
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['IsMarketOpen'] = ((df['Hour'] >= 0) & (df['Hour'] < 24)).astype(int)

        logger.info("Data preprocessing completed")
        return df

    def validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Validate data quality across all currency pairs.

        Args:
            data: Dictionary of DataFrames for each currency pair

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        for pair, df in data.items():
            results = {
                'total_records': len(df),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_records': df.duplicated().sum(),
                'price_data_quality': {},
                'indicators_coverage': {},
                'date_range': {
                    'start': df.index.min(),
                    'end': df.index.max(),
                    'total_days': (df.index.max() - df.index.min()).days
                }
            }

            # Price data quality checks
            results['price_data_quality'] = {
                'ohlc_consistency': (df['High'] >= df[['Open', 'Close']].max(axis=1)).all(),
                'price_gaps': (df['Close'].pct_change().abs() > 0.05).sum(),  # >5% moves
                'zero_volume_candles': (df['Volume'] == 0).sum(),
                'negative_spread': (df['Low'] > df['High']).sum()
            }

            # Indicators coverage
            indicator_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
            for col in indicator_columns:
                coverage = (df[col].notna().sum() / len(df)) * 100
                results['indicators_coverage'][col] = f"{coverage:.1f}%"

            validation_results[pair] = results

            # Log summary
            logger.info(f"{pair} validation: {results['total_records']} records, "
                       f"{results['missing_values']} missing values, "
                       f"{len(indicator_columns)} indicators")

        return validation_results

    def plot_indicators(self, data: Dict[str, pd.DataFrame], pair: str = 'EURUSD', days: int = 30):
        """
        Create comprehensive plots of price data and technical indicators.

        Args:
            data: Dictionary of DataFrames for each currency pair
            pair: Currency pair to plot
            days: Number of recent days to plot
        """
        if pair not in data:
            logger.error(f"Data not available for {pair}")
            return

        df = data[pair].tail(days * 96)  # 96 candles per day (15-min timeframe)

        # Create subplot layout
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle(f'{pair} Technical Analysis - Last {days} Days', fontsize=16)

        # 1. Price and Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['Close'], label='Close', linewidth=2)
        if 'SMA_20' in df.columns:
            ax1.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
        if 'EMA_12' in df.columns:
            ax1.plot(df.index, df['EMA_12'], label='EMA 12', alpha=0.7)
        ax1.set_title('Price & Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. RSI
        ax2 = axes[0, 1]
        if 'RSI' in df.columns:
            ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.set_ylim(0, 100)
        ax2.set_title('RSI (14)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. MACD
        ax3 = axes[1, 0]
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
            ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='red')
            if 'MACD_Hist' in df.columns:
                ax3.bar(df.index, df['MACD_Hist'], label='Histogram', alpha=0.3)
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Bollinger Bands
        ax4 = axes[1, 1]
        ax4.plot(df.index, df['Close'], label='Close', linewidth=2)
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            ax4.plot(df.index, df['BB_Upper'], label='BB Upper', alpha=0.7)
            ax4.plot(df.index, df['BB_Lower'], label='BB Lower', alpha=0.7)
            ax4.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1)
        ax4.set_title('Bollinger Bands')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Stochastic
        ax5 = axes[2, 0]
        if all(col in df.columns for col in ['Stoch_K', 'Stoch_D']):
            ax5.plot(df.index, df['Stoch_K'], label='%K', color='blue')
            ax5.plot(df.index, df['Stoch_D'], label='%D', color='red')
            ax5.axhline(y=80, color='r', linestyle='--', alpha=0.5)
            ax5.axhline(y=20, color='g', linestyle='--', alpha=0.5)
            ax5.set_ylim(0, 100)
        ax5.set_title('Stochastic Oscillator')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. ATR (Volatility)
        ax6 = axes[2, 1]
        if 'ATR' in df.columns:
            ax6.plot(df.index, df['ATR'], label='ATR', color='orange')
        if 'Volatility_20' in df.columns:
            ax6.plot(df.index, df['Volatility_20'], label='Volatility 20', color='red', alpha=0.7)
        ax6.set_title('Volatility Measures')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Volume
        ax7 = axes[3, 0]
        ax7.bar(df.index, df['Volume'], alpha=0.6, color='gray')
        ax7.set_title('Volume')
        ax7.grid(True, alpha=0.3)

        # 8. Price Position & Support/Resistance
        ax8 = axes[3, 1]
        ax8.plot(df.index, df['Close'], label='Close', linewidth=2)
        if 'Support' in df.columns and 'Resistance' in df.columns:
            ax8.plot(df.index, df['Support'], label='Support', color='green', alpha=0.7)
            ax8.plot(df.index, df['Resistance'], label='Resistance', color='red', alpha=0.7)
        ax8.set_title('Support & Resistance')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()
        plt.show()

        logger.info(f"Technical indicators plotted for {pair}")

    def get_multi_pair_data(self, days: int = 365, csv_files: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Get processed data for all currency pairs.

        Args:
            days: Number of days of historical data (for MT5 or sample data)
            csv_files: Optional dictionary mapping pair names to CSV file paths
                      e.g., {'EURUSD': 'data/EURUSD_15M.csv', 'GBPUSD': 'data/GBPUSD_15M.csv'}

        Returns:
            Dictionary with processed data for each currency pair
        """
        all_data = {}

        # Priority 1: Use user-provided CSV files
        if csv_files:
            logger.info("Loading data from user-provided CSV files...")
            csv_data = self.load_csv_data(csv_files)

            for pair, df in csv_data.items():
                df = self.calculate_technical_indicators(df)
                df = self.preprocess_data(df)
                all_data[pair] = df

            if all_data:
                logger.info("✅ Using CSV data for training")
                self._log_data_summary(all_data)
                validation_results = self.validate_data_quality(all_data)
                return all_data

        # Priority 2: Try to get MT5 real data
        if self.initialize_mt5():
            logger.info("Loading data from MetaTrader 5...")
            for pair in self.currency_pairs:
                df = self.get_historical_data(pair, days)
                if not df.empty:
                    df = self.calculate_technical_indicators(df)
                    df = self.preprocess_data(df)
                    all_data[pair] = df
                else:
                    logger.warning(f"No MT5 data for {pair}")

        # Priority 3: Use sample data as fallback
        if not all_data:
            logger.info("Using generated sample data for all pairs")
            sample_data = self.load_sample_data()

            for pair, df in sample_data.items():
                df = self.calculate_technical_indicators(df)
                df = self.preprocess_data(df)
                all_data[pair] = df

        # Validate data quality
        validation_results = self.validate_data_quality(all_data)

        logger.info(f"Multi-pair data ready: {list(all_data.keys())}")
        self._log_data_summary(all_data)
        return all_data

    def _log_data_summary(self, data: Dict[str, pd.DataFrame]):
        """Log summary of loaded data."""
        for pair, df in data.items():
            date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
            total_days = (df.index.max() - df.index.min()).days
            features = len([col for col in df.columns if col != 'Symbol'])
            logger.info(f"📊 {pair}: {len(df):,} candles | {total_days} days | {features} features | {date_range}")

    def cleanup(self):
        """Clean up resources and close MT5 connection."""
        if self.mt5_initialized:
            mt5.shutdown()
            logger.info("MT5 connection closed")


if __name__ == "__main__":
    # Test the DataManager
    print("Testing DataManager...")

    # Initialize
    dm = DataManager()

    # Test sample data generation
    print("\n1. Testing sample data generation...")
    sample_data = dm.load_sample_data()

    for pair, df in sample_data.items():
        print(f"{pair}: {len(df)} candles, {len(df.columns)} columns")
        print(f"  Price range: {df['Close'].min():.5f} - {df['Close'].max():.5f}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Test indicator calculation
    print("\n2. Testing technical indicators...")
    for pair, df in sample_data.items():
        df_with_indicators = dm.calculate_technical_indicators(df)
        indicator_count = len([col for col in df_with_indicators.columns
                             if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']])
        print(f"{pair}: {indicator_count} indicators calculated")

        # Show some indicator values
        if len(df_with_indicators) > 50:
            recent = df_with_indicators.iloc[-1]
            print(f"  Recent RSI: {recent.get('RSI', 'N/A'):.2f}")
            print(f"  Recent MACD: {recent.get('MACD', 'N/A'):.6f}")

    # Test data preprocessing
    print("\n3. Testing data preprocessing...")
    for pair, df in sample_data.items():
        df_with_indicators = dm.calculate_technical_indicators(df)
        df_processed = dm.preprocess_data(df_with_indicators)
        normalized_cols = [col for col in df_processed.columns if '_normalized' in col]
        print(f"{pair}: {len(normalized_cols)} normalized indicators")

    # Test validation
    print("\n4. Testing data validation...")
    processed_data = {}
    for pair, df in sample_data.items():
        df_with_indicators = dm.calculate_technical_indicators(df)
        processed_data[pair] = dm.preprocess_data(df_with_indicators)

    validation_results = dm.validate_data_quality(processed_data)
    for pair, results in validation_results.items():
        print(f"{pair}: {results['total_records']} records, "
              f"{results['missing_values']} missing values")

    print("\n5. DataManager test completed successfully!")
    print("You can now use: dm.plot_indicators(sample_data) to see charts")