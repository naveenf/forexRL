#!/usr/bin/env python3
"""
Forex Trading System - Main Entry Point

AI-powered forex trading system targeting $100 profit in 4 candles (1 hour)
using a 2B LLM trained with PPO reinforcement learning.

Usage:
    python main.py              # Start desktop application
    python main.py --mode train # Training mode (use Google Colab instead)
    python main.py --mode test  # Test mode for development
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

# Project root directory
PROJECT_ROOT = Path(__file__).parent


def setup_logging(config_path: Path) -> None:
    """Configure logging based on config file."""
    try:
        with open(config_path / "config" / "inference.yaml", 'r') as f:
            config = yaml.safe_load(f)

        log_level = config.get('logging', {}).get('level', 'INFO')
        log_file = config.get('logging', {}).get('file', 'logs/forex_trading.log')

        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)

        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.add(
            log_file,
            level=log_level,
            rotation="100 MB",
            retention="10 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )

    except Exception as e:
        # Fallback logging setup
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"Could not load logging config: {e}. Using fallback.")


def validate_config_files() -> bool:
    """Validate that all required config files exist and are readable."""
    required_configs = [
        "config/environment.yaml",
        "config/training.yaml",
        "config/inference.yaml"
    ]

    for config_file in required_configs:
        config_path = PROJECT_ROOT / config_file
        if not config_path.exists():
            logger.error(f"Missing config file: {config_path}")
            return False

        try:
            with open(config_path, 'r') as f:
                yaml.safe_load(f)
            logger.info(f"✅ Config file validated: {config_file}")
        except Exception as e:
            logger.error(f"Invalid YAML in {config_file}: {e}")
            return False

    return True


def run_desktop_app() -> int:
    """Launch the desktop trading application."""
    try:
        logger.info("🚀 Starting Forex RL Trading System - Desktop Mode")

        # Check if UI is implemented
        ui_path = PROJECT_ROOT / "src" / "ui" / "main_window.py"
        if not ui_path.exists():
            logger.warning("⚠️  Desktop UI not implemented yet (Phase 5)")
            logger.info("Currently available:")
            logger.info("  - Phase 1: ✅ DataManager (completed)")
            logger.info("  - Phase 2: ⏳ RL Environment (next)")
            logger.info("  - Phase 3: ⏳ Training")
            logger.info("  - Phase 4: ⏳ Inference Engine")
            logger.info("  - Phase 5: ⏳ Desktop UI")
            logger.info("")
            logger.info("For now, you can test the DataManager:")
            logger.info("  python -c \"from src.data_manager import DataManager; dm = DataManager(); data = dm.load_sample_data(); print(f'✅ {len(data)} pairs loaded')\"")
            return 0

        # Import here to avoid issues if PySide6 not installed
        from src.ui.main_window import ForexTradingApp
        from PySide6.QtWidgets import QApplication

        app = QApplication(sys.argv)
        window = ForexTradingApp()
        window.show()

        logger.info("Desktop application launched successfully")
        return app.exec()

    except ImportError as e:
        logger.error(f"Missing dependencies for desktop app: {e}")
        logger.info("Please install requirements: pip install -r requirements.txt")
        return 1
    except Exception as e:
        logger.error(f"Failed to start desktop application: {e}")
        return 1


def run_training_mode() -> int:
    """Training mode - should redirect to Google Colab."""
    logger.warning("🔄 Training mode detected")
    logger.info("For training, please use Google Colab with training/forex_training.ipynb")
    logger.info("Local training is not recommended due to computational requirements")
    logger.info("See TRAINING_GUIDE_COMPREHENSIVE.md for detailed instructions")
    return 0


def run_test_mode() -> int:
    """Test mode for development and validation."""
    logger.info("🧪 Test mode - Running system validation")

    try:
        # Test basic imports
        logger.info("Testing basic imports...")
        import pandas
        import numpy
        logger.info("✅ Basic dependencies imported successfully")

        # Test DataManager
        logger.info("Testing DataManager...")
        from src.data_manager import DataManager
        dm = DataManager()
        data = dm.load_sample_data()
        logger.info(f"✅ DataManager working: {len(data)} pairs loaded")

        # Test optional RL imports
        try:
            import stable_baselines3
            import gymnasium
            logger.info("✅ RL dependencies imported successfully")
        except ImportError:
            logger.warning("⚠️  RL dependencies not installed (optional for now)")

        # Test YAML loading
        logger.info("Testing configuration loading...")
        if not validate_config_files():
            return 1

        logger.info("✅ All tests passed")
        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1


def main() -> int:
    """Main entry point for the Forex Trading System."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI-Powered Forex Trading System",
        epilog="For more information, see CLAUDE.md and documentation files"
    )
    parser.add_argument(
        "--mode",
        choices=["desktop", "train", "test"],
        default="desktop",
        help="Application mode (default: desktop)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "config",
        help="Configuration directory path"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(PROJECT_ROOT)
    logger.info(f"Forex RL Trading System starting in {args.mode} mode")

    # Validate configuration
    if not validate_config_files():
        logger.error("Configuration validation failed")
        return 1

    # Route to appropriate mode
    if args.mode == "desktop":
        return run_desktop_app()
    elif args.mode == "train":
        return run_training_mode()
    elif args.mode == "test":
        return run_test_mode()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())