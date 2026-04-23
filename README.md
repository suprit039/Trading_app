# Crypto Strategy Forward Testing

A robust Python implementation for forward testing (paper trading) cryptocurrency trading strategies. This system connects to the Binance exchange (or others via CCXT) to fetch real-time market data and execute simulated trades based on momentum and mean-reversion signals.

## Features

- **Real-time Market Data**: Integrates with CCXT for live price fetching and historical OHLCV data.
- **Market Regime Analysis**: Automatically classifies market conditions (e.g., Strong Bull, Mild Bear, Sideways) to adjust strategy aggressiveness.
- **Dual-Signal Strategy**: Combines momentum-based entry/exit with mean-reversion (Bollinger Bands & RSI) logic.
- **Automated Paper Trading**: Simulates order execution with fee and slippage modeling.
- **State Persistence**: Automatically saves trading state (positions, capital, history) to JSON for continuity across restarts.
- **Comprehensive Reporting**: Generates detailed performance reports including Sharpe Ratio, Max Drawdown, and Win Rate.
- **Logging**: Full execution logging to both console and `forward_test.log`.

## Setup

1. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Modify the `CRYPTO_SYMBOLS` and `INITIAL_CAPITAL` constants in the `main()` function of the script to customize your setup.

## Usage

Run the main strategy script with different arguments:

- **Start Forward Testing**: Starts the continuous loop (default 60-min intervals).
  ```bash
  python "General_Strategy_3rdPapertrading (1).py"
  ```

- **Run Single Cycle**: Executes one strategy check immediately and exits.
  ```bash
  python "General_Strategy_3rdPapertrading (1).py" run
  ```

- **Check Status**: Generates and displays a performance report based on saved data.
  ```bash
  python "General_Strategy_3rdPapertrading (1).py" status
  ```

- **Reset Data**: Clears all saved trading state and logs.
  ```bash
  python "General_Strategy_3rdPapertrading (1).py" reset
  ```

## Project Structure

- `General_Strategy_3rdPapertrading (1).py`: Core strategy logic and execution engine.
- `requirements.txt`: Project dependencies.
- `forward_test.log`: Real-time execution logs.
- `forward_test_data/`: Directory containing saved state, trade history, and performance reports.

## Disclaimer

This software is for **paper trading and educational purposes only**. Do not use it for live trading without thorough validation and risk assessment.
