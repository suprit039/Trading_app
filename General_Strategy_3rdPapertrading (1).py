import pandas as pd
import numpy as np
import ccxt
import requests
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import logging
import json
import os
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
import schedule

warnings.filterwarnings('ignore')

# Set up enhanced logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forward_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    shares: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_price(self, new_price: float):
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.entry_price) * self.shares

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    action: str  # BUY/SELL
    shares: float
    price: float
    value: float
    timestamp: datetime
    regime: str
    fees: float = 0.0

class ForwardTestingStrategy:
    """
    Forward Testing (Paper Trading) Implementation of the Crypto Strategy
    """
    
    def __init__(self, initial_capital: float = 100000):
        # Initialize strategy parameters (same as backtest)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        
        # Strategy parameters (same as backtest)
        self.fee_rate = 0.001
        self.slippage = 0.0005
        self.lookback_period = 20
        self.volatility_threshold = 0.01
        self.momentum_threshold = 0.02
        self.max_position_per_asset = 0.30
        self.max_total_exposure = 0.90
        self.stop_loss_pct = 0.10
        self.portfolio_heat = 0.20
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        
        # Data storage
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.last_signals: Dict[str, Dict] = {}
        self.current_regime = 'SIDEWAYS'
        
        # Exchange connection
        try:
            self.exchange = ccxt.binance({
                'rateLimit': 1200,
                'enableRateLimit': True,
                'sandbox': False,  # Set to True for testing
            })
            logger.info("Exchange connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
        
        # Create data directories
        Path('forward_test_data').mkdir(exist_ok=True)
        
        # Load previous state if exists
        self.load_state()
        
        logger.info(f"Forward testing initialized with ${initial_capital:,.2f}")
    
    def save_state(self):
        """Save current trading state to file"""
        try:
            state = {
                'current_capital': self.current_capital,
                'cash': self.cash,
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'completed_trades': [asdict(trade) for trade in self.completed_trades],
                'portfolio_history': self.portfolio_history,
                'last_update': datetime.now().isoformat()
            }
            
            with open('forward_test_data/trading_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.debug("Trading state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self):
        """Load previous trading state if exists"""
        try:
            state_file = Path('forward_test_data/trading_state.json')
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.current_capital = state.get('current_capital', self.initial_capital)
                self.cash = state.get('cash', self.initial_capital)
                self.portfolio_history = state.get('portfolio_history', [])
                
                # Restore positions
                for symbol, pos_data in state.get('positions', {}).items():
                    self.positions[symbol] = Position(
                        symbol=pos_data['symbol'],
                        shares=pos_data['shares'],
                        entry_price=pos_data['entry_price'],
                        entry_time=datetime.fromisoformat(pos_data['entry_time']),
                        current_price=pos_data.get('current_price', 0.0)
                    )
                
                # Restore completed trades
                for trade_data in state.get('completed_trades', []):
                    self.completed_trades.append(Trade(
                        symbol=trade_data['symbol'],
                        action=trade_data['action'],
                        shares=trade_data['shares'],
                        price=trade_data['price'],
                        value=trade_data['value'],
                        timestamp=datetime.fromisoformat(trade_data['timestamp']),
                        regime=trade_data['regime'],
                        fees=trade_data.get('fees', 0.0)
                    ))
                
                logger.info(f"Loaded previous state: ${self.current_capital:,.2f} capital, "
                           f"{len(self.positions)} positions, {len(self.completed_trades)} completed trades")
            
        except Exception as e:
            logger.warning(f"Could not load previous state: {e}")
    
    def fetch_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current market prices"""
        prices = {}
        
        for symbol in symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                prices[symbol] = ticker['last']
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to fetch price for {symbol}: {e}")
                # Use last known price if available
                if symbol in self.price_history and not self.price_history[symbol].empty:
                    prices[symbol] = self.price_history[symbol]['close'].iloc[-1]
        
        return prices
    
    def fetch_historical_data(self, symbol: str, timeframe: str = '1d', limit: int = 500) -> pd.DataFrame:
        """Fetch recent historical data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Store for later use
            self.price_history[symbol] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_market_regime(self, btc_data: pd.DataFrame) -> str:
        """Calculate current market regime (same logic as backtest)"""
        try:
            if len(btc_data) < 50:
                return 'SIDEWAYS'
            
            data = btc_data.copy()
            
            # Calculate moving averages
            data['sma_20'] = data['close'].rolling(20, min_periods=10).mean()
            data['sma_50'] = data['close'].rolling(50, min_periods=25).mean()
            data['sma_200'] = data['close'].rolling(200, min_periods=100).mean()
            
            # Get latest values
            latest = data.iloc[-1]
            
            # Price position relative to trend
            price_vs_sma50 = (latest['close'] - latest['sma_50']) / latest['sma_50']
            sma50_vs_sma200 = latest['sma_50'] > latest['sma_200']
            
            # Regime classification (same thresholds as backtest)
            if price_vs_sma50 > 0.02 and sma50_vs_sma200:
                return 'STRONG_BULL'
            elif price_vs_sma50 > -0.02 and sma50_vs_sma200:
                return 'MILD_BULL'
            elif price_vs_sma50 < -0.02 and not sma50_vs_sma200:
                return 'STRONG_BEAR'
            elif price_vs_sma50 < 0.02 and not sma50_vs_sma200:
                return 'MILD_BEAR'
            else:
                return 'SIDEWAYS'
            
        except Exception as e:
            logger.error(f"Error calculating regime: {e}")
            return 'SIDEWAYS'
    
    def calculate_signals(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Calculate trading signals for a symbol (same logic as backtest)"""
        try:
            if len(data) < 50:
                return {}
            
            df = data.copy()
            
            # Momentum signals
            df['momentum_5d'] = df['close'].pct_change(5)
            df['momentum_10d'] = df['close'].pct_change(10)
            df['momentum_20d'] = df['close'].pct_change(20)
            
            # Momentum consistency
            df['momentum_consistency'] = (
                (df['momentum_5d'] > -0.02).astype(int) +
                (df['momentum_10d'] > -0.02).astype(int) +
                (df['momentum_20d'] > -0.02).astype(int)
            ) / 3
            
            # Volatility-adjusted momentum
            volatility = df['close'].pct_change().rolling(20, min_periods=10).std() + 0.001
            df['vol_adj_momentum'] = df['momentum_10d'] / volatility
            df['volatility'] = volatility - 0.001
            
            # Mean reversion signals
            df['bb_middle'] = df['close'].rolling(20, min_periods=10).mean()
            df['bb_std'] = df['close'].rolling(20, min_periods=10).std()
            df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
            df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
            
            bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
            df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=7).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Get latest signals
            latest = df.iloc[-1]
            
            signals = {
                'momentum_10d': latest.get('momentum_10d', 0),
                'momentum_consistency': latest.get('momentum_consistency', 0.5),
                'vol_adj_momentum': latest.get('vol_adj_momentum', 0),
                'volatility': latest.get('volatility', 0.02),
                'bb_position': latest.get('bb_position', 0.5),
                'rsi': latest.get('rsi', 50),
                'current_price': latest['close']
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating signals for {symbol}: {e}")
            return {}
    
    def calculate_position_size(self, symbol: str, signals: Dict, regime: str) -> float:
        """Calculate desired position size (same logic as backtest)"""
        try:
            # Regime-based exposure multiplier
            regime_multipliers = {
                'STRONG_BULL': 1.0,
                'MILD_BULL': 0.9,
                'SIDEWAYS': 0.7,
                'MILD_BEAR': 0.5,
                'STRONG_BEAR': 0.3
            }
            
            max_exposure = self.max_total_exposure * regime_multipliers.get(regime, 0.7)
            
            momentum_score = 0
            mean_reversion_score = 0
            
            # Calculate signals based on regime
            if regime in ['STRONG_BULL', 'MILD_BULL']:
                momentum_consistency = signals.get('momentum_consistency', 0)
                vol_adj_momentum = signals.get('vol_adj_momentum', 0)
                
                if momentum_consistency > 0.5 and vol_adj_momentum > 0.2:
                    momentum_score = momentum_consistency * abs(vol_adj_momentum) * 0.5
            
            elif regime in ['SIDEWAYS', 'MILD_BEAR']:
                bb_position = signals.get('bb_position', 0.5)
                rsi = signals.get('rsi', 50)
                
                # Oversold conditions
                if bb_position < 0.3 and rsi < 40:
                    mean_reversion_score = (0.3 - bb_position) * (40 - rsi) / 40 * 0.5
                # Overbought conditions
                elif bb_position > 0.7 and rsi > 60:
                    mean_reversion_score = -(bb_position - 0.7) * (rsi - 60) / 40 * 0.5
            
            # Baseline momentum
            baseline_momentum = signals.get('momentum_10d', 0)
            if abs(baseline_momentum) > 0.01:
                momentum_score += baseline_momentum * 0.2
            
            # Combine signals
            total_signal = momentum_score + mean_reversion_score
            
            if abs(total_signal) > 0.05:
                # Adjust for volatility
                volatility = signals.get('volatility', 0.02)
                vol_adjustment = min(0.03 / max(volatility, 0.005), 3.0)
                adjusted_score = total_signal * vol_adjustment
                
                # Calculate position size
                position_size = min(abs(adjusted_score) * max_exposure, self.max_position_per_asset)
                return position_size * np.sign(adjusted_score)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def update_positions(self, current_prices: Dict[str, float]):
        """Update position values with current prices"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_price(current_prices[symbol])
    
    def execute_trade(self, symbol: str, target_position: float, current_price: float):
        """Execute a paper trade"""
        try:
            current_position_value = 0
            if symbol in self.positions:
                current_position_value = self.positions[symbol].shares * current_price
            
            # Calculate target position value
            portfolio_value = self.get_portfolio_value()
            target_position_value = portfolio_value * target_position
            
            # Calculate required trade
            trade_value = target_position_value - current_position_value
            
            if abs(trade_value) > portfolio_value * 0.01:  # Minimum trade threshold
                trade_shares = trade_value / current_price
                trade_cost = abs(trade_value) * self.fee_rate
                
                # Check cash availability
                cash_needed = max(0, trade_value) + trade_cost
                
                if self.cash >= cash_needed:
                    # Execute the trade
                    if symbol in self.positions:
                        # Update existing position
                        old_shares = self.positions[symbol].shares
                        new_shares = old_shares + trade_shares
                        
                        if abs(new_shares) < 1e-6:  # Close position
                            del self.positions[symbol]
                        else:
                            # Update position
                            self.positions[symbol].shares = new_shares
                            if old_shares * new_shares < 0:  # Sign change - new entry
                                self.positions[symbol].entry_price = current_price
                                self.positions[symbol].entry_time = datetime.now()
                    else:
                        # Create new position
                        if abs(trade_shares) > 1e-6:
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                shares=trade_shares,
                                entry_price=current_price,
                                entry_time=datetime.now(),
                                current_price=current_price
                            )
                    
                    # Update cash
                    self.cash -= trade_value + trade_cost
                    
                    # Log the trade
                    trade = Trade(
                        symbol=symbol,
                        action='BUY' if trade_shares > 0 else 'SELL',
                        shares=abs(trade_shares),
                        price=current_price,
                        value=abs(trade_value),
                        timestamp=datetime.now(),
                        regime=self.current_regime,
                        fees=trade_cost
                    )
                    
                    self.completed_trades.append(trade)
                    
                    logger.info(f"TRADE EXECUTED: {trade.action} {trade.shares:.4f} {symbol} @ ${trade.price:.2f} "
                               f"(Value: ${trade.value:,.2f}, Fees: ${trade.fees:.2f})")
                    
                else:
                    logger.warning(f"Insufficient cash for {symbol} trade. Need ${cash_needed:,.2f}, have ${self.cash:,.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.cash
        
        for position in self.positions.values():
            portfolio_value += position.shares * position.current_price
        
        return portfolio_value
    
    def log_portfolio_status(self):
        """Log detailed portfolio status"""
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value / self.initial_capital - 1) * 100
        
        logger.info("=" * 60)
        logger.info("PORTFOLIO STATUS")
        logger.info("=" * 60)
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Current Value:  ${portfolio_value:,.2f}")
        logger.info(f"Cash:          ${self.cash:,.2f}")
        logger.info(f"Total Return:   {total_return:+.2f}%")
        logger.info(f"Active Positions: {len(self.positions)}")
        logger.info(f"Completed Trades: {len(self.completed_trades)}")
        logger.info(f"Current Regime: {self.current_regime}")
        
        if self.positions:
            logger.info("\nACTIVE POSITIONS:")
            logger.info("-" * 60)
            total_position_value = 0
            
            for symbol, position in self.positions.items():
                position_value = position.shares * position.current_price
                total_position_value += abs(position_value)
                pnl_pct = (position.current_price / position.entry_price - 1) * 100
                
                logger.info(f"{symbol:<12} | {position.shares:>10.4f} | ${position.current_price:>8.2f} | "
                           f"${position_value:>10,.2f} | {pnl_pct:>+6.2f}% | {position.entry_time.strftime('%Y-%m-%d')}")
            
            logger.info(f"Total Position Value: ${total_position_value:,.2f}")
            logger.info(f"Position Exposure: {total_position_value/portfolio_value*100:.1f}%")
        
        # Recent trades
        if self.completed_trades:
            recent_trades = self.completed_trades[-5:]  # Last 5 trades
            logger.info(f"\nRECENT TRADES (Last {len(recent_trades)}):")
            logger.info("-" * 60)
            for trade in recent_trades:
                logger.info(f"{trade.timestamp.strftime('%Y-%m-%d %H:%M')} | {trade.action:<4} | "
                           f"{trade.symbol:<12} | {trade.shares:>8.4f} | ${trade.price:>8.2f} | "
                           f"${trade.value:>8,.2f}")
        
        logger.info("=" * 60)
    
    def run_strategy_cycle(self, symbols: List[str]):
        """Run one complete strategy cycle"""
        try:
            logger.info(f"Starting strategy cycle at {datetime.now()}")
            
            # Fetch current prices
            current_prices = self.fetch_current_prices(symbols)
            if not current_prices:
                logger.error("Could not fetch current prices")
                return
            
            # Update existing positions
            self.update_positions(current_prices)
            
            # Fetch historical data for analysis
            btc_data = self.fetch_historical_data('BTC/USDT')
            if btc_data.empty:
                logger.error("Could not fetch BTC data for regime analysis")
                return
            
            # Calculate market regime
            self.current_regime = self.calculate_market_regime(btc_data)
            logger.info(f"Current market regime: {self.current_regime}")
            
            # Calculate signals and positions for each symbol
            for symbol in symbols:
                try:
                    # Fetch historical data
                    hist_data = self.fetch_historical_data(symbol)
                    if hist_data.empty:
                        continue
                    
                    # Calculate signals
                    signals = self.calculate_signals(symbol, hist_data)
                    if not signals:
                        continue
                    
                    self.last_signals[symbol] = signals
                    
                    # Calculate target position
                    target_position = self.calculate_position_size(symbol, signals, self.current_regime)
                    
                    # Execute trade if needed
                    if abs(target_position) > 0.01 or symbol in self.positions:
                        self.execute_trade(symbol, target_position, current_prices[symbol])
                    
                    # Small delay between symbols
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Update portfolio history
            portfolio_value = self.get_portfolio_value()
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'num_positions': len(self.positions),
                'regime': self.current_regime
            })
            
            # Save state
            self.save_state()
            
            # Log status
            self.log_portfolio_status()
            
            logger.info(f"Strategy cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in strategy cycle: {e}")
    
    def start_forward_testing(self, symbols: List[str], run_interval_minutes: int = 60):
        """Start the forward testing loop"""
        logger.info(f"Starting forward testing with {len(symbols)} symbols")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Update interval: {run_interval_minutes} minutes")
        
        # Schedule the strategy to run
        schedule.every(run_interval_minutes).minutes.do(self.run_strategy_cycle, symbols)
        
        # Run initial cycle
        self.run_strategy_cycle(symbols)
        
        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Forward testing stopped by user")
            self.save_state()
        except Exception as e:
            logger.error(f"Forward testing error: {e}")
            self.save_state()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            if not self.portfolio_history:
                logger.warning("No portfolio history available for report")
                return
            
            df = pd.DataFrame(self.portfolio_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate metrics
            initial_value = df['portfolio_value'].iloc[0] if len(df) > 0 else self.initial_capital
            current_value = df['portfolio_value'].iloc[-1] if len(df) > 0 else self.get_portfolio_value()
            
            total_return = (current_value / initial_value - 1) * 100
            
            # Calculate daily returns if we have enough data
            if len(df) > 1:
                df['returns'] = df['portfolio_value'].pct_change()
                returns = df['returns'].dropna()
                
                if len(returns) > 0:
                    mean_return = returns.mean()
                    std_return = returns.std()
                    
                    # Annualized metrics (approximate)
                    periods_per_year = 365 * 24 * 60 / 60  # Assuming hourly updates
                    if std_return > 0:
                        sharpe_ratio = np.sqrt(periods_per_year) * mean_return / std_return
                    else:
                        sharpe_ratio = 0
                    
                    # Drawdown
                    rolling_max = df['portfolio_value'].expanding().max()
                    drawdown = (df['portfolio_value'] - rolling_max) / rolling_max
                    max_drawdown = drawdown.min() * 100
                    
                    # Win rate
                    win_rate = (returns > 0).mean() * 100
                else:
                    sharpe_ratio = 0
                    max_drawdown = 0
                    win_rate = 0
            else:
                sharpe_ratio = 0
                max_drawdown = 0
                win_rate = 0
            
            # Trade statistics
            total_trades = len(self.completed_trades)
            total_fees = sum(trade.fees for trade in self.completed_trades)
            
            # Generate report
            report = f"""
FORWARD TESTING PERFORMANCE REPORT
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO SUMMARY:
- Initial Capital: ${initial_value:,.2f}
- Current Value:   ${current_value:,.2f}
- Cash Balance:    ${self.cash:,.2f}
- Total Return:    {total_return:+.2f}%

RISK METRICS:
- Max Drawdown:    {max_drawdown:.2f}%
- Sharpe Ratio:    {sharpe_ratio:.2f}
- Win Rate:        {win_rate:.1f}%

TRADING ACTIVITY:
- Total Trades:    {total_trades}
- Total Fees:      ${total_fees:,.2f}
- Active Positions: {len(self.positions)}

CURRENT POSITIONS:
"""
            
            if self.positions:
                for symbol, position in self.positions.items():
                    pnl = (position.current_price / position.entry_price - 1) * 100
                    position_value = position.shares * position.current_price
                    report += f"- {symbol}: {position.shares:.4f} shares @ ${position.current_price:.2f} ({pnl:+.2f}%, ${position_value:,.2f})\n"
            else:
                report += "- No active positions\n"
            
            print(report)
            
            # Save report to file
            with open('forward_test_data/performance_report.txt', 'w') as f:
                f.write(report)
            
            # Save portfolio history
            if not df.empty:
                df.to_csv('forward_test_data/portfolio_history.csv')
            
            # Save trades
            if self.completed_trades:
                trades_df = pd.DataFrame([asdict(trade) for trade in self.completed_trades])
                trades_df.to_csv('forward_test_data/completed_trades.csv', index=False)
            
            logger.info("Performance report generated and saved")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")


def main():
    """Main function to run forward testing"""
    
    # Same crypto assets as in the backtest
    CRYPTO_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    
    # Configuration
    INITIAL_CAPITAL = 100000  # $100,000 paper money
    UPDATE_INTERVAL = 60      # Run strategy every 60 minutes
    
    try:
        # Initialize strategy
        logger.info("Initializing Forward Testing Strategy")
        strategy = ForwardTestingStrategy(initial_capital=INITIAL_CAPITAL)
        
        # Generate initial report
        strategy.generate_performance_report()
        
        print("\n" + "="*60)
        print("CRYPTO STRATEGY FORWARD TESTING")
        print("="*60)
        print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"Assets: {CRYPTO_SYMBOLS}")
        print(f"Update Interval: {UPDATE_INTERVAL} minutes")
        print("="*60)
        print("\nStarting forward testing... Press Ctrl+C to stop")
        print("Logs are saved to 'forward_test.log'")
        print("Performance data saved to 'forward_test_data/' directory")
        print("="*60)
        
        # Start forward testing
        strategy.start_forward_testing(CRYPTO_SYMBOLS, UPDATE_INTERVAL)
        
    except KeyboardInterrupt:
        print("\n\nForward testing stopped by user")
        logger.info("Forward testing interrupted by user")
        
        # Generate final report
        try:
            strategy = ForwardTestingStrategy(initial_capital=INITIAL_CAPITAL)
            strategy.generate_performance_report()
        except:
            pass
            
    except Exception as e:
        logger.error(f"Critical error in forward testing: {e}")
        print(f"\nError: {e}")


# Additional utility functions
def quick_status_check():
    """Quick function to check current status without starting full forward test"""
    try:
        strategy = ForwardTestingStrategy(initial_capital=100000)
        strategy.generate_performance_report()
    except Exception as e:
        print(f"Error checking status: {e}")


def manual_strategy_run():
    """Manually run one strategy cycle for testing"""
    CRYPTO_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    
    try:
        print("Running manual strategy cycle...")
        strategy = ForwardTestingStrategy(initial_capital=100000)
        strategy.run_strategy_cycle(CRYPTO_SYMBOLS)
        strategy.generate_performance_report()
        print("Manual run completed!")
        
    except Exception as e:
        print(f"Error in manual run: {e}")


def reset_forward_test():
    """Reset forward testing state (use with caution!)"""
    import shutil
    
    confirm = input("Are you sure you want to reset all forward testing data? (yes/no): ")
    if confirm.lower() == 'yes':
        try:
            if os.path.exists('forward_test_data'):
                shutil.rmtree('forward_test_data')
            if os.path.exists('forward_test.log'):
                os.remove('forward_test.log')
            print("Forward testing data reset successfully!")
        except Exception as e:
            print(f"Error resetting data: {e}")
    else:
        print("Reset cancelled.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'status':
            quick_status_check()
        elif sys.argv[1] == 'run':
            manual_strategy_run()
        elif sys.argv[1] == 'reset':
            reset_forward_test()
        else:
            print("Usage: python forward_test.py [status|run|reset]")
            print("  status: Show current performance status")
            print("  run:    Run one strategy cycle manually")  
            print("  reset:  Reset all forward testing data")
    else:
        main()