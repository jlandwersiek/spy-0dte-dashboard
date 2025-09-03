import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pytz
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Rate Limiter and Safe Functions
class YahooRateLimiter:
    """Rate limiter for Yahoo Finance to prevent API abuse"""
    
    def __init__(self):
        self.last_request = {}
        self.min_interval = 1.5  # Minimum 1.5 seconds between requests
        self.request_count = 0
        self.session_start = time.time()
        self.max_requests_per_hour = 100
    
    def can_make_request(self, symbol: str) -> bool:
        """Check if we can make a request for this symbol"""
        current_time = time.time()
        
        # Check session limits
        if self.request_count >= self.max_requests_per_hour:
            if (current_time - self.session_start) < 3600:  # 1 hour
                return False
            else:
                # Reset counters after an hour
                self.request_count = 0
                self.session_start = current_time
        
        # Check per-symbol rate limit
        if symbol in self.last_request:
            time_since_last = current_time - self.last_request[symbol]
            if time_since_last < self.min_interval:
                return False
        
        return True
    
    def wait_if_needed(self, symbol: str):
        """Wait if needed to respect rate limits"""
        if not self.can_make_request(symbol):
            wait_time = self.min_interval
            time.sleep(wait_time)
    
    def record_request(self, symbol: str):
        """Record that we made a request"""
        self.last_request[symbol] = time.time()
        self.request_count += 1

# Initialize global rate limiter
yahoo_limiter = YahooRateLimiter()

# Safe Yahoo Finance download function
def safe_yahoo_download(symbol: str, **kwargs):
    """Safe Yahoo Finance download with rate limiting"""
    try:
        yahoo_limiter.wait_if_needed(symbol)
        yahoo_limiter.record_request(symbol)
        
        # Add progress=False to reduce noise
        kwargs['progress'] = False
        return yf.download(symbol, **kwargs)
        
    except Exception as e:
        if "Rate limit" in str(e) or "Too Many Requests" in str(e):
            # Return empty dataframe silently
            return pd.DataFrame()
        else:
            raise e

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_cached_yahoo_data(symbol: str, period: str, interval: str):
    """Cached Yahoo Finance data to reduce API calls"""
    return safe_yahoo_download(symbol, period=period, interval=interval)

# Page configuration
st.set_page_config(
    page_title="SPY 0DTE Gap Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradierAPI:
    """Enhanced Tradier API client - PRIMARY data source"""
    
    def __init__(self, token: str, sandbox: bool = False):
        self.token = token
        self.base_url = "https://sandbox.tradier.com/v1" if sandbox else "https://api.tradier.com/v1"
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json'
        }
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test API connection and return status"""
        try:
            response = requests.get(
                f"{self.base_url}/user/profile",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "✅ Connected to Tradier API"
            elif response.status_code == 401:
                return False, "❌ Invalid API token"
            else:
                return False, f"❌ API Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "❌ Connection timeout"
        except Exception as e:
            return False, f"❌ Connection error: {str(e)}"
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for single symbol"""
        try:
            response = requests.get(
                f"{self.base_url}/markets/quotes",
                params={'symbols': symbol},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                quotes = data.get('quotes', {})
                if 'quote' in quotes:
                    return quotes['quote']
                return {}
            else:
                return None
                
        except Exception as e:
            return None
    
    def get_quotes_bulk(self, symbols: str) -> Optional[Dict]:
        """Get real-time quotes for multiple symbols (comma-separated)"""
        try:
            response = requests.get(
                f"{self.base_url}/markets/quotes",
                params={'symbols': symbols},
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                quotes = data.get('quotes', {})
                if 'quote' in quotes:
                    quote_data = quotes['quote']
                    # Handle both single quote and list of quotes
                    if isinstance(quote_data, list):
                        return {q['symbol']: q for q in quote_data}
                    else:
                        return {quote_data['symbol']: quote_data}
                return {}
            else:
                return None
                
        except Exception as e:
            return None
    
    def get_historical_quotes(self, symbol: str, interval: str = '1min', start: str = None, end: str = None) -> Optional[Dict]:
        """Get historical data from Tradier"""
        try:
            params = {
                'symbol': symbol,
                'interval': interval
            }
            if start:
                params['start'] = start
            if end:
                params['end'] = end
            
            response = requests.get(
                f"{self.base_url}/markets/history",
                params=params,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            return None
    
    def get_options_chain(self, symbol: str, expiration: str) -> Optional[Dict]:
        """Get options chain for symbol and expiration"""
        try:
            response = requests.get(
                f"{self.base_url}/markets/options/chains",
                params={
                    'symbol': symbol,
                    'expiration': expiration,
                    'greeks': 'true'
                },
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            return None

class ProxyDataProvider:
    """Provides reasonable proxy estimates when all APIs fail"""
    
    @staticmethod
    def get_spy_proxy_data(current_time: datetime) -> Dict:
        """Generate reasonable SPY proxy data"""
        # Base on typical market conditions
        hour = current_time.hour
        
        # Market closed - use reasonable proxy
        if hour < 9 or hour >= 16:
            return {
                'current_price': 640.27,
                'volume': 45000000,
                'gap_pct': -1.17,
                'source': 'Proxy (Market Closed)'
            }
        
        # Market hours - more dynamic proxy
        return {
            'current_price': 640.27 + (hour - 12) * 0.1,  # Slight variation by hour
            'volume': 50000000,
            'gap_pct': -1.17,
            'source': 'Proxy (Live Estimate)'
        }
    
    @staticmethod
    def get_sector_proxy_data() -> Dict:
        """Generate sector proxy data with realistic variations"""
        return {
            'XLK': {'change_pct': -1.01, 'volume_ratio': 1.5},
            'XLF': {'change_pct': -0.75, 'volume_ratio': 1.1}, 
            'XLV': {'change_pct': 0.10, 'volume_ratio': 1.0},
            'XLY': {'change_pct': -0.75, 'volume_ratio': 1.4},
            'XLI': {'change_pct': -0.96, 'volume_ratio': 0.9},
            'XLP': {'change_pct': -0.17, 'volume_ratio': 1.1},
            'XLE': {'change_pct': 0.15, 'volume_ratio': 0.7},
            'XLU': {'change_pct': -0.34, 'volume_ratio': 0.9},
            'XLRE': {'change_pct': -1.71, 'volume_ratio': 1.3}
        }
    
    @staticmethod
    def get_internals_proxy_data() -> Dict:
        """Generate market internals proxy based on sector sentiment"""
        # Calculate based on sector data
        sector_data = ProxyDataProvider.get_sector_proxy_data()
        positive_sectors = sum(1 for data in sector_data.values() if data['change_pct'] > 0)
        total_sectors = len(sector_data)
        
        # Generate proxy internals
        breadth_ratio = positive_sectors / total_sectors
        
        return {
            'tick': {'value': ((breadth_ratio - 0.5) * 2000), 'signal': 'BEARISH' if breadth_ratio < 0.4 else 'NEUTRAL'},
            'trin': {'value': 1.0 / (0.8 + breadth_ratio * 0.4), 'signal': 'NEUTRAL'},
            'nyad': {'value': ((breadth_ratio - 0.5) * 4000), 'signal': 'BEARISH' if breadth_ratio < 0.4 else 'NEUTRAL'},
            'vold': {'value': 0.3 + breadth_ratio * 0.4, 'signal': 'BEARISH' if breadth_ratio < 0.4 else 'NEUTRAL'}
        }

class ExitSignalManager:
    """Dedicated exit signal management for 0DTE trades"""
    
    def __init__(self, api):
        self.api = api
        self.miami_tz = pytz.timezone('US/Eastern')
    
    def get_exit_signals(self, entry_decision: str, entry_price: float, current_price: float, 
                        entry_time: datetime, targets: Dict) -> Dict:
        """Generate comprehensive exit signals"""
        
        current_time = datetime.now(self.miami_tz)
        time_in_trade = (current_time - entry_time).total_seconds() / 60  # minutes
        
        exit_signals = {
            'primary_signal': 'HOLD',
            'urgency': 'LOW',
            'reasons': [],
            'profit_loss_pct': 0,
            'time_warnings': [],
            'technical_exits': [],
            'should_exit': False,
            'exit_score': 0  # -10 to +10, +10 = IMMEDIATE EXIT
        }
        
        # Calculate P&L
        if 'LONG' in entry_decision:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # SHORT
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        exit_signals['profit_loss_pct'] = pnl_pct
        
        # TIME-BASED EXITS (Critical for 0DTE)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        minutes_to_close = (market_close - current_time).total_seconds() / 60
        
        if minutes_to_close <= 30:
            exit_signals['exit_score'] += 8
            exit_signals['primary_signal'] = 'IMMEDIATE EXIT'
            exit_signals['urgency'] = 'CRITICAL'
            exit_signals['time_warnings'].append("🚨 FINAL 30 MINUTES - Theta burn accelerating")
            
        elif minutes_to_close <= 60:
            exit_signals['exit_score'] += 5
            exit_signals['primary_signal'] = 'STRONG EXIT'
            exit_signals['urgency'] = 'HIGH'
            exit_signals['time_warnings'].append("⚠️ Final hour - Start looking for exit")
            
        elif minutes_to_close <= 90:
            exit_signals['exit_score'] += 3
            exit_signals['time_warnings'].append("🕐 90 minutes left - Prepare for exit")
        
        # PROFIT TARGET EXITS
        if pnl_pct >= 100:  # 100% gain
            exit_signals['exit_score'] += 6
            exit_signals['reasons'].append(f"🎯 MASSIVE WIN: {pnl_pct:.1f}% profit - Take it!")
            
        elif pnl_pct >= 50:  # 50% gain
            exit_signals['exit_score'] += 4
            exit_signals['reasons'].append(f"✅ STRONG PROFIT: {pnl_pct:.1f}% - Consider taking profits")
            
        elif pnl_pct >= 25:  # 25% gain
            exit_signals['exit_score'] += 2
            exit_signals['reasons'].append(f"💰 Good profit: {pnl_pct:.1f}% - Watch for reversal")
        
        # STOP LOSS EXITS
        if pnl_pct <= -50:  # 50% loss
            exit_signals['exit_score'] += 7
            exit_signals['primary_signal'] = 'IMMEDIATE EXIT'
            exit_signals['urgency'] = 'CRITICAL'
            exit_signals['reasons'].append(f"🛑 STOP LOSS HIT: {pnl_pct:.1f}% loss")
            
        elif pnl_pct <= -30:  # 30% loss
            exit_signals['exit_score'] += 4
            exit_signals['reasons'].append(f"⚠️ Significant loss: {pnl_pct:.1f}% - Consider exit")
        
        # TECHNICAL REVERSAL EXITS
        try:
            reversal_signals = self._check_technical_reversals(current_price, entry_decision, targets)
            exit_signals['technical_exits'] = reversal_signals['signals']
            exit_signals['exit_score'] += reversal_signals['score']
        except:
            pass
        
        # TARGET ACHIEVEMENT EXITS
        upside_target = targets.get('upside_target', entry_price * 1.01)
        downside_target = targets.get('downside_target', entry_price * 0.99)
        
        if 'LONG' in entry_decision and current_price >= upside_target:
            exit_signals['exit_score'] += 5
            exit_signals['reasons'].append(f"🎯 UPSIDE TARGET HIT: ${current_price:.2f} >= ${upside_target:.2f}")
            
        elif 'SHORT' in entry_decision and current_price <= downside_target:
            exit_signals['exit_score'] += 5
            exit_signals['reasons'].append(f"🎯 DOWNSIDE TARGET HIT: ${current_price:.2f} <= ${downside_target:.2f}")
        
        # FINAL EXIT DECISION
        if exit_signals['exit_score'] >= 8:
            exit_signals['should_exit'] = True
            exit_signals['primary_signal'] = 'IMMEDIATE EXIT'
            exit_signals['urgency'] = 'CRITICAL'
        elif exit_signals['exit_score'] >= 5:
            exit_signals['should_exit'] = True
            exit_signals['primary_signal'] = 'STRONG EXIT'
            exit_signals['urgency'] = 'HIGH'
        elif exit_signals['exit_score'] >= 3:
            exit_signals['primary_signal'] = 'CONSIDER EXIT'
            exit_signals['urgency'] = 'MEDIUM'
        
        return exit_signals
    
    def _check_technical_reversals(self, current_price: float, entry_direction: str, targets: Dict) -> Dict:
        """Check for technical reversal patterns using Tradier first, Yahoo fallback, Proxy last"""
        reversal_data = {'signals': [], 'score': 0}
        
        spy_data = None
        
        # PRIMARY: Tradier historical data
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            tradier_hist = self.api.get_historical_quotes('SPY', '1min', today)
            
            if tradier_hist and 'history' in tradier_hist:
                hist_data = tradier_hist['history']['day']
                if isinstance(hist_data, dict):
                    hist_data = [hist_data]  # Single day response
                
                # Process last 10 data points for momentum
                if len(hist_data) >= 10:
                    recent_prices = [float(d['close']) for d in hist_data[-10:]]
                    recent_volumes = [float(d['volume']) for d in hist_data[-10:]]
                    
                    # Calculate momentum
                    momentum_5min = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] * 100
                    momentum_10min = (recent_prices[-1] - recent_prices[-10]) / recent_prices[-10] * 100
                    
                    # Volume analysis
                    volume_trend = "INCREASING" if recent_volumes[-1] > np.mean(recent_volumes) else "DECREASING"
                    
                    spy_data = {
                        'momentum_5min': momentum_5min,
                        'momentum_10min': momentum_10min,
                        'volume_trend': volume_trend,
                        'source': 'Tradier'
                    }
                    
        except:
            pass
        
        # FALLBACK: Yahoo Finance only if Tradier fails
        if not spy_data:
            try:
                spy_hist = get_cached_yahoo_data('SPY', '1d', '1m')
                
                if not spy_hist.empty and spy_hist.shape[0] > 20:
                    recent_closes = spy_hist['Close'].tail(10)
                    momentum_5min = (recent_closes.iloc[-1] - recent_closes.iloc[-5]) / recent_closes.iloc[-5] * 100
                    momentum_10min = (recent_closes.iloc[-1] - recent_closes.iloc[-10]) / recent_closes.iloc[-10] * 100
                    
                    recent_volumes = spy_hist['Volume'].tail(10)
                    volume_trend = "INCREASING" if recent_volumes.iloc[-1] > recent_volumes.mean() else "DECREASING"
                    
                    spy_data = {
                        'momentum_5min': momentum_5min,
                        'momentum_10min': momentum_10min,
                        'volume_trend': volume_trend,
                        'source': 'Yahoo (Fallback)'
                    }
            except:
                pass
        
        # FINAL FALLBACK: Proxy estimates
        if not spy_data:
            spy_data = {
                'momentum_5min': -0.05,  # Slight bearish momentum
                'momentum_10min': -0.12,
                'volume_trend': "DECREASING",
                'source': 'Proxy Estimate'
            }
        
        # Analyze reversal signals
        if 'LONG' in entry_direction:
            if spy_data['momentum_5min'] < -0.1 and spy_data['momentum_10min'] < -0.15:
                reversal_data['signals'].append(f"📉 Short-term momentum turning negative (Source: {spy_data['source']})")
                reversal_data['score'] += 2
            
            if spy_data['volume_trend'] == "INCREASING" and spy_data['momentum_5min'] < 0:
                reversal_data['signals'].append(f"📉 Volume surge on downturn (Source: {spy_data['source']})")
                reversal_data['score'] += 1
                
        else:  # SHORT position
            if spy_data['momentum_5min'] > 0.1 and spy_data['momentum_10min'] > 0.15:
                reversal_data['signals'].append(f"📈 Short-term momentum turning positive (Source: {spy_data['source']})")
                reversal_data['score'] += 2
            
            if spy_data['volume_trend'] == "INCREASING" and spy_data['momentum_5min'] > 0:
                reversal_data['signals'].append(f"📈 Volume surge on upturn (Source: {spy_data['source']})")
                reversal_data['score'] += 1
                
        return reversal_data

class GapTradingAnalyzer:
    """Enhanced gap trading analyzer - Tradier PRIMARY, Yahoo FALLBACK, Proxy FINAL"""
    
    def __init__(self, api: TradierAPI):
        self.api = api
        self.miami_tz = pytz.timezone('US/Eastern')
        self.current_time = datetime.now(self.miami_tz)
        self.proxy_provider = ProxyDataProvider()
    
    def check_trading_window_enhanced(self):
        """Enhanced trading window with Miami time and after-hours logic"""
        miami_time = datetime.now(pytz.timezone('US/Eastern'))  # Miami is EDT
        
        # Check if it's a trading day
        trading_day = miami_time.weekday() < 5  # Monday = 0, Friday = 4
        
        if not trading_day:
            next_trading_day = "Monday" if miami_time.weekday() == 6 else "Tomorrow"
            return False, f"Weekend - Market opens {next_trading_day} at 9:30 AM EDT"
        
        # Market hours: 9:30 AM - 4:00 PM EDT
        market_open = miami_time.replace(hour=9, minute=30, second=0, microsecond=0)
        prime_window_end = miami_time.replace(hour=11, minute=0, second=0, microsecond=0)  # 11:00 AM
        danger_zone_start = miami_time.replace(hour=14, minute=30, second=0, microsecond=0)  # 2:30 PM  
        market_close = miami_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if miami_time < market_open:
            hours_until = (market_open - miami_time).total_seconds() / 3600
            return False, f"Market opens in {hours_until:.1f} hours (9:30 AM EDT)"
        elif miami_time > market_close:
            hours_since_close = (miami_time - market_close).total_seconds() / 3600
            return False, f"Market closed {hours_since_close:.1f} hours ago - Next session: Tomorrow 9:30 AM EDT"
        elif miami_time <= prime_window_end:
            mins_left = (prime_window_end - miami_time).total_seconds() / 60
            return True, f"🟢 PRIME 0DTE WINDOW - {mins_left:.0f} minutes left in optimal zone"
        elif miami_time < danger_zone_start:
            return True, f"🟡 STANDARD WINDOW - Good for trading but watch for lunch-time chop"
        else:
            mins_to_close = (market_close - miami_time).total_seconds() / 60
            return True, f"🔴 DANGER ZONE - HIGH THETA BURN - {mins_to_close:.0f} minutes to close"

    def check_trading_window(self):
        """Legacy method for compatibility"""
        return self.check_trading_window_enhanced()

    def get_spy_data_enhanced(self) -> Dict:
        """Get SPY data with Tradier PRIMARY, Yahoo fallback, Proxy final"""
        spy_data = {
            'current_price': 0,
            'historical': None,
            'volume_data': {},
            'vwap_data': {},
            'source': 'unknown',
            'error': None
        }
        
        # PRIMARY: Try Tradier first
        try:
            spy_quote = self.api.get_quote('SPY')
            if spy_quote and spy_quote.get('last'):
                spy_data['current_price'] = float(spy_quote['last'])
                spy_data['volume_data'] = {
                    'current_volume': float(spy_quote.get('volume', 0)),
                    'avg_volume': float(spy_quote.get('avgvolume', 0))
                }
                spy_data['source'] = 'Tradier'
                
                # Try to get historical data from Tradier for VWAP calculation
                today = datetime.now().strftime('%Y-%m-%d')
                tradier_hist = self.api.get_historical_quotes('SPY', '1min', today)
                
                if tradier_hist and 'history' in tradier_hist:
                    hist_data = tradier_hist['history']['day']
                    if isinstance(hist_data, dict):
                        hist_data = [hist_data]
                    
                    # Calculate VWAP from Tradier data
                    if hist_data:
                        total_pv = sum(float(d['close']) * float(d['volume']) for d in hist_data)
                        total_volume = sum(float(d['volume']) for d in hist_data)
                        
                        if total_volume > 0:
                            vwap = total_pv / total_volume
                            vwap_distance = ((spy_data['current_price'] - vwap) / vwap) * 100
                            
                            spy_data['vwap_data'] = {
                                'vwap': vwap,
                                'distance_pct': vwap_distance,
                                'source': 'Tradier'
                            }
                
                return spy_data
                
        except Exception as e:
            spy_data['error'] = f"Tradier error: {str(e)}"
        
        # FALLBACK: Use Yahoo Finance only if Tradier fails
        try:
            spy_ticker = yf.Ticker('SPY')
            spy_hist = get_cached_yahoo_data('SPY', '2d', '1m')
            
            if not spy_hist.empty:
                spy_data['current_price'] = spy_hist['Close'].iloc[-1]
                spy_data['historical'] = spy_hist
                spy_data['volume_data'] = {
                    'current_volume': spy_hist['Volume'].iloc[-1],
                    'avg_volume': spy_hist['Volume'].mean()
                }
                
                # Calculate VWAP from Yahoo data
                typical_price = (spy_hist['High'] + spy_hist['Low'] + spy_hist['Close']) / 3
                vwap = (typical_price * spy_hist['Volume']).sum() / spy_hist['Volume'].sum()
                vwap_distance = ((spy_data['current_price'] - vwap) / vwap) * 100
                
                spy_data['vwap_data'] = {
                    'vwap': vwap,
                    'distance_pct': vwap_distance,
                    'source': 'Yahoo (Fallback)'
                }
                
                spy_data['source'] = 'Yahoo (Fallback)'
                spy_data['error'] = None
                
                return spy_data
                
        except Exception as e:
            spy_data['error'] = f"Yahoo also failed: {str(e)}"
        
        # FINAL FALLBACK: Use Proxy Data
        proxy_data = self.proxy_provider.get_spy_proxy_data(self.current_time)
        spy_data.update({
            'current_price': proxy_data['current_price'],
            'volume_data': {
                'current_volume': proxy_data['volume'],
                'avg_volume': proxy_data['volume'] * 0.9
            },
            'vwap_data': {
                'vwap': proxy_data['current_price'],
                'distance_pct': 0.000,
                'source': 'Proxy Estimate'
            },
            'source': proxy_data['source'],
            'error': None
        })
        
        return spy_data
    
    def get_sector_data_enhanced(self) -> Dict:
        """Get sector data with Tradier PRIMARY, Yahoo fallback, Proxy final"""
        sector_symbols = ['XLK', 'XLF', 'XLV', 'XLY', 'XLI', 'XLP', 'XLE', 'XLU', 'XLRE']
        sector_weights = {
            'XLK': 0.25, 'XLF': 0.15, 'XLV': 0.15, 'XLY': 0.12, 'XLI': 0.10,
            'XLP': 0.08, 'XLE': 0.05, 'XLU': 0.05, 'XLRE': 0.05
        }
        
        sector_data = {}
        
        # PRIMARY: Try Tradier for all sectors first
        try:
            symbols_str = ','.join(sector_symbols)
            tradier_quotes = self.api.get_quotes_bulk(symbols_str)
            
            if tradier_quotes:
                for symbol in sector_symbols:
                    if symbol in tradier_quotes:
                        quote = tradier_quotes[symbol]
                        sector_data[symbol] = {
                            'current_price': float(quote.get('last', 0)),
                            'change_pct': float(quote.get('change_percentage', 0)),
                            'volume': float(quote.get('volume', 0)),
                            'avg_volume': float(quote.get('avgvolume', 0)),
                            'source': 'Tradier',
                            'weight': sector_weights.get(symbol, 0.05)
                        }
                
                if len(sector_data) == len(sector_symbols):
                    return sector_data  # All data successfully from Tradier
                    
        except:
            pass  # Silent fallback
        
        # FALLBACK: Only use Yahoo for missing data
        for symbol in sector_symbols:
            if symbol not in sector_data:
                try:
                    hist = get_cached_yahoo_data(symbol, '1d', '1m')
                    if not hist.empty:
                        change_pct = ((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0]) * 100
                        sector_data[symbol] = {
                            'current_price': hist['Close'].iloc[-1],
                            'change_pct': change_pct,
                            'volume': hist['Volume'].iloc[-1],
                            'avg_volume': hist['Volume'].mean(),
                            'source': 'Yahoo (Fallback)',
                            'weight': sector_weights.get(symbol, 0.05)
                        }
                except:
                    pass  # Use proxy fallback
        
        # FINAL FALLBACK: Use proxy data for any remaining missing sectors
        proxy_sectors = self.proxy_provider.get_sector_proxy_data()
        for symbol in sector_symbols:
            if symbol not in sector_data:
                proxy = proxy_sectors.get(symbol, {'change_pct': 0, 'volume_ratio': 1.0})
                sector_data[symbol] = {
                    'current_price': 100.0,  # Placeholder
                    'change_pct': proxy['change_pct'],
                    'volume': 1000000,  # Placeholder
                    'avg_volume': 1000000,
                    'source': 'Proxy Estimate',
                    'weight': sector_weights.get(symbol, 0.05)
                }
        
        return sector_data
    
    def get_market_internals_yahoo_proxy(self) -> Dict:
        """Market internals - Yahoo first, then proxy estimates"""
        internals = {
            'tick': {'value': 0, 'source': 'Yahoo Proxy'},
            'trin': {'value': 1.0, 'source': 'Yahoo Proxy'},
            'nyad': {'value': 0, 'source': 'Yahoo Proxy'},
            'vold': {'value': 1.0, 'source': 'Yahoo Proxy'}
        }
        
        # Try Yahoo calculation first
        try:
            indices = ['SPY', 'QQQ', 'IWM', 'DIA']
            index_data = {}
            
            for idx in indices:
                hist = get_cached_yahoo_data(idx, '1d', '1m')
                if not hist.empty:
                    change_pct = ((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0]) * 100
                    volume_ratio = hist['Volume'].iloc[-1] / hist['Volume'].mean() if len(hist) > 1 else 1.0
                    index_data[idx] = {'change_pct': change_pct, 'volume_ratio': volume_ratio}
            
            if index_data:
                positive_indices = sum(1 for data in index_data.values() if data['change_pct'] > 0)
                total_indices = len(index_data)
                
                # Calculate proxy values
                internals['tick']['value'] = ((positive_indices / total_indices) - 0.5) * 2000
                internals['nyad']['value'] = ((positive_indices / total_indices) - 0.5) * 4000
                
                avg_volume_ratio = sum(data['volume_ratio'] for data in index_data.values()) / len(index_data)
                internals['trin']['value'] = 1.0 / avg_volume_ratio if avg_volume_ratio > 0 else 1.0
                internals['vold']['value'] = (index_data.get('QQQ', {}).get('volume_ratio', 1) + 
                                             index_data.get('SPY', {}).get('volume_ratio', 1)) / 2
                
                return internals
                
        except:
            pass
        
        # FINAL FALLBACK: Use proxy estimates
        proxy_internals = self.proxy_provider.get_internals_proxy_data()
        for key in internals:
            internals[key].update(proxy_internals[key])
            internals[key]['source'] = 'Proxy Estimate'
        
        return internals

    def calculate_gap_analysis(self) -> Dict:
        """Calculate gap analysis with transparent points breakdown"""
        try:
            spy_data = self.get_spy_data_enhanced()
            current_price = spy_data.get('current_price', 640.27)
            
            # Initialize with proxy/default values
            yesterday_close = current_price * 1.005  # Default small gap
            today_open = current_price
            today_volume = 50000000
            avg_volume = 45000000
            gap_pct = -1.2  # Based on typical gap down
            
            # Try Tradier first (silently)
            try:
                start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                tradier_hist = self.api.get_historical_quotes('SPY', '1day', start_date, end_date)
                
                if tradier_hist and 'history' in tradier_hist:
                    hist_data = tradier_hist['history']['day']
                    if isinstance(hist_data, dict):
                        hist_data = [hist_data]
                    
                    if len(hist_data) >= 2:
                        yesterday_close = float(hist_data[-2]['close'])
                        today_open = float(hist_data[-1]['open'])
                        today_volume = float(hist_data[-1]['volume'])
                        gap_pct = ((today_open - yesterday_close) / yesterday_close) * 100
            except:
                pass  # Silent fallback
            
            # Try Yahoo fallback (silently)
            if gap_pct == -1.2:  # Still using default
                try:
                    spy_hist = get_cached_yahoo_data('SPY', '5d', '1d')
                    if not spy_hist.empty and spy_hist.shape[0] >= 2:
                        yesterday_close = spy_hist['Close'].iloc[-2]
                        today_open = spy_hist['Open'].iloc[-1]
                        today_volume = spy_hist['Volume'].iloc[-1]
                        avg_volume = spy_hist['Volume'].tail(5).mean()
                        gap_pct = ((today_open - yesterday_close) / yesterday_close) * 100
                except:
                    pass  # Use proxy values
            
            # POINTS BREAKDOWN - Make this very transparent
            points_breakdown = {
                'gap_size': {'points': 0, 'reason': ''},
                'statistical_significance': {'points': 0, 'reason': ''},
                'volume_confirmation': {'points': 0, 'reason': ''},
                'vwap_alignment': {'points': 0, 'reason': ''},
                'es_alignment': {'points': 0.5, 'reason': 'ES futures alignment (assumed)'},
                'total_base_points': 0,
                'direction_multiplier': 1,
                'final_points': 0
            }
            
            # Volume analysis
            volume_surge_ratio = today_volume / avg_volume if avg_volume > 0 else 1.39
            
            # VWAP calculation
            vwap_data = spy_data.get('vwap_data', {})
            vwap_distance = vwap_data.get('distance_pct', 0.0)
            
            # VWAP status
            if abs(vwap_distance) < 0.05:
                vwap_status = "AT VWAP"
            elif vwap_distance > 0.25:
                vwap_status = "STRONG ABOVE"
            elif vwap_distance > 0.05:
                vwap_status = "ABOVE"
            elif vwap_distance < -0.25:
                vwap_status = "STRONG BELOW"
            else:
                vwap_status = "BELOW"
            
            # GAP SIZE POINTS (Transparent calculation)
            abs_gap = abs(gap_pct)
            if abs_gap >= 2.5:
                gap_size_category = "MONSTER"
                points_breakdown['gap_size']['points'] = 1.0
                points_breakdown['gap_size']['reason'] = f"Monster gap {abs_gap:.2f}% (≥2.5%) = 1.0 pts (high unpredictability)"
            elif abs_gap >= 1.5:
                gap_size_category = "LARGE"
                points_breakdown['gap_size']['points'] = 2.0
                points_breakdown['gap_size']['reason'] = f"Large gap {abs_gap:.2f}% (1.5-2.5%) = 2.0 pts (optimal risk/reward)"
            elif abs_gap >= 0.75:
                gap_size_category = "MEDIUM"
                points_breakdown['gap_size']['points'] = 1.5
                points_breakdown['gap_size']['reason'] = f"Medium gap {abs_gap:.2f}% (0.75-1.5%) = 1.5 pts (good probability)"
            elif abs_gap >= 0.5:
                gap_size_category = "SMALL"
                points_breakdown['gap_size']['points'] = 1.0
                points_breakdown['gap_size']['reason'] = f"Small gap {abs_gap:.2f}% (0.5-0.75%) = 1.0 pts (moderate edge)"
            else:
                gap_size_category = "MINIMAL"
                points_breakdown['gap_size']['points'] = 0.0
                points_breakdown['gap_size']['reason'] = f"Minimal gap {abs_gap:.2f}% (<0.5%) = 0.0 pts (no edge)"
            
            # STATISTICAL SIGNIFICANCE POINTS
            if abs_gap >= 1.5:
                statistical_significance = "MODERATE"
                points_breakdown['statistical_significance']['points'] = 0.5
                points_breakdown['statistical_significance']['reason'] = f"Gap {abs_gap:.2f}% shows moderate significance = 0.5 pts"
            else:
                statistical_significance = "LOW"
                points_breakdown['statistical_significance']['points'] = 0.0
                points_breakdown['statistical_significance']['reason'] = f"Gap {abs_gap:.2f}% shows low significance = 0.0 pts"
            
            # VOLUME CONFIRMATION POINTS
            if volume_surge_ratio >= 2.0:
                points_breakdown['volume_confirmation']['points'] = 1.0
                points_breakdown['volume_confirmation']['reason'] = f"Strong volume surge {volume_surge_ratio:.2f}x (≥2.0x) = 1.0 pts"
            elif volume_surge_ratio >= 1.5:
                points_breakdown['volume_confirmation']['points'] = 0.5
                points_breakdown['volume_confirmation']['reason'] = f"Moderate volume surge {volume_surge_ratio:.2f}x (1.5-2.0x) = 0.5 pts"
            else:
                points_breakdown['volume_confirmation']['points'] = 0.0
                points_breakdown['volume_confirmation']['reason'] = f"Weak volume {volume_surge_ratio:.2f}x (<1.5x) = 0.0 pts"
            
            # VWAP ALIGNMENT POINTS
            if abs(vwap_distance) > 0.25:
                points_breakdown['vwap_alignment']['points'] = 1.0
                points_breakdown['vwap_alignment']['reason'] = f"Strong VWAP distance {vwap_distance:+.3f}% (>0.25%) = 1.0 pts"
            elif abs(vwap_distance) > 0.1:
                points_breakdown['vwap_alignment']['points'] = 0.5
                points_breakdown['vwap_alignment']['reason'] = f"Moderate VWAP distance {vwap_distance:+.3f}% (>0.1%) = 0.5 pts"
            else:
                points_breakdown['vwap_alignment']['points'] = 0.0
                points_breakdown['vwap_alignment']['reason'] = f"Neutral VWAP distance {vwap_distance:+.3f}% (≤0.1%) = 0.0 pts"
            
            # CALCULATE TOTAL BASE POINTS
            points_breakdown['total_base_points'] = (
                points_breakdown['gap_size']['points'] + 
                points_breakdown['statistical_significance']['points'] + 
                points_breakdown['volume_confirmation']['points'] + 
                points_breakdown['vwap_alignment']['points'] + 
                points_breakdown['es_alignment']['points']
            )
            
            # APPLY DIRECTION MULTIPLIER
            if gap_pct < 0:  # Gap down
                if vwap_distance < 0:  # Price below VWAP - alignment
                    points_breakdown['direction_multiplier'] = -1.0
                    points_breakdown['final_points'] = -points_breakdown['total_base_points']
                else:  # Price above VWAP - misalignment
                    points_breakdown['direction_multiplier'] = -0.5
                    points_breakdown['final_points'] = -points_breakdown['total_base_points'] * 0.5
            else:  # Gap up
                if vwap_distance > 0:  # Price above VWAP - alignment
                    points_breakdown['direction_multiplier'] = 1.0
                    points_breakdown['final_points'] = points_breakdown['total_base_points']
                else:  # Price below VWAP - misalignment
                    points_breakdown['direction_multiplier'] = 0.5
                    points_breakdown['final_points'] = points_breakdown['total_base_points'] * 0.5
            
            return {
                'gap_pct': gap_pct,
                'gap_size_category': gap_size_category,
                'statistical_significance': statistical_significance,
                'volume_surge_ratio': volume_surge_ratio,
                'vwap_distance_pct': vwap_distance,
                'vwap_status': vwap_status,
                'es_alignment': True,
                'total_points': points_breakdown['final_points'],
                'points_breakdown': points_breakdown,  # NEW: Detailed breakdown
                'data_source': spy_data.get('source', 'Proxy'),
                'error': None
            }
            
        except Exception as e:
            # Return proxy data that works
            return {
                'gap_pct': -1.17,
                'gap_size_category': 'MEDIUM',
                'statistical_significance': 'LOW',
                'volume_surge_ratio': 1.39,
                'vwap_distance_pct': 0.000,
                'vwap_status': 'AT VWAP',
                'es_alignment': True,
                'total_points': -2.0,
                'points_breakdown': {
                    'gap_size': {'points': 1.5, 'reason': 'Medium gap (proxy estimate)'},
                    'final_points': -2.0
                },
                'data_source': 'Proxy Fallback',
                'error': None
            }

    def analyze_market_internals_enhanced(self) -> Dict:
        """Enhanced market internals with transparent points breakdown"""
        try:
            internals_data = self.get_market_internals_yahoo_proxy()
            
            # Points breakdown for transparency
            internals_breakdown = {
                'tick': {'points': 0, 'reason': ''},
                'trin': {'points': 0, 'reason': ''},
                'nyad': {'points': 0, 'reason': ''},
                'vold': {'points': 0, 'reason': ''},
                'total_points': 0
            }
            
            # Analyze each internal with transparent scoring
            tick_value = internals_data['tick']['value']
            trin_value = internals_data['trin']['value']
            nyad_value = internals_data['nyad']['value']
            vold_value = internals_data['vold']['value']
            
            # TICK analysis with transparent points
            if tick_value >= 1000:
                tick_signal = "EXTREME BULLISH"
                internals_breakdown['tick']['points'] = 2.0
                internals_breakdown['tick']['reason'] = f"$TICK {tick_value:.0f} ≥ +1000 (Extreme Bullish) = +2.0 pts"
            elif tick_value >= 800:
                tick_signal = "BULLISH"
                internals_breakdown['tick']['points'] = 1.5
                internals_breakdown['tick']['reason'] = f"$TICK {tick_value:.0f} ≥ +800 (Strong Bullish) = +1.5 pts"
            elif tick_value >= 200:
                tick_signal = "MODERATE BULLISH"
                internals_breakdown['tick']['points'] = 1.0
                internals_breakdown['tick']['reason'] = f"$TICK {tick_value:.0f} ≥ +200 (Moderate Bullish) = +1.0 pts"
            elif tick_value <= -1000:
                tick_signal = "EXTREME BEARISH"
                internals_breakdown['tick']['points'] = -2.0
                internals_breakdown['tick']['reason'] = f"$TICK {tick_value:.0f} ≤ -1000 (Extreme Bearish) = -2.0 pts"
            elif tick_value <= -800:
                tick_signal = "BEARISH"
                internals_breakdown['tick']['points'] = -1.5
                internals_breakdown['tick']['reason'] = f"$TICK {tick_value:.0f} ≤ -800 (Strong Bearish) = -1.5 pts"
            elif tick_value <= -200:
                tick_signal = "MODERATE BEARISH"
                internals_breakdown['tick']['points'] = -1.0
                internals_breakdown['tick']['reason'] = f"$TICK {tick_value:.0f} ≤ -200 (Moderate Bearish) = -1.0 pts"
            else:
                tick_signal = "NEUTRAL"
                internals_breakdown['tick']['points'] = 0.0
                internals_breakdown['tick']['reason'] = f"$TICK {tick_value:.0f} in neutral range (-200 to +200) = 0.0 pts"
            
            # TRIN analysis with transparent points
            if trin_value <= 0.8:
                trin_signal = "BULLISH"
                internals_breakdown['trin']['points'] = 1.0
                internals_breakdown['trin']['reason'] = f"$TRIN {trin_value:.3f} ≤ 0.8 (Bullish volume flow) = +1.0 pts"
            elif trin_value >= 1.2:
                trin_signal = "BEARISH"
                internals_breakdown['trin']['points'] = -1.0
                internals_breakdown['trin']['reason'] = f"$TRIN {trin_value:.3f} ≥ 1.2 (Bearish volume flow) = -1.0 pts"
            else:
                trin_signal = "NEUTRAL"
                internals_breakdown['trin']['points'] = 0.0
                internals_breakdown['trin']['reason'] = f"$TRIN {trin_value:.3f} neutral (0.8-1.2) = 0.0 pts"
            
            # NYAD analysis with transparent points
            if nyad_value > 1000:
                nyad_signal = "BULLISH"
                internals_breakdown['nyad']['points'] = 1.0
                internals_breakdown['nyad']['reason'] = f"NYAD {nyad_value:.0f} > +1000 (Broad bullish participation) = +1.0 pts"
            elif nyad_value < -1000:
                nyad_signal = "BEARISH"
                internals_breakdown['nyad']['points'] = -1.0
                internals_breakdown['nyad']['reason'] = f"NYAD {nyad_value:.0f} < -1000 (Broad bearish participation) = -1.0 pts"
            else:
                nyad_signal = "NEUTRAL"
                internals_breakdown['nyad']['points'] = 0.0
                internals_breakdown['nyad']['reason'] = f"NYAD {nyad_value:.0f} neutral (-1000 to +1000) = 0.0 pts"
            
            # VOLD analysis with transparent points
            if vold_value > 1.5:
                vold_signal = "BULLISH"
                internals_breakdown['vold']['points'] = 1.0
                internals_breakdown['vold']['reason'] = f"Volume Flow {vold_value:.2f} > 1.5 (Risk-on flow) = +1.0 pts"
            elif vold_value < 0.7:
                vold_signal = "BEARISH"
                internals_breakdown['vold']['points'] = -1.0
                internals_breakdown['vold']['reason'] = f"Volume Flow {vold_value:.2f} < 0.7 (Risk-off flow) = -1.0 pts"
            else:
                vold_signal = "NEUTRAL"
                internals_breakdown['vold']['points'] = 0.0
                internals_breakdown['vold']['reason'] = f"Volume Flow {vold_value:.2f} neutral (0.7-1.5) = 0.0 pts"
            
            # Calculate total points
            internals_breakdown['total_points'] = (
                internals_breakdown['tick']['points'] + 
                internals_breakdown['trin']['points'] + 
                internals_breakdown['nyad']['points'] + 
                internals_breakdown['vold']['points']
            )
            
            # Update signals
            internals_data['tick']['signal'] = tick_signal
            internals_data['trin']['signal'] = trin_signal
            internals_data['nyad']['signal'] = nyad_signal
            internals_data['vold']['signal'] = vold_signal
            
            return {
                'tick': internals_data['tick'],
                'trin': internals_data['trin'],
                'nyad': internals_data['nyad'],
                'vold': internals_data['vold'],
                'total_points': internals_breakdown['total_points'],
                'points_breakdown': internals_breakdown  # NEW: Detailed breakdown
            }
            
        except Exception as e:
            # Return proxy data with breakdown
            return {
                'total_points': -1.5,
                'points_breakdown': {
                    'tick': {'points': 0, 'reason': 'Neutral proxy estimate'},
                    'trin': {'points': 0, 'reason': 'Neutral proxy estimate'},
                    'nyad': {'points': -1.0, 'reason': 'Bearish proxy estimate'},
                    'vold': {'points': -0.5, 'reason': 'Bearish proxy estimate'},
                    'total_points': -1.5
                },
                'tick': {'value': 0, 'signal': 'NEUTRAL', 'source': 'Proxy'},
                'trin': {'value': 1.0, 'signal': 'NEUTRAL', 'source': 'Proxy'},
                'nyad': {'value': -2000, 'signal': 'BEARISH', 'source': 'Proxy'},
                'vold': {'value': 0.30, 'signal': 'BEARISH', 'source': 'Proxy'}
            }

    def analyze_sectors_enhanced(self) -> Dict:
        """Enhanced sector analysis with transparent points breakdown"""
        try:
            sector_data = self.get_sector_data_enhanced()
            
            # Get SPY performance for relative strength
            spy_change = 0
            try:
                spy_quote = self.api.get_quote('SPY')
                if spy_quote and spy_quote.get('change_percentage'):
                    spy_change = float(spy_quote['change_percentage'])
                    spy_source = 'Tradier'
                else:
                    # Fallback to Yahoo
                    spy_data = get_cached_yahoo_data('SPY', '1d', '1m')
                    if not spy_data.empty:
                        spy_change = ((spy_data['Close'].iloc[-1] - spy_data['Open'].iloc[0]) / spy_data['Open'].iloc[0]) * 100
                        spy_source = 'Yahoo (Fallback)'
            except:
                spy_change = -0.75  # Proxy fallback
                spy_source = 'Proxy Estimate'
            
            sector_analysis = {}
            total_weighted_score = 0
            strong_sectors = 0
            weak_sectors = 0
            
            # Points breakdown for transparency
            sectors_breakdown = {
                'individual_scores': {},
                'leadership_calculation': '',
                'rotation_logic': '',
                'total_weighted_points': 0
            }
            
            # Analyze each sector with transparent scoring
            for symbol, data in sector_data.items():
                change_pct = data['change_pct']
                volume_ratio = data['volume'] / data['avg_volume'] if data['avg_volume'] > 0 else 1.0
                vwap_distance = 0.0  # Would need intraday data for proper VWAP
                relative_strength = change_pct - spy_change
                
                # Transparent sector strength calculation
                if relative_strength > 0.5 and volume_ratio > 1.2:
                    strength_score = 2
                    strength_reason = f"Strong leader: +{relative_strength:.2f}% vs SPY, {volume_ratio:.1f}x volume = +2 pts"
                    strong_sectors += 1
                elif relative_strength > 0.2:
                    strength_score = 1
                    strength_reason = f"Moderate leader: +{relative_strength:.2f}% vs SPY = +1 pts"
                elif relative_strength < -0.5 and volume_ratio > 1.2:
                    strength_score = -2
                    strength_reason = f"Strong laggard: {relative_strength:.2f}% vs SPY, {volume_ratio:.1f}x volume = -2 pts"
                    weak_sectors += 1
                elif relative_strength < -0.2:
                    strength_score = -1
                    strength_reason = f"Moderate laggard: {relative_strength:.2f}% vs SPY = -1 pts"
                else:
                    strength_score = 0
                    strength_reason = f"Neutral: {relative_strength:.2f}% vs SPY = 0 pts"
                
                # Weight the score
                weighted_score = strength_score * data['weight']
                total_weighted_score += weighted_score
                
                sector_analysis[symbol] = {
                    'name': self._get_sector_name(symbol),
                    'change_pct': change_pct,
                    'relative_strength': relative_strength,
                    'volume_ratio': volume_ratio,
                    'vwap_distance': vwap_distance,
                    'strength_score': strength_score,
                    'weight': data['weight'],
                    'source': data['source']
                }
                
                # Store breakdown
                sectors_breakdown['individual_scores'][symbol] = {
                    'strength_score': strength_score,
                    'weight': data['weight'],
                    'weighted_score': weighted_score,
                    'reason': strength_reason
                }
            
            # Leadership score calculation
            leadership_score = strong_sectors - weak_sectors
            sectors_breakdown['leadership_calculation'] = f"Strong sectors ({strong_sectors}) - Weak sectors ({weak_sectors}) = {leadership_score}"
            
            # Determine market regime
            if leadership_score >= 2:
                rotation_signal = "RISK ON"
                sectors_breakdown['rotation_logic'] = f"Leadership score {leadership_score} ≥ 2 = RISK ON (growth/cyclical leading)"
            elif leadership_score <= -2:
                rotation_signal = "RISK OFF"
                sectors_breakdown['rotation_logic'] = f"Leadership score {leadership_score} ≤ -2 = RISK OFF (defensive leading)"
            else:
                rotation_signal = "NEUTRAL"
                sectors_breakdown['rotation_logic'] = f"Leadership score {leadership_score} in neutral range (-2 to +2) = NEUTRAL"
            
            sectors_breakdown['total_weighted_points'] = total_weighted_score
            
            return {
                'sectors': sector_analysis,
                'total_points': total_weighted_score,
                'leadership_score': leadership_score,
                'rotation_signal': rotation_signal,
                'strong_sectors': strong_sectors,
                'weak_sectors': weak_sectors,
                'spy_source': spy_source,
                'points_breakdown': sectors_breakdown  # NEW: Detailed breakdown
            }
            
        except Exception as e:
            # Return proxy data with breakdown
            return {
                'error': f'Sector analysis error: {str(e)}',
                'total_points': 0.0,
                'leadership_score': 0.0,
                'rotation_signal': 'NEUTRAL',
                'points_breakdown': {
                    'total_weighted_points': 0.0,
                    'leadership_calculation': 'Using proxy estimates',
                    'rotation_logic': 'Neutral due to data limitations'
                }
            }
    
    def _get_sector_name(self, symbol: str) -> str:
        """Get sector name from symbol"""
        names = {
            'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
            'XLY': 'Consumer Discretionary', 'XLI': 'Industrials', 'XLP': 'Consumer Staples',
            'XLE': 'Energy', 'XLU': 'Utilities', 'XLRE': 'Real Estate'
        }
        return names.get(symbol, 'Unknown')

    def analyze_technicals_enhanced(self) -> Dict:
        """Enhanced technical analysis with transparent points breakdown"""
        try:
            spy_data = self.get_spy_data_enhanced()
            
            if spy_data.get('error'):
                # Use proxy fallback
                current_price = 640.27
                vwap = 640.27
                vwap_distance_pct = 0.000
                yesterday_high = 647.84
                yesterday_low = 643.14
                yesterday_close = 645.05
                volume_confirmation = 1.0
                data_source = 'Proxy Fallback'
            else:
                current_price = spy_data['current_price']
                vwap_data = spy_data.get('vwap_data', {})
                vwap = vwap_data.get('vwap', current_price)
                vwap_distance_pct = vwap_data.get('distance_pct', 0.0)
                data_source = spy_data['source']
                
                # Support/Resistance Analysis - try Tradier first
                yesterday_high = yesterday_low = yesterday_close = None
                
                try:
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                    tradier_hist = self.api.get_historical_quotes('SPY', '1day', yesterday)
                    
                    if tradier_hist and 'history' in tradier_hist:
                        hist_data = tradier_hist['history']['day']
                        if isinstance(hist_data, dict):
                            yesterday_high = float(hist_data['high'])
                            yesterday_low = float(hist_data['low'])
                            yesterday_close = float(hist_data['close'])
                except:
                    pass  # Will use Yahoo fallback
                
                # FALLBACK: Yahoo if Tradier fails
                if yesterday_high is None:
                    try:
                        spy_hist = get_cached_yahoo_data('SPY', '2d', '1d')
                        if not spy_hist.empty and spy_hist.shape[0] >= 2:
                            yesterday_high = spy_hist['High'].iloc[-2]
                            yesterday_low = spy_hist['Low'].iloc[-2]
                            yesterday_close = spy_hist['Close'].iloc[-2]
                    except:
                        # Ultimate fallback
                        yesterday_high = current_price * 1.015
                        yesterday_low = current_price * 0.985
                        yesterday_close = current_price * 1.005
                
                # Volume confirmation
                volume_data = spy_data.get('volume_data', {})
                current_volume = volume_data.get('current_volume', 0)
                avg_volume = volume_data.get('avg_volume', 1)
                volume_confirmation = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Points breakdown for transparency
            technicals_breakdown = {
                'vwap': {'points': 0, 'reason': ''},
                'support_resistance': {'points': 0, 'reason': ''},
                'volume': {'points': 0, 'reason': ''},
                'total_points': 0
            }
            
            # VWAP signal strength with transparent points
            if abs(vwap_distance_pct) > 0.3:
                vwap_signal = "STRONG"
                vwap_points = 2.0 if vwap_distance_pct > 0 else -2.0
                technicals_breakdown['vwap']['points'] = vwap_points
                technicals_breakdown['vwap']['reason'] = f"Strong VWAP distance {vwap_distance_pct:+.3f}% (>0.3%) = {vwap_points:+.1f} pts"
            elif abs(vwap_distance_pct) > 0.15:
                vwap_signal = "MODERATE"
                vwap_points = 1.0 if vwap_distance_pct > 0 else -1.0
                technicals_breakdown['vwap']['points'] = vwap_points
                technicals_breakdown['vwap']['reason'] = f"Moderate VWAP distance {vwap_distance_pct:+.3f}% (0.15-0.3%) = {vwap_points:+.1f} pts"
            elif abs(vwap_distance_pct) > 0.05:
                vwap_signal = "WEAK"
                vwap_points = 0.5 if vwap_distance_pct > 0 else -0.5
                technicals_breakdown['vwap']['points'] = vwap_points
                technicals_breakdown['vwap']['reason'] = f"Weak VWAP distance {vwap_distance_pct:+.3f}% (0.05-0.15%) = {vwap_points:+.1f} pts"
            else:
                vwap_signal = "NEUTRAL"
                vwap_points = 0.0
                technicals_breakdown['vwap']['points'] = vwap_points
                technicals_breakdown['vwap']['reason'] = f"Neutral VWAP distance {vwap_distance_pct:+.3f}% (≤0.05%) = 0.0 pts"
            
            # Support/Resistance points with transparency
            current_vs_yesterday_high = ((current_price - yesterday_high) / yesterday_high) * 100
            current_vs_yesterday_low = ((current_price - yesterday_low) / yesterday_low) * 100
            
            if current_price > yesterday_high:
                sr_points = 1.0
                technicals_breakdown['support_resistance']['points'] = sr_points
                technicals_breakdown['support_resistance']['reason'] = f"Breakout above yesterday high ${yesterday_high:.2f} = +1.0 pts"
            elif current_price < yesterday_low:
                sr_points = -1.0
                technicals_breakdown['support_resistance']['points'] = sr_points
                technicals_breakdown['support_resistance']['reason'] = f"Breakdown below yesterday low ${yesterday_low:.2f} = -1.0 pts"
            else:
                sr_points = 0.0
                technicals_breakdown['support_resistance']['points'] = sr_points
                technicals_breakdown['support_resistance']['reason'] = f"Price within yesterday range ${yesterday_low:.2f}-${yesterday_high:.2f} = 0.0 pts"
            
            # Volume confirmation points with transparency
            if volume_confirmation > 1.5:
                volume_points = 0.5
                technicals_breakdown['volume']['points'] = volume_points
                technicals_breakdown['volume']['reason'] = f"Strong volume {volume_confirmation:.1f}x average (>1.5x) = +0.5 pts"
            elif volume_confirmation < 0.8:
                volume_points = -0.5
                technicals_breakdown['volume']['points'] = volume_points
                technicals_breakdown['volume']['reason'] = f"Weak volume {volume_confirmation:.1f}x average (<0.8x) = -0.5 pts"
            else:
                volume_points = 0.0
                technicals_breakdown['volume']['points'] = volume_points
                technicals_breakdown['volume']['reason'] = f"Normal volume {volume_confirmation:.1f}x average (0.8-1.5x) = 0.0 pts"
            
            # Calculate total
            total_technical_points = vwap_points + sr_points + volume_points
            technicals_breakdown['total_points'] = total_technical_points
            
            return {
                'total_points': total_technical_points,
                'vwap_analysis': {
                    'current_price': current_price,
                    'vwap': vwap,
                    'distance_pct': vwap_distance_pct,
                    'signal_strength': vwap_signal,
                    'source': data_source
                },
                'support_resistance': {
                    'yesterday_high': yesterday_high,
                    'yesterday_low': yesterday_low,
                    'yesterday_close': yesterday_close,
                    'current_vs_yesterday_high': current_vs_yesterday_high,
                    'current_vs_yesterday_low': current_vs_yesterday_low
                },
                'volume_analysis': {
                    'recent_vs_average': volume_confirmation,
                    'signal': 'STRONG' if volume_confirmation > 1.5 else 'WEAK' if volume_confirmation < 0.8 else 'NORMAL'
                },
                'data_source': data_source,
                'points_breakdown': technicals_breakdown,  # NEW: Detailed breakdown
                'error': None
            }
            
        except Exception as e:
            # Return proxy data with breakdown
            return {
                'total_points': -1.0,
                'points_breakdown': {
                    'vwap': {'points': 0.0, 'reason': 'Neutral VWAP (proxy estimate)'},
                    'support_resistance': {'points': -0.5, 'reason': 'Below resistance level (proxy estimate)'},
                    'volume': {'points': -0.5, 'reason': 'Weak volume (proxy estimate)'},
                    'total_points': -1.0
                },
                'vwap_analysis': {
                    'current_price': 640.27,
                    'vwap': 640.27,
                    'distance_pct': 0.000,
                    'signal_strength': 'NEUTRAL'
                },
                'support_resistance': {
                    'yesterday_high': 647.84,
                    'yesterday_low': 643.14,
                    'yesterday_close': 645.05,
                    'current_vs_yesterday_high': -1.17,
                    'current_vs_yesterday_low': -0.45
                },
                'data_source': 'Proxy Fallback',
                'error': None
            }

    def calculate_price_targets_enhanced(self, current_price: float) -> Dict:
        """Enhanced price targets with options integration and proxy fallbacks"""
        analysis = self.get_final_decision()
        
        # Get volatility data with fallbacks
        volatility = atr_proxy = 1.2
        
        try:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            tradier_hist = self.api.get_historical_quotes('SPY', '1day', start_date)
            
            if tradier_hist and 'history' in tradier_hist:
                hist_data = tradier_hist['history']['day']
                if isinstance(hist_data, dict):
                    hist_data = [hist_data]
                
                if len(hist_data) > 5:
                    closes = [float(d['close']) for d in hist_data]
                    highs = [float(d['high']) for d in hist_data]
                    lows = [float(d['low']) for d in hist_data]
                    
                    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                    volatility = np.std(returns) * 100 if returns else 1.2
                    
                    atr_values = [(highs[i] - lows[i]) for i in range(len(highs))]
                    atr_proxy = np.mean(atr_values[-20:]) / current_price * 100 if len(atr_values) >= 20 else 1.0
                    
        except:
            try:
                hist_data = get_cached_yahoo_data('SPY', '30d', '1d')
                if not hist_data.empty and hist_data.shape[0] > 5:
                    daily_returns = hist_data['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * 100
                    atr_proxy = (hist_data['High'] - hist_data['Low']).tail(20).mean() / current_price * 100
            except:
                pass  # Use defaults
        
        # Base move calculation
        base_move = max(
            current_price * volatility / 100,
            current_price * atr_proxy / 100,
            current_price * 0.005
        )
        
        gap_pct = analysis['gap_analysis'].get('gap_pct', 0)
        bullish_points = analysis['bullish_points']
        bearish_points = analysis['bearish_points']
        
        # Gap continuation factor
        gap_continuation_factor = 1.0
        if abs(gap_pct) > 1.0:
            gap_continuation_factor = 1.2 if abs(gap_pct) < 2.0 else 1.1
        
        targets = {
            'upside_target': current_price,
            'downside_target': current_price,
            'upside_probability': 50,
            'downside_probability': 50,
            'reasoning': [],
            'options_suggestions': {}
        }
        
        # Target calculation logic
        if bullish_points >= 8:
            targets['upside_target'] = current_price + (base_move * 1.4)
            targets['downside_target'] = current_price - (base_move * 0.6)
            targets['upside_probability'] = 75
            targets['downside_probability'] = 25
            targets['reasoning'].append("Strong bullish signals suggest larger upside move")
            
        elif bullish_points >= 6:
            targets['upside_target'] = current_price + (base_move * 1.1)
            targets['downside_target'] = current_price - (base_move * 0.7)
            targets['upside_probability'] = 65
            targets['downside_probability'] = 35
            targets['reasoning'].append("Moderate bullish signals")
            
        elif bearish_points >= 8:
            targets['downside_target'] = current_price - (base_move * 1.4 * gap_continuation_factor)
            targets['upside_target'] = current_price + (base_move * 0.6)
            targets['upside_probability'] = 25
            targets['downside_probability'] = 75
            targets['reasoning'].append("Strong bearish signals suggest larger downside move")
            
        elif bearish_points >= 6:
            targets['downside_target'] = current_price - (base_move * 1.1 * gap_continuation_factor)
            targets['upside_target'] = current_price + (base_move * 0.7)
            targets['upside_probability'] = 35
            targets['downside_probability'] = 65
            targets['reasoning'].append("Moderate bearish signals")
            
        else:
            targets['upside_target'] = current_price + (base_move * 0.8)
            targets['downside_target'] = current_price - (base_move * 0.8)
            targets['reasoning'].append("Mixed signals - wider neutral range")
        
        # Add gap-specific reasoning
        if gap_pct > 1.5:
            targets['reasoning'].append(f"Large {gap_pct:.1f}% gap up - expect initial continuation then potential reversal")
            if bearish_points < bullish_points:
                targets['downside_target'] *= 0.9
        elif gap_pct < -1.5:
            targets['reasoning'].append(f"Large {gap_pct:.1f}% gap down - expect initial continuation then potential bounce")
            if bullish_points < bearish_points:
                targets['upside_target'] *= 0.9
        
        # Add VWAP reasoning
        vwap_status = analysis['gap_analysis'].get('vwap_status', 'UNKNOWN')
        if 'ABOVE' in vwap_status:
            targets['reasoning'].append("Price above VWAP supports upside bias")
        elif 'BELOW' in vwap_status:
            targets['reasoning'].append("Price below VWAP supports downside bias")
        
        # Add options suggestions
        targets['options_suggestions'] = self._generate_options_suggestions(
            current_price, targets, analysis['decision']
        )
        
        return targets

    def _generate_options_suggestions(self, current_price: float, targets: Dict, decision: str) -> Dict:
        """Generate options suggestions with Tradier data when available"""
        suggestions = {
            'calls': [],
            'puts': [],
            'recommended_strategy': '',
            'risk_warnings': [],
            'tradier_data_available': False
        }
        
        # Try to get actual options data from Tradier
        try:
            options_data = self.get_options_data_tradier_only()
            if not options_data.get('error') and options_data.get('processed'):
                suggestions['tradier_data_available'] = True
                
                calls = options_data['processed']['calls']
                puts = options_data['processed']['puts']
                
                reasonable_calls = [c for c in calls if abs(float(c['strike']) - current_price) <= 10]
                reasonable_puts = [p for p in puts if abs(float(p['strike']) - current_price) <= 10]
                
                if 'LONG' in decision and reasonable_calls:
                    atm_calls = [c for c in reasonable_calls if abs(float(c['strike']) - current_price) <= 1]
                    
                    if atm_calls:
                        best_call = min(atm_calls, key=lambda x: abs(float(x['strike']) - current_price))
                        suggestions['recommended_strategy'] = f"ATM Call ${best_call['strike']} - Bid: ${best_call.get('bid', 'N/A')}, Ask: ${best_call.get('ask', 'N/A')}"
                
                elif 'SHORT' in decision and reasonable_puts:
                    atm_puts = [p for p in reasonable_puts if abs(float(p['strike']) - current_price) <= 1]
                    
                    if atm_puts:
                        best_put = min(atm_puts, key=lambda x: abs(float(x['strike']) - current_price))
                        suggestions['recommended_strategy'] = f"ATM Put ${best_put['strike']} - Bid: ${best_put.get('bid', 'N/A')}, Ask: ${best_put.get('ask', 'N/A')}"
                        
        except:
            pass  # Use fallback suggestions
        
        # Fallback to theoretical suggestions
        if not suggestions['tradier_data_available']:
            if 'LONG' in decision:
                itm_strike = int(current_price) - 1
                atm_strike = int(current_price + 0.5)
                otm_strike = int(targets['upside_target'])
                
                suggestions['calls'] = [
                    {'strike': itm_strike, 'type': 'ITM', 'risk': 'LOWER', 'reward': 'MODERATE'},
                    {'strike': atm_strike, 'type': 'ATM', 'risk': 'MODERATE', 'reward': 'HIGH'},
                    {'strike': otm_strike, 'type': 'OTM', 'risk': 'HIGH', 'reward': 'VERY HIGH'}
                ]
                suggestions['recommended_strategy'] = f"ATM Call ${atm_strike} (Theoretical - no live data)"
                
            elif 'SHORT' in decision:
                itm_strike = int(current_price) + 1
                atm_strike = int(current_price - 0.5)
                otm_strike = int(targets['downside_target'])
                
                suggestions['puts'] = [
                    {'strike': itm_strike, 'type': 'ITM', 'risk': 'LOWER', 'reward': 'MODERATE'},
                    {'strike': atm_strike, 'type': 'ATM', 'risk': 'MODERATE', 'reward': 'HIGH'},
                    {'strike': otm_strike, 'type': 'OTM', 'risk': 'HIGH', 'reward': 'VERY HIGH'}
                ]
                suggestions['recommended_strategy'] = f"ATM Put ${atm_strike} (Theoretical - no live data)"
        
        # Universal risk warnings
        suggestions['risk_warnings'] = [
            "Set stop loss at 50% of premium paid",
            "Close position 90 minutes before market close",
            "Watch for volume confirmation on any breakout",
            f"Current implied volatility may be {'HIGH' if abs(gap_pct := targets.get('gap_pct', 0)) > 1 else 'NORMAL'}",
            "Use limit orders only - never market orders on 0DTE options"
        ]
        
        return suggestions

    def get_options_data_tradier_only(self, symbol: str = 'SPY') -> Dict:
        """Get options data - Tradier ONLY"""
        options_data = {
            'chains': {},
            'expirations': [],
            'source': 'Tradier',
            'error': None
        }
        
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            chain = self.api.get_options_chain(symbol, today)
            
            if chain and 'options' in chain:
                options_data['chains'] = chain['options']
                
                if 'option' in chain['options']:
                    option_list = chain['options']['option']
                    if not isinstance(option_list, list):
                        option_list = [option_list]
                    
                    calls = [opt for opt in option_list if opt.get('option_type') == 'call']
                    puts = [opt for opt in option_list if opt.get('option_type') == 'put']
                    
                    options_data['processed'] = {
                        'calls': calls,
                        'puts': puts,
                        'count': len(calls) + len(puts)
                    }
                else:
                    options_data['error'] = 'No option data in response'
            else:
                options_data['error'] = 'No options data available from Tradier for today'
                
        except Exception as e:
            options_data['error'] = f"Tradier options error: {str(e)}"
        
        return options_data

    def get_final_decision(self) -> Dict:
        """Final trading decision with transparent points breakdown"""
        gap_analysis = self.calculate_gap_analysis()
        internals = self.analyze_market_internals_enhanced()
        sectors = self.analyze_sectors_enhanced()
        technicals = self.analyze_technicals_enhanced()
        
        # Simplified volatility analysis
        volatility = {'total_points': 0.5}
        
        # Calculate points with transparency
        gap_points = gap_analysis.get('total_points', 0)
        internals_points = internals.get('total_points', 0)
        sectors_points = sectors.get('total_points', 0)
        technicals_points = technicals.get('total_points', 0)
        volatility_points = volatility['total_points']
        
        total_bullish_points = (
            max(0, gap_points) + 
            max(0, internals_points) + 
            max(0, volatility_points) +
            max(0, sectors_points) +
            max(0, technicals_points)
        )
        
        total_bearish_points = (
            max(0, -gap_points) + 
            max(0, -internals_points) + 
            max(0, -volatility_points) +
            max(0, -sectors_points) +
            max(0, -technicals_points)
        )
        
        # Decision logic with transparency
        decision_breakdown = {
            'gap_contribution': gap_points,
            'internals_contribution': internals_points,
            'sectors_contribution': sectors_points,
            'technicals_contribution': technicals_points,
            'volatility_contribution': volatility_points,
            'bullish_total': total_bullish_points,
            'bearish_total': total_bearish_points,
            'decision_logic': ''
        }
        
        # Final decision
        if total_bullish_points >= 8:
            decision = 'STRONG LONG'
            confidence = 'HIGH'
            decision_breakdown['decision_logic'] = f"Bullish points {total_bullish_points:.1f} ≥ 8.0 = STRONG LONG"
        elif total_bullish_points >= 6:
            decision = 'MODERATE LONG'
            confidence = 'MEDIUM'
            decision_breakdown['decision_logic'] = f"Bullish points {total_bullish_points:.1f} ≥ 6.0 = MODERATE LONG"
        elif total_bearish_points >= 8:
            decision = 'STRONG SHORT'
            confidence = 'HIGH'
            decision_breakdown['decision_logic'] = f"Bearish points {total_bearish_points:.1f} ≥ 8.0 = STRONG SHORT"
        elif total_bearish_points >= 6:
            decision = 'MODERATE SHORT'
            confidence = 'MEDIUM'
            decision_breakdown['decision_logic'] = f"Bearish points {total_bearish_points:.1f} ≥ 6.0 = MODERATE SHORT"
        else:
            decision = 'NO TRADE'
            confidence = 'LOW'
            decision_breakdown['decision_logic'] = f"Neither bullish ({total_bullish_points:.1f}) nor bearish ({total_bearish_points:.1f}) points reach 6.0 threshold = NO TRADE"
        
        return {
            'decision': decision,
            'confidence': confidence,
            'bullish_points': total_bullish_points,
            'bearish_points': total_bearish_points,
            'gap_analysis': gap_analysis,
            'internals': internals,
            'volatility': volatility,
            'sectors_enhanced': sectors,
            'technicals_enhanced': technicals,
            'decision_breakdown': decision_breakdown  # NEW: Transparent decision logic
        }

def display_points_breakdown_ui(analysis: Dict):
    """Display transparent points breakdown"""
    st.markdown("---")
    st.header("🔍 Points System Breakdown")
    
    # Overall decision breakdown
    decision_breakdown = analysis.get('decision_breakdown', {})
    
    with st.expander("📊 Decision Logic Breakdown", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Bullish Points")
            st.metric("Gap Analysis", f"{max(0, decision_breakdown.get('gap_contribution', 0)):+.1f}")
            st.metric("Market Internals", f"{max(0, decision_breakdown.get('internals_contribution', 0)):+.1f}")
            st.metric("Sector Leadership", f"{max(0, decision_breakdown.get('sectors_contribution', 0)):+.1f}")
            st.metric("Technical Analysis", f"{max(0, decision_breakdown.get('technicals_contribution', 0)):+.1f}")
            st.metric("Volatility Factor", f"{decision_breakdown.get('volatility_contribution', 0):+.1f}")
            st.metric("**TOTAL BULLISH**", f"**{decision_breakdown.get('bullish_total', 0):.1f}**")
        
        with col2:
            st.markdown("### 🔴 Bearish Points")
            st.metric("Gap Analysis", f"{max(0, -decision_breakdown.get('gap_contribution', 0)):+.1f}")
            st.metric("Market Internals", f"{max(0, -decision_breakdown.get('internals_contribution', 0)):+.1f}")
            st.metric("Sector Leadership", f"{max(0, -decision_breakdown.get('sectors_contribution', 0)):+.1f}")
            st.metric("Technical Analysis", f"{max(0, -decision_breakdown.get('technicals_contribution', 0)):+.1f}")
            st.metric("Volatility Factor", f"0.0")
            st.metric("**TOTAL BEARISH**", f"**{decision_breakdown.get('bearish_total', 0):.1f}**")
        
        st.info(f"**Decision Logic:** {decision_breakdown.get('decision_logic', 'Logic not available')}")
    
    # Detailed component breakdowns
    col1, col2 = st.columns(2)
    
    with col1:
        # Gap Analysis Details
        gap_breakdown = analysis['gap_analysis'].get('points_breakdown', {})
        if gap_breakdown:
            with st.expander("📊 Gap Analysis Points Detail"):
                for component, data in gap_breakdown.items():
                    if isinstance(data, dict) and 'points' in data:
                        st.write(f"**{component.replace('_', ' ').title()}:** {data['points']:+.1f} pts")
                        if 'reason' in data:
                            st.caption(data['reason'])
        
        # Internals Details
        internals_breakdown = analysis['internals'].get('points_breakdown', {})
        if internals_breakdown:
            with st.expander("🏛️ Market Internals Points Detail"):
                for component, data in internals_breakdown.items():
                    if isinstance(data, dict) and 'points' in data:
                        st.write(f"**{component.upper()}:** {data['points']:+.1f} pts")
                        if 'reason' in data:
                            st.caption(data['reason'])
    
    with col2:
        # Sectors Details
        sectors_breakdown = analysis['sectors_enhanced'].get('points_breakdown', {})
        if sectors_breakdown:
            with st.expander("🏢 Sector Analysis Points Detail"):
                st.write(f"**Total Weighted Points:** {sectors_breakdown.get('total_weighted_points', 0):+.2f}")
                st.caption(sectors_breakdown.get('leadership_calculation', ''))
                st.caption(sectors_breakdown.get('rotation_logic', ''))
        
        # Technicals Details
        technicals_breakdown = analysis['technicals_enhanced'].get('points_breakdown', {})
        if technicals_breakdown:
            with st.expander("📈 Technical Analysis Points Detail"):
                for component, data in technicals_breakdown.items():
                    if isinstance(data, dict) and 'points' in data:
                        st.write(f"**{component.replace('_', ' ').title()}:** {data['points']:+.1f} pts")
                        if 'reason' in data:
                            st.caption(data['reason'])

def display_data_sources_info():
    """Show users what data comes from where"""
    with st.expander("📊 Data Sources & Reliability", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔴 Tradier API (Primary)")
            st.markdown("""
            ✅ **Real-time SPY quotes**
            ✅ **Live sector ETF data** 
            ✅ **Historical OHLCV data**
            ✅ **Options chains with Greeks**
            ✅ **Volume & trading data**
            ✅ **Market hours status**
            
            *Most reliable for trading decisions*
            """)
        
        with col2:
            st.markdown("### 🟡 Yahoo Finance (Fallback)")
            st.markdown("""
            ⚠️ **Market internals proxies** (Tradier unavailable)
            ⚠️ **Backup price data** (if Tradier fails)
            ⚠️ **Historical data backup**
            
            ❌ **No real options data**
            ❌ **No real market internals**
            ❌ **Can be rate limited**
            
            *Used when Tradier unavailable*
            """)
        
        with col3:
            st.markdown("### 🟢 Proxy Estimates (Final Fallback)")
            st.markdown("""
            📊 **Reasonable market estimates**
            📊 **Based on typical conditions**
            📊 **Ensures system always works**
            📊 **Transparent about limitations**
            
            ✅ **Never fails completely**
            ✅ **Maintains functionality**
            
            *Used when all APIs fail*
            """)

def display_exit_signals_ui(exit_manager, entry_data=None):
    """UI component for exit signals with proper data sourcing"""
    
    st.markdown("---")
    st.header("🚪 EXIT SIGNAL DASHBOARD")
    
    if not entry_data:
        st.info("""
        **👋 No Active Position Detected**
        
        To get exit signals, you need to track your entry:
        • Entry time
        • Entry price  
        • Direction (LONG/SHORT)
        • Target levels
        
        *Use the form below to input your trade details*
        """)
        
        # Manual trade entry form
        with st.form("trade_entry_form"):
            st.subheader("📝 Enter Your Trade Details")
            
            col1, col2 = st.columns(2)
            with col1:
                entry_direction = st.selectbox("Direction", ["STRONG LONG", "MODERATE LONG", "STRONG SHORT", "MODERATE SHORT"])
                entry_price = st.number_input("Entry Price ($)", value=640.0, step=0.01)
                
            with col2:
                entry_time_input = st.time_input("Entry Time (EDT)", value=datetime.now().time())
                upside_target = st.number_input("Upside Target ($)", value=642.0, step=0.01)
                downside_target = st.number_input("Downside Target ($)", value=638.0, step=0.01)
            
            submitted = st.form_submit_button("🎯 Start Exit Tracking")
            
            if submitted:
                today = datetime.now(pytz.timezone('US/Eastern')).date()
                entry_datetime = datetime.combine(today, entry_time_input).replace(tzinfo=pytz.timezone('US/Eastern'))
                
                st.session_state['active_trade'] = {
                    'entry_decision': entry_direction,
                    'entry_price': entry_price,
                    'entry_time': entry_datetime,
                    'targets': {
                        'upside_target': upside_target,
                        'downside_target': downside_target
                    }
                }
                st.rerun()
                
    else:
        # Show exit signals for active trade
        try:
            # Get current SPY price using proper hierarchy
            current_spy = 640.27  # Fallback
            price_source = "Static Fallback"
            
            # Try Tradier first
            try:
                spy_quote = exit_manager.api.get_quote('SPY')
                if spy_quote and spy_quote.get('last'):
                    current_spy = float(spy_quote['last'])
                    price_source = "Tradier"
            except:
                # Fallback to Yahoo
                try:
                    spy_current = get_cached_yahoo_data('SPY', '1d', '1m')
                    if not spy_current.empty:
                        current_spy = spy_current['Close'].iloc[-1]
                        price_source = "Yahoo (Fallback)"
                except:
                    pass  # Use static fallback
            
            exit_signals = exit_manager.get_exit_signals(
                entry_data['entry_decision'],
                entry_data['entry_price'], 
                current_spy,
                entry_data['entry_time'],
                entry_data['targets']
            )
            
            # Main exit signal display
            signal_color = {
                'IMMEDIATE EXIT': '#dc3545',
                'STRONG EXIT': '#fd7e14',
                'CONSIDER EXIT': '#ffc107',
                'HOLD': '#28a745'
            }.get(exit_signals['primary_signal'], '#6c757d')
            
            st.markdown(f"""
            <div style="text-align: center; padding: 25px; background: {signal_color}20; 
                        border: 3px solid {signal_color}; border-radius: 15px; margin: 20px 0;">
                <h1 style="color: {signal_color}; margin: 0;">
                    {'🚨' if exit_signals['should_exit'] else '✋' if exit_signals['primary_signal'] == 'CONSIDER EXIT' else '✅'}
                </h1>
                <h2 style="color: {signal_color}; margin: 10px 0;">{exit_signals['primary_signal']}</h2>
                <h3 style="margin: 10px 0;">Urgency: {exit_signals['urgency']}</h3>
                <p style="font-size: 1.5em; margin: 15px 0;">
                    <strong>P&L: {exit_signals['profit_loss_pct']:+.1f}%</strong>
                </p>
                <p style="font-size: 0.9em; margin: 5px 0; opacity: 0.8;">
                    Price Source: {price_source}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed breakdown
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Entry Price", f"${entry_data['entry_price']:.2f}")
                st.metric("Current Price", f"${current_spy:.2f}")
                
            with col2:
                st.metric("Profit/Loss", f"{exit_signals['profit_loss_pct']:+.1f}%")
                st.metric("Exit Score", f"{exit_signals['exit_score']}/10")
                
            with col3:
                time_in_trade = (datetime.now(pytz.timezone('US/Eastern')) - entry_data['entry_time']).total_seconds() / 3600
                st.metric("Time in Trade", f"{time_in_trade:.1f} hours")
                st.metric("Direction", entry_data['entry_decision'])
            
            # Exit reasons
            if exit_signals['reasons']:
                st.subheader("📋 Exit Reasons:")
                for reason in exit_signals['reasons']:
                    st.write(f"• {reason}")
            
            # Time warnings (critical for 0DTE)
            if exit_signals['time_warnings']:
                st.subheader("⏰ Time Warnings:")
                for warning in exit_signals['time_warnings']:
                    st.warning(warning)
            
            # Technical exits
            if exit_signals['technical_exits']:
                st.subheader("📈 Technical Signals:")
                for signal in exit_signals['technical_exits']:
                    st.write(f"• {signal}")
            
        except Exception as e:
            st.error(f"Error calculating exit signals: {str(e)}")

def main():
    st.title("📈 SPY 0DTE Gap Trading Dashboard")
    st.markdown("*Tradier PRIMARY + Smart Yahoo Fallbacks + Exit Signals*")
    
    # Show data sources info
    display_data_sources_info()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ API Configuration")
        
        api_token = st.text_input(
            "Tradier API Token",
            type="password",
            help="Enter your Tradier API token (Production recommended)"
        )
        
        sandbox_mode = st.checkbox(
            "Use Sandbox Mode", 
            value=False, 
            help="Uncheck for production API (recommended for real data)"
        )
        
        # API Connection Test
        if api_token:
            api = TradierAPI(api_token, sandbox_mode)
            connection_ok, connection_msg = api.test_connection()
            
            if connection_ok:
                st.success(connection_msg)
            else:
                st.error(connection_msg)
        
        st.markdown("---")
        st.subheader("📊 Settings")
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        st.markdown("---")
        st.subheader("💰 Trading Preferences")
        
        # Trading method selection
        trade_method = st.radio(
            "Trading Method:",
            ["Options Contracts", "Stock/ETF"],
            help="Select whether you're trading options or the underlying"
        )
        
        if trade_method == "Options Contracts":
            st.markdown("**📋 Tradier Options Integration:**")
            st.markdown("""
            ✅ **Live options chain data**
            ✅ **Real bid/ask spreads**
            ✅ **Actual volume & open interest**
            ✅ **Greeks (Delta, Theta, Gamma)**
            ✅ **Quality scoring & warnings**
            ✅ **Fair value estimates**
            """)
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["Conservative (30% max loss)", "Moderate (50% max loss)", "Aggressive (70% max loss)"],
                index=1,
                help="Maximum % of option premium you're willing to lose"
            )
        
        st.markdown("---")
        if st.button("🔄 Refresh Data", type="primary", key="refresh_btn"):
            st.rerun()
    
    if not api_token:
        st.warning("⚠️ Please enter your Tradier API token in the sidebar to continue.")
        st.info("""
        **How to get your Tradier API Token:**
        1. Sign up at [Tradier](https://tradier.com)
        2. Go to Account Settings → API Access
        3. Generate a new API token
        4. **Important**: Use Production API (not Sandbox) for real market data
        5. Enter the token in the sidebar
        """)
        return
    
    # Initialize analyzers
    api = TradierAPI(api_token, sandbox_mode)
    gap_analyzer = GapTradingAnalyzer(api)
    exit_manager = ExitSignalManager(api)
    
    # Get SPY current price with proper hierarchy
    current_spy = 640.27  # Fallback
    price_source = "Static Fallback"
    
    try:
        # PRIMARY: Tradier
        spy_quotes = api.get_quote('SPY')
        if spy_quotes and spy_quotes.get('last'):
            current_spy = float(spy_quotes['last'])
            price_source = "Tradier"
    except:
        # FALLBACK: Yahoo
        try:
            spy_data = get_cached_yahoo_data('SPY', '1d', '1m')
            if not spy_data.empty:
                current_spy = spy_data['Close'].iloc[-1]
                price_source = "Yahoo (Fallback)"
        except:
            st.error("Unable to fetch current SPY price from both Tradier and Yahoo - using last known price $640.27")
    
    # Current time and trading window
    trading_window_ok, window_message = gap_analyzer.check_trading_window_enhanced()
    current_time = gap_analyzer.current_time.strftime("%Y-%m-%d %H:%M:%S ET")
    
    # Header info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info(f"🕐 Current Time: {current_time}")
    with col2:
        if trading_window_ok:
            st.success(f"✅ Trading Window Open")
        else:
            st.warning(f"⚠️ {window_message.split('(')[0]}")
    with col3:
        api_status = "Production" if not sandbox_mode else "Sandbox"
        st.info(f"📡 API: {api_status}")
        st.caption(f"Price: {price_source}")
    
    # Auto-refresh logic
    if auto_refresh and trading_window_ok:
        time.sleep(30)
        st.rerun()
    
    # Get analysis and calculate targets
    with st.spinner("🔍 Analyzing market conditions..."):
        analysis = gap_analyzer.get_final_decision()
        price_targets = gap_analyzer.calculate_price_targets_enhanced(current_spy)
    
    # Main Decision Display
    st.markdown("---")
    decision_color = {
        'STRONG LONG': '🟢',
        'MODERATE LONG': '🟡', 
        'STRONG SHORT': '🔴',
        'MODERATE SHORT': '🟠',
        'NO TRADE': '⚪'
    }.get(analysis['decision'], '⚪')
    
    decision_bg_color = {
        'STRONG LONG': '#d4edda',
        'MODERATE LONG': '#fff3cd', 
        'STRONG SHORT': '#f8d7da',
        'MODERATE SHORT': '#ffeaa7',
        'NO TRADE': '#e2e3e5'
    }.get(analysis['decision'], '#e2e3e5')
    
    st.markdown(f"""
    <div style="text-align: center; padding: 30px; background: {decision_bg_color}; 
                border-radius: 15px; border: 3px solid #007bff; margin: 20px 0;">
        <h1 style="margin: 0; font-size: 4em; color: #333;">{decision_color}</h1>
        <h2 style="margin: 10px 0; color: #333;">{analysis['decision']}</h2>
        <h3 style="margin: 10px 0; color: #666;">Confidence: {analysis['confidence']}</h3>
        <p style="font-size: 1.3em; margin: 15px 0; color: #333;">
            <strong>Bullish Points: {analysis['bullish_points']:.1f} | Bearish Points: {analysis['bearish_points']:.1f}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System-Calculated Price Targets
    st.markdown("---")
    st.header("🎯 System-Calculated Price Targets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current SPY", 
            f"${current_spy:.2f}",
            delta=f"Source: {price_source}"
        )
    
    with col2:
        upside_move = price_targets['upside_target'] - current_spy
        st.metric(
            "Upside Target", 
            f"${price_targets['upside_target']:.2f}",
            delta=f"+${upside_move:.2f} ({price_targets['upside_probability']}% prob)"
        )
    
    with col3:
        downside_move = current_spy - price_targets['downside_target']
        st.metric(
            "Downside Target", 
            f"${price_targets['downside_target']:.2f}",
            delta=f"-${downside_move:.2f} ({price_targets['downside_probability']}% prob)"
        )
    
    # Show target reasoning
    with st.expander("📋 Target Calculation Reasoning", expanded=False):
        for reason in price_targets['reasoning']:
            st.write(f"• {reason}")
    
    # Options Suggestions (if options trading selected)
    if trade_method == "Options Contracts" and price_targets['options_suggestions']:
        st.markdown("---")
        st.header("📋 Options Recommendations")
        
        options_data = price_targets['options_suggestions']
        
        if options_data.get('tradier_data_available'):
            st.success("✅ **Live Tradier Options Data Available**")
        else:
            st.warning("⚠️ **Using Theoretical Data - Live options unavailable**")
        
        if options_data['recommended_strategy']:
            st.info(f"🎯 **Recommended:** {options_data['recommended_strategy']}")
        
        if options_data['calls']:
            st.subheader("📈 Call Options:")
            calls_df = pd.DataFrame(options_data['calls'])
            st.dataframe(calls_df, hide_index=True, use_container_width=True)
        
        if options_data['puts']:
            st.subheader("📉 Put Options:")
            puts_df = pd.DataFrame(options_data['puts'])
            st.dataframe(puts_df, hide_index=True, use_container_width=True)
        
        if options_data['risk_warnings']:
            st.warning("⚠️ **Risk Management:**")
            for warning in options_data['risk_warnings']:
                st.write(f"• {warning}")
    
    # Transparent Points Breakdown
    display_points_breakdown_ui(analysis)
    
    # Market Analysis Details with proper data source attribution
    st.markdown("---")
    st.header("📋 Market Analysis Breakdown")
    
    # Gap Analysis Detail
    with st.expander("📊 Gap Analysis - Statistical Edge Detection", expanded=False):
        gap = analysis['gap_analysis']
        
        if gap.get('error'):
            st.error(f"⚠️ {gap['error']}")
        else:
            st.info(f"📊 **Data Source:** {gap.get('data_source', 'Unknown')}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Gap %", f"{gap['gap_pct']:.2f}%")
                st.caption(f"Category: {gap['gap_size_category']}")
                
            with col2:
                st.metric("VWAP Distance", f"{gap['vwap_distance_pct']:.3f}%")
                st.caption(gap['vwap_status'])
                
            with col3:
                st.metric("Volume Surge Ratio", f"{gap['volume_surge_ratio']:.2f}x")
                st.caption("vs average")
                    
            with col4:
                st.metric("Total Gap Points", f"{gap['total_points']:.1f}")
                st.caption("Weighted score")
    
    # Market Internals Detail
    with st.expander("🏛️ Market Internals - Proxy Calculations", expanded=False):
        internals = analysis['internals']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("$TICK Proxy", f"{internals['tick']['value']:,.0f}")
            st.caption(f"Signal: {internals['tick']['signal']}")
                
        with col2:
            st.metric("$TRIN Proxy", f"{internals['trin']['value']:.3f}")
            st.caption(f"Signal: {internals['trin']['signal']}")
                
        with col3:
            st.metric("NYAD Proxy", f"{internals['nyad']['value']:,.0f}")
            st.caption(f"Signal: {internals['nyad']['signal']}")
            
        with col4:
            st.metric("Volume Flow", f"{internals['vold']['value']:.2f}")
            st.caption(f"Signal: {internals['vold']['signal']}")
    
    # Sector Analysis
    with st.expander("🏢 Sector Leadership Analysis", expanded=False):
        sectors = analysis.get('sectors_enhanced', {'error': 'Sector data not available'})
        
        if sectors.get('error'):
            st.warning(f"⚠️ {sectors['error']}")
        else:
            st.info(f"📊 **SPY Data Source:** {sectors.get('spy_source', 'Unknown')}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Leadership Score", f"{sectors['leadership_score']:.2f}")
            with col2:
                st.metric("Rotation Signal", sectors['rotation_signal'])
            with col3:
                st.metric("Total Points", f"{sectors['total_points']:.1f}")
            
            # Individual sector details
            if 'sectors' in sectors:
                st.markdown("**Individual Sector Analysis:**")
                sector_df_data = []
                for symbol, data in sectors['sectors'].items():
                    sector_df_data.append({
                        'Sector': f"{symbol} ({data['name']})",
                        'Change %': f"{data['change_pct']:+.2f}%",
                        'vs SPY': f"{data['relative_strength']:+.2f}%",
                        'Volume': f"{data['volume_ratio']:.1f}x",
                        'Source': data['source']
                    })
                
                if sector_df_data:
                    sector_df = pd.DataFrame(sector_df_data)
                    st.dataframe(sector_df, use_container_width=True, hide_index=True)
    
    # Technical Analysis
    with st.expander("📈 Technical Analysis", expanded=False):
        technicals = analysis.get('technicals_enhanced', {'error': 'Technical data not available'})
        
        if technicals.get('error'):
            st.warning(f"⚠️ {technicals['error']}")
        else:
            st.info(f"📊 **Data Source:** {technicals.get('data_source', 'Unknown')}")
            
            vwap_data = technicals.get('vwap_analysis', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${vwap_data.get('current_price', 0):.2f}")
                st.metric("VWAP", f"${vwap_data.get('vwap', 0):.2f}")
                
            with col2:
                distance = vwap_data.get('distance_pct', 0)
                st.metric("VWAP Distance", f"{distance:+.3f}%")
                st.caption(f"Signal: {vwap_data.get('signal_strength', 'UNKNOWN')}")
                
            with col3:
                st.metric("Technical Points", f"{technicals['total_points']:+.1f}")
                st.caption("Price action score")
    
    # Exit Signals Dashboard
    active_trade = st.session_state.get('active_trade', None)
    display_exit_signals_ui(exit_manager, active_trade)
    
    # Clear trade button
    if active_trade:
        if st.button("🗑️ Clear Active Trade", type="secondary"):
            del st.session_state['active_trade']
            st.rerun()
    
    # Final recommendation
    st.markdown("---")
    st.header("🎯 Final Trading Recommendation")
    
    if analysis['decision'] != 'NO TRADE' and trading_window_ok:
        st.success(f"""
        ✅ **EXECUTE TRADE: {analysis['decision']}**
        
        **Trade Details:**
        • Direction: {analysis['decision']}
        • Confidence: {analysis['confidence']}
        • Current SPY: ${current_spy:.2f} (Source: {price_source})
        • Upside Target: ${price_targets['upside_target']:.2f} ({price_targets['upside_probability']}% prob)
        • Downside Target: ${price_targets['downside_target']:.2f} ({price_targets['downside_probability']}% prob)
        
        **Signal Breakdown:**
        • Gap Analysis: {analysis['gap_analysis'].get('total_points', 0):.1f} points
        • Market Internals: {analysis['internals'].get('total_points', 0):.1f} points  
        • Sector Leadership: {analysis['sectors_enhanced'].get('total_points', 0):.1f} points
        • Technical Analysis: {analysis['technicals_enhanced'].get('total_points', 0):.1f} points
        """)
    elif not trading_window_ok:
        st.info(f"⏰ **MARKET STATUS** - {window_message}")
        
        if "closed" in window_message.lower():
            st.markdown("### 📊 After-Hours Analysis Available")
            st.write("Current analysis is based on last available market data.")
            st.write("This can help you prepare for tomorrow's trading session.")
    else:
        st.error("❌ **NO TRADE** - Insufficient signals or unfavorable conditions")
    
    # Footer
    st.markdown("---")
    st.caption(f"""
    Dashboard last updated: {current_time} | API Mode: {'Sandbox' if sandbox_mode else 'Production'} | 
    **Data Hierarchy:** Tradier PRIMARY → Yahoo FALLBACK → Static FALLBACK | Exit Signals Active
    """)

if __name__ == "__main__":
    main()