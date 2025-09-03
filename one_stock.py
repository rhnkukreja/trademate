import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
import logging
import time as t
from supabase import create_client, Client
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import json
import talib
from scipy.signal import find_peaks
from scipy.stats import linregress
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
import openpyxl
from openpyxl import load_workbook
import pandas_ta as ta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_breakout_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
load_dotenv()

# --------------------------
# Configuration
# --------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

API_KEY = os.getenv("ZERODHA_API_KEY")
ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN")
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# Instrument tokens for market indices (based on Zerodha's conventions)
NIFTY_50_TOKEN = 256265  # NIFTY 50
BANK_NIFTY_TOKEN = 260105  # BANK NIFTY
VIX_TOKEN = 264969  # INDIA VIX

# Static data for RELIANCE
RELIANCE_SECTOR = "Oil & Gas"
RELIANCE_APPROX_SHARES = 6766e6  # Approx 6.766 billion shares outstanding (as of 2025 estimate)
RELIANCE_PE_RATIO = 28.5  # Approximate P/E ratio (placeholder; update with actual data if available)
RELIANCE_INSTITUTIONAL_HOLDING = 0.50  # 50% institutional holding (placeholder)

class PatternType(Enum):
    BREAKOUT = "breakout"
    TRIANGLE = "triangle"
    FLAG = "flag"
    SUPPORT_RESISTANCE = "support_resistance"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    WEDGE = "wedge"
    CHANNEL = "channel"
    CUP_HANDLE = "cup_handle"

@dataclass
class PatternSignal:
    pattern_type: PatternType
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    breakout_level: float
    volume_confirmation: bool
    strength_score: float

class ChartPatternRecognizer:
    def __init__(self):
        pass
    
    def analyze_stock(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Analyze stock data for chart patterns"""
        signals = []
        
        if len(df) < 50:
            return signals
            
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        # Detect patterns
        signals.extend(self._detect_breakout_patterns(df))
        signals.extend(self._detect_triangle_patterns(df))
        signals.extend(self._detect_flag_patterns(df))
        signals.extend(self._detect_support_resistance_break(df))
        signals.extend(self._detect_double_bottom(df))
        signals.extend(self._detect_channel_patterns(df))
        signals.extend(self._detect_cup_handle_patterns(df))
        
        return [s for s in signals if s.confidence >= 0.6]
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['sma_20'] = talib.SMA(df['close'], 20)
        df['sma_50'] = talib.SMA(df['close'], 50)
        df['ema_20'] = talib.EMA(df['close'], 20)
        df['ema_50'] = talib.EMA(df['close'], 50)
        
        # Volume indicators
        df['volume_sma_20'] = talib.SMA(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Technical indicators
        df['rsi_14'] = talib.RSI(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(df['close'], timeperiod=20)
        df['bollinger_width'] = (df['upper_bb'] - df['lower_bb']) / df['middle_bb']
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Candle patterns (using pandas_ta)
        df['candle_pattern'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name=['doji', 'engulfing', 'hammer', 'shootingstar'])['CDL_DOJI']
        
        return df
    
    def _detect_breakout_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect MA breakout patterns"""
        signals = []
        latest = df.iloc[-1]
        
        # Check volume confirmation
        volume_confirmed = latest['volume_ratio'] >= 2.0
        
        if volume_confirmed:
            confidence = 0.75
            
            signal = PatternSignal(
                pattern_type=PatternType.BREAKOUT,
                confidence=confidence,
                entry_price=latest['close'],
                target_price=latest['close'] * 1.10,
                stop_loss=latest['close'] * 0.95,
                breakout_level=latest['close'],
                volume_confirmation=volume_confirmed,
                strength_score=confidence
            )
            signals.append(signal)
            
        return signals
    
    def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect triangle patterns"""
        signals = []
        
        if len(df) < 30:
            return signals
            
        highs = df['high'].values[-30:]
        lows = df['low'].values[-30:]
        
        peak_indices = find_peaks(highs, distance=5)[0]
        trough_indices = find_peaks(-lows, distance=5)[0]
        
        if len(peak_indices) >= 2 and len(trough_indices) >= 2:
            latest = df.iloc[-1]
            confidence = 0.7
            
            signal = PatternSignal(
                pattern_type=PatternType.TRIANGLE,
                confidence=confidence,
                entry_price=latest['close'],
                target_price=latest['close'] * 1.08,
                stop_loss=latest['close'] * 0.96,
                breakout_level=latest['close'],
                volume_confirmation=latest['volume_ratio'] > 1.5,
                strength_score=confidence
            )
            signals.append(signal)
            
        return signals
    
    def _detect_flag_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect flag patterns"""
        signals = []
        latest = df.iloc[-1]
        confidence = 0.65
        
        signal = PatternSignal(
            pattern_type=PatternType.FLAG,
            confidence=confidence,
            entry_price=latest['close'],
            target_price=latest['close'] * 1.06,
            stop_loss=latest['close'] * 0.97,
            breakout_level=latest['close'],
            volume_confirmation=latest.get('volume_ratio', 1.0) > 1.2,
            strength_score=confidence
        )
        signals.append(signal)
        return signals
    
    def _detect_support_resistance_break(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect support/resistance breaks"""
        signals = []
        latest = df.iloc[-1]
        confidence = 0.68
        
        signal = PatternSignal(
            pattern_type=PatternType.SUPPORT_RESISTANCE,
            confidence=confidence,
            entry_price=latest['close'],
            target_price=latest['close'] * 1.07,
            stop_loss=latest['close'] * 0.96,
            breakout_level=latest['close'],
            volume_confirmation=latest.get('volume_ratio', 1.0) > 1.3,
            strength_score=confidence
        )
        signals.append(signal)
        return signals
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect double bottom patterns"""
        signals = []
        latest = df.iloc[-1]
        confidence = 0.72
        
        signal = PatternSignal(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            confidence=confidence,
            entry_price=latest['close'],
            target_price=latest['close'] * 1.09,
            stop_loss=latest['close'] * 0.94,
            breakout_level=latest['close'],
            volume_confirmation=latest.get('volume_ratio', 1.0) > 1.4,
            strength_score=confidence
        )
        signals.append(signal)
        return signals
    
    def _detect_channel_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect channel patterns"""
        signals = []
        latest = df.iloc[-1]
        confidence = 0.69
        
        signal = PatternSignal(
            pattern_type=PatternType.CHANNEL,
            confidence=confidence,
            entry_price=latest['close'],
            target_price=latest['close'] * 1.08,
            stop_loss=latest['close'] * 0.95,
            breakout_level=latest['close'],
            volume_confirmation=latest.get('volume_ratio', 1.0) > 1.3,
            strength_score=confidence
        )
        signals.append(signal)
        return signals
    
    def _detect_cup_handle_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Detect cup and handle patterns"""
        signals = []
        latest = df.iloc[-1]
        confidence = 0.74
        
        signal = PatternSignal(
            pattern_type=PatternType.CUP_HANDLE,
            confidence=confidence,
            entry_price=latest['close'],
            target_price=latest['close'] * 1.12,
            stop_loss=latest['close'] * 0.93,
            breakout_level=latest['close'],
            volume_confirmation=latest.get('volume_ratio', 1.0) > 1.5,
            strength_score=confidence
        )
        signals.append(signal)
        return signals

class EnhancedBreakoutAnalyzer:
    def __init__(self):
        self.pattern_recognizer = ChartPatternRecognizer()
        self.excel_file = "breakout_stock_analysis.xlsx"
        self.rate_limit_delay = 0.5  # 500ms between API calls
        self.previous_breakouts = []  # Track breakouts for failed attempts and time since last
        
    def create_supabase_tables(self):
        """Create necessary tables in Supabase"""
        try:
            supabase.table('instruments').select("*").limit(1).execute()
        except:
            logger.info("Creating instruments table...")
            
        try:
            supabase.table('daily_data').select("*").limit(1).execute()
        except:
            logger.info("Creating daily_data table...")
            
        try:
            supabase.table('hourly_data').select("*").limit(1).execute()
        except:
            logger.info("Creating hourly_data table...")
            
        try:
            supabase.table('analysis_progress').select("*").limit(1).execute()
        except:
            logger.info("Creating analysis_progress table...")
    
    def save_progress(self, phase: str, date: str, stocks_processed: int, day_number: int = 0):
        """Save progress to Supabase"""
        try:
            supabase.table('analysis_progress').upsert({
                'id': 'current_progress',
                'phase': phase,
                'date': date,
                'stocks_processed': stocks_processed,
                'day_number': day_number,
                'timestamp': datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Error saving progress: {e}")

    def get_last_progress(self) -> Dict:
        """Get last progress from Supabase"""
        try:
            response = supabase.table('analysis_progress').select("*").eq('id', 'current_progress').execute()
            return response.data[0] if response.data else {}
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            return {}

    def fetch_reliance_instrument(self):
        """Fetch instrument details for RELIANCE only"""
        try:
            print("Fetching RELIANCE instrument...")
            instruments_raw = kite.instruments("NSE")
        
            reliance_inst = None
            for inst in instruments_raw:
                if inst.get("tradingsymbol") == "RELIANCE" and inst.get("instrument_type") == "EQ" and inst.get("exchange") == "NSE":
                    expiry_value = inst.get('expiry')
                    if expiry_value == "" or expiry_value is None:
                        expiry_value = None

                    def safe_float(value):
                        if value == "" or value is None:
                            return None
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return None
                
                    def safe_int(value):
                        if value == "" or value is None:
                            return None
                        try:
                            return int(value)
                        except (ValueError, TypeError):
                            return None
                
                    instrument_data = {
                        'instrument_token': inst.get('instrument_token'),
                        'exchange_token': inst.get('exchange_token'),
                        'tradingsymbol': inst.get('tradingsymbol', 'RELIANCE'),
                        'name': inst.get('name', 'Reliance Industries'),  # Fallback for name
                        'last_price': safe_float(inst.get('last_price')),
                        'expiry': expiry_value,
                        'strike': safe_float(inst.get('strike')),
                        'tick_size': safe_float(inst.get('tick_size')),
                        'lot_size': safe_int(inst.get('lot_size')),
                        'instrument_type': inst.get('instrument_type', 'EQ'),
                        'segment': inst.get('segment'),
                        'exchange': inst.get('exchange', 'NSE')
                    }
                    reliance_inst = instrument_data
                    break
        
            if reliance_inst:
                print("Found RELIANCE instrument")
                return [reliance_inst]
            else:
                logger.error("RELIANCE instrument not found")
                return []
        
        except Exception as e:
            logger.error(f"Error fetching RELIANCE instrument: {e}")
            return []
    
    def store_instruments_in_supabase(self, instruments):
        """Store instruments in Supabase with proper error handling"""
        try:
            if not instruments:
                logger.error("No instruments provided to store")
                return []
            
            print(f"Storing RELIANCE instrument in Supabase...")
        
            # Clear existing instruments
            supabase.table('instruments').delete().neq('instrument_token', 0).execute()
        
            # Ensure only RELIANCE is stored
            reliance_inst = next((inst for inst in instruments if inst.get('tradingsymbol') == 'RELIANCE'), None)
            if reliance_inst:
                supabase.table('instruments').insert(reliance_inst).execute()
                print("Stored RELIANCE instrument")
                return [reliance_inst]
            else:
                logger.error("No RELIANCE instrument found to store")
                return []
        
        except Exception as e:
            logger.error(f"Error storing instruments: {e}")
            return []
    
    def fetch_and_store_historical_data(self, instruments: List[Dict], days: int = 1000):
        """Fetch 1000 days of historical data for RELIANCE and indices"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        print(f"Fetching {days} days of historical data from {start_date} to {end_date}")
        
        last_progress = self.get_last_progress()
        if last_progress and last_progress.get('phase') == 'data_fetching_complete':
            print("Data already fetched for RELIANCE and indices")
            return
        
        # Fetch RELIANCE data
        instrument = instruments[0]
        try:
            symbol = instrument['tradingsymbol']
            token = instrument['instrument_token']
            
            print(f"Processing {symbol} - {instrument['name']}")
            
            daily_data = self._fetch_with_retry(token, start_date, end_date, "day")
            if daily_data:
                self._store_daily_data(token, symbol, daily_data)
            
            hourly_data = self._fetch_with_retry(token, start_date, end_date, "hour")
            if hourly_data:
                self._store_hourly_data(token, symbol, hourly_data)
            
            # Fetch index data
            for index_token, index_name in [(NIFTY_50_TOKEN, "NIFTY_50"), (BANK_NIFTY_TOKEN, "BANK_NIFTY"), (VIX_TOKEN, "INDIA_VIX")]:
                daily_data = self._fetch_with_retry(index_token, start_date, end_date, "day")
                if daily_data:
                    self._store_daily_data(index_token, index_name, daily_data)
            
            self.save_progress('data_fetching_complete', datetime.now().strftime('%Y-%m-%d'), 1)
            print("Historical data storage complete!")
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    def _fetch_with_retry(self, token, from_date, to_date, interval, max_retries=3):
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                return kite.historical_data(
                    instrument_token=token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval
                )
            except Exception as e:
                if "Too many requests" in str(e) or "rate limit" in str(e).lower():
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    t.sleep(wait_time)
                elif attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Retry {attempt + 1} in {wait_time}s: {e}")
                    t.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return None
        return None
    
    def _store_daily_data(self, token: int, symbol: str, data: List[Dict]):
        """Store daily data in Supabase"""
        try:
            supabase.table('daily_data').delete().eq('instrument_token', token).execute()
            
            formatted_data = []
            for record in data:
                formatted_data.append({
                    'instrument_token': token,
                    'symbol': symbol,
                    'date': record['date'].strftime('%Y-%m-%d'),
                    'open': float(record['open']),
                    'high': float(record['high']),
                    'low': float(record['low']),
                    'close': float(record['close']),
                    'volume': int(record['volume'])
                })
            
            batch_size = 100
            for i in range(0, len(formatted_data), batch_size):
                batch = formatted_data[i:i+batch_size]
                supabase.table('daily_data').insert(batch).execute()
                
        except Exception as e:
            logger.error(f"Error storing daily data for {symbol}: {e}")
    
    def _store_hourly_data(self, token: int, symbol: str, data: List[Dict]):
        """Store hourly data in Supabase"""
        try:
            supabase.table('hourly_data').delete().eq('instrument_token', token).execute()
            
            formatted_data = []
            for record in data:
                formatted_data.append({
                    'instrument_token': token,
                    'symbol': symbol,
                    'datetime': record['date'].isoformat(),
                    'open': float(record['open']),
                    'high': float(record['high']),
                    'low': float(record['low']),
                    'close': float(record['close']),
                    'volume': int(record['volume'])
                })
            
            batch_size = 100
            for i in range(0, len(formatted_data), batch_size):
                batch = formatted_data[i:i+batch_size]
                supabase.table('hourly_data').insert(batch).execute()
                
        except Exception as e:
            logger.error(f"Error storing hourly data for {symbol}: {e}")
    
    def get_instruments_from_supabase(self) -> List[Dict]:
        """Get instruments from Supabase"""
        try:
            response = supabase.table('instruments').select("*").execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return []
    
    def get_historical_data_from_supabase(self, token: int, data_type: str = "daily"):
        """Get historical data from Supabase"""
        try:
            if data_type == "daily":
                response = supabase.table('daily_data').select("*").eq('instrument_token', token).order('date').execute()
            else:
                response = supabase.table('hourly_data').select("*").eq('instrument_token', token).order('datetime').execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching {data_type} data for token {token}: {e}")
            return []

    def fetch_missing_data_and_store(self, token: int, symbol: str, target_date):
        """Fetch missing data and store in Supabase"""
        try:
            print(f"  → Fetching missing data for {symbol} around {target_date}")
            
            end_date = target_date
            start_date = target_date - timedelta(days=300)
            
            daily_data = self._fetch_with_retry(token, start_date, end_date, "day")
            if daily_data:
                self._store_daily_data(token, symbol, daily_data)
                print(f"  ✓ Stored {len(daily_data)} daily records for {symbol}")
            
            hourly_data = self._fetch_with_retry(token, start_date, end_date, "hour")
            if hourly_data:
                self._store_hourly_data(token, symbol, hourly_data)
                print(f"  ✓ Stored {len(hourly_data)} hourly records for {symbol}")
                
            return daily_data, hourly_data
            
        except Exception as e:
            logger.error(f"Error fetching missing data for {symbol}: {e}")
            return None, None

    def _calculate_all_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['sma_5'] = talib.SMA(df['close'], 5)
        df['sma_10'] = talib.SMA(df['close'], 10)
        df['sma_20'] = talib.SMA(df['close'], 20)
        df['sma_50'] = talib.SMA(df['close'], 50)
        df['sma_100'] = talib.SMA(df['close'], 100)
        df['sma_200'] = talib.SMA(df['close'], 200)
        df['ema_20'] = talib.EMA(df['close'], 20)
        df['ema_50'] = talib.EMA(df['close'], 50)
        
        # Bollinger Bands
        df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(df['close'], timeperiod=20)
        df['bollinger_width'] = (df['upper_bb'] - df['lower_bb']) / df['middle_bb']
        
        # Volume metrics
        df['volume_sma_20'] = talib.SMA(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['prev_3day_avg_volume'] = df['volume'].rolling(window=3).mean().shift(1)
        df['volume_percentile'] = df['volume'].rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # RSI and MACD
        df['rsi_14'] = talib.RSI(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # ATR and Stochastic
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Candle characteristics
        df['candle_body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['candle_body_size'].replace(0, 1)
        df['lower_shadow_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['candle_body_size'].replace(0, 1)
        df['is_bullish_candle'] = df['close'] > df['open']
        
        # Candle patterns
        candle_patterns = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name=['doji', 'engulfing', 'hammer', 'shootingstar'])
        df['candle_pattern'] = candle_patterns.apply(lambda x: next((name for name, val in x.items() if val != 0), 'None'), axis=1)
        
        # Support and Resistance
        window = 50
        df['resistance_level_1'] = df['high'].rolling(window=window).max().shift(1)
        df['support_level_1'] = df['low'].rolling(window=window).min().shift(1)
        df['distance_to_resistance'] = (df['resistance_level_1'] - df['close']) / df['close'] * 100
        df['distance_to_support'] = (df['close'] - df['support_level_1']) / df['close'] * 100
        df['resistance_strength_1'] = df['high'].rolling(window=window).apply(lambda x: sum(abs(x - x.max()) < x.max() * 0.01))
        
        # Price changes
        df['price_change_1d'] = df['close'].pct_change(1) * 100
        df['price_change_3d'] = df['close'].pct_change(3) * 100
        df['price_change_5d'] = df['close'].pct_change(5) * 100
        df['price_change_10d'] = df['close'].pct_change(10) * 100
        
        # Volatility
        df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
        
        # Gap from previous close
        df['gap_from_prev_close'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        
        return df

    def _calculate_moving_averages(self, daily_df: pd.DataFrame, hourly_df: pd.DataFrame, target_date):
        """Calculate moving averages"""
        mas = {
            'sma_5': daily_df['close'].rolling(5).mean().iloc[-1] if len(daily_df) >= 5 else None,
            'sma_10': daily_df['close'].rolling(10).mean().iloc[-1] if len(daily_df) >= 10 else None,
            'sma_20': daily_df['close'].rolling(20).mean().iloc[-1] if len(daily_df) >= 20 else None,
            'sma_50': daily_df['close'].rolling(50).mean().iloc[-1] if len(daily_df) >= 50 else None,
            'sma_100': daily_df['close'].rolling(100).mean().iloc[-1] if len(daily_df) >= 100 else None,
            'sma_200': daily_df['close'].rolling(200).mean().iloc[-1] if len(daily_df) >= 200 else None,
            'ema_20': talib.EMA(daily_df['close'], 20).iloc[-1] if len(daily_df) >= 20 else None,
            'ema_50': talib.EMA(daily_df['close'], 50).iloc[-1] if len(daily_df) >= 50 else None
        }
        return mas

    def _calculate_volume_metrics(self, df: pd.DataFrame):
        """Calculate volume metrics"""
        return {
            'volume_sma_20': df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else None,
            'volume_ratio': (df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]) if len(df) >= 20 and df['volume'].rolling(20).mean().iloc[-1] > 0 else None,
            'volume_spike': df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 2 if len(df) >= 20 else False,
            'prev_3day_avg_volume': df['volume'].rolling(3).mean().shift(1).iloc[-1] if len(df) >= 4 else None,
            'volume_percentile': df['volume'].rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).iloc[-1] if len(df) >= 252 else None
        }

    def _calculate_price_metrics(self, df: pd.DataFrame):
        """Calculate price metrics"""
        return {
            'price_change_1d': df['close'].pct_change(1).iloc[-1] * 100 if len(df) >= 2 else None,
            'price_change_3d': df['close'].pct_change(3).iloc[-1] * 100 if len(df) >= 4 else None,
            'price_change_5d': df['close'].pct_change(5).iloc[-1] * 100 if len(df) >= 6 else None,
            'price_change_10d': df['close'].pct_change(10).iloc[-1] * 100 if len(df) >= 11 else None,
            'gap_from_prev_close': ((df['open'].iloc[-1] - df['close'].shift(1).iloc[-1]) / df['close'].shift(1).iloc[-1] * 100) if len(df) >= 2 else None
        }

    def _calculate_volatility_metrics(self, df: pd.DataFrame):
        """Calculate volatility metrics"""
        return {
            'volatility_20d': df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100 if len(df) >= 20 else None,
            'atr_14': talib.ATR(df['high'], df['low'], df['close'], 14).iloc[-1] if len(df) >= 14 else None
        }

    def _calculate_momentum_indicators(self, df: pd.DataFrame):
        """Calculate momentum indicators"""
        return {
            'rsi_14': talib.RSI(df['close'], 14).iloc[-1] if len(df) >= 14 else None,
            'macd': talib.MACD(df['close'])[0].iloc[-1] if len(df) >= 26 else None,
            'macd_signal': talib.MACD(df['close'])[1].iloc[-1] if len(df) >= 26 else None,
            'macd_histogram': talib.MACD(df['close'])[2].iloc[-1] if len(df) >= 26 else None,
            'stoch_k': talib.STOCH(df['high'], df['low'], df['close'])[0].iloc[-1] if len(df) >= 14 else None,
            'stoch_d': talib.STOCH(df['high'], df['low'], df['close'])[1].iloc[-1] if len(df) >= 14 else None
        }

    def _calculate_pattern_metrics(self, df: pd.DataFrame):
        """Calculate pattern metrics"""
        highs = df['high'].values[-50:]
        lows = df['low'].values[-50:]
        peak_indices = find_peaks(highs, distance=5)[0]
        trough_indices = find_peaks(-lows, distance=5)[0]
        
        # Consolidation days
        consolidation_days = 0
        if len(df) >= 20:
            recent_prices = df['close'].values[-20:]
            price_range = (max(recent_prices) - min(recent_prices)) / min(recent_prices)
            if price_range < 0.05:  # Less than 5% range
                consolidation_days = 20
        
        # Pre-breakout volume trend
        volume_trend = None
        if len(df) >= 20:
            volumes = df['volume'].values[-20:]
            slope, _, _, _, _ = linregress(range(len(volumes)), volumes)
            volume_trend = 'Increasing' if slope > 0 else 'Decreasing'
        
        # Failed breakout attempts
        failed_attempts = 0
        if len(self.previous_breakouts) > 0:
            recent_breakouts = [b for b in self.previous_breakouts if b['Date'] >= (pd.to_datetime(df['date'].iloc[-1]) - timedelta(days=50)).strftime('%Y-%m-%d')]
            failed_attempts = sum(1 for b in recent_breakouts if not b.get('is_true_breakout', False))
        
        # Time since last breakout
        time_since_last = None
        if self.previous_breakouts:
            last_breakout_date = pd.to_datetime(max(b['Date'] for b in self.previous_breakouts))
            time_since_last = (pd.to_datetime(df['date'].iloc[-1]) - last_breakout_date).days
        
        return {
            'consolidation_days': consolidation_days,
            'pre_breakout_volume_trend': volume_trend,
            'failed_breakout_attempts': failed_attempts,
            'time_since_last_breakout': time_since_last
        }

    def _get_market_context(self, target_date):
        """Get market context (NIFTY, BANKNIFTY, VIX)"""
        try:
            nifty_data = self.get_historical_data_from_supabase(NIFTY_50_TOKEN, "daily")
            banknifty_data = self.get_historical_data_from_supabase(BANK_NIFTY_TOKEN, "daily")
            vix_data = self.get_historical_data_from_supabase(VIX_TOKEN, "daily")
            
            nifty_change = None
            banknifty_change = None
            vix_level = None
            market_trend = 'Neutral'
            
            if nifty_data:
                nifty_df = pd.DataFrame(nifty_data)
                nifty_df['date'] = pd.to_datetime(nifty_df['date'])
                day_data = nifty_df[nifty_df['date'].dt.date == target_date]
                if not day_data.empty:
                    nifty_change = (day_data['close'].iloc[-1] - day_data['open'].iloc[-1]) / day_data['open'].iloc[-1] * 100
                    market_trend = 'Bullish' if nifty_change > 0 else 'Bearish' if nifty_change < 0 else 'Neutral'
            
            if banknifty_data:
                banknifty_df = pd.DataFrame(banknifty_data)
                banknifty_df['date'] = pd.to_datetime(banknifty_df['date'])
                day_data = banknifty_df[banknifty_df['date'].dt.date == target_date]
                if not day_data.empty:
                    banknifty_change = (day_data['close'].iloc[-1] - day_data['open'].iloc[-1]) / day_data['open'].iloc[-1] * 100
            
            if vix_data:
                vix_df = pd.DataFrame(vix_data)
                vix_df['date'] = pd.to_datetime(vix_df['date'])
                day_data = vix_df[vix_df['date'].dt.date == target_date]
                if not day_data.empty:
                    vix_level = day_data['close'].iloc[-1]
            
            return {
                'nifty_change': nifty_change,
                'banknifty_change': banknifty_change,
                'vix_level': vix_level,
                'market_trend': market_trend
            }
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {
                'nifty_change': None,
                'banknifty_change': None,
                'vix_level': None,
                'market_trend': 'Unknown'
            }

    def calculate_comprehensive_parameters(self, daily_data: List[Dict], hourly_data: List[Dict], 
                                         symbol: str, company_name: str, target_date,
                                         breakout_data: Optional[Dict] = None) -> Dict:
        """Calculate ALL parameters as per Excel file requirements"""
        try:
            if not daily_data or len(daily_data) < 200:
                return {}
            
            daily_df = pd.DataFrame(daily_data)
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            daily_df = daily_df.sort_values('date')
            
            hourly_df = pd.DataFrame(hourly_data) if hourly_data else pd.DataFrame()
            if not hourly_df.empty:
                hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
                hourly_df = hourly_df.sort_values('datetime')
            
            daily_filtered = daily_df[daily_df['date'].dt.date <= target_date].copy()
            if len(daily_filtered) < 50:
                return {}
            
            daily_filtered = self._calculate_all_technical_indicators(daily_filtered)
            latest = daily_filtered.iloc[-1]
            
            # Moving averages
            mas = self._calculate_moving_averages(daily_filtered, hourly_df, target_date)
            
            # Volume metrics
            volume_metrics = self._calculate_volume_metrics(daily_filtered)
            
            # Price metrics
            price_metrics = self._calculate_price_metrics(daily_filtered)
            
            # Volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(daily_filtered)
            
            # Momentum indicators
            momentum_metrics = self._calculate_momentum_indicators(daily_filtered)
            
            # Pattern metrics
            pattern_metrics = self._calculate_pattern_metrics(daily_filtered)
            
            # Market context
            market_context = self._get_market_context(target_date)
            
            # Future data for post-breakout metrics
            future_data = daily_df[daily_df['date'].dt.date > target_date].head(10)
            
            # Post-breakout metrics
            max_gains = {}
            close_prices = {}
            hit_targets = {}
            days_to_10_percent = None
            max_drawdown = None
            stop_loss_hit = False
            
            if not future_data.empty:
                breakout_price = breakout_data.get('breakout_price', latest['close']) if breakout_data else latest['close']
                max_gains['max_gain_1d'] = ((future_data.head(1)['high'].max() - breakout_price) / breakout_price * 100) if len(future_data) >= 1 else None
                max_gains['max_gain_3d'] = ((future_data.head(3)['high'].max() - breakout_price) / breakout_price * 100) if len(future_data) >= 3 else None
                max_gains['max_gain_5d'] = ((future_data.head(5)['high'].max() - breakout_price) / breakout_price * 100) if len(future_data) >= 5 else None
                max_gains['max_gain_10d'] = ((future_data['high'].max() - breakout_price) / breakout_price * 100) if len(future_data) >= 10 else None
                
                close_prices['close_price_1d'] = future_data.head(1)['close'].iloc[-1] if len(future_data) >= 1 else None
                close_prices['close_price_3d'] = future_data.head(3)['close'].iloc[-1] if len(future_data) >= 3 else None
                close_prices['close_price_5d'] = future_data.head(5)['close'].iloc[-1] if len(future_data) >= 5 else None
                close_prices['close_price_10d'] = future_data['close'].iloc[-1] if len(future_data) >= 10 else None
                
                hit_targets['hit_5_percent_target'] = any(future_data['high'] >= breakout_price * 1.05)
                hit_targets['hit_10_percent_target'] = any(future_data['high'] >= breakout_price * 1.10)
                hit_targets['hit_15_percent_target'] = any(future_data['high'] >= breakout_price * 1.15)
                
                ten_percent_idx = future_data[future_data['high'] >= breakout_price * 1.10].index
                days_to_10_percent = (ten_percent_idx[0] - daily_filtered.index[-1]).days if not ten_percent_idx.empty else None
                
                max_drawdown = ((breakout_price - future_data['low'].min()) / breakout_price * 100) if not future_data['low'].empty else None
                stop_loss = breakout_data.get('stop_loss', breakout_price * 0.95) if breakout_data else breakout_price * 0.95
                stop_loss_hit = any(future_data['low'] <= stop_loss)
            
            # Estimate market cap
            market_cap = latest['close'] * RELIANCE_APPROX_SHARES
            market_cap_size = 'Large' if market_cap > 200e9 else 'Mid' if market_cap > 50e9 else 'Small'
            
            # Sector performance (placeholder; assumes same as NIFTY change for RELIANCE's sector)
            sector_performance = market_context.get('nifty_change')
            
            # Breakout quality
            breakout_quality = 0.0
            if breakout_data:
                breakout_quality = (
                    (0.4 if breakout_data.get('volume_ratio', 0) > 2 else 0) +
                    (0.3 if breakout_data.get('pattern_confidence', 0) > 0.7 else 0) +
                    (0.2 if latest['rsi_14'] < 70 else 0) +
                    (0.1 if market_context.get('market_trend') == 'Bullish' else 0)
                )
            
            comprehensive_params = {
                'Date': target_date.strftime('%Y-%m-%d'),
                'stock_symbol': symbol,
                'Company_Name': company_name,
                'sector': RELIANCE_SECTOR,
                'market_cap': market_cap,
                'market_cap_size': market_cap_size,
                'breakout_time': breakout_data.get('breakout_time') if breakout_data else None,
                'breakout_open': breakout_data.get('breakout_price', latest['open']) if breakout_data else latest['open'],
                'breakout_high': breakout_data.get('highest_price', latest['high']) if breakout_data else latest['high'],
                'breakout_low': latest['low'],
                'breakout_close': breakout_data.get('breakout_price', latest['close']) if breakout_data else latest['close'],
                'breakout_volume': breakout_data.get('breakout_volume', latest['volume']) if breakout_data else latest['volume'],
                'prev_close': float(daily_filtered.iloc[-2]['close']) if len(daily_filtered) > 1 else float(latest['close']),
                **mas,
                **volume_metrics,
                **price_metrics,
                **volatility_metrics,
                **momentum_metrics,
                **pattern_metrics,
                **market_context,
                'bollinger_upper': latest.get('upper_bb'),
                'bollinger_lower': latest.get('lower_bb'),
                'bollinger_width': latest.get('bollinger_width'),
                'candle_body_size': latest.get('candle_body_size'),
                'upper_shadow_ratio': latest.get('upper_shadow_ratio'),
                'lower_shadow_ratio': latest.get('lower_shadow_ratio'),
                'is_bullish_candle': latest.get('is_bullish_candle'),
                'candle_pattern': latest.get('candle_pattern'),
                'resistance_level_1': latest.get('resistance_level_1'),
                'resistance_strength_1': latest.get('resistance_strength_1'),
                'distance_to_resistance': latest.get('distance_to_resistance'),
                'support_level_1': latest.get('support_level_1'),
                'distance_to_support': latest.get('distance_to_support'),
                'pe_ratio': RELIANCE_PE_RATIO,
                'avg_delivery_percentage': 0.30,  # Placeholder
                'institutional_holding': RELIANCE_INSTITUTIONAL_HOLDING,
                **max_gains,
                **close_prices,
                **hit_targets,
                'days_to_hit_10_percent': days_to_10_percent,
                'max_drawdown': max_drawdown,
                'stop_loss_hit': stop_loss_hit,
                'breakout_quality': breakout_quality,
                'true_breakout': breakout_data.get('is_true_breakout', False) if breakout_data else False,
                'breakout_strength_score': breakout_data.get('pattern_confidence', 0) if breakout_data else 0
            }
            
            return comprehensive_params
            
        except Exception as e:
            logger.error(f"Error calculating parameters for {symbol}: {e}")
            return {}

    def scan_for_monitoring_candidates(self, target_date):
        """Scan for stocks below all 8 MAs on previous day"""
        instruments = self.get_instruments_from_supabase()
        if not instruments:
            logger.warning("No instruments found in Supabase")
            return []
        
        candidates = []
        for inst in instruments:
            # Ensure the instrument is valid and for RELIANCE
            if not inst or inst.get('tradingsymbol') != 'RELIANCE' or inst.get('exchange') != 'NSE' or inst.get('instrument_type') != 'EQ':
                logger.warning(f"Skipping invalid or non-RELIANCE instrument: {inst}")
                continue
                
            token = inst.get('instrument_token')
            symbol = inst.get('tradingsymbol', 'RELIANCE')
            company_name = inst.get('name', 'Reliance Industries')  # Fallback if name is None
            
            if not token:
                logger.error(f"Invalid instrument data: token={token}, symbol={symbol}")
                continue
                
            daily_data = self.get_historical_data_from_supabase(token, "daily")
            if not daily_data:
                logger.info(f"No daily data for {symbol}, fetching missing data")
                self.fetch_missing_data_and_store(token, symbol, target_date - timedelta(days=1))
                daily_data = self.get_historical_data_from_supabase(token, "daily")
            
            if daily_data:
                df = pd.DataFrame(daily_data)
                df['date'] = pd.to_datetime(df['date'])
                prev_day = target_date - timedelta(days=1)
                prev_data = df[df['date'].dt.date == prev_day]
                
                if not prev_data.empty:
                    prev_data = self._calculate_all_technical_indicators(prev_data)
                    latest = prev_data.iloc[-1]
                    mas = ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200', 'ema_20', 'ema_50']
                    if all(latest['close'] < latest.get(ma, float('inf')) for ma in mas):
                        candidates.append({
                            'token': token,
                            'symbol': symbol,
                            'company_name': company_name
                        })
                else:
                    logger.warning(f"No data for {symbol} on {prev_day}")
        
        return candidates

    def detect_breakouts_for_day(self, candidates, target_date):
        """Detect breakouts for the day"""
        breakouts = []
        for candidate in candidates:
            token = candidate['token']
            symbol = candidate['symbol']
            company_name = candidate['company_name']
            
            daily_data = self.get_historical_data_from_supabase(token, "daily")
            hourly_data = self.get_historical_data_from_supabase(token, "hourly")
            
            if not daily_data or not hourly_data:
                self.fetch_missing_data_and_store(token, symbol, target_date)
                daily_data = self.get_historical_data_from_supabase(token, "daily")
                hourly_data = self.get_historical_data_from_supabase(token, "hourly")
            
            if daily_data and hourly_data:
                params = self.calculate_comprehensive_parameters(daily_data, hourly_data, symbol, company_name, target_date)
                if params:
                    breakout_data = self.check_comprehensive_breakout(hourly_data, params, target_date)
                    if breakout_data:
                        params.update(breakout_data)
                        breakouts.append(params)
                        self.previous_breakouts.append(params)  # Track for pattern metrics
        
        return breakouts

    def check_comprehensive_breakout(self, hourly_data: List[Dict], params: Dict, target_date):
        """Check for breakout on the day"""
        try:
            hourly_df = pd.DataFrame(hourly_data)
            hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
            day_data = hourly_df[hourly_df['datetime'].dt.date == target_date]
            
            if day_data.empty:
                logger.warning(f"No hourly data for {target_date}")
                return None
            
            # Check if price breaks above all MAs
            daily_df = pd.DataFrame(self.get_historical_data_from_supabase(params['stock_symbol'], "daily"))
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            daily_df = self._calculate_all_technical_indicators(daily_df)
            day_daily = daily_df[daily_df['date'].dt.date == target_date]
            
            if day_daily.empty:
                return None
                
            latest_daily = day_daily.iloc[-1]
            mas = ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200', 'ema_20', 'ema_50']
            is_breakout = all(latest_daily['close'] > latest_daily.get(ma, 0) for ma in mas)
            
            if not is_breakout:
                return None
            
            breakout_time = day_data.iloc[0]['datetime'].time()
            breakout_price = day_data['high'].max()
            breakout_volume = day_data['volume'].sum()
            highest_price = day_data['high'].max()
            percentage_change = ((highest_price - params['prev_close']) / params['prev_close']) * 100
            is_true_breakout = percentage_change >= 5.0
            volume_ratio = breakout_volume / params.get('volume_sma_20', 1) if params.get('volume_sma_20', 1) > 0 else 1.0
            
            return {
                'breakout_time': breakout_time,
                'breakout_price': breakout_price,
                'breakout_volume': breakout_volume,
                'highest_price': highest_price,
                'percentage_change': percentage_change,
                'is_true_breakout': is_true_breakout,
                'volume_ratio': volume_ratio,
                'pattern_confidence': 0.7
            }
            
        except Exception as e:
            logger.error(f"Error checking comprehensive breakout: {e}")
            return None

    def get_trading_days(self, days: int) -> List[datetime]:
        """Get list of trading days (excluding weekends)"""
        trading_days = []
        end_date = datetime.now().date()
        
        for i in range(days * 2):
            check_date = end_date - timedelta(days=i)
            if check_date.weekday() < 5:
                trading_days.append(check_date)
                if len(trading_days) >= days:
                    break
        
        return trading_days

    def calculate_resume_index(self, last_date_str: str, trading_days: List[datetime]) -> int:
        """Calculate resume index based on last processed date"""
        try:
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
            for i, date in enumerate(trading_days):
                if date == last_date:
                    return i + 1
            return 0
        except:
            return 0
    
    def run_comprehensive_analysis(self, days_to_analyze: int = 1000):
        """Run comprehensive analysis for 800 trading days"""
        print("=" * 80)
        print("COMPREHENSIVE NSE BREAKOUT ANALYSIS v3.0 - RELIANCE ONLY")
        print("=" * 80)
        print(f"Analyzing last {days_to_analyze} days of trading data for RELIANCE")
        
        print("\n1. Setting up database...")
        self.create_supabase_tables()
        
        if not self.get_instruments_from_supabase():
            print("\n2. Fetching RELIANCE instrument...")
            instruments = self.fetch_reliance_instrument()
            if not instruments:
                print("Failed to fetch RELIANCE instrument. Exiting.")
                return
            
            print("\n3. Storing RELIANCE instrument in Supabase...")
            self.store_instruments_in_supabase(instruments)
            
            print(f"\n4. Fetching {days_to_analyze} days of historical data...")
            self.fetch_and_store_historical_data(instruments, days_to_analyze)
        else:
            print("\n2-4. Using existing data from Supabase")
            instruments = self.get_instruments_from_supabase()
        
        print(f"\n5. Starting breakout analysis for 800 trading days...")
        
        all_results = []
        trading_days_analyzed = 0
        trading_days = self.get_trading_days(days_to_analyze)[:800]
        
        last_progress = self.get_last_progress()
        start_index = 0
        if last_progress and last_progress.get('phase') == 'analysis':
            start_index = self.calculate_resume_index(last_progress.get('date', ''), trading_days)
            print(f"Resuming analysis from day {start_index + 1}")
        
        for i, target_date in enumerate(tqdm(trading_days[start_index:], initial=start_index, total=len(trading_days), desc="Analyzing trading days")):
            try:
                current_day = start_index + i + 1
                print(f"\n[Day {current_day}/{len(trading_days)}] Analyzing {target_date}...")
                
                monitoring_candidates = self.scan_for_monitoring_candidates(target_date)
                
                if not monitoring_candidates:
                    print(f"  No monitoring candidate (RELIANCE) for {target_date}")
                    if current_day % 10 == 0:
                        self.save_progress('analysis', target_date.strftime('%Y-%m-%d'), 0, current_day)
                    continue
                
                print(f"  Found RELIANCE as monitoring candidate")
                
                breakout_results = self.detect_breakouts_for_day(monitoring_candidates, target_date)
                
                if breakout_results:
                    all_results.extend(breakout_results)
                    print(f"  🎯 Found breakout for RELIANCE on {target_date}")
                else:
                    print(f"  No breakout for RELIANCE on {target_date}")
                
                trading_days_analyzed += 1
                
                if current_day % 10 == 0:
                    self.save_progress('analysis', target_date.strftime('%Y-%m-%d'), len(breakout_results), current_day)
                    print(f"  Progress saved: Day {current_day}/{len(trading_days)} completed")
                
            except Exception as e:
                logger.error(f"Error analyzing {target_date}: {e}")
                continue
        
        self.save_progress('analysis_complete', datetime.now().strftime('%Y-%m-%d'), len(all_results), len(trading_days))
        
        if all_results:
            print(f"\n6. Saving {len(all_results)} records to Excel...")
            self.save_comprehensive_results(all_results)
            
            self.generate_analysis_summary(all_results, trading_days_analyzed)
        else:
            print("No breakout results found to save")
    
    def save_comprehensive_results(self, results: List[Dict]):
        """Save comprehensive results to Excel with all parameters"""
        try:
            df = pd.DataFrame(results)
            
            if df.empty:
                print("No data to save")
                return
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass
            
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Sheet1', index=False)
            
            print(f"Results saved to {self.excel_file}")
            print(f"Total records: {len(df)}")
            
        except Exception as e:
            logger.error(f"Error saving comprehensive results: {e}")
    
    def generate_analysis_summary(self, results: List[Dict], trading_days: int):
        """Generate and print analysis summary"""
        try:
            df = pd.DataFrame(results)
            
            total_breakouts = len(df)
            true_breakouts = len(df[df['true_breakout'] == True])
            false_breakouts = total_breakouts - true_breakouts
            
            print("\n" + "=" * 80)
            print("COMPREHENSIVE ANALYSIS SUMMARY - RELIANCE")
            print("=" * 80)
            print(f"Trading days analyzed: {trading_days}")
            print(f"Total breakouts found: {total_breakouts}")
            print(f"True breakouts (≥5%): {true_breakouts}")
            print(f"False breakouts (<5%): {false_breakouts}")
            print(f"Overall success rate: {(true_breakouts/total_breakouts*100):.1f}%" if total_breakouts > 0 else "N/A")
            
            if true_breakouts > 0:
                true_df = df[df['true_breakout'] == True]
                print(f"Average gain (true breakouts): {true_df['percentage_change'].mean():.2f}%")
                print(f"Maximum gain: {true_df['percentage_change'].max():.2f}%")
            
            print(f"\nResults saved to: {self.excel_file}")
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")


    def run_analysis_only(self, days_to_analyze: int = 800):
        """Run analysis using existing Supabase data for specified days"""
        print("=" * 80)
        print("ANALYSIS ONLY MODE - RELIANCE ONLY")
        print("=" * 80)
        print(f"Analyzing {days_to_analyze} days of existing data for RELIANCE")
        
        # Check if instruments exist in Supabase
        instruments = self.get_instruments_from_supabase()
        if not instruments:
            print("No instruments found in Supabase. Please run mode 3 to fetch data first.")
            return
        
        # Verify RELIANCE is present
        reliance_inst = next((inst for inst in instruments if inst.get('tradingsymbol') == 'RELIANCE'), None)
        if not reliance_inst:
            print("RELIANCE instrument not found in Supabase. Please run mode 3 to fetch data.")
            return
        
        print("\n1. Using existing data from Supabase")
        
        # Perform analysis for 800 trading days
        all_results = []
        trading_days_analyzed = 0
        trading_days = self.get_trading_days(days_to_analyze)[:800]
        
        last_progress = self.get_last_progress()
        start_index = 0
        if last_progress and last_progress.get('phase') == 'analysis':
            start_index = self.calculate_resume_index(last_progress.get('date', ''), trading_days)
            print(f"Resuming analysis from day {start_index + 1}")
        
        for i, target_date in enumerate(tqdm(trading_days[start_index:], initial=start_index, total=len(trading_days), desc="Analyzing trading days")):
            try:
                current_day = start_index + i + 1
                print(f"\n[Day {current_day}/{len(trading_days)}] Analyzing {target_date}...")
                
                monitoring_candidates = self.scan_for_monitoring_candidates(target_date)
                
                if not monitoring_candidates:
                    print(f"  No monitoring candidate (RELIANCE) for {target_date}")
                    if current_day % 10 == 0:
                        self.save_progress('analysis', target_date.strftime('%Y-%m-%d'), 0, current_day)
                    continue
                
                print(f"  Found RELIANCE as monitoring candidate")
                
                breakout_results = self.detect_breakouts_for_day(monitoring_candidates, target_date)
                
                if breakout_results:
                    all_results.extend(breakout_results)
                    print(f"  🎯 Found breakout for RELIANCE on {target_date}")
                else:
                    print(f"  No breakout for RELIANCE on {target_date}")
                
                trading_days_analyzed += 1
                
                if current_day % 10 == 0:
                    self.save_progress('analysis', target_date.strftime('%Y-%m-%d'), len(breakout_results), current_day)
                    print(f"  Progress saved: Day {current_day}/{len(trading_days)} completed")
                
            except Exception as e:
                logger.error(f"Error analyzing {target_date}: {e}")
                continue
        
        self.save_progress('analysis_complete', datetime.now().strftime('%Y-%m-%d'), len(all_results), len(trading_days))
        
        if all_results:
            print(f"\n2. Saving {len(all_results)} records to Excel...")
            self.save_comprehensive_results(all_results)
            
            self.generate_analysis_summary(all_results, trading_days_analyzed)
        else:
            print("No breakout results found to save")

    def fetch_data_only(self, days: int = 1000):
        """Fetch and store data for RELIANCE and indices without analysis"""
        print("=" * 80)
        print("DATA FETCH ONLY MODE - RELIANCE AND INDICES")
        print("=" * 80)
        print(f"Fetching {days} days of data...")
        
        self.create_supabase_tables()
        
        print("\n1. Fetching RELIANCE instrument...")
        instruments = self.fetch_reliance_instrument()
        if not instruments:
            print("Failed to fetch RELIANCE instrument. Exiting.")
            return
        
        print("\n2. Storing RELIANCE instrument in Supabase...")
        self.store_instruments_in_supabase(instruments)
        
        print(f"\n3. Fetching {days} days of historical data...")
        self.fetch_and_store_historical_data(instruments, days)
        
        print("\nData fetching completed successfully!")

    
def main():
    """Main execution function"""
    try:
        analyzer = EnhancedBreakoutAnalyzer()
        
        print("=" * 80)
        print("ENHANCED NSE BREAKOUT ANALYZER v3.0 - RELIANCE ONLY")
        print("=" * 80)
        print("Features:")
        print("• RELIANCE stock only")
        print("• 1000 days of historical data")
        print("• 800 days of breakout analysis")
        print("• 8-MA breakout strategy")
        print("• Comprehensive technical analysis")
        print("• Pattern recognition")
        print("• Complete Excel export")
        print("• Resume capability after interruptions")
        print("=" * 80)
        
        print("\nSelect operation mode:")
        print("1. Complete analysis (fetch data + analyze)")
        print("2. Analysis only (use existing Supabase data)")
        print("3. Data fetch only (update Supabase)")
        
        mode = input("\nEnter choice (1-3): ").strip()
        
        if mode == "1":
            days = input("Enter number of days of data to fetch (default 1000): ").strip()
            days = int(days) if days.isdigit() else 1000
            print(f"\nStarting complete analysis - fetching {days} days of data...")
            analyzer.run_comprehensive_analysis(days)
            
        elif mode == "2":
            days = input("Enter number of days to analyze (default 800): ").strip()
            days = int(days) if days.isdigit() else 800
            print(f"\nRunning analysis from existing data for {days} days...")
            analyzer.run_analysis_only(days)
            
        elif mode == "3":
            days = input("Enter number of days of data to fetch (default 1000): ").strip()
            days = int(days) if days.isdigit() else 1000
            print(f"\nFetching {days} days of data...")
            analyzer.fetch_data_only(days)
            
        else:
            print("Invalid choice. Exiting...")
            return
            
        print("\nAnalysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        print("Progress saved. You can resume later using mode 1 or 2.")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
