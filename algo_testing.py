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
SUPABASE_URL = os.environ.get("MY_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("MY_SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

API_KEY = os.getenv("ZERODHA_API_KEY")
ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN")
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

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
        
        # Volume indicators
        df['volume_sma'] = talib.SMA(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Technical indicators
        df['rsi'] = talib.RSI(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        
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
        self.excel_file = "comprehensive_breakout_analysis.xlsx"
        self.rate_limit_delay = 0.25  # 250ms between API calls to respect rate limits
        
    def create_supabase_tables(self):
        """Create necessary tables in Supabase"""
        try:
            # Create instruments table
            supabase.table('instruments').select("*").limit(1).execute()
        except:
            logger.info("Creating instruments table...")
            
        try:
            # Create daily_data table  
            supabase.table('daily_data').select("*").limit(1).execute()
        except:
            logger.info("Creating daily_data table...")
            
        try:
            # Create hourly_data table
            supabase.table('hourly_data').select("*").limit(1).execute()
        except:
            logger.info("Creating hourly_data table...")
    
    def fetch_all_nse_instruments(self):
        """Fetch ALL NSE equity instruments (~2100 stocks)"""
        try:
            print("Fetching ALL NSE instruments...")
            instruments_raw = kite.instruments("NSE")
        
            # Filter for equity stocks only - be more precise
            instruments = []
            for inst in instruments_raw:
                # More specific filtering for NSE equities
                if (inst.get("instrument_type") == "EQ" 
                    and inst.get("exchange") == "NSE"
                    and inst.get("segment") == "NSE"  # Ensure NSE segment
                    and "-" not in inst.get("tradingsymbol", "")
                    and inst.get("lot_size", 0) > 0  # Active stocks have lot size
                    and len(inst.get("tradingsymbol", "")) <= 20):  # Filter out unusually long symbols
                
                    # Handle expiry field - convert empty string to None
                    expiry_value = inst.get('expiry')
                    if expiry_value == "" or expiry_value is None:
                        expiry_value = None

                    # Skip indices
                    if inst.get("segment") == "INDICES":
                        continue
                        
                    # Handle numeric fields that might be empty strings
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
                        'tradingsymbol': inst.get('tradingsymbol'),
                        'name': inst.get('name'),
                        'last_price': safe_float(inst.get('last_price')),
                        'expiry': expiry_value,
                        'strike': safe_float(inst.get('strike')),
                        'tick_size': safe_float(inst.get('tick_size')),
                        'lot_size': safe_int(inst.get('lot_size')),
                        'instrument_type': inst.get('instrument_type'),
                        'segment': inst.get('segment'),
                        'exchange': inst.get('exchange')
                    }
                
                    instruments.append(instrument_data)
        
            print(f"Found {len(instruments)} active NSE equity instruments")
            return instruments
        
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return []
    
    def store_instruments_in_supabase(self, instruments):
        """Store instruments in Supabase with proper error handling"""
        try:
            print(f"Storing {len(instruments)} instruments in Supabase...")
        
            # Clear existing data
            supabase.table('instruments').delete().neq('instrument_token', 0).execute()
        
            # Insert in smaller batches to avoid timeouts
            batch_size = 50
            for i in range(0, len(instruments), batch_size):
                batch = instruments[i:i+batch_size]
                try:
                    supabase.table('instruments').insert(batch).execute()
                    print(f"Stored batch {i//batch_size + 1}/{(len(instruments)-1)//batch_size + 1}")
                    t.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
                    continue
        
            print(f"Successfully stored {len(instruments)} instruments")
            return instruments
        
        except Exception as e:
            logger.error(f"Error storing instruments: {e}")
            return []
    
    def fetch_and_store_historical_data(self, instruments: List[Dict], days: int = 1000):
        """Fetch 1000 days of historical data for all instruments"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        print(f"Fetching {days} days of historical data from {start_date} to {end_date}")
        print(f"Processing {len(instruments)} instruments...")
        
        # Use tqdm for better progress tracking
        for i, instrument in enumerate(tqdm(instruments, desc="Fetching historical data")):
            try:
                symbol = instrument['tradingsymbol']
                token = instrument['instrument_token']
                
                print(f"Processing {symbol} ({i+1}/{len(instruments)})...")
                
                # Fetch daily data (1000 days)
                daily_data = self._fetch_with_retry(token, start_date, end_date, "day")
                if daily_data:
                    self._store_daily_data(token, symbol, daily_data)
                
                # Fetch hourly data (1000 days worth)
                hourly_data = self._fetch_with_retry(token, start_date, end_date, "hour")
                if hourly_data:
                    self._store_hourly_data(token, symbol, hourly_data)
                
                # Respect rate limits
                t.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        print("Historical data storage complete!")
    
    def _fetch_with_retry(self, token, from_date, to_date, interval, max_retries=3):
        """Fetch data with retry logic and better error handling"""
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
                    # Exponential backoff for rate limiting
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
        """Store daily data in Supabase with error handling"""
        try:
            # Clear existing data for this symbol
            supabase.table('daily_data').delete().eq('instrument_token', token).execute()
            
            # Prepare data for insertion
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
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(formatted_data), batch_size):
                batch = formatted_data[i:i+batch_size]
                supabase.table('daily_data').insert(batch).execute()
                
        except Exception as e:
            logger.error(f"Error storing daily data for {symbol}: {e}")
    
    def _store_hourly_data(self, token: int, symbol: str, data: List[Dict]):
        """Store hourly data in Supabase with error handling"""
        try:
            # Clear existing data for this symbol
            supabase.table('hourly_data').delete().eq('instrument_token', token).execute()
            
            # Prepare data for insertion
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
            
            # Insert in batches
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
    
    def calculate_8_moving_averages(self, daily_data: List[Dict], hourly_data: List[Dict], target_date) -> Dict:
        """Calculate 8 MAs for a specific date."""
        try:
            daily_df = pd.DataFrame(daily_data)
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            daily_filtered = daily_df[daily_df['date'].dt.date <= target_date]
            
            if len(daily_filtered) < 200:
                return {}
            
            latest_daily = daily_filtered.iloc[-1]
            
            hourly_df = pd.DataFrame(hourly_data) if hourly_data else pd.DataFrame()
            mas = {
                'current_price': float(latest_daily['close']),
                'ma_20': float(talib.SMA(daily_filtered['close'], 20).iloc[-1]) if len(daily_filtered) >= 20 else None,
                'ma_50': float(talib.SMA(daily_filtered['close'], 50).iloc[-1]) if len(daily_filtered) >= 50 else None,
                'ma_100': float(talib.SMA(daily_filtered['close'], 100).iloc[-1]) if len(daily_filtered) >= 100 else None,
                'ma_200': float(talib.SMA(daily_filtered['close'], 200).iloc[-1]) if len(daily_filtered) >= 200 else None
            }
            
            if not hourly_df.empty and len(hourly_df) >= 200:
                hourly_filtered = hourly_df[hourly_df['datetime'].dt.date <= target_date]
                if len(hourly_filtered) >= 200:
                    mas.update({
                        'ma_20h': float(talib.SMA(hourly_filtered['close'], 20).iloc[-1]) if len(hourly_filtered) >= 20 else None,
                        'ma_50h': float(talib.SMA(hourly_filtered['close'], 50).iloc[-1]) if len(hourly_filtered) >= 50 else None,
                        'ma_100h': float(talib.SMA(hourly_filtered['close'], 100).iloc[-1]) if len(hourly_filtered) >= 100 else None,
                        'ma_200h': float(talib.SMA(hourly_filtered['close'], 200).iloc[-1]) if len(hourly_filtered) >= 200 else None
                    })
            
            return mas
        except Exception as e:
            logger.error(f"Error calculating 8 MAs: {e}")
            return {}

    def is_below_all_8_mas(self, analysis: Dict) -> bool:
        """Check if price is below all 8 MAs."""
        try:
            price = analysis.get('current_price', 0)
            ma_keys = ['ma_20', 'ma_50', 'ma_100', 'ma_200', 'ma_20h', 'ma_50h', 'ma_100h', 'ma_200h']
            return all(price < analysis.get(key) for key in ma_keys if analysis.get(key) is not None)
        except Exception as e:
            logger.error(f"Error checking below all 8 MAs: {e}")
            return False

    def calculate_comprehensive_parameters(self, daily_data: List[Dict], hourly_data: List[Dict], 
                                         symbol: str, company_name: str, target_date,
                                         breakout_data: Optional[Dict] = None) -> Dict:
        """Calculate ALL parameters as per Excel file requirements"""
        try:
            if not daily_data or len(daily_data) < 200:
                return {}
            
            # Create DataFrames
            daily_df = pd.DataFrame(daily_data)
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            daily_df = daily_df.sort_values('date')
            
            hourly_df = pd.DataFrame(hourly_data) if hourly_data else pd.DataFrame()
            if not hourly_df.empty:
                hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
                hourly_df = hourly_df.sort_values('datetime')
            
            # Filter data up to target date
            daily_filtered = daily_df[daily_df['date'].dt.date <= target_date].copy()
            if len(daily_filtered) < 50:
                return {}
            
            # Calculate all technical indicators
            daily_filtered = self._calculate_all_technical_indicators(daily_filtered)
            
            # Get latest data point
            latest = daily_filtered.iloc[-1]
            
            # Calculate moving averages
            mas = self._calculate_moving_averages(daily_filtered, hourly_df, target_date)
            
            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(daily_filtered)
            
            # Calculate price metrics
            price_metrics = self._calculate_price_metrics(daily_filtered)
            
            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(daily_filtered)
            
            # Calculate momentum indicators
            momentum_metrics = self._calculate_momentum_indicators(daily_filtered)
            
            # Calculate pattern analysis
            pattern_metrics = self._calculate_pattern_metrics(daily_filtered)
            
            # Market context (if available)
            market_metrics = self._get_market_context()
            
            # Compile comprehensive parameter set
            comprehensive_params = {
                # Basic Information
                'Date': target_date.strftime('%Y-%m-%d'),
                'Stock_Symbol': symbol,
                'Company_Name': company_name,
                'Sector': 'Unknown',  # Can be enhanced with sector mapping
                'Market_Cap_Category': 'Unknown',  # Can be enhanced with market cap data
                
                # Current Price Data
                'Current_Price': float(latest['close']),
                'Open': float(latest['open']),
                'High': float(latest['high']),
                'Low': float(latest['low']),
                'Volume': int(latest['volume']),
                'Previous_Close': float(daily_filtered.iloc[-2]['close']) if len(daily_filtered) > 1 else float(latest['close']),
                
                # Moving Averages (Daily)
                'SMA_5': mas.get('sma_5'),
                'SMA_10': mas.get('sma_10'),
                'SMA_20': mas.get('sma_20'),
                'SMA_50': mas.get('sma_50'),
                'SMA_100': mas.get('sma_100'),
                'SMA_200': mas.get('sma_200'),
                'EMA_20': mas.get('ema_20'),
                'EMA_50': mas.get('ema_50'),
                
                # Moving Averages (Hourly)
                'SMA_20H': mas.get('sma_20h'),
                'SMA_50H': mas.get('sma_50h'),
                'SMA_100H': mas.get('sma_100h'),
                'SMA_200H': mas.get('sma_200h'),
                
                # Volume Analysis
                'Volume_SMA_20': volume_metrics.get('volume_sma_20'),
                'Volume_SMA_50': volume_metrics.get('volume_sma_50'),
                'Volume_Ratio': volume_metrics.get('volume_ratio'),
                'Volume_Spike': volume_metrics.get('volume_spike'),
                'Volume_Percentile_20D': volume_metrics.get('volume_percentile_20d'),
                'Volume_Percentile_100D': volume_metrics.get('volume_percentile_100d'),
                'Avg_Volume_10D': volume_metrics.get('avg_volume_10d'),
                'Avg_Volume_30D': volume_metrics.get('avg_volume_30d'),
                
                # Technical Indicators
                'RSI_14': float(latest.get('rsi', 50)) if not pd.isna(latest.get('rsi', np.nan)) else None,
                'RSI_21': float(latest.get('rsi_21', 50)) if not pd.isna(latest.get('rsi_21', np.nan)) else None,
                'MACD': float(latest.get('macd', 0)) if not pd.isna(latest.get('macd', np.nan)) else None,
                'MACD_Signal': float(latest.get('macd_signal', 0)) if not pd.isna(latest.get('macd_signal', np.nan)) else None,
                'MACD_Histogram': float(latest.get('macd_hist', 0)) if not pd.isna(latest.get('macd_hist', np.nan)) else None,
                'Stochastic_K': float(latest.get('stoch_k', 50)) if not pd.isna(latest.get('stoch_k', np.nan)) else None,
                'Stochastic_D': float(latest.get('stoch_d', 50)) if not pd.isna(latest.get('stoch_d', np.nan)) else None,
                'Williams_R': float(latest.get('williams_r', -50)) if not pd.isna(latest.get('williams_r', np.nan)) else None,
                'ATR_14': float(latest.get('atr', 0)) if not pd.isna(latest.get('atr', np.nan)) else None,
                'ATR_Percentage': float(latest.get('atr_pct', 0)) if not pd.isna(latest.get('atr_pct', np.nan)) else None,
                
                # Bollinger Bands
                'BB_Upper': float(latest.get('bb_upper', 0)) if not pd.isna(latest.get('bb_upper', np.nan)) else None,
                'BB_Middle': float(latest.get('bb_middle', 0)) if not pd.isna(latest.get('bb_middle', np.nan)) else None,
                'BB_Lower': float(latest.get('bb_lower', 0)) if not pd.isna(latest.get('bb_lower', np.nan)) else None,
                'BB_Width': float(latest.get('bb_width', 0)) if not pd.isna(latest.get('bb_width', np.nan)) else None,
                'BB_Position': float(latest.get('bb_position', 0)) if not pd.isna(latest.get('bb_position', np.nan)) else None,
                
                # Price Momentum
                'Price_Change_1D': price_metrics.get('price_change_1d'),
                'Price_Change_3D': price_metrics.get('price_change_3d'),
                'Price_Change_5D': price_metrics.get('price_change_5d'),
                'Price_Change_10D': price_metrics.get('price_change_10d'),
                'Price_Change_20D': price_metrics.get('price_change_20d'),
                'Price_Change_50D': price_metrics.get('price_change_50d'),
                
                # High/Low Analysis
                '52_Week_High': float(daily_filtered['high'].tail(252).max()) if len(daily_filtered) >= 252 else float(daily_filtered['high'].max()),
                '52_Week_Low': float(daily_filtered['low'].tail(252).min()) if len(daily_filtered) >= 252 else float(daily_filtered['low'].min()),
                'Distance_From_52W_High': price_metrics.get('distance_from_52w_high'),
                'Distance_From_52W_Low': price_metrics.get('distance_from_52w_low'),
                '20D_High': float(daily_filtered['high'].tail(20).max()),
                '20D_Low': float(daily_filtered['low'].tail(20).min()),
                
                # Volatility Measures
                'Volatility_10D': volatility_metrics.get('volatility_10d'),
                'Volatility_20D': volatility_metrics.get('volatility_20d'),
                'Volatility_50D': volatility_metrics.get('volatility_50d'),
                'Historical_Volatility': volatility_metrics.get('historical_volatility'),
                
                # Momentum Indicators
                'Momentum_10D': momentum_metrics.get('momentum_10d'),
                'ROC_10D': momentum_metrics.get('roc_10d'),
                'ROC_20D': momentum_metrics.get('roc_20d'),
                'CCI_20': float(latest.get('cci', 0)) if not pd.isna(latest.get('cci', np.nan)) else None,
                
                # Support/Resistance
                'Support_Level_1': pattern_metrics.get('support_level_1'),
                'Support_Level_2': pattern_metrics.get('support_level_2'),
                'Resistance_Level_1': pattern_metrics.get('resistance_level_1'),
                'Resistance_Level_2': pattern_metrics.get('resistance_level_2'),
                
                # Pattern Analysis
                'Pattern_Detected': pattern_metrics.get('pattern_detected'),
                'Pattern_Confidence': pattern_metrics.get('pattern_confidence'),
                'Trend_Direction': pattern_metrics.get('trend_direction'),
                'Trend_Strength': pattern_metrics.get('trend_strength'),
                
                # Breakout Specific (if breakout occurred)
                'Breakout_Occurred': breakout_data is not None,
                'Breakout_Time': str(breakout_data.get('breakout_time', '')) if breakout_data else '',
                'Breakout_Price': float(breakout_data.get('breakout_price', 0)) if breakout_data else None,
                'Breakout_Volume': int(breakout_data.get('breakout_volume', 0)) if breakout_data else None,
                'Percentage_Gain_At_Breakout': float(breakout_data.get('percentage_change', 0)) if breakout_data else None,
                'Is_True_Breakout': bool(breakout_data.get('is_true_breakout', False)) if breakout_data else False,
                'Above_All_8_MAs': self._is_above_all_8_mas(latest, mas),
                'Below_All_8_MAs': self._is_below_all_8_mas(latest, mas),
                
                # Market Context
                'Market_Trend': market_metrics.get('market_trend'),
                'Nifty_Change': market_metrics.get('nifty_change'),
                'Bank_Nifty_Change': market_metrics.get('bank_nifty_change'),
                'VIX_Level': market_metrics.get('vix_level'),
                'Market_Volatility': market_metrics.get('market_volatility'),
                
                # Risk Metrics
                'Beta': self._calculate_beta(daily_filtered),
                'Sharpe_Ratio': self._calculate_sharpe_ratio(daily_filtered),
                'Max_Drawdown': self._calculate_max_drawdown(daily_filtered),
                'Risk_Score': self._calculate_risk_score(daily_filtered),
                
                # Additional Metrics
                'Liquidity_Score': volume_metrics.get('liquidity_score'),
                'Price_Efficiency': self._calculate_price_efficiency(daily_filtered),
                'Breakout_Strength_Score': self._calculate_breakout_strength_score(daily_filtered, breakout_data),
                'Overall_Score': self._calculate_overall_score(daily_filtered, mas, volume_metrics, breakout_data)
            }
            
            return comprehensive_params
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive parameters for {symbol}: {e}")
            return {}
    
    def _calculate_all_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            # Moving Averages
            df['sma_5'] = talib.SMA(df['close'], 5)
            df['sma_10'] = talib.SMA(df['close'], 10)
            df['sma_20'] = talib.SMA(df['close'], 20)
            df['sma_50'] = talib.SMA(df['close'], 50)
            df['sma_100'] = talib.SMA(df['close'], 100)
            df['sma_200'] = talib.SMA(df['close'], 200)
            df['ema_20'] = talib.EMA(df['close'], 20)
            df['ema_50'] = talib.EMA(df['close'], 50)
            
            # Oscillators
            df['rsi'] = talib.RSI(df['close'], 14)
            df['rsi_21'] = talib.RSI(df['close'], 21)
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], 20)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            
            # Volatility
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], 20, 2, 2)
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100
            df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def _calculate_moving_averages(self, daily_df: pd.DataFrame, hourly_df: pd.DataFrame, target_date) -> Dict:
        """Calculate all moving averages"""
        try:
            latest_daily = daily_df.iloc[-1]
            mas = {
                'sma_5': float(latest_daily.get('sma_5', 0)) if not pd.isna(latest_daily.get('sma_5', np.nan)) else None,
                'sma_10': float(latest_daily.get('sma_10', 0)) if not pd.isna(latest_daily.get('sma_10', np.nan)) else None,
                'sma_20': float(latest_daily.get('sma_20', 0)) if not pd.isna(latest_daily.get('sma_20', np.nan)) else None,
                'sma_50': float(latest_daily.get('sma_50', 0)) if not pd.isna(latest_daily.get('sma_50', np.nan)) else None,
                'sma_100': float(latest_daily.get('sma_100', 0)) if not pd.isna(latest_daily.get('sma_100', np.nan)) else None,
                'sma_200': float(latest_daily.get('sma_200', 0)) if not pd.isna(latest_daily.get('sma_200', np.nan)) else None,
                'ema_20': float(latest_daily.get('ema_20', 0)) if not pd.isna(latest_daily.get('ema_20', np.nan)) else None,
                'ema_50': float(latest_daily.get('ema_50', 0)) if not pd.isna(latest_daily.get('ema_50', np.nan)) else None,
            }
            
            # Calculate hourly MAs if hourly data available
            if not hourly_df.empty:
                hourly_filtered = hourly_df[hourly_df['datetime'].dt.date <= target_date]
                if len(hourly_filtered) >= 200:
                    hourly_filtered['sma_20h'] = talib.SMA(hourly_filtered['close'], 20)
                    hourly_filtered['sma_50h'] = talib.SMA(hourly_filtered['close'], 50)
                    hourly_filtered['sma_100h'] = talib.SMA(hourly_filtered['close'], 100)
                    hourly_filtered['sma_200h'] = talib.SMA(hourly_filtered['close'], 200)
                    
                    latest_hourly = hourly_filtered.iloc[-1]
                    mas.update({
                        'sma_20h': float(latest_hourly.get('sma_20h', 0)) if not pd.isna(latest_hourly.get('sma_20h', np.nan)) else None,
                        'sma_50h': float(latest_hourly.get('sma_50h', 0)) if not pd.isna(latest_hourly.get('sma_50h', np.nan)) else None,
                        'sma_100h': float(latest_hourly.get('sma_100h', 0)) if not pd.isna(latest_hourly.get('sma_100h', np.nan)) else None,
                        'sma_200h': float(latest_hourly.get('sma_200h', 0)) if not pd.isna(latest_hourly.get('sma_200h', np.nan)) else None,
                    })
            
            return mas
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive volume metrics"""
        try:
            latest = df.iloc[-1]
            
            # Volume averages
            volume_sma_20 = df['volume'].tail(20).mean()
            volume_sma_50 = df['volume'].tail(50).mean() if len(df) >= 50 else volume_sma_20
            
            metrics = {
                'volume_sma_20': float(volume_sma_20),
                'volume_sma_50': float(volume_sma_50),
                'volume_ratio': float(latest['volume'] / volume_sma_20) if volume_sma_20 > 0 else 1.0,
                'volume_spike': bool(latest['volume'] > volume_sma_20 * 2),
                'avg_volume_10d': float(df['volume'].tail(10).mean()),
                'avg_volume_30d': float(df['volume'].tail(30).mean()) if len(df) >= 30 else float(df['volume'].tail(10).mean()),
                'volume_percentile_20d': float(self._calculate_volume_percentile(df, 20)),
                'volume_percentile_100d': float(self._calculate_volume_percentile(df, 100)),
                'liquidity_score': self._calculate_liquidity_score(df)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volume metrics: {e}")
            return {}
    
    def _calculate_price_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate price movement metrics"""
        try:
            latest = df.iloc[-1]
            
            metrics = {}
            
            # Price changes over different periods
            for days in [1, 3, 5, 10, 20, 50]:
                if len(df) > days:
                    old_price = df.iloc[-(days+1)]['close']
                    change = ((latest['close'] - old_price) / old_price) * 100
                    metrics[f'price_change_{days}d'] = float(change)
                else:
                    metrics[f'price_change_{days}d'] = 0.0
            
            # 52-week high/low analysis
            if len(df) >= 252:
                high_52w = df['high'].tail(252).max()
                low_52w = df['low'].tail(252).min()
            else:
                high_52w = df['high'].max()
                low_52w = df['low'].min()
            
            metrics.update({
                'distance_from_52w_high': float(((latest['close'] - high_52w) / high_52w) * 100),
                'distance_from_52w_low': float(((latest['close'] - low_52w) / low_52w) * 100)
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating price metrics: {e}")
            return {}
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility metrics"""
        try:
            metrics = {}
            
            # Calculate volatility over different periods
            for days in [10, 20, 50]:
                if len(df) >= days:
                    returns = df['close'].tail(days).pct_change().dropna()
                    volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized
                    metrics[f'volatility_{days}d'] = volatility
                else:
                    metrics[f'volatility_{days}d'] = 0.0
            
            # Historical volatility (200 days if available)
            if len(df) >= 200:
                returns = df['close'].tail(200).pct_change().dropna()
                metrics['historical_volatility'] = float(returns.std() * np.sqrt(252) * 100)
            else:
                metrics['historical_volatility'] = metrics.get('volatility_50d', 0.0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {}
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate momentum indicators"""
        try:
            latest = df.iloc[-1]
            
            metrics = {}
            
            # Momentum (price difference)
            if len(df) > 10:
                momentum_10d = latest['close'] - df.iloc[-11]['close']
                metrics['momentum_10d'] = float(momentum_10d)
            else:
                metrics['momentum_10d'] = 0.0
            
            # Rate of Change
            for days in [10, 20]:
                if len(df) > days:
                    old_price = df.iloc[-(days+1)]['close']
                    roc = ((latest['close'] - old_price) / old_price) * 100
                    metrics[f'roc_{days}d'] = float(roc)
                else:
                    metrics[f'roc_{days}d'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    def _calculate_pattern_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate pattern and support/resistance metrics"""
        try:
            # Detect patterns using the pattern recognizer
            pattern_signals = self.pattern_recognizer.analyze_stock(df)
            
            metrics = {
                'pattern_detected': 'None',
                'pattern_confidence': 0.0,
                'trend_direction': self._determine_trend_direction(df),
                'trend_strength': self._calculate_trend_strength(df),
                'support_level_1': self._find_support_level(df, 1),
                'support_level_2': self._find_support_level(df, 2),
                'resistance_level_1': self._find_resistance_level(df, 1),
                'resistance_level_2': self._find_resistance_level(df, 2)
            }
            
            if pattern_signals:
                best_pattern = max(pattern_signals, key=lambda x: x.confidence)
                metrics['pattern_detected'] = best_pattern.pattern_type.value
                metrics['pattern_confidence'] = float(best_pattern.confidence)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating pattern metrics: {e}")
            return {}
    
    def _get_market_context(self) -> Dict:
        """Get market context (placeholder - can be enhanced with real market data)"""
        return {
            'market_trend': 'Neutral',
            'nifty_change': 0.0,
            'bank_nifty_change': 0.0,
            'vix_level': 15.0,
            'market_volatility': 'Medium'
        }
    
    def _is_above_all_8_mas(self, latest_row, mas: Dict) -> bool:
        """Check if price is above all 8 moving averages"""
        try:
            price = latest_row['close']
            ma_keys = ['sma_20', 'sma_50', 'sma_100', 'sma_200', 'sma_20h', 'sma_50h', 'sma_100h', 'sma_200h']
            
            for key in ma_keys:
                ma_value = mas.get(key)
                if ma_value is None or price <= ma_value:
                    return False
            return True
            
        except Exception as e:
            logger.error(f"Error checking above all MAs: {e}")
            return False
    
    def _is_below_all_8_mas(self, latest_row, mas: Dict) -> bool:
        """Check if price is below all 8 moving averages"""
        try:
            price = latest_row['close']
            ma_keys = ['sma_20', 'sma_50', 'sma_100', 'sma_200', 'sma_20h', 'sma_50h', 'sma_100h', 'sma_200h']
            
            for key in ma_keys:
                ma_value = mas.get(key)
                if ma_value is None or price >= ma_value:
                    return False
            return True
            
        except Exception as e:
            logger.error(f"Error checking below all MAs: {e}")
            return False
    
    def _calculate_volume_percentile(self, df: pd.DataFrame, days: int) -> float:
        """Calculate volume percentile"""
        try:
            if len(df) < days:
                days = len(df)
            
            volumes = df['volume'].tail(days).values
            current_volume = df.iloc[-1]['volume']
            
            percentile = (np.sum(volumes <= current_volume) / len(volumes)) * 100
            return percentile
            
        except Exception as e:
            logger.error(f"Error calculating volume percentile: {e}")
            return 50.0
    
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate liquidity score (0-10)"""
        try:
            avg_volume = df['volume'].tail(20).mean()
            avg_turnover = (df['close'] * df['volume']).tail(20).mean()
            
            # Simple scoring based on volume and turnover
            volume_score = min(avg_volume / 100000, 5.0)  # Up to 5 points for volume
            turnover_score = min(avg_turnover / 10000000, 5.0)  # Up to 5 points for turnover
            
            return float(volume_score + turnover_score)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 5.0
    
    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine trend direction"""
        try:
            if len(df) < 20:
                return 'Sideways'
            
            sma_20 = df['close'].tail(20).mean()
            sma_50 = df['close'].tail(50).mean() if len(df) >= 50 else sma_20
            
            latest_price = df.iloc[-1]['close']
            
            if latest_price > sma_20 > sma_50:
                return 'Bullish'
            elif latest_price < sma_20 < sma_50:
                return 'Bearish'
            else:
                return 'Sideways'
                
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return 'Unknown'
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-10)"""
        try:
            if len(df) < 20:
                return 5.0
            
            # Calculate slope of price over last 20 days
            y = df['close'].tail(20).values
            x = np.arange(len(y))
            
            slope, _, r_value, _, _ = linregress(x, y)
            
            # Convert to strength score
            strength = abs(r_value) * 10
            return float(min(strength, 10.0))
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 5.0
    
    def _find_support_level(self, df: pd.DataFrame, level: int) -> float:
        """Find support levels using pivot points"""
        try:
            if len(df) < 50:
                return df['low'].min()
            
            lows = df['low'].tail(50)
            pivot_lows = []
            
            for i in range(2, len(lows)-2):
                if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1] and
                    lows.iloc[i] < lows.iloc[i-2] and lows.iloc[i] < lows.iloc[i+2]):
                    pivot_lows.append(lows.iloc[i])
            
            if not pivot_lows:
                return float(df['low'].tail(20).min())
            
            pivot_lows.sort()
            
            if level == 1:
                return float(pivot_lows[-1] if pivot_lows else df['low'].tail(20).min())
            elif level == 2:
                return float(pivot_lows[-2] if len(pivot_lows) > 1 else pivot_lows[-1] if pivot_lows else df['low'].tail(20).min())
            
            return float(df['low'].tail(20).min())
            
        except Exception as e:
            logger.error(f"Error finding support level: {e}")
            return float(df['low'].tail(20).min())
    
    def _find_resistance_level(self, df: pd.DataFrame, level: int) -> float:
        """Find resistance levels using pivot points"""
        try:
            if len(df) < 50:
                return df['high'].max()
            
            highs = df['high'].tail(50)
            pivot_highs = []
            
            for i in range(2, len(highs)-2):
                if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1] and
                    highs.iloc[i] > highs.iloc[i-2] and highs.iloc[i] > highs.iloc[i+2]):
                    pivot_highs.append(highs.iloc[i])
            
            if not pivot_highs:
                return float(df['high'].tail(20).max())
            
            pivot_highs.sort(reverse=True)
            
            if level == 1:
                return float(pivot_highs[0] if pivot_highs else df['high'].tail(20).max())
            elif level == 2:
                return float(pivot_highs[1] if len(pivot_highs) > 1 else pivot_highs[0] if pivot_highs else df['high'].tail(20).max())
            
            return float(df['high'].tail(20).max())
            
        except Exception as e:
            logger.error(f"Error finding resistance level: {e}")
            return float(df['high'].tail(20).max())
    
    def _calculate_beta(self, df: pd.DataFrame) -> float:
        """Calculate beta (placeholder - needs market index data)"""
        # This would require market index data for proper calculation
        # For now, return a placeholder value
        return 1.0
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(df) < 50:
                return 0.0
            
            returns = df['close'].pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            excess_return = returns.mean() - 0.05/252  # Assuming 5% risk-free rate
            volatility = returns.std()
            
            if volatility == 0:
                return 0.0
            
            sharpe = (excess_return / volatility) * np.sqrt(252)
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        try:
            prices = df['close']
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak * 100
            return float(drawdown.min())
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> float:
        """Calculate overall risk score (1-10)"""
        try:
            volatility = df['close'].pct_change().std() * np.sqrt(252) * 100
            max_dd = abs(self._calculate_max_drawdown(df))
            
            # Simple risk scoring
            vol_score = min(volatility / 5, 5)  # Up to 5 points for volatility
            dd_score = min(max_dd / 10, 5)      # Up to 5 points for drawdown
            
            return float(vol_score + dd_score)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 5.0
    
    def _calculate_price_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate price efficiency"""
        try:
            if len(df) < 20:
                return 0.5
            
            # Simple efficiency calculation
            start_price = df.iloc[-20]['close']
            end_price = df.iloc[-1]['close']
            
            linear_distance = abs(end_price - start_price)
            actual_distance = sum(abs(df['close'].iloc[i] - df['close'].iloc[i-1]) 
                                for i in range(-19, 0))
            
            if actual_distance == 0:
                return 1.0
            
            efficiency = linear_distance / actual_distance
            return float(min(efficiency, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating price efficiency: {e}")
            return 0.5
    
    def _calculate_breakout_strength_score(self, df: pd.DataFrame, breakout_data: Optional[Dict]) -> float:
        """Calculate breakout strength score"""
        try:
            if not breakout_data:
                return 0.0
            
            base_score = 5.0
            
            # Volume factor
            volume_ratio = breakout_data.get('volume_ratio', 1.0)
            if volume_ratio >= 3:
                base_score += 2.0
            elif volume_ratio >= 2:
                base_score += 1.0
            
            # Price movement factor
            pct_change = breakout_data.get('percentage_change', 0)
            if pct_change >= 10:
                base_score += 2.0
            elif pct_change >= 5:
                base_score += 1.0
            
            # Pattern confidence
            pattern_conf = breakout_data.get('pattern_confidence', 0)
            base_score += pattern_conf
            
            return float(min(base_score, 10.0))
            
        except Exception as e:
            logger.error(f"Error calculating breakout strength score: {e}")
            return 5.0
    
    def _calculate_overall_score(self, df: pd.DataFrame, mas: Dict, volume_metrics: Dict, 
                                breakout_data: Optional[Dict]) -> float:
        """Calculate overall score combining multiple factors"""
        try:
            score = 5.0
            
            # Technical strength
            latest = df.iloc[-1]
            if mas.get('sma_20') and latest['close'] > mas['sma_20']:
                score += 0.5
            if mas.get('sma_50') and latest['close'] > mas['sma_50']:
                score += 0.5
            
            # Volume strength
            volume_ratio = volume_metrics.get('volume_ratio', 1.0)
            if volume_ratio > 2:
                score += 1.0
            elif volume_ratio > 1.5:
                score += 0.5
            
            # RSI
            rsi = latest.get('rsi', 50)
            if not pd.isna(rsi) and 40 <= rsi <= 60:
                score += 0.5
            
            # Breakout bonus
            if breakout_data and breakout_data.get('is_true_breakout'):
                score += 2.0
            
            return float(min(score, 10.0))
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 5.0
    
    def scan_for_monitoring_candidates(self, target_date) -> List[Dict]:
        """Scan all stocks to find monitoring candidates (below all 8 MAs)"""
        monitoring_list = []
        
        # Get instruments from Supabase
        instruments = self.get_instruments_from_supabase()
        
        if not instruments:
            print("No instruments found in Supabase")
            return []
        
        print(f"Scanning {len(instruments)} stocks for monitoring candidates on {target_date}...")
        
        # Use tqdm for better progress tracking
        for i, instrument in enumerate(tqdm(instruments, desc="Scanning stocks")):
            try:
                symbol = instrument['tradingsymbol']
                token = instrument['instrument_token']
                company_name = instrument['name']
                
                print(f"Analyzing {symbol} - {company_name}")
                
                # Get data from Supabase
                daily_data = self.get_historical_data_from_supabase(token, "daily")
                hourly_data = self.get_historical_data_from_supabase(token, "hourly")
                
                if not daily_data or len(daily_data) < 200:
                    continue
                
                # Calculate comprehensive parameters
                params = self.calculate_comprehensive_parameters(
                    daily_data, hourly_data, symbol, company_name, target_date
                )
                
                if not params:
                    continue
                
                # Check if below all 8 MAs for monitoring
                if params.get('Below_All_8_MAs', False):
                    monitoring_list.append({
                        'symbol': symbol,
                        'company_name': company_name,
                        'token': token,
                        'params': params,
                        'daily_data': daily_data,
                        'hourly_data': hourly_data
                    })
                
                if i % 50 == 0 and i > 0:
                    print(f"Scanned {i}/{len(instruments)} stocks... Found {len(monitoring_list)} candidates")
                    
            except Exception as e:
                logger.error(f"Error scanning {instrument.get('tradingsymbol', 'unknown')}: {e}")
                continue
        
        print(f"Found {len(monitoring_list)} stocks below all 8 MAs for monitoring")
        return monitoring_list
    
    def detect_breakouts_for_day(self, monitoring_list: List[Dict], target_date) -> List[Dict]:
        """Detect breakouts for a specific day"""
        breakout_stocks = []
        
        print(f"Checking {len(monitoring_list)} stocks for breakouts on {target_date}...")
        
        for stock in tqdm(monitoring_list, desc="Checking breakouts"):
            try:
                symbol = stock['symbol']
                token = stock['token']
                company_name = stock['company_name']
                
                print(f"Checking breakout for {symbol} - {company_name}")
                
                # Get minute data for the day
                minute_data = self._get_minute_data_for_day(token, target_date)
                
                if not minute_data:
                    continue
                
                # Check for breakout
                breakout_info = self._check_for_breakout_comprehensive(stock, minute_data, target_date)
                
                if breakout_info:
                    # Calculate comprehensive parameters with breakout data
                    comprehensive_params = self.calculate_comprehensive_parameters(
                        stock['daily_data'], 
                        stock['hourly_data'], 
                        symbol, 
                        company_name, 
                        target_date,
                        breakout_info
                    )
                    
                    if comprehensive_params:
                        breakout_stocks.append(comprehensive_params)
                    
            except Exception as e:
                logger.error(f"Error checking breakout for {symbol}: {e}")
                continue
        
        return breakout_stocks
    
    def _get_minute_data_for_day(self, token: int, target_date) -> List[Dict]:
        """Get minute data for the trading day"""
        try:
            from_datetime = datetime.combine(target_date, time(9, 15))
            to_datetime = datetime.combine(target_date, time(15, 30))
            
            return self._fetch_with_retry(token, from_datetime, to_datetime, "minute")
        except Exception as e:
            logger.error(f"Error fetching minute data: {e}")
            return []
    
    def _check_for_breakout_comprehensive(self, stock: Dict, minute_data: List[Dict], target_date) -> Optional[Dict]:
        """Check for breakout with comprehensive analysis"""
        try:
            if not minute_data:
                return None
            
            symbol = stock['symbol']
            params = stock['params']
            
            # Extract MA values
            mas = {
                'sma_20': params.get('SMA_20'),
                'sma_50': params.get('SMA_50'),
                'sma_100': params.get('SMA_100'),
                'sma_200': params.get('SMA_200'),
                'sma_20h': params.get('SMA_20H'),
                'sma_50h': params.get('SMA_50H'),
                'sma_100h': params.get('SMA_100H'),
                'sma_200h': params.get('SMA_200H')
            }
            
            # Find highest price during the day
            highest_price = max([candle['high'] for candle in minute_data])
            
            # Check if broke above all 8 MAs
            above_all_mas = all([
                highest_price > mas['sma_20'] if mas['sma_20'] else False,
                highest_price > mas['sma_50'] if mas['sma_50'] else False,
                highest_price > mas['sma_100'] if mas['sma_100'] else False,
                highest_price > mas['sma_200'] if mas['sma_200'] else False,
                highest_price > mas['sma_20h'] if mas['sma_20h'] else False,
                highest_price > mas['sma_50h'] if mas['sma_50h'] else False,
                highest_price > mas['sma_100h'] if mas['sma_100h'] else False,
                highest_price > mas['sma_200h'] if mas['sma_200h'] else False
            ])
            
            if not above_all_mas:
                return None
            
            # Find exact breakout time and details
            breakout_time = None
            breakout_price = None
            breakout_volume = 0
            
            for candle in minute_data:
                candle_high = candle['high']
                
                if all([
                    candle_high > mas['sma_20'] if mas['sma_20'] else False,
                    candle_high > mas['sma_50'] if mas['sma_50'] else False,
                    candle_high > mas['sma_100'] if mas['sma_100'] else False,
                    candle_high > mas['sma_200'] if mas['sma_200'] else False,
                    candle_high > mas['sma_20h'] if mas['sma_20h'] else False,
                    candle_high > mas['sma_50h'] if mas['sma_50h'] else False,
                    candle_high > mas['sma_100h'] if mas['sma_100h'] else False,
                    candle_high > mas['sma_200h'] if mas['sma_200h'] else False
                ]):
                    breakout_time = pd.to_datetime(candle['date']).time()
                    breakout_price = candle_high
                    breakout_volume = candle['volume']
                    break
            
            if not breakout_time:
                return None
            
            # Calculate percentage change
            price_at_9am = params.get('Current_Price', 0)
            if price_at_9am <= 0:
                return None
                
            percentage_change = ((highest_price - price_at_9am) / price_at_9am) * 100
            is_true_breakout = percentage_change >= 5.0
            
            # Calculate volume ratio at breakout
            avg_volume = params.get('Volume_SMA_20', 1)
            volume_ratio = breakout_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                'breakout_time': breakout_time,
                'breakout_price': breakout_price,
                'breakout_volume': breakout_volume,
                'highest_price': highest_price,
                'percentage_change': percentage_change,
                'is_true_breakout': is_true_breakout,
                'volume_ratio': volume_ratio,
                'pattern_confidence': 0.7  # Default confidence
            }
            
        except Exception as e:
            logger.error(f"Error checking comprehensive breakout: {e}")
            return None
    
    def run_comprehensive_analysis(self, days_to_analyze: int = 1000):
        """Run comprehensive analysis for the specified number of days"""
        print("=" * 80)
        print("COMPREHENSIVE NSE BREAKOUT ANALYSIS")
        print("=" * 80)
        print(f"Analyzing last {days_to_analyze} days of trading data")
        print("Features: All NSE stocks, 8-MA breakouts, comprehensive metrics")
        
        # Step 1: Setup
        print("\n1. Setting up database...")
        self.create_supabase_tables()
        
        # Step 2: Fetch all NSE instruments
        print("\n2. Fetching all NSE instruments...")
        instruments = self.fetch_all_nse_instruments()
        if not instruments:
            print("Failed to fetch instruments. Exiting.")
            return
        
        print(f"Found {len(instruments)} NSE equity instruments")
        
        # Step 3: Store instruments
        print("\n3. Storing instruments in Supabase...")
        self.store_instruments_in_supabase(instruments)
        
        # Step 4: Fetch historical data
        print(f"\n4. Fetching {days_to_analyze} days of historical data...")
        self.fetch_and_store_historical_data(instruments, days_to_analyze)
        
        # Step 5: Analyze each trading day
        print(f"\n5. Starting breakout analysis for {days_to_analyze} days...")
        
        end_date = datetime.now().date()
        all_results = []
        trading_days_analyzed = 0
        
        # Get trading days for analysis
        trading_days = []
        for i in range(days_to_analyze):
            check_date = end_date - timedelta(days=i)
            if check_date.weekday() < 5:  # Monday=0, Sunday=6
                trading_days.append(check_date)
        
        print(f"Will analyze {len(trading_days)} trading days")
        
        # Analyze each trading day
        for target_date in tqdm(trading_days[:min(len(trading_days), 800)], desc="Analyzing trading days"):
            try:
                print(f"\nAnalyzing {target_date}...")
                
                # Find monitoring candidates (stocks below all 8 MAs)
                monitoring_candidates = self.scan_for_monitoring_candidates(target_date)
                
                if not monitoring_candidates:
                    print(f"No monitoring candidates found for {target_date}")
                    continue
                
                print(f"Found {len(monitoring_candidates)} monitoring candidates")
                
                # Check for breakouts
                breakout_results = self.detect_breakouts_for_day(monitoring_candidates, target_date)
                
                if breakout_results:
                    all_results.extend(breakout_results)
                    print(f"Found {len(breakout_results)} breakouts on {target_date}")
                
                trading_days_analyzed += 1
                
            except Exception as e:
                logger.error(f"Error analyzing {target_date}: {e}")
                continue
        
        # Step 6: Save results
        if all_results:
            print(f"\n6. Saving {len(all_results)} records to Excel...")
            self.save_comprehensive_results(all_results)
            
            # Step 7: Generate analysis summary
            self.generate_analysis_summary(all_results, trading_days_analyzed)
        else:
            print("No breakout results found to save")
    
    def save_comprehensive_results(self, results: List[Dict]):
        """Save comprehensive results to Excel with all parameters"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            if df.empty:
                print("No data to save")
                return
            
            # Ensure all columns are properly formatted
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                # Main sheet with all data
                df.to_excel(writer, sheet_name='All_Breakouts', index=False)
                
                # True breakouts only
                true_breakouts = df[df['Is_True_Breakout'] == True]
                if not true_breakouts.empty:
                    true_breakouts.to_excel(writer, sheet_name='True_Breakouts_5%+', index=False)
                
                # Summary statistics
                summary_df = self.create_summary_statistics(df)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Pattern analysis
                pattern_df = self.create_pattern_analysis(df)
                pattern_df.to_excel(writer, sheet_name='Pattern_Analysis', index=False)
            
            print(f"Results saved to {self.excel_file}")
            print(f"Total records: {len(df)}")
            print(f"True breakouts (≥5%): {len(true_breakouts)}")
            
        except Exception as e:
            logger.error(f"Error saving comprehensive results: {e}")
    
    def create_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics DataFrame"""
        try:
            stats = []
            
            total_breakouts = len(df)
            true_breakouts = len(df[df['Is_True_Breakout'] == True])
            false_breakouts = total_breakouts - true_breakouts
            
            stats.append(['Total Breakouts', total_breakouts])
            stats.append(['True Breakouts (≥5%)', true_breakouts])
            stats.append(['False Breakouts (<5%)', false_breakouts])
            stats.append(['Success Rate (%)', (true_breakouts/total_breakouts*100) if total_breakouts > 0 else 0])
            
            if true_breakouts > 0:
                true_df = df[df['Is_True_Breakout'] == True]
                stats.append(['Avg Gain - True Breakouts (%)', true_df['Percentage_Gain_At_Breakout'].mean()])
                stats.append(['Max Gain - True Breakouts (%)', true_df['Percentage_Gain_At_Breakout'].max()])
                stats.append(['Min Gain - True Breakouts (%)', true_df['Percentage_Gain_At_Breakout'].min()])
            
            if false_breakouts > 0:
                false_df = df[df['Is_True_Breakout'] == False]
                stats.append(['Avg Gain - False Breakouts (%)', false_df['Percentage_Gain_At_Breakout'].mean()])
            
            # Volume analysis
            stats.append(['Avg Volume Ratio', df['Volume_Ratio'].mean()])
            stats.append(['High Volume Breakouts (>2x)', len(df[df['Volume_Ratio'] > 2])])
            
            # Pattern analysis
            pattern_counts = df['Pattern_Detected'].value_counts()
            stats.append(['Most Common Pattern', pattern_counts.index[0] if len(pattern_counts) > 0 else 'None'])
            
            return pd.DataFrame(stats, columns=['Metric', 'Value'])
            
        except Exception as e:
            logger.error(f"Error creating summary statistics: {e}")
            return pd.DataFrame()
    
    def create_pattern_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pattern analysis DataFrame"""
        try:
            pattern_stats = []
            
            # Pattern frequency
            patterns = df['Pattern_Detected'].value_counts()
            
            for pattern, count in patterns.items():
                pattern_df = df[df['Pattern_Detected'] == pattern]
                true_count = len(pattern_df[pattern_df['Is_True_Breakout'] == True])
                success_rate = (true_count / count * 100) if count > 0 else 0
                avg_gain = pattern_df['Percentage_Gain_At_Breakout'].mean()
                
                pattern_stats.append([
                    pattern, count, true_count, f"{success_rate:.1f}%", f"{avg_gain:.2f}%"
                ])
            
            columns = ['Pattern', 'Total_Count', 'True_Breakouts', 'Success_Rate', 'Avg_Gain']
            return pd.DataFrame(pattern_stats, columns=columns)
            
        except Exception as e:
            logger.error(f"Error creating pattern analysis: {e}")
            return pd.DataFrame()
    
    def generate_analysis_summary(self, results: List[Dict], trading_days: int):
        """Generate and print analysis summary"""
        try:
            df = pd.DataFrame(results)
            
            total_breakouts = len(df)
            true_breakouts = len(df[df['Is_True_Breakout'] == True])
            false_breakouts = total_breakouts - true_breakouts
            
            print("\n" + "=" * 80)
            print("COMPREHENSIVE ANALYSIS SUMMARY")
            print("=" * 80)
            print(f"Trading days analyzed: {trading_days}")
            print(f"Total breakouts found: {total_breakouts}")
            print(f"True breakouts (≥5%): {true_breakouts}")
            print(f"False breakouts (<5%): {false_breakouts}")
            print(f"Overall success rate: {(true_breakouts/total_breakouts*100):.1f}%")
            
            if true_breakouts > 0:
                true_df = df[df['Is_True_Breakout'] == True]
                print(f"Average gain (true breakouts): {true_df['Percentage_Gain_At_Breakout'].mean():.2f}%")
                print(f"Maximum gain: {true_df['Percentage_Gain_At_Breakout'].max():.2f}%")
            
            # Pattern analysis
            print(f"\nPattern Distribution:")
            patterns = df['Pattern_Detected'].value_counts()
            for pattern, count in patterns.head(5).items():
                percentage = (count / total_breakouts * 100)
                print(f"  {pattern}: {count} ({percentage:.1f}%)")
            
            # Volume analysis
            high_vol_breakouts = len(df[df['Volume_Ratio'] > 2])
            print(f"\nHigh volume breakouts (>2x avg): {high_vol_breakouts} ({high_vol_breakouts/total_breakouts*100:.1f}%)")
            
            print(f"\nResults saved to: {self.excel_file}")
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")

    def run_analysis_only(self, days: int):
            """Run analysis using existing Supabase data"""
            try:
                print(f"Running analysis for {days} days using existing data...")
                
                end_date = datetime.now().date()
                all_results = []
                trading_days_analyzed = 0
                
                # Get trading days
                trading_days = []
                for i in range(days):
                    check_date = end_date - timedelta(days=i)
                    if check_date.weekday() < 5:
                        trading_days.append(check_date)
                
                # Analyze each day
                for target_date in tqdm(trading_days, desc="Analyzing days"):
                    try:
                        monitoring_candidates = self.scan_for_monitoring_candidates(target_date)
                        if monitoring_candidates:
                            breakouts = self.detect_breakouts_for_day(monitoring_candidates, target_date)
                            if breakouts:
                                all_results.extend(breakouts)
                        trading_days_analyzed += 1
                    except Exception as e:
                        logger.error(f"Error analyzing {target_date}: {e}")
                        continue
                
                # Save results
                if all_results:
                    self.save_comprehensive_results(all_results)
                    self.generate_analysis_summary(all_results, trading_days_analyzed)
                else:
                    print("No results found")
                    
            except Exception as e:
                logger.error(f"Error in analysis-only mode: {e}")
        
    def fetch_data_only(self, days: int):
            """Fetch and store data without analysis"""
            try:
                print(f"Fetching {days} days of data...")
                
                # Setup
                self.create_supabase_tables()
                
                # Fetch instruments
                instruments = self.fetch_all_nse_instruments()
                if instruments:
                    self.store_instruments_in_supabase(instruments)
                    self.fetch_and_store_historical_data(instruments, days)
                    print("Data fetch completed successfully!")
                else:
                    print("Failed to fetch instruments")
                    
            except Exception as e:
                logger.error(f"Error in fetch-only mode: {e}")

def test_specific_breakouts(test_cases: List[Dict]):
    """Test algo on specific stocks and dates."""
    analyzer = EnhancedBreakoutAnalyzer()
    results = []
    
    for case in test_cases:
        symbol = case['symbol']
        target_date = datetime.strptime(case['date'], '%Y-%m-%d').date()
        expected_time = case['expected_time']
        expected_gain = case['expected_gain']
        
        print(f"\nTesting {symbol} on {target_date}")
        
        # Get instrument token (from Supabase or API)
        instruments = analyzer.get_instruments_from_supabase()
        instrument = next((inst for inst in instruments if inst['tradingsymbol'] == symbol), None)
        if not instrument:
            print(f"❌ Instrument not found for {symbol}")
            continue
        token = instrument['instrument_token']
        
        # Get data
        daily_data = analyzer.get_historical_data_from_supabase(token, "daily")
        hourly_data = analyzer.get_historical_data_from_supabase(token, "hourly")
        
        # If hourly data missing, fetch on-demand (as in script)
        if len(hourly_data) < 200:
            hourly_start = target_date - timedelta(days=100)
            hourly_data_raw = analyzer._fetch_with_retry(token, hourly_start, target_date, "hour")
            if hourly_data_raw:
                analyzer._store_hourly_data(token, symbol, hourly_data_raw)
                hourly_data = [{'datetime': rec['date'].isoformat(), 'open': rec['open'], 'high': rec['high'], 'low': rec['low'], 'close': rec['close'], 'volume': rec['volume']} for rec in hourly_data_raw]
        
        # Calculate 8 MAs at 9:15 AM
        analysis_9am = analyzer.calculate_8_moving_averages(daily_data, hourly_data, target_date)
        if not analysis_9am:
            print(f"❌ Insufficient data for {symbol} on {target_date}")
            continue
        
        # Check if below all 8 MAs at 9:15 AM
        below_all = analyzer.is_below_all_8_mas(analysis_9am)
        print(f"Below all 8 MAs at 9:15 AM: {'✅ Yes' if below_all else '❌ No'}")
        
        if not below_all:
            results.append({'symbol': symbol, 'date': target_date, 'result': 'Not a candidate (not below all MAs)'})
            continue
        
        # Get minute data for the day
        minute_data = analyzer._get_minute_data_for_day(token, target_date)
        if not minute_data:
            print(f"❌ No minute data for {symbol} on {target_date}")
            continue
        
        # Create stock dict for breakout check
        stock_dict = {
            'symbol': symbol,
            'company_name': instrument['name'],
            'token': token,
            'price_at_9am': analysis_9am['current_price'],
            'mas': {k: analysis_9am[k] for k in ['ma_20', 'ma_50', 'ma_100', 'ma_200', 'ma_20h', 'ma_50h', 'ma_100h', 'ma_200h']},
            'daily_data': daily_data,
            'hourly_data': hourly_data
        }
        
        # Check for breakout
        breakout_info = analyzer._check_for_breakout_comprehensive(stock_dict, minute_data, target_date)
        if breakout_info:
            is_true = breakout_info['is_true_breakout']
            detected_gain = breakout_info['percentage_change']
            detected_time = breakout_info['breakout_time']
            pattern = breakout_info.get('pattern_detected', 'None')
            print(f"Detected Breakout: {'✅ True' if is_true else '❌ False'} (Gain: {detected_gain:.2f}%, Time: {detected_time}, Pattern: {pattern})")
            print(f"Expected: Gain {expected_gain} at {expected_time}")
            results.append({'symbol': symbol, 'date': target_date, 'result': f"{'True' if is_true else 'False'} Breakout (Detected Gain: {detected_gain:.2f}%)"})
        else:
            print("❌ No breakout detected")
            results.append({'symbol': symbol, 'date': target_date, 'result': 'No breakout'})
    
    # Summary
    print("\nTest Summary:")
    for res in results:
        print(f"{res['symbol']} ({res['date']}): {res['result']}")

def main():
    """Main execution function with enhanced user interface"""
    try:
        analyzer = EnhancedBreakoutAnalyzer()
        
        print("=" * 80)
        print("ENHANCED NSE BREAKOUT ANALYZER v2.0")
        print("=" * 80)
        print("Features:")
        print("• All ~2100 NSE equity stocks")
        print("• 1000 days of historical data")
        print("• 8-MA breakout strategy")
        print("• Comprehensive technical analysis")
        print("• Pattern recognition")
        print("• Complete Excel export")
        print("=" * 80)
        
        print("\nSelect operation mode:")
        print("1. Complete analysis (fetch data + analyze)")
        print("2. Analysis only (use existing Supabase data)")
        print("3. Data fetch only (update Supabase)")
        
        mode = input("\nEnter choice (1-3): ").strip()
        
        if mode == "1":
            days = input("Enter number of days to analyze (default 1000): ").strip()
            days = int(days) if days.isdigit() else 1000
            
            print(f"\nStarting complete analysis for {days} days...")
            analyzer.run_comprehensive_analysis(days)
            
        elif mode == "2":
            days = input("Enter number of days to analyze from existing data (default 200): ").strip()
            days = int(days) if days.isdigit() else 200
            
            print(f"\nRunning analysis from existing data for {days} days...")
            # Run analysis without data fetching
            analyzer.run_analysis_only(days)
            
        elif mode == "3":
            days = input("Enter number of days of data to fetch (default 1000): ").strip()
            days = int(days) if days.isdigit() else 1000
            
            print(f"\nFetching {days} days of data...")
            analyzer.fetch_data_only(days)
            
        # In main(), after mode selection
        elif mode == "4":  # Add this
            print("Running specific breakout tests...")
            test_cases = [
                {'symbol': 'KCP', 'date': '2025-08-04', 'expected_time': '14:34:00', 'expected_gain': '+5.13%'},
                {'symbol': 'MOTOGENFIN', 'date': '2025-08-04', 'expected_time': '10:48:00', 'expected_gain': '+11.42%'},
                {'symbol': 'SURANAT&P', 'date': '2025-08-04', 'expected_time': '13:29:00', 'expected_gain': '+10.53%'},
                {'symbol': 'MTARTECH', 'date': '2025-08-05', 'expected_time': '14:36:00', 'expected_gain': '+9.60%'},
                {'symbol': 'AMBICAAGAR', 'date': '2025-08-05', 'expected_time': '10:47:00', 'expected_gain': '+9.24%'},
                {'symbol': 'JOCIL', 'date': '2025-08-05', 'expected_time': '13:09:00', 'expected_gain': '+12.66%'},
                {'symbol': 'SANDUMA', 'date': '2025-08-05', 'expected_time': '15:16:00', 'expected_gain': '+6.19%'},
                {'symbol': 'RELIABLE', 'date': '2025-08-05', 'expected_time': '10:38:00', 'expected_gain': '+2.89%'},
                {'symbol': 'ADL', 'date': '2025-08-06', 'expected_time': '12:42:00', 'expected_gain': '+4.43%'},
                {'symbol': 'JETFREIGHT', 'date': '2025-08-06', 'expected_time': '13:35:00', 'expected_gain': '+8.97%'},
                {'symbol': 'GMRP&UI', 'date': '2025-08-07', 'expected_time': '15:07:00', 'expected_gain': '+5.68%'},
                {'symbol': 'TREL', 'date': '2025-08-07', 'expected_time': '09:15:00', 'expected_gain': '+9.35%'}
            ]
            test_specific_breakouts(test_cases)    
        else:
            print("Invalid choice. Exiting...")
            return
            
        print("\nAnalysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()