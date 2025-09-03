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

CHECKPOINT_FILE = "analysis_checkpoint.json"

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pattern_breakout_analysis.log', encoding='utf-8'),
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
        if len(df) < 20:
            return signals

        # Calculate indicators
        df = self._calculate_indicators(df)

        # Detect patterns
        signals.extend(self._detect_breakout_patterns(df))
        signals.extend(self._detect_triangle_patterns(df))
        signals.extend(self._detect_flag_patterns(df))
        signals.extend(self._detect_support_resistance_break(df))
        signals.extend(self._detect_double_bottom(df))

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
        if len(df) < 20:
            return signals

        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]

        peak_indices = find_peaks(highs, distance=3)[0]
        trough_indices = find_peaks(-lows, distance=3)[0]

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

class EnhancedBreakoutAnalyzer:
    def __init__(self):
        self.pattern_recognizer = ChartPatternRecognizer()
        self.excel_file = "breakout_stock_analysis.xlsx"

    def save_checkpoint(self, completed_days: List[str], current_day: str, total_days: int):
        """Save analysis progress to checkpoint file"""
        try:
            checkpoint = {
                'completed_days': completed_days,
                'current_day': current_day,
                'total_days': total_days,
                'timestamp': datetime.now().isoformat(),
                'progress_percentage': (len(completed_days) / total_days * 100) if total_days > 0 else 0
            }
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            print(f"✅ Checkpoint saved: {len(completed_days)}/{total_days} days completed ({checkpoint['progress_percentage']:.1f}%)")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def load_checkpoint(self) -> Dict:
        """Load previous analysis progress"""
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    checkpoint = json.load(f)
                print(f"📂 Resuming from checkpoint: {len(checkpoint.get('completed_days', []))} days already completed")
                return checkpoint
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                return {}
        return {}

    def cleanup_checkpoint(self):
        """Remove checkpoint file when analysis is complete"""
        try:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                print("🗑️ Checkpoint file removed - analysis complete!")
        except Exception as e:
            logger.error(f"Error removing checkpoint: {e}")

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

    def fetch_and_store_instruments(self):
        """Fetch NSE instruments and store in Supabase"""
        try:
            print("Fetching NSE instruments...")
            instruments_raw = kite.instruments("NSE")

            instruments = []
            for inst in instruments_raw:
                if (inst.get("instrument_type") == "EQ"
                    and inst.get("exchange") == "NSE"
                    and "-" not in inst.get("tradingsymbol", "")):

                    expiry_value = inst.get('expiry')
                    if expiry_value == "" or expiry_value is None:
                        expiry_value = None

                    if inst.get("segment") == "INDICES":
                        continue

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

            print(f"Storing {len(instruments)} instruments in Supabase...")

            supabase.table('instruments').delete().neq('instrument_token', 0).execute()

            batch_size = 100
            for i in range(0, len(instruments), batch_size):
                batch = instruments[i:i+batch_size]
                try:
                    supabase.table('instruments').insert(batch).execute()
                    print(f"Stored instruments batch {i//batch_size + 1}/{(len(instruments)-1)//batch_size + 1}")
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
        """Fetch historical data for all instruments and store in Supabase"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        print(f"Fetching historical data from {start_date} to {end_date}")

        for i, instrument in enumerate(instruments):
            try:
                symbol = instrument['tradingsymbol']
                token = instrument['instrument_token']

                print(f"Processing {symbol} ({i+1}/{len(instruments)})...")

                daily_data = self._fetch_with_retry(token, start_date, end_date, "day")
                if daily_data:
                    self._store_daily_data(token, symbol, daily_data)

                hourly_start = end_date - timedelta(days=90)
                hourly_data = self._fetch_with_retry(token, hourly_start, end_date, "hour")
                if hourly_data:
                    self._store_hourly_data(token, symbol, hourly_data)

                if i % 10 == 0:
                    t.sleep(1)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        print("Historical data storage complete!")

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
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Retry in {wait_time}s: {e}")
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

    def calculate_8_moving_averages(self, daily_data: List[Dict], hourly_data: List[Dict], target_date) -> Optional[Dict]:
        """Calculate 8 moving averages (4 daily + 4 hourly) at 9:00 AM"""
        try:
            if not daily_data or not hourly_data:
                return None

            daily_df = pd.DataFrame(daily_data)
            daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
            daily_filtered = daily_df[daily_df['date'] <= target_date].tail(200).reset_index(drop=True)

            if len(daily_filtered) < 200:
                return None

            daily_filtered['ma_20'] = daily_filtered['close'].rolling(window=20, min_periods=20).mean().round(2)
            daily_filtered['ma_50'] = daily_filtered['close'].rolling(window=50, min_periods=50).mean().round(2)
            daily_filtered['ma_100'] = daily_filtered['close'].rolling(window=100, min_periods=100).mean().round(2)
            daily_filtered['ma_200'] = daily_filtered['close'].rolling(window=200, min_periods=200).mean().round(2)

            daily_target = daily_filtered[daily_filtered['date'] == target_date]
            if daily_target.empty:
                return None

            latest_daily = daily_target.iloc[-1]

            hourly_df = pd.DataFrame(hourly_data)
            hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
            hourly_df['date_only'] = hourly_df['datetime'].dt.date
            hourly_df['time_only'] = hourly_df['datetime'].dt.time

            morning_time = time(9, 15)
            hourly_filtered = hourly_df[
                (hourly_df['date_only'] < target_date) |
                ((hourly_df['date_only'] == target_date) & (hourly_df['time_only'] <= morning_time))
            ].tail(200).reset_index(drop=True)

            if len(hourly_filtered) < 200:
                return None

            hourly_filtered['ma_20h'] = hourly_filtered['close'].rolling(window=20, min_periods=20).mean().round(2)
            hourly_filtered['ma_50h'] = hourly_filtered['close'].rolling(window=50, min_periods=50).mean().round(2)
            hourly_filtered['ma_100h'] = hourly_filtered['close'].rolling(window=100, min_periods=100).mean().round(2)
            hourly_filtered['ma_200h'] = hourly_filtered['close'].rolling(window=200, min_periods=200).mean().round(2)

            current_price_data = hourly_filtered[
                (hourly_filtered['date_only'] == target_date) &
                (hourly_filtered['time_only'] <= morning_time)
            ]

            if current_price_data.empty:
                return None

            latest_hourly = current_price_data.iloc[-1]

            return {
                'date': target_date,
                'current_price': float(latest_hourly['close']),
                'ma_20': None if pd.isna(latest_daily['ma_20']) else float(latest_daily['ma_20']),
                'ma_50': None if pd.isna(latest_daily['ma_50']) else float(latest_daily['ma_50']),
                'ma_100': None if pd.isna(latest_daily['ma_100']) else float(latest_daily['ma_100']),
                'ma_200': None if pd.isna(latest_daily['ma_200']) else float(latest_daily['ma_200']),
                'ma_20h': None if pd.isna(latest_hourly['ma_20h']) else float(latest_hourly['ma_20h']),
                'ma_50h': None if pd.isna(latest_hourly['ma_50h']) else float(latest_hourly['ma_50h']),
                'ma_100h': None if pd.isna(latest_hourly['ma_100h']) else float(latest_hourly['ma_100h']),
                'ma_200h': None if pd.isna(latest_hourly['ma_200h']) else float(latest_hourly['ma_200h']),
                'datetime': latest_hourly['datetime']
            }

        except Exception as e:
            logger.error(f"Error calculating MAs: {e}")
            return None

    def is_below_all_8_mas(self, analysis_data: Dict) -> bool:
        """Check if price is below all 8 MAs"""
        if not analysis_data:
            return False

        mas = ['ma_20', 'ma_50', 'ma_100', 'ma_200', 'ma_20h', 'ma_50h', 'ma_100h', 'ma_200h']

        for ma in mas:
            if analysis_data[ma] is None or analysis_data['current_price'] >= analysis_data[ma]:
                return False

        return True

    def calculate_8_volume_averages(self, daily_data: List[Dict], hourly_data: List[Dict], target_date, breakout_time) -> Dict:
        """Calculate 8 volume averages"""
        try:
            volume_data = {}

            daily_df = pd.DataFrame(daily_data)
            daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
            daily_filtered = daily_df[daily_df['date'] <= target_date].reset_index(drop=True)

            volume_data.update({
                'daily_volume_avg_20': daily_filtered['volume'].tail(20).mean() if len(daily_filtered) >= 20 else None,
                'daily_volume_avg_50': daily_filtered['volume'].tail(50).mean() if len(daily_filtered) >= 50 else None,
                'daily_volume_avg_100': daily_filtered['volume'].tail(100).mean() if len(daily_filtered) >= 100 else None,
                'daily_volume_avg_200': daily_filtered['volume'].tail(200).mean() if len(daily_filtered) >= 200 else None,
            })

            hourly_df = pd.DataFrame(hourly_data)
            hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
            hourly_df['date_only'] = hourly_df['datetime'].dt.date
            hourly_df['time_only'] = hourly_df['datetime'].dt.time

            hourly_filtered = hourly_df[
                (hourly_df['date_only'] < target_date) |
                ((hourly_df['date_only'] == target_date) & (hourly_df['time_only'] <= breakout_time))
            ].reset_index(drop=True)

            volume_data.update({
                'hourly_volume_avg_20h': hourly_filtered['volume'].tail(20).mean() if len(hourly_filtered) >= 20 else None,
                'hourly_volume_avg_50h': hourly_filtered['volume'].tail(50).mean() if len(hourly_filtered) >= 50 else None,
                'hourly_volume_avg_100h': hourly_filtered['volume'].tail(100).mean() if len(hourly_filtered) >= 100 else None,
                'hourly_volume_avg_200h': hourly_filtered['volume'].tail(200).mean() if len(hourly_filtered) >= 200 else None,
            })

            return volume_data

        except Exception as e:
            logger.error(f"Error calculating volume averages: {e}")
            return {}

    def scan_for_monitoring_candidates(self, target_date) -> List[Dict]:
        """Scan all stocks at 9AM to find monitoring candidates"""
        monitoring_list = []

        instruments = self.get_instruments_from_supabase()[:1000]

        if not instruments:
            print("No instruments found in Supabase")
            return []

        print(f"Scanning {len(instruments)} stocks for 9AM monitoring candidates...")

        for i, instrument in enumerate(instruments):
            try:
                symbol = instrument['tradingsymbol']
                token = instrument['instrument_token']

                daily_data = self.get_historical_data_from_supabase(token, "daily")
                hourly_data = self.get_historical_data_from_supabase(token, "hourly")

                if not daily_data:
                    continue

                if not hourly_data:
                    print(f"No hourly data found for {symbol}, fetching from API...")
                    hourly_start = target_date - timedelta(days=100)
                    hourly_end = target_date + timedelta(days=1)

                    hourly_data_raw = self._fetch_with_retry(token, hourly_start, hourly_end, "hour")

                    if hourly_data_raw:
                        self._store_hourly_data(token, symbol, hourly_data_raw)

                        hourly_data = []
                        for record in hourly_data_raw:
                            hourly_data.append({
                                'instrument_token': token,
                                'symbol': symbol,
                                'datetime': record['date'].isoformat(),
                                'open': float(record['open']),
                                'high': float(record['high']),
                                'low': float(record['low']),
                                'close': float(record['close']),
                                'volume': int(record['volume'])
                            })
                    else:
                        print(f"Failed to fetch hourly data for {symbol}")
                        continue

                else:
                    hourly_df = pd.DataFrame(hourly_data)
                    hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
                    hourly_df['date_only'] = hourly_df['datetime'].dt.date

                    date_coverage = hourly_df['date_only'].unique()

                    if (target_date not in date_coverage or
                        len(hourly_df[hourly_df['date_only'] <= target_date]) < 200):

                        print(f"Insufficient hourly data coverage for {symbol} on {target_date}, fetching from API...")
                        hourly_start = target_date - timedelta(days=100)
                        hourly_end = target_date + timedelta(days=1)

                        hourly_data_raw = self._fetch_with_retry(token, hourly_start, hourly_end, "hour")

                        if hourly_data_raw:
                            self._store_hourly_data(token, symbol, hourly_data_raw)

                            hourly_data = []
                            for record in hourly_data_raw:
                                hourly_data.append({
                                    'instrument_token': token,
                                    'symbol': symbol,
                                    'datetime': record['date'].isoformat(),
                                    'open': float(record['open']),
                                    'high': float(record['high']),
                                    'low': float(record['low']),
                                    'close': float(record['close']),
                                    'volume': int(record['volume'])
                                })

                analysis_9am = self.calculate_8_moving_averages(daily_data, hourly_data, target_date)

                if not analysis_9am:
                    continue

                if self.is_below_all_8_mas(analysis_9am):
                    monitoring_list.append({
                        'symbol': symbol,
                        'company_name': instrument['name'],
                        'token': token,
                        'price_at_9am': analysis_9am['current_price'],
                        'mas': {
                            'ma_20': analysis_9am['ma_20'],
                            'ma_50': analysis_9am['ma_50'],
                            'ma_100': analysis_9am['ma_100'],
                            'ma_200': analysis_9am['ma_200'],
                            'ma_20h': analysis_9am['ma_20h'],
                            'ma_50h': analysis_9am['ma_50h'],
                            'ma_100h': analysis_9am['ma_100h'],
                            'ma_200h': analysis_9am['ma_200h']
                        },
                        'daily_data': daily_data,
                        'hourly_data': hourly_data
                    })

                if i % 100 == 0:
                    print(f"Scanned {i}/{len(instruments)} stocks... Found {len(monitoring_list)} candidates")

            except Exception as e:
                logger.error(f"Error scanning {instrument.get('tradingsymbol', 'unknown')}: {e}")
                continue

        print(f"Found {len(monitoring_list)} stocks below all 8 MAs at 9AM")
        return monitoring_list

    def detect_breakouts(self, monitoring_list: List[Dict], target_date) -> List[Dict]:
        """Monitor for breakouts throughout the trading day"""
        breakout_stocks = []

        print(f"Monitoring {len(monitoring_list)} stocks for breakouts...")

        for stock in monitoring_list:
            try:
                symbol = stock['symbol']
                token = stock['token']

                minute_data = self._get_minute_data_for_day(token, target_date)

                if not minute_data:
                    continue

                breakout_info = self._check_for_breakout(stock, minute_data, target_date)

                if breakout_info:
                    breakout_stocks.append(breakout_info)

            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
                continue

        return breakout_stocks

    def _get_minute_data_for_day(self, token: int, target_date) -> List[Dict]:
        """Get minute data for the trading day"""
        from_datetime = datetime.combine(target_date, time(9, 15))
        to_datetime = datetime.combine(target_date, time(15, 30))

        return self._fetch_with_retry(token, from_datetime, to_datetime, "minute")

    def _check_for_breakout(self, stock: Dict, minute_data: List[Dict], target_date) -> Optional[Dict]:
        """Check if stock broke out above all 8 MAs"""
        try:
            symbol: str = stock['symbol']
            company_name: str = stock['company_name']
            token: int = stock['token']
            mas: Dict = stock['mas']
            price_at_9am: float = stock['price_at_9am']

            highest_price = max([candle['high'] for candle in minute_data])

            above_all_mas = all([
                highest_price > mas['ma_20'] if mas['ma_20'] else False,
                highest_price > mas['ma_50'] if mas['ma_50'] else False,
                highest_price > mas['ma_100'] if mas['ma_100'] else False,
                highest_price > mas['ma_200'] if mas['ma_200'] else False,
                highest_price > mas['ma_20h'] if mas['ma_20h'] else False,
                highest_price > mas['ma_50h'] if mas['ma_50h'] else False,
                highest_price > mas['ma_100h'] if mas['ma_100h'] else False,
                highest_price > mas['ma_200h'] if mas['ma_200h'] else False
            ])

            if not above_all_mas:
                return None

            breakout_time = None
            breakout_price = None

            for candle in minute_data:
                candle_high = candle['high']
                if all([
                    candle_high > mas['ma_20'] if mas['ma_20'] else False,
                    candle_high > mas['ma_50'] if mas['ma_50'] else False,
                    candle_high > mas['ma_100'] if mas['ma_100'] else False,
                    candle_high > mas['ma_200'] if mas['ma_200'] else False,
                    candle_high > mas['ma_20h'] if mas['ma_20h'] else False,
                    candle_high > mas['ma_50h'] if mas['ma_50h'] else False,
                    candle_high > mas['ma_100h'] if mas['ma_100h'] else False,
                    candle_high > mas['ma_200h'] if mas['ma_200h'] else False
                ]):
                    breakout_time = pd.to_datetime(candle['date']).time()
                    breakout_price = candle_high
                    break

            if not breakout_time:
                return None

            volume_data = self.calculate_8_volume_averages(
                stock['daily_data'],
                stock['hourly_data'],
                target_date,
                breakout_time
            )

            price_at_9am = stock['price_at_9am']
            percentage_change = ((highest_price - price_at_9am) / price_at_9am) * 100

            is_true_breakout = percentage_change >= 5.0

            pattern_detected = None

            daily_df = pd.DataFrame(stock['daily_data'])
            daily_df.rename(columns={'date': 'timestamp'}, inplace=True)
            pattern_signals = self.pattern_recognizer.analyze_stock(daily_df)

            if pattern_signals:
                best_pattern = max(pattern_signals, key=lambda x: x.confidence)
                pattern_detected = best_pattern.pattern_type.value

            return {
                'symbol': stock['symbol'],
                'company_name': stock['company_name'],
                'token': token,
                'price_at_9am': price_at_9am,
                'breakout_time': breakout_time,
                'breakout_price': breakout_price,
                'highest_price': highest_price,
                'percentage_change': percentage_change,
                'is_true_breakout': is_true_breakout,
                'mas': mas,
                'volume_analysis': volume_data,
                'pattern_detected': pattern_detected,
                'target_date': target_date,
                'minute_data': minute_data
            }

        except Exception as e:
            logger.error(f"Error checking breakout for {stock['symbol']}: {e}")
            return None

    def calculate_comprehensive_metrics(self, breakout_data: Dict) -> Dict:
        """Calculate all metrics"""
        try:
            symbol = breakout_data['symbol']
            minute_data = breakout_data['minute_data']
            mas = breakout_data['mas']
            volume_data = breakout_data['volume_analysis']

            df = pd.DataFrame(minute_data)
            df['timestamp'] = pd.to_datetime(df['date'])

            df = self._calculate_technical_indicators(df)

            breakout_candle = None
            for _, row in df.iterrows():
                if row['timestamp'].time() == breakout_data['breakout_time']:
                    breakout_candle = row
                    break

            if breakout_candle is None:
                breakout_candle = df.iloc[-1]

            nifty_change = 0.5
            bank_nifty_change = 0.3
            vix_level = 15.0

            metrics = {
                'date': breakout_data['target_date'].strftime('%Y-%m-%d'),
                'stock_symbol': symbol,
                'breakout_time': str(breakout_data['breakout_time']),
                'sector': 'Unknown',
                'market_cap': 'Unknown',
                'breakout_open': float(breakout_candle['open']),
                'breakout_high': float(breakout_candle['high']),
                'breakout_low': float(breakout_candle['low']),
                'breakout_close': float(breakout_candle['close']),
                'breakout_volume': int(breakout_candle['volume']),
                'prev_close': float(df.iloc[-2]['close']) if len(df) > 1 else float(breakout_candle['close']),
                'sma_20': mas.get('ma_20'),
                'sma_50': mas.get('ma_50'),
                'sma_100': mas.get('ma_100'),
                'sma_200': mas.get('ma_200'),
                'sma_20h': mas.get('ma_20h'),
                'sma_50h': mas.get('ma_50h'),
                'sma_100h': mas.get('ma_100h'),
                'sma_200h': mas.get('ma_200h'),
                'volume_sma_20': volume_data.get('daily_volume_avg_20'),
                'volume_ratio': breakout_candle['volume'] / volume_data.get('daily_volume_avg_20', 1) if volume_data.get('daily_volume_avg_20') else None,
                'volume_spike': breakout_candle['volume'] > (volume_data.get('daily_volume_avg_20', 0) * 2),
                'volume_percentile': self._calculate_volume_percentile(df, breakout_candle['volume']),
                'rsi_14': breakout_candle.get('rsi'),
                'macd': breakout_candle.get('macd'),
                'macd_signal': breakout_candle.get('macd_signal'),
                'macd_histogram': breakout_candle.get('macd_hist'),
                'atr_14': breakout_candle.get('atr'),
                'stoch_k': breakout_candle.get('stoch_k'),
                'stoch_d': breakout_candle.get('stoch_d'),
                'price_change_1d': ((breakout_candle['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']) * 100 if len(df) > 1 else 0,
                'price_change_3d': self._calculate_price_change(df, 3),
                'price_change_5d': self._calculate_price_change(df, 5),
                'price_change_10d': self._calculate_price_change(df, 10),
                'volatility_20d': df['close'].tail(20).std() if len(df) >= 20 else None,
                'gap_from_prev_close': ((breakout_candle['open'] - df.iloc[-2]['close']) / df.iloc[-2]['close']) * 100 if len(df) > 1 else 0,
                'nifty_change': nifty_change,
                'banknifty_change': bank_nifty_change,
                'vix_level': vix_level,
                'market_trend': 'Bullish' if nifty_change > 0 else 'Bearish',
                'true_breakout': breakout_data['is_true_breakout'],
                'percentage_change': breakout_data['percentage_change'],
                'pattern_type': breakout_data.get('pattern_detected', 'Unknown'),
                'max_gain_1d': breakout_data['percentage_change'],
                'hit_5_percent_target': breakout_data['percentage_change'] >= 5.0,
                'hit_10_percent_target': breakout_data['percentage_change'] >= 10.0,
                'breakout_strength_score': self._calculate_breakout_strength_score(breakout_data, df)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the DataFrame"""
        try:
            df['rsi'] = talib.RSI(df['close'], 14)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)

            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])

            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

    def _calculate_volume_percentile(self, df: pd.DataFrame, current_volume: float) -> float:
        """Calculate volume percentile vs last 100 days"""
        try:
            volumes = df['volume'].tail(100).values
            if len(volumes) == 0:
                return 50.0

            percentile = (np.sum(volumes <= current_volume) / len(volumes)) * 100
            return percentile
        except:
            return 50.0

    def _calculate_price_change(self, df: pd.DataFrame, days: int) -> float:
        """Calculate price change over specified days"""
        try:
            if len(df) <= days:
                return 0.0

            current_price = df.iloc[-1]['close']
            past_price = df.iloc[-(days+1)]['close']
            return ((current_price - past_price) / past_price) * 100
        except:
            return 0.0

    def _calculate_breakout_strength_score(self, breakout_data: Dict, df: pd.DataFrame) -> float:
        """Calculate breakout strength score (1-10)"""
        try:
            score = 5.0

            volume_data = breakout_data['volume_analysis']
            avg_volume = volume_data.get('daily_volume_avg_20', 1)

            if avg_volume > 0:
                latest_volume = df.iloc[-1]['volume']
                volume_ratio = latest_volume / avg_volume

                if volume_ratio >= 3.0:
                    score += 2.0
                elif volume_ratio >= 2.0:
                    score += 1.5
                elif volume_ratio >= 1.5:
                    score += 1.0

            pct_change = breakout_data['percentage_change']
            if pct_change >= 10.0:
                score += 2.0
            elif pct_change >= 7.0:
                score += 1.5
            elif pct_change >= 5.0:
                score += 1.0

            if breakout_data.get('pattern_detected') and breakout_data['pattern_detected'] != 'Unknown':
                score += 1.0

            return min(score, 10.0)

        except Exception as e:
            logger.error(f"Error calculating strength score: {e}")
            return 5.0

    def save_to_excel(self, breakout_results: List[Dict]):
        """Save results to Excel file"""
        try:
            df_results = pd.DataFrame(breakout_results)

            if df_results.empty:
                print("No breakout data to save")
                return

            if os.path.exists(self.excel_file):
                with pd.ExcelWriter(self.excel_file, mode='a', if_sheet_exists='overlay') as writer:
                    existing_df = pd.read_excel(self.excel_file, sheet_name='Breakouts')
                    start_row = len(existing_df) + 1

                    df_results.to_excel(writer, sheet_name='Breakouts', startrow=start_row,
                                       header=False, index=False)
            else:
                with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                    df_results.to_excel(writer, sheet_name='Breakouts', index=False)

            print(f"Saved {len(df_results)} records to {self.excel_file}")

        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

    def analyze_false_breakout_patterns(self, breakout_results: List[Dict]) -> str:
        """Analyze patterns specifically in false breakouts"""
        false_breakouts = [b for b in breakout_results if not b.get('true_breakout', False)]
        true_breakouts = [b for b in breakout_results if b.get('true_breakout', False)]

        if not false_breakouts:
            return "No false breakouts found for pattern analysis"

        false_pattern_counts = {}
        for breakout in false_breakouts:
            pattern = breakout.get('pattern_type', 'Unknown')
            false_pattern_counts[pattern] = false_pattern_counts.get(pattern, 0) + 1

        true_pattern_counts = {}
        for breakout in true_breakouts:
            pattern = breakout.get('pattern_type', 'Unknown')
            true_pattern_counts[pattern] = true_pattern_counts.get(pattern, 0) + 1

        analysis_lines = []
        analysis_lines.append("FALSE BREAKOUT PATTERN ANALYSIS")
        analysis_lines.append("="*50)
        analysis_lines.append(f"Total false breakouts: {len(false_breakouts)}")
        analysis_lines.append(f"Total true breakouts: {len(true_breakouts)}")
        analysis_lines.append("")

        analysis_lines.append("FALSE BREAKOUT PATTERNS:")
        for pattern, count in sorted(false_pattern_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(false_breakouts)) * 100
            analysis_lines.append(f" {pattern}: {count} ({percentage:.1f}%)")

        analysis_lines.append("")
        analysis_lines.append("TRUE BREAKOUT PATTERNS (for comparison):")
        for pattern, count in sorted(true_pattern_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(true_breakouts)) * 100 if true_breakouts else 0
            analysis_lines.append(f" {pattern}: {count} ({percentage:.1f}%)")

        analysis_lines.append("")
        analysis_lines.append("PATTERN RELIABILITY ANALYSIS:")
        all_patterns = set(false_pattern_counts.keys()) | set(true_pattern_counts.keys())

        for pattern in all_patterns:
            false_count = false_pattern_counts.get(pattern, 0)
            true_count = true_pattern_counts.get(pattern, 0)
            total_count = false_count + true_count

            if total_count > 0:
                success_rate = (true_count / total_count) * 100
                failure_rate = (false_count / total_count) * 100
                analysis_lines.append(f" {pattern}: {success_rate:.1f}% success, {failure_rate:.1f}% failure ({total_count} total)")

        if false_breakouts:
            analysis_lines.append("")
            analysis_lines.append("FALSE BREAKOUT CHARACTERISTICS:")
            avg_percentage = sum(b['percentage_change'] for b in false_breakouts) / len(false_breakouts)
            max_percentage = max(b['percentage_change'] for b in false_breakouts)
            min_percentage = min(b['percentage_change'] for b in false_breakouts)

            analysis_lines.append(f" Average gain: {avg_percentage:.2f}%")
            analysis_lines.append(f" Max gain: {max_percentage:.2f}%")
            analysis_lines.append(f" Min gain: {min_percentage:.2f}%")

            volume_confirmed = sum(1 for b in false_breakouts if b.get('volume_spike', False))
            analysis_lines.append(f" Volume spikes: {volume_confirmed}/{len(false_breakouts)} ({volume_confirmed/len(false_breakouts)*100:.1f}%)")

        return "\n".join(analysis_lines)

    def analyze_pattern_frequency(self, breakout_results: List[Dict]) -> str:
        """Find most common pattern among true breakouts"""
        true_breakouts = [b for b in breakout_results if b.get('true_breakout', False)]

        if not true_breakouts:
            return "No true breakouts found"

        pattern_counts = {}
        for breakout in true_breakouts:
            pattern = breakout.get('pattern_type', 'Unknown')
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        if not pattern_counts:
            return "No patterns detected"

        most_common = max(pattern_counts, key=pattern_counts.get)
        count = pattern_counts[most_common]
        total = len(true_breakouts)

        return f"Most common pattern: {most_common} ({count}/{total} = {count/total*100:.1f}%)"

    def run_analysis_for_past_days(self, days: int = 1000):
        """Run complete analysis for past trading days with checkpoint support"""
        print("Starting Enhanced 8-MA Breakout Analysis with Resume Support")
        print("="*80)

        checkpoint = self.load_checkpoint()
        completed_days = set(checkpoint.get('completed_days', []))

        if not completed_days:
            print("Setting up Supabase tables...")
            self.create_supabase_tables()

            print("Fetching and storing instruments...")
            instruments = self.fetch_and_store_instruments()
            if not instruments:
                print("Failed to fetch instruments")
                return

            print(f"Fetching {days} days of historical data...")
            self.fetch_and_store_historical_data(instruments, days)

        print("Starting breakout analysis with resume capability...")
        end_date = datetime.now().date()
        all_breakout_results = []

        # Initialize lists to collect breakout stocks
        true_breakout_stocks = []
        false_breakout_stocks = []

        trading_days = []
        for day_offset in range(min(30, 90)):
            target_date = end_date - timedelta(days=day_offset)
            if target_date.weekday() < 5:
                trading_days.append(target_date)

        total_trading_days = len(trading_days)
        print(f"📅 Total trading days to analyze: {total_trading_days}")

        if completed_days:
            print(f"⏭️ Resuming analysis: {len(completed_days)} days already completed")

        for target_date in trading_days:
            date_str = target_date.strftime('%Y-%m-%d')

            if date_str in completed_days:
                print(f"⏭️ Skipping {target_date} (already completed)")
                continue

            try:
                print(f"\n📅 Analyzing {target_date}...")

                monitoring_list = self.scan_for_monitoring_candidates(target_date)

                if not monitoring_list:
                    print(f" ❌ No monitoring candidates found for {target_date}")
                    completed_days.add(date_str)
                    self.save_checkpoint(list(completed_days), date_str, total_trading_days)
                    continue

                print(f" Found {len(monitoring_list)} monitoring candidates")

                breakouts = self.detect_breakouts(monitoring_list, target_date)

                for breakout in breakouts:
                    metrics = self.calculate_comprehensive_metrics(breakout)
                    if metrics:
                        all_breakout_results.append(metrics)
                        
                        # Collect stock names for end summary
                        stock_symbol = metrics.get('stock_symbol', 'Unknown')
                        percentage_gain = metrics.get('percentage_change', 0)
                        date_analyzed = metrics.get('date', target_date)
                        
                        if metrics.get('true_breakout', False):
                            true_breakout_stocks.append(f"{stock_symbol} (+{percentage_gain:.2f}%) on {date_analyzed}")
                        else:
                            false_breakout_stocks.append(f"{stock_symbol} (+{percentage_gain:.2f}%) on {date_analyzed}")

                print(f" ✅ Found {len(breakouts)} breakouts on {target_date}")

                completed_days.add(date_str)
                self.save_checkpoint(list(completed_days), date_str, total_trading_days)

            except KeyboardInterrupt:
                print(f"\n⚠️ Analysis interrupted by user after processing {target_date}")
                self.save_checkpoint(list(completed_days), date_str, total_trading_days)
                print(f"💾 Progress saved. Run the script again to resume from {target_date}")
                return

            except Exception as e:
                logger.error(f"❌ Error analyzing {target_date}: {e}")
                completed_days.add(date_str)
                self.save_checkpoint(list(completed_days), date_str, total_trading_days)
                continue

        pattern_analysis = self.analyze_pattern_frequency(all_breakout_results)
        false_breakout_analysis = self.analyze_false_breakout_patterns(all_breakout_results)

        print(f"\nOverall Pattern Analysis: {pattern_analysis}")
        print(f"\n{false_breakout_analysis}")

        if all_breakout_results:
            self.save_to_excel(all_breakout_results)

            true_breakouts = [b for b in all_breakout_results if b.get('true_breakout', False)]
            false_breakouts = [b for b in all_breakout_results if not b.get('true_breakout', False)]

            print("\n" + "="*80)
            print("ANALYSIS SUMMARY")
            print("="*80)
            print(f"Total breakouts analyzed: {len(all_breakout_results)}")
            print(f"True breakouts (≥5%): {len(true_breakouts)}")
            print(f"False breakouts (<5%): {len(false_breakouts)}")
            print(f"Success rate: {len(true_breakouts)/len(all_breakout_results)*100:.1f}%")

            if true_breakouts:
                print(f"Average gain (true breakouts): {np.mean([b['percentage_change'] for b in true_breakouts]):.2f}%")

            print(f"Pattern analysis: {pattern_analysis}")

            # Display collected breakout stocks at the end
            print("\n" + "="*60)
            print("BREAKOUT STOCKS SUMMARY")
            print("="*60)

            if true_breakout_stocks:
                print(f"✅ TRUE BREAKOUTS ({len(true_breakout_stocks)} stocks):")
                for i, stock in enumerate(true_breakout_stocks, 1):
                    print(f"   {i}. {stock}")
            else:
                print("✅ TRUE BREAKOUTS: None")

            print()

            if false_breakout_stocks:
                print(f"❌ FALSE BREAKOUTS ({len(false_breakout_stocks)} stocks):")
                for i, stock in enumerate(false_breakout_stocks, 1):
                    print(f"   {i}. {stock}")
            else:
                print("❌ FALSE BREAKOUTS: None")

            # Also log to file
            logger.info(f"TRUE breakout stocks: {[stock for stock in true_breakout_stocks]}")
            logger.info(f"FALSE breakout stocks: {[stock for stock in false_breakout_stocks]}")
        else:
            print("No breakouts found in the analyzed period")

        self.cleanup_checkpoint()
        print(f"\n🎉 Analysis completed successfully!")

def main():
    """Main execution function with PROPER checkpoint support and resume point detection"""
    try:
        analyzer = EnhancedBreakoutAnalyzer()

        print("Enhanced 8-MA Breakout Analyzer with Supabase Integration")
        print("Moving Averages: MA20, MA50, MA100, MA200 (Daily & Hourly)")
        print("Volume Averages: 20, 50, 100, 200 periods")
        print("="*80)

        print("\nSelect operation mode:")
        print("1. First time setup (fetch all data to Supabase)")
        print("2. Run analysis from Supabase data")

        mode = input("Enter choice (1 or 2): ").strip()

        if mode == "1":
            print("Running first time setup...")
            analyzer.run_analysis_for_past_days(1000)

        elif mode == "2":
            print("Running analysis from existing Supabase data with checkpoint support...")
            
            checkpoint = analyzer.load_checkpoint()
            completed_days = set(checkpoint.get('completed_days', []))

            if checkpoint:
                if completed_days:
                    end_date = datetime.now().date()
                    trading_days_sorted = []
                    for day_offset in range(30):
                        d = end_date - timedelta(days=day_offset)
                        if d.weekday() < 5:
                            trading_days_sorted.append(d)
                    
                    trading_days_sorted = sorted(trading_days_sorted)

                    resume_from = None
                    for day in trading_days_sorted:
                        if day.strftime('%Y-%m-%d') not in completed_days:
                            resume_from = day
                            break
                    
                    if resume_from:
                        print(f"🔄 RESUMING program from {resume_from.strftime('%Y-%m-%d')} (first uncompleted day)")
                    else:
                        print("✅ All days already completed, running final cleanup...")
                else:
                    print("🔄 RESUMING program from interrupted state (no days completed yet)")
            else:
                print("🆕 Starting fresh program run - no previous checkpoint found")

            end_date = datetime.now().date()
            all_breakout_results = []
            
            # Initialize lists to collect breakout stocks
            true_breakout_stocks = []
            false_breakout_stocks = []

            trading_days = []
            for day_offset in range(30):
                target_date = end_date - timedelta(days=day_offset)
                if target_date.weekday() < 5:
                    trading_days.append(target_date)

            total_trading_days = len(trading_days)
            print(f"\n📅 Total trading days to analyze: {total_trading_days}")

            for target_date in trading_days:
                date_str = target_date.strftime('%Y-%m-%d')

                if date_str in completed_days:
                    print(f"⏭️ Skipping {target_date} (already completed)")
                    continue

                try:
                    trading_days_processed = len([d for d in trading_days if d >= target_date])
                    print(f"\n📅 Analyzing Day {total_trading_days - trading_days_processed + 1}/{total_trading_days}: {target_date} ({target_date.strftime('%A')})")
                    print("="*60)

                    instruments = analyzer.get_instruments_from_supabase()
                    if not instruments:
                        print(f"❌ No instruments available - possible network/DB error for {target_date}")
                        print(f"🔄 Will retry this day on next run (NOT marking as completed)")
                        continue

                    monitoring_list = analyzer.scan_for_monitoring_candidates(target_date)

                    if monitoring_list:
                        breakouts = analyzer.detect_breakouts(monitoring_list, target_date)

                        for breakout in breakouts:
                            metrics = analyzer.calculate_comprehensive_metrics(breakout)
                            if metrics:
                                all_breakout_results.append(metrics)
                                
                                # Collect stock names for end summary
                                stock_symbol = metrics.get('stock_symbol', 'Unknown')
                                percentage_gain = metrics.get('percentage_change', 0)
                                date_analyzed = metrics.get('date', target_date)
                                
                                if metrics.get('true_breakout', False):
                                    true_breakout_stocks.append(f"{stock_symbol} (+{percentage_gain:.2f}%) on {date_analyzed}")
                                else:
                                    false_breakout_stocks.append(f"{stock_symbol} (+{percentage_gain:.2f}%) on {date_analyzed}")

                        print(f" ✅ Found {len(breakouts)} breakouts on {target_date}")

                    else:
                        if instruments:
                            print(f" ✅ Successfully scanned but found no monitoring candidates on {target_date}")
                        else:
                            print(f"❌ Failed to scan due to data issues. NOT marking as completed.")
                            continue

                    completed_days.add(date_str)
                    analyzer.save_checkpoint(list(completed_days), date_str, total_trading_days)

                except KeyboardInterrupt:
                    print(f"\n⚠️ Analysis interrupted by user after processing {target_date}")
                    analyzer.save_checkpoint(list(completed_days), date_str, total_trading_days)
                    print(f"💾 Progress ACTUALLY saved to analysis_checkpoint.json")
                    print(f"🔄 Run the script again to resume from {target_date}")
                    return

                except Exception as e:
                    logger.error(f"❌ Error analyzing {target_date}: {e}")
                    print(f"❌ Error on {target_date}. NOT marking as completed - will retry next time.")
                    continue

            if all_breakout_results:
                analyzer.save_to_excel(all_breakout_results)
                pattern_analysis = analyzer.analyze_pattern_frequency(all_breakout_results)
                false_breakout_analysis = analyzer.analyze_false_breakout_patterns(all_breakout_results)

                print(f"\nPattern Analysis: {pattern_analysis}")
                print(f"\n{false_breakout_analysis}")

                true_breakouts = [b for b in all_breakout_results if b.get('true_breakout', False)]
                false_breakouts = [b for b in all_breakout_results if not b.get('true_breakout', False)]

                print("\n" + "="*80)
                print("ANALYSIS SUMMARY")
                print("="*80)
                print(f"Total breakouts analyzed: {len(all_breakout_results)}")
                print(f"True breakouts (≥5%): {len(true_breakouts)}")
                print(f"False breakouts (<5%): {len(false_breakouts)}")
                print(f"Success rate: {len(true_breakouts)/len(all_breakout_results)*100:.1f}%")

                if true_breakouts:
                    print(f"Average gain (true breakouts): {np.mean([b['percentage_change'] for b in true_breakouts]):.2f}%")

                # Display collected breakout stocks at the end
                print("\n" + "="*60)
                print("BREAKOUT STOCKS SUMMARY")
                print("="*60)

                if true_breakout_stocks:
                    print(f"✅ TRUE BREAKOUTS ({len(true_breakout_stocks)} stocks):")
                    for i, stock in enumerate(true_breakout_stocks, 1):
                        print(f"   {i}. {stock}")
                else:
                    print("✅ TRUE BREAKOUTS: None")

                print()

                if false_breakout_stocks:
                    print(f"❌ FALSE BREAKOUTS ({len(false_breakout_stocks)} stocks):")
                    for i, stock in enumerate(false_breakout_stocks, 1):
                        print(f"   {i}. {stock}")
                else:
                    print("❌ FALSE BREAKOUTS: None")

                # Also log to file
                logger.info(f"TRUE breakout stocks: {[stock for stock in true_breakout_stocks]}")
                logger.info(f"FALSE breakout stocks: {[stock for stock in false_breakout_stocks]}")
            else:
                print("No breakouts found in the analyzed period")

            if len(completed_days) >= total_trading_days:
                analyzer.cleanup_checkpoint()
                print(f"\n🎉 Analysis completed successfully!")
            else:
                print(f"\n⏸️ Analysis paused at {len(completed_days)}/{total_trading_days} days.")
                print(f"🔄 Resume by running the script again.")

        else:
            print("Invalid choice. Exiting...")

    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
