import datetime
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import logging
from supabase import create_client, Client
from decimal import Decimal, ROUND_CEILING
from multiprocessing import cpu_count
import concurrent.futures
import time
from kiteconnect import KiteConnect
from collections import Counter
import subprocess
import sys
import requests
import io

# -------------------------- Logger Setup --------------------------
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    
    file_handler = logging.FileHandler('kite_algo_breakouts.log', encoding='utf-8', mode='a')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# -------------------------- Config & Auth --------------------------
load_dotenv()
SUPABASE_URL = os.getenv("N_SUPABASE_URL")
SUPABASE_KEY = os.getenv("N_SUPABASE_ANON_KEY")
KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

if not all([SUPABASE_URL, SUPABASE_KEY, KITE_API_KEY, KITE_ACCESS_TOKEN]):
    raise ValueError("Missing Supabase or Kite credentials in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Create placeholder for global kite client (do NOT initialize at import-time)
kite = None

def init_kite():
    """Initialize the global Kite client. Call this from main() — not at import time."""
    global kite
    if not all([KITE_API_KEY, KITE_ACCESS_TOKEN]):
        raise ValueError("Missing Kite credentials in .env")
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(KITE_ACCESS_TOKEN)
    try:
        # Validate token once (this will do the network call) — keep it inside main()
        kite.profile()
    except Exception as e:
        logger.error(f"❌ Kite authentication failed: {e}")
        raise ValueError("Failed to authenticate with Kite. Check your access token.")


# -------------------------- Helper Functions --------------------------

def next_price_above(value, tick):
    """Round up to next valid tick."""
    dv = Decimal(str(value))
    dt = Decimal(str(tick))
    n = (dv / dt).to_integral_value(rounding=ROUND_CEILING)
    candidate = n * dt
    if candidate <= dv:
        candidate = (n + 1) * dt
    return float(candidate)

def get_kite_instruments(nse_symbols):
    """Fetches active NSE stocks (~2200) AND Nifty 50 Index from NSE list."""
    logger.info("Fetching instruments from Kite...")
    
    nse_symbol_set = set(nse_symbols)
    max_retries = 3
    nse_instruments = None

    # Fetch NSE with retry
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2)
            nse_instruments = kite.instruments("NSE")
            break
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch NSE (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error("❌ Failed to fetch NSE instruments after all retries.")
                return {}
    
    if not nse_instruments:
        return {}
    
    symbol_map = {}
    nifty_token = None

    # A. Find Nifty 50 Token from NSE list
    for ins in nse_instruments:
        if ins.get('tradingsymbol') == 'NIFTY 50' or ins.get('name') == 'NIFTY 50':
            nifty_token = ins['instrument_token']
            logger.info(f"✅ Found NIFTY 50 token: {nifty_token}")
            break

    # B. Filter Stocks from NSE list
    for ins in nse_instruments:
        symbol = ins["tradingsymbol"]
        if symbol not in nse_symbol_set:
            continue
        symbol_map[symbol] = {
            "token": ins["instrument_token"],
            "tick_size": ins.get("tick_size", 0.05),
            "tradingsymbol": symbol
        }

    logger.info(f"Filtered down to {len(symbol_map)} active NSE equity symbols.")
    
    # C. Add Nifty 50 to the map
    if nifty_token:
        symbol_map["^NSEI"] = {
            "token": nifty_token, 
            "tick_size": 0.05, 
            "tradingsymbol": "NIFTY 50"
        }
    else:
        logger.warning("⚠️ WARNING: NIFTY 50 token not found in NSE list!")
        
    return symbol_map

# -------------------------- Technical Indicators --------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    if df.empty or not all(c in df.columns for c in ['high', 'low', 'close']):
        return np.nan
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_prev_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_prev_close'] = np.abs(df['low'] - df['close'].shift(1))
    tr = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.iloc[-1] if not atr.empty else np.nan

def calculate_pivots(prev_high, prev_low, prev_close):
    if pd.isna(prev_high) or pd.isna(prev_low) or pd.isna(prev_close):
        return {}
    pivot = (prev_high + prev_low + prev_close) / 3
    s1 = (pivot * 2) - prev_high
    r1 = (pivot * 2) - prev_low
    s2 = pivot - (prev_high - prev_low)
    r2 = pivot + (prev_high - prev_low)
    return {
        'pivot': round(pivot, 2),
        's1': round(s1, 2), 'r1': round(r1, 2),
        's2': round(s2, 2), 'r2': round(r2, 2)
    }

# -------------------------- Kite Data Fetching --------------------------

def fetch_history_safe(token, from_date, to_date, interval):
    """
    Wrapper for kite.historical_data with retry logic and error handling.
    Takes in the time period like minute, daily and hourly.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Kite historical_data returns list of dicts
            data = kite.historical_data(token, from_date, to_date, interval)
            df = pd.DataFrame(data)
            if not df.empty:
                # Normalize columns to lowercase
                df.rename(columns=str.lower, inplace=True)
                # Ensure date is timezone naive for compatibility with existing logic
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['date'] = df['date'].dt.tz_localize(None)
            return df
        except Exception as e:
            if "Too many requests" in str(e):
                time.sleep(1 + attempt) # Backoff
            elif attempt == max_retries - 1:
                logger.error(f"Failed to fetch {token} data: {e}")
                return pd.DataFrame()
            time.sleep(0.5)
    return pd.DataFrame()

def preload_data_kite(symbols_batch, instrument_map, from_date, to_date):
    """
    Fetches daily and 15m data for a batch of symbols using Kite.
    Respects rate limits (approx 3 req/sec).
    """
    data_dict = {}
    
    # Nifty Token (ensure we have it)
    nifty_meta = instrument_map.get("^NSEI")
    
    # List of symbols to fetch (Batch + Nifty)
    targets = [s for s in symbols_batch if s in instrument_map]
    if nifty_meta and "^NSEI" not in targets:
        targets.insert(0, "^NSEI")

    for symbol in targets:
        meta = instrument_map[symbol]
        token = meta['token']
        
        # 1. Fetch Daily Data
        df_daily = fetch_history_safe(token, from_date, to_date, "day")
        if df_daily.empty:
            continue
            
        # 2. Fetch Intraday (15minute) - Kite allows max ~100 days for 15m.
        intraday_start = max(from_date, to_date - datetime.timedelta(days=60))
        df_minute = fetch_history_safe(token, intraday_start, to_date, "15minute")
        
        if df_minute.empty:
            continue

        # Create Hourly aggregate from minute data for Mas
        df_minute_sorted = df_minute.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        df_hourly = df_minute_sorted.set_index('date').resample('60min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()

        key = 'nifty' if symbol == "^NSEI" else symbol
        data_dict[key] = {
            'daily': df_daily,
            'minute': df_minute_sorted,
            'hourly': df_hourly,
            'tick_size': meta['tick_size']
        }
        
        time.sleep(0.4) 

    return data_dict

# -------------------------- MA Precompute --------------------------
def precompute_mas(data_dict, symbols):
    ma_dict = {}
    
    # Handle Nifty separately if needed, or include in loop
    keys = list(symbols)
    if 'nifty' in data_dict:
        keys.append('nifty')

    for symbol in keys:
        if symbol not in data_dict:
            continue
        ma_dict[symbol] = {}
        for interval in ['daily', 'hourly']:
            df = data_dict[symbol].get(interval)
            if df is None or df.empty or 'close' not in df.columns:
                continue
            ma_df = pd.DataFrame({'date': df['date'].copy()})
            close = df['close']
            for p in [20, 50, 100, 200]:
                if len(close) >= p:
                    # Shift(1) because we use yesterday's MA to judge today's entry
                    ma_df[f'ma{p}'] = close.rolling(p).mean().shift(1)
                else:
                    ma_df[f'ma{p}'] = np.nan
            ma_dict[symbol][interval] = ma_df
    return ma_dict

# -------------------------- Logic Components --------------------------
def get_ma_thresholds(ma_dict, symbol, analysis_date):
    daily = ma_dict.get(symbol, {}).get('daily')
    hourly = ma_dict.get(symbol, {}).get('hourly')
    if daily is None or daily.empty:
        return 'no_data', None

    # Date comparison: ensure both are dates
    day_row = daily[daily['date'].dt.date == analysis_date]
    if day_row.empty:
        return 'no_data', None
    
    day_mas = [day_row.iloc[0][f'ma{p}'] for p in [20,50,100,200]]

    # For hourly, we take the last closed candle *before* the analysis date starts
    if hourly is not None and not hourly.empty:
        prev_hourly = hourly[hourly['date'] < pd.Timestamp(analysis_date)]
        if not prev_hourly.empty:
            last = prev_hourly.iloc[-1]
            day_mas.extend([last[f'ma{p}'] for p in [20,50,100,200]])

    if any(pd.isna(x) for x in day_mas):
        return 'nan', None
    return 'success', (min(day_mas), max(day_mas))

def check_volume_spike(df_subset, lookback=20):
    if len(df_subset) <= lookback:
        return False, 1.0
    
    historical_volume = df_subset['volume'].iloc[:-1]
    if len(historical_volume) < lookback:
        return False, 1.0
    
    avg_vol = historical_volume.tail(lookback).mean()
    current_vol = df_subset['volume'].iloc[-1]
    
    if pd.isna(avg_vol) or avg_vol == 0:
        return False, 1.0
    
    multiplier = current_vol / avg_vol
    is_spike = multiplier > 1.5
    return is_spike, round(multiplier, 2)

def calculate_bb_squeeze(df, period=20, std_dev=2, kc_period=20, kc_atr_mult=1.5):
    """
    Detects Bollinger Band Squeeze.
    Returns True if BB width is narrower than Keltner Channel width.
    """
    if len(df) < period:
        return False
    
    # Bollinger Bands
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    bb_upper = sma + (std_dev * std)
    bb_lower = sma - (std_dev * std)
    bb_width = bb_upper - bb_lower
    
    # Keltner Channels
    kc_middle = df['close'].rolling(kc_period).mean()
    atr = calculate_atr(df, kc_period)
    kc_upper = kc_middle.iloc[-1] + (kc_atr_mult * atr) if pd.notna(atr) else np.nan
    kc_lower = kc_middle.iloc[-1] - (kc_atr_mult * atr) if pd.notna(atr) else np.nan
    kc_width = kc_upper - kc_lower

    # Squeeze occurs when BB is inside KC
    if pd.notna(bb_width.iloc[-1]) and pd.notna(kc_width):
        return bb_width.iloc[-1] < kc_width
    return False

# -------------------------- Core Processing --------------------------
def process_stock_day(task):
    symbol = task['symbol']
    date = task['analysis_date']
    intraday = task['intraday_data']
    daily = task['daily_data']
    hourly = task['hourly_data'] # Ensure this is passed in main()
    tick_size = task['tick_size']
    nifty_intraday = task.get('nifty_intraday')
    nifty_50ma = task.get('nifty_50ma')

    if intraday.empty or daily.empty or hourly is None or hourly.empty:
        return ('no_data', symbol)

    # --- 1. GET TODAY'S OPEN ---
    df_today = intraday[intraday['date'].dt.date == date].sort_values('date').reset_index(drop=True)
    if df_today.empty: return ('no_intraday_today', symbol)
    day_open = float(df_today.iloc[0]['open'])

    # --- 2. RE-CALCULATE MAs (History + Day Open) ---
    
    # A. Daily MAs
    daily_hist = daily[daily['date'].dt.date < date].sort_values('date').reset_index(drop=True)
    if len(daily_hist) < 200: return ('not_enough_history', symbol)
    
    daily_prices = list(daily_hist['close'].values)
    daily_prices.append(day_open) 
    daily_series = pd.Series(daily_prices)
    
    daily_mas = []
    for p in [20, 50, 100, 200]:
        val = daily_series.rolling(window=p).mean().iloc[-1]
        daily_mas.append(float(val))

    # B. Hourly MAs
    hourly_hist = hourly[hourly['date'] < pd.Timestamp(date)].sort_values('date').reset_index(drop=True)
    if len(hourly_hist) < 200: return ('not_enough_history', symbol)
    
    hourly_prices = list(hourly_hist['close'].values)
    hourly_prices.append(day_open)
    hourly_series = pd.Series(hourly_prices)
    
    hourly_mas = []
    for p in [20, 50, 100, 200]:
        val = hourly_series.rolling(window=p).mean().iloc[-1]
        hourly_mas.append(float(val))
        
    # --- 3. DETERMINE THRESHOLDS ---
    mas_list = daily_mas + hourly_mas
    if any(pd.isna(x) for x in mas_list): return ('nan_mas', symbol)
    
    # In this script, we typically look for the Min MA (Dip) and Max MA (Breakout)
    min_ma = min(mas_list)
    max_ma = max(mas_list)

    # --- 4. MONITOR CHECK ---
    # Stock must start below the Lowest MA
    if day_open < min_ma:
        entry_time = df_today.iloc[0]['date'] - pd.Timedelta(seconds=1)
    else:
        return ('no_dip', symbol)

    # --- 5. BREAKOUT LOGIC ---
    post_dip = df_today[df_today['date'] > entry_time].copy()

    # Calculate the precise breakout price (1 tick above max_ma)
    breakout_threshold = next_price_above(max_ma, tick_size)

    # Check if either CLOSE or OPEN crossed the threshold
    post_dip['breakout'] = (post_dip['close'] >= breakout_threshold) | (post_dip['open'] >= breakout_threshold)

    breakout_candles = post_dip[post_dip['breakout']]
    if breakout_candles.empty:
        return ('no_breakout', symbol)    
    
    first_breakout = breakout_candles.iloc[0]
    breakout_time = first_breakout['date']

    # --- 6. PRECISE PRICE CALCULATION ---
    breakout_threshold = next_price_above(max_ma, tick_size)
    candle_open = float(first_breakout['open'])
    breakout_price = max(candle_open, breakout_threshold)
    
    # --- 7. METRICS ---
    # Calculate Day High after breakout
    post_breakout_data = df_today[df_today['date'] >= breakout_time]
    day_high = post_breakout_data['high'].max() if not post_breakout_data.empty else breakout_price
    day_low = df_today['low'].min()
    
    # Indicators
    atr_14 = calculate_atr(daily_hist.tail(30), period=14)
    is_bb_squeeze = bool(calculate_bb_squeeze(daily_hist.tail(50)))
    
    # RSI
    hist_to_entry = intraday[intraday['date'] <= entry_time]
    rsi_entry = calculate_rsi(hist_to_entry['close']).iloc[-1] if len(hist_to_entry) >= 14 else np.nan
    
    hist_to_breakout = intraday[intraday['date'] <= breakout_time]
    rsi_breakout = calculate_rsi(hist_to_breakout['close']).iloc[-1] if len(hist_to_breakout) >= 14 else np.nan
    
    percent_rsi_move = 0.0
    if pd.notna(rsi_entry) and pd.notna(rsi_breakout) and rsi_entry != 0:
        percent_rsi_move = ((rsi_breakout - rsi_entry) / rsi_entry) * 100

    # Nifty Metrics
    nifty_at_entry = None
    nifty_at_breakout = None
    nifty_move = 0.0
    nifty_is_above_val = False

    if nifty_intraday is not None and not nifty_intraday.empty:
        # Nifty Entry
        n_entry_rows = nifty_intraday[nifty_intraday.index <= entry_time]
        if not n_entry_rows.empty:
            nifty_at_entry = float(n_entry_rows.iloc[-1]['close'])
        else:
            nifty_at_entry = float(nifty_intraday.iloc[0]['open'])
            
        # Nifty Breakout
        n_break_rows = nifty_intraday[nifty_intraday.index <= breakout_time]
        if not n_break_rows.empty:
            nifty_at_breakout = float(n_break_rows.iloc[-1]['close'])
        
        # Calculation
        if nifty_at_entry and nifty_at_breakout and nifty_at_entry != 0:
            nifty_move = ((nifty_at_breakout - nifty_at_entry) / nifty_at_entry) * 100
            
        if nifty_at_breakout and nifty_50ma:
            nifty_is_above_val = nifty_at_breakout > nifty_50ma

    # Volume
    spike, vol_mult = check_volume_spike(hist_to_breakout)
    avg_vol_20 = daily_hist['volume'].tail(20).mean()
    avg_daily_vol_20d = float(avg_vol_20) if pd.notna(avg_vol_20) else None
    
    vol_at_breakout = df_today[df_today['date'] <= breakout_time]['volume'].sum()
    vol_vs_avg_pct = (vol_at_breakout / avg_daily_vol_20d * 100) if avg_daily_vol_20d and avg_daily_vol_20d > 0 else 0.0
    
    # Previous Day & Pivots
    prev_day = daily_hist.iloc[-1]
    pivots = calculate_pivots(prev_day['high'], prev_day['low'], prev_day['close'])
    
    df_hist_10 = daily_hist.tail(10).copy()
    df_hist_10['range_pct'] = ((df_hist_10['high'] - df_hist_10['low']) / df_hist_10['low']) * 100
    avg_intraday_range_pct = df_hist_10['range_pct'].mean()

    # Store MAs for reference
    mas_dict = {}
    periods = [20, 50, 100, 200]
    for i, val in enumerate(daily_mas): mas_dict[f'daily_ma{periods[i]}'] = val
    for i, val in enumerate(hourly_mas): mas_dict[f'hourly_ma{periods[i]}'] = val

    return 'success', {
        'symbol': symbol,
        'breakout_date': date.strftime('%Y-%m-%d'),
        'monitor_entry_time': entry_time.isoformat(),
        'breakout_time': breakout_time.isoformat(),
        'breakout_price': float(breakout_price),
        'min_ma': float(min_ma),
        'max_ma': float(max_ma),
        'day_high_price': float(day_high),
        'day_low_price': float(day_low),
        'percent_move': round(((day_high - breakout_price) / breakout_price) * 100, 2),
        'rsi_at_entry': float(rsi_entry) if pd.notna(rsi_entry) else None,
        'rsi_at_breakout': float(rsi_breakout) if pd.notna(rsi_breakout) else None,
        'percent_rsi_move': float(percent_rsi_move),
        'nifty_percent_move': float(nifty_move),
        'nifty_value_at_entry': float(nifty_at_entry) if nifty_at_entry else None,
        'nifty_value_at_breakout': float(nifty_at_breakout) if nifty_at_breakout else None,
        'nifty_above_50ma': bool(nifty_is_above_val), 
        'volume_spike': bool(spike),
        'is_bb_squeeze': bool(is_bb_squeeze),
        'volume_multiplier': float(vol_mult),
        **mas_dict,
        'atr_14': float(atr_14) if pd.notna(atr_14) else None,
        'prev_day_high': float(prev_day['high']),
        'prev_day_low': float(prev_day['low']),
        'avg_intraday_range_pct_10d': float(avg_intraday_range_pct),
        'vol_vs_avg_pct_at_breakout': float(vol_vs_avg_pct),
        'pivot_points': pivots,
        'avg_daily_vol_20d': avg_daily_vol_20d
    }

# -------------------------- Main Pipeline --------------------------

def trigger_screenshot_scraper(symbol, breakout_date):
    """
    Calls the screenshot_scrapper_2.py script for a specific stock.
    Converts 'RELIANCE.NS' -> 'NSE:RELIANCE' for TradingView compatibility.
    """
    try:
        # Convert Yahoo format (RELIANCE.NS) to TradingView format (NSE:RELIANCE)
        tv_symbol = f"NSE:{symbol.replace('.NS', '')}"
        
        logger.info(f"📸 Taking screenshots for {tv_symbol}...")
        
        # We use subprocess.run (blocking) to prevent opening 50 browsers at once and crashing RAM.
        # If you want it to run in background, change .run() to .Popen()
        subprocess.run([
            sys.executable, "screenshot_scrapper_2.py",
            "--symbol", tv_symbol,
            "--dt", breakout_date,
            "--out", "./breakout_ss",  # Images will be saved here
            "--headless"                   # Run without opening visible browser windows
        ], check=True, timeout=120)
        
    except Exception as e:
        logger.error(f"❌ Screenshot failed for {symbol}: {e}")

def save_breakouts_to_excel(data):
    try:
        new_df = pd.DataFrame(data)

        if os.path.exists("custom_ma_breakouts_backup.xlsx"):
            old_df = pd.read_excel("custom_ma_breakouts_backup.xlsx")
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df

        # Deduplicate by symbol + breakout_date
        combined = combined.drop_duplicates(
            subset=["symbol", "breakout_date"],
            keep="last"
        )

        combined.to_excel("custom_ma_breakouts_backup.xlsx", index=False)

        logger.info("📁 Saved Excel fallback: custom_ma_breakouts_backup.xlsx")

    except Exception as e:
        logger.error(f"❌ Failed writing Excel backup: {e}")

def batch_upsert_supabase(data):
    if not data:
        return

    unique_data = list({ (d['symbol'], d['breakout_date']): d for d in data }.values())

    try:
        supabase.table("algo_version2_breakouts") \
            .upsert(unique_data, on_conflict='symbol,breakout_date') \
            .execute()

        logger.info(f"Upserted {len(unique_data)} records.")

    except Exception as e:
        logger.error(f"Supabase upsert error: {e}")
        save_breakouts_to_excel(unique_data)

def get_all_nse_symbols():
    """Fetch all active NSE equity symbols from NSE (EQ series only)."""
    logger.info("Fetching NSE symbol list from NSE CSV...")
    
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text))
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt+1}/{max_retries} failed to fetch NSE CSV: {e}")
                time.sleep(5)
            else:
                logger.error("Failed to fetch NSE symbol list.")
                raise

    df.columns = df.columns.str.strip()
    df = df[df["SERIES"] == "EQ"]
    symbols = df["SYMBOL"].tolist()

    logger.info(f"Fetched {len(symbols)} NSE EQ symbols from NSE.")
    return symbols

def main():
    logger.info("=== KITE BREAKOUT SCANNER (DEBUG MODE) ===")

    init_kite()   # initialize Kite client here (safe — happens only in the main process)    
    nse_symbols = get_all_nse_symbols()        # ~2065 symbols
    instrument_map = get_kite_instruments(nse_symbols)

    # Only symbols that Kite actually supports
    all_symbols = [s for s in nse_symbols if s in instrument_map]
    logger.info(f"NSE symbols: {len(nse_symbols)} | Kite-mapped symbols: {len(all_symbols)}")

    target_date = datetime.date(2026, 2, 3)
    target_dates = [target_date]
    logger.info(f"Targeting dates: {target_dates}")

    fetch_start = target_date - datetime.timedelta(days=400)
    fetch_end = target_date
    
    batch_size = 50
    
    for i in range(0, len(all_symbols), batch_size):
        batch_symbols = all_symbols[i:i+batch_size]
        logger.info(f"--- Processing Batch {i//batch_size + 1} ({len(batch_symbols)} symbols) ---")
        
        data_dict = preload_data_kite(batch_symbols, instrument_map, fetch_start, fetch_end)
        
        # --- DEBUG CHECK: Are we actually getting data? ---
        if not data_dict:
            logger.error("CRITICAL: Data Dictionary is empty! Check API Keys or Internet.")
            continue
        
        if 'nifty' not in data_dict:
            logger.warning("Nifty data fetch failed, skipping batch.")
            continue
            
        
        nifty_daily = data_dict['nifty']['daily']
        nifty_daily['ma50'] = nifty_daily['close'].rolling(50).mean().shift(1)
        nifty_50ma_lookup = dict(zip(nifty_daily['date'].dt.date, nifty_daily['ma50']))
        nifty_intraday_full = data_dict['nifty']['minute']
        
        tasks = []
        for sym in batch_symbols:
            if sym not in data_dict:
                continue
            
            for d in target_dates:
                daily_dates = data_dict[sym]['daily']['date'].dt.date.values
                if d not in daily_dates:
                    continue

                task = {
                    'symbol': sym,
                    'analysis_date': d,
                    'intraday_data': data_dict[sym]['minute'],
                    'daily_data': data_dict[sym]['daily'],
                    'hourly_data': data_dict[sym]['hourly'],
                    'nifty_intraday': nifty_intraday_full[nifty_intraday_full['date'].dt.date == d].set_index('date'),
                    'nifty_50ma': nifty_50ma_lookup.get(d),
                    'tick_size': data_dict[sym]['tick_size']
                }
                tasks.append(task)
        
        results = []
        failure_reasons = Counter() # Track why stocks fail

        if tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                futures = {executor.submit(process_stock_day, t): t for t in tasks}
                for f in concurrent.futures.as_completed(futures):
                    status, res = f.result()
                    if status == 'success':
                        results.append(res)
                    else:
                        failure_reasons[status] += 1 # Count the failure reason
        
        # --- DEBUG LOGGING ---
        if results:
            logger.info(f"✅ Found {len(results)} breakouts!")
            
            cleaned_results = []
            for r in results:
                # 1. Trigger the Screenshot Scraper
                trigger_screenshot_scraper(r['symbol'], r['breakout_time'])
                
                # 2. Clean up data for DB
                cleaned_results.append(r)
            
            # 3. Upload to Supabase
            batch_upsert_supabase(cleaned_results)
        else:
            logger.info("❌ No breakouts.")
            # PRINT THE REASONS WHY
            logger.info(f"   [Rejection Stats]: {dict(failure_reasons)}")
            
        time.sleep(1)

    logger.info("=== SCAN COMPLETE ===")

if __name__ == "__main__":
    main()