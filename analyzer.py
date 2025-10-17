"""This file contains only the pure analysis logic. It is now completely decoupled from data fetching and database writing.
The main function find_breakout_for_day is designed to be called by multiprocessing.Pool"""

# analyzer.py
import pandas as pd
import pandas_ta as ta
from decimal import Decimal, ROUND_CEILING
from config import Config
import logging

logger = logging.getLogger(__name__)

# --- Helper Functions (moved from the original script) ---

def next_price_above(value: float, tick: float) -> float:
    """Calculates the next valid price tick above a given value."""
    # This function is well-formed, no changes needed.
    dv = Decimal(str(value))
    dt = Decimal(str(tick))
    n = (dv / dt).to_integral_value(rounding=ROUND_CEILING)
    candidate = n * dt
    if candidate <= dv:
        candidate = (n + 1) * dt
    return float(candidate)

def get_ma_thresholds(ma_dict, symbol, analysis_date):
    """Extracts precomputed MAs for a specific date."""
    daily_mas = ma_dict.get(symbol, {}).get('daily')
    hourly_mas = ma_dict.get(symbol, {}).get('hourly')

    if daily_mas is None or daily_mas.empty:
        logger.warning(f"[{symbol} on {analysis_date.strftime('%Y-%m-%d')}] FAILED: No daily MA data exists.")
        return 'no_data', None # Return a specific status for this case

    start_of_day = pd.Timestamp(analysis_date)
    daily_row_df = daily_mas[daily_mas['date'].dt.date == analysis_date]

    if daily_row_df.empty:
        return 'no_data', None # Non-trading day, normal occurrence.
    
    daily_row = daily_row_df.iloc[0]
    
    # ✅ IMPROVEMENT: Using MA periods from the central config file.
    all_mas = [daily_row[f'ma{p}'] for p in Config.MA_PERIODS]
    
    if hourly_mas is not None and not hourly_mas.empty:
        hourly_rows_before_day = hourly_mas[hourly_mas['date'] < start_of_day]
        if not hourly_rows_before_day.empty:
            last_hourly_row_prev_day = hourly_rows_before_day.iloc[-1]
            all_mas.extend([last_hourly_row_prev_day[f'ma{p}'] for p in Config.MA_PERIODS])

    if any(pd.isna(ma) for ma in all_mas):
        return 'nan_failure', None

    return 'success', (min(all_mas), max(all_mas))

def analyze_breakout_conditions_vectorized(df_subset: pd.DataFrame):
    """Vectorized analysis of RSI, BB Squeeze, and Volume Spike at breakout."""
    rsi_at_breakout = None
    if len(df_subset) >= Config.RSI_PERIOD:
        rsi = ta.rsi(df_subset['close'], length=Config.RSI_PERIOD)
        if rsi is not None and not rsi.empty:
            rsi_at_breakout = rsi.iloc[-1]
    
    is_bb_squeeze = None
    if len(df_subset) >= Config.BB_SQUEEZE_LOOKBACK + Config.BBANDS_PERIOD:
        bbands = ta.bbands(df_subset['close'], length=Config.BBANDS_PERIOD, std=Config.BBANDS_STD_DEV)
        
        # ✅ IMPROVEMENT: Dynamically generate column names for robustness
        bbu_col = f'BBU_{Config.BBANDS_PERIOD}_{float(Config.BBANDS_STD_DEV)}'
        bbl_col = f'BBL_{Config.BBANDS_PERIOD}_{float(Config.BBANDS_STD_DEV)}'
        bbm_col = f'BBM_{Config.BBANDS_PERIOD}_{float(Config.BBANDS_STD_DEV)}'
        required_cols = [bbu_col, bbl_col, bbm_col]

        if bbands is not None and not bbands.empty and all(col in bbands.columns for col in required_cols):
            bandwidth = ((bbands[bbu_col] - bbands[bbl_col]) / bbands[bbm_col]).dropna()
            
            if len(bandwidth) >= Config.BB_SQUEEZE_LOOKBACK:
                quantile_10 = bandwidth.rolling(window=Config.BB_SQUEEZE_LOOKBACK).quantile(0.1)
                
                if pd.notna(bandwidth.iloc[-1]) and pd.notna(quantile_10.iloc[-1]):
                    is_bb_squeeze = bandwidth.iloc[-1] < quantile_10.iloc[-1]

    volume_spike = None
    if len(df_subset) > Config.AVG_VOL_LOOKBACK:
        avg_volume = df_subset['volume'].shift(1).rolling(window=Config.AVG_VOL_LOOKBACK).mean().iloc[-1]
        volume_spike = df_subset['volume'].iloc[-1] > avg_volume if pd.notna(avg_volume) else None
    
    return {
        'rsi_at_breakout': rsi_at_breakout,
        'is_bb_squeeze': is_bb_squeeze,
        'volume_spike': volume_spike
    }

# --- Main Analysis Function for Multiprocessing ---
def find_breakout_for_day(task_data: dict):
    """Processes a single stock/day and returns a status tuple."""
    symbol = task_data['symbol']
    analysis_date = task_data['analysis_date']
    ma_data_for_stock = task_data['ma_data']
    intraday_data_full_history = task_data['intraday_data']
    nifty_intraday_for_day = task_data['nifty_intraday']
    nifty_50ma_for_day = task_data['nifty_50ma']
    tick_size = task_data['tick_size']

    if intraday_data_full_history.empty:
        return ('no_intraday_data', symbol)
    
    status, ma_values = get_ma_thresholds(ma_data_for_stock, symbol, analysis_date)
    if status != 'success':
        return (status, symbol)
    
    min_ma, max_ma = ma_values
    breakout_price = next_price_above(max_ma, tick_size)
    
    df_intraday_today = intraday_data_full_history[
        intraday_data_full_history['date'].dt.date == analysis_date
    ].set_index('date')
    
    if df_intraday_today.empty:
        return ('no_intraday_data', symbol)
    
    day_high_price = df_intraday_today['high'].max()
    day_low_price = df_intraday_today['low'].min()
    
    if pd.isna(nifty_50ma_for_day) or nifty_intraday_for_day.empty:
        return ('no_nifty_data', symbol)
    
    df_intraday_today = df_intraday_today.copy()
    df_intraday_today['below_min_ma'] = df_intraday_today['low'] < min_ma
    df_intraday_today['above_breakout'] = df_intraday_today['high'] >= breakout_price
    
    dip_rows = df_intraday_today[df_intraday_today['below_min_ma']]
    if dip_rows.empty:
        return ('no_dip', symbol)
    
    monitor_entry_time = dip_rows.index[0]
    
    df_history_up_to_entry = intraday_data_full_history[intraday_data_full_history['date'] <= monitor_entry_time]
    rsi_at_entry = None
    if not df_history_up_to_entry.empty and len(df_history_up_to_entry) >= Config.RSI_PERIOD:
        rsi_result = ta.rsi(df_history_up_to_entry['close'], length=Config.RSI_PERIOD)
        if rsi_result is not None and not rsi_result.empty:
            rsi_at_entry = rsi_result.iloc[-1]
            
    nifty_value_at_entry = nifty_intraday_for_day['close'].asof(monitor_entry_time)
    
    post_dip = df_intraday_today[df_intraday_today.index >= monitor_entry_time]
    breakout_rows = post_dip[post_dip['above_breakout']]
    if breakout_rows.empty:
        return ('no_breakout', symbol)
    
    breakout_time = breakout_rows.index[0]
    
    df_history_up_to_breakout = intraday_data_full_history[intraday_data_full_history['date'] <= breakout_time]
    breakout_conditions = analyze_breakout_conditions_vectorized(df_history_up_to_breakout)

    percent_move = ((day_high_price - breakout_price) / breakout_price) * 100 if breakout_price > 0 else 0
    nifty_value_at_breakout = nifty_intraday_for_day['close'].asof(breakout_time)
    nifty_above_50ma = nifty_value_at_breakout > nifty_50ma_for_day if pd.notna(nifty_value_at_breakout) and pd.notna(nifty_50ma_for_day) else None
    
    result = {
        'symbol': symbol,
        'breakout_date': analysis_date.strftime('%Y-%m-%d'),
        'monitor_entry_time': monitor_entry_time.isoformat() if monitor_entry_time else None,
        'rsi_at_entry': float(rsi_at_entry) if pd.notna(rsi_at_entry) else None,
        'nifty_value_at_entry': float(nifty_value_at_entry) if pd.notna(nifty_value_at_entry) else None,
        'breakout_time': breakout_time.isoformat() if breakout_time else None,
        'day_high_price': float(day_high_price) if pd.notna(day_high_price) else None,
        'day_low_price': float(day_low_price) if pd.notna(day_low_price) else None,
        'min_ma': float(min_ma),
        'max_ma': float(max_ma),
        'breakout_price': float(breakout_price),
        'percent_move': float(percent_move),
        'rsi_at_breakout': float(breakout_conditions['rsi_at_breakout']) if pd.notna(breakout_conditions['rsi_at_breakout']) else None,
        'nifty_above_50ma': bool(nifty_above_50ma) if nifty_above_50ma is not None else None,
        'is_bb_squeeze': bool(breakout_conditions['is_bb_squeeze']) if breakout_conditions['is_bb_squeeze'] is not None else None,
        'volume_spike': bool(breakout_conditions['volume_spike']) if breakout_conditions['volume_spike'] is not None else None,
    }
    return 'success', result