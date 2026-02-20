import datetime
import pandas as pd
import numpy as np
import os
import logging
import io
import requests
from kiteconnect import KiteConnect
from dotenv import load_dotenv
from decimal import Decimal, ROUND_CEILING
import concurrent.futures

# -------------------------- Setup & Config --------------------------
load_dotenv()
KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Kite
kite = KiteConnect(api_key=KITE_API_KEY)
kite.set_access_token(KITE_ACCESS_TOKEN)

# -------------------------- Helper Functions --------------------------

def next_price_above(value, tick):
    """Calculates the exact entry price 1 tick above the MA."""
    dv = Decimal(str(value))
    dt = Decimal(str(tick))
    n = (dv / dt).to_integral_value(rounding=ROUND_CEILING)
    candidate = n * dt
    if candidate <= dv:
        candidate = (n + 1) * dt
    return float(candidate)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period-1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period-1, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_pivots(prev_high, prev_low, prev_close):
    pivot = (prev_high + prev_low + prev_close) / 3
    return {
        'pivot': round(pivot, 2),
        's1': round((pivot * 2) - prev_high, 2),
        'r1': round((pivot * 2) - prev_low, 2)
    }

def calculate_atr(df, period=14):
    if df.empty or not all(c in df.columns for c in ['high', 'low', 'close']):
        return np.nan
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_prev_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_prev_close'] = np.abs(df['low'] - df['close'].shift(1))
    tr = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    return tr.rolling(window=period).mean().iloc[-1]

def calculate_bb_squeeze(df, period=20, std_dev=2, kc_atr_mult=1.5):
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    bb_width = (sma + (std_dev * std)) - (sma - (std_dev * std))
    atr = calculate_atr(df, period)
    if pd.isna(atr) or pd.isna(sma.iloc[-1]) or pd.isna(bb_width.iloc[-1]):
        return False

    kc_width = (sma.iloc[-1] + (kc_atr_mult * atr)) - (sma.iloc[-1] - (kc_atr_mult * atr))

    return bb_width.iloc[-1] < kc_width


def get_active_nse_symbols():
    """Fetches the list of EQ series stocks from NSE and cleans column names."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(res.text))
        
        df.columns = df.columns.str.strip()
        
        return df[df["SERIES"] == "EQ"]["SYMBOL"].tolist()
    except Exception as e:
        logger.error(f"Failed to fetch NSE symbols: {e}")
        return []

def fetch_history(token, start, end, interval):
    cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    try:
        data = kite.historical_data(token, start, end, interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df.rename(columns=str.lower, inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            return df
        return pd.DataFrame(columns=cols) # Return empty with columns
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return pd.DataFrame(columns=cols) # Return empty with columns

# -------------------------- Trade Logic --------------------------

def process_paper_trade(df_today, breakout_price, breakout_time):
    """
    Standard Exit Logic: 
    - SL: -1% | Target: +3% | Time: 3:15 PM EOD
    """

    df_today = df_today.copy()
    df_today['ma8'] = df_today['close'].rolling(window=8).mean()
    
    trade_data = df_today[df_today['date'] >= breakout_time].sort_values('date')
    if trade_data.empty:
        return df_today.iloc[-1]['close'], "Late Entry/No Data", 0, 0

    target_price = breakout_price * 1.03
    sl_price = breakout_price
    
    is_in_trade = True
    total_pnl_pct = 0
    exit_reason = "Market Close"
    last_exit_price = breakout_price

    for idx, row in trade_data.iterrows():
        # Current candle data
        high, low, close, ma8 = row['high'], row['low'], row['close'], row['ma8']
        
        if is_in_trade:
            # CHECK EXIT: Hit Stop Loss
            if low <= sl_price:
                is_in_trade = False
                last_exit_price = sl_price
                exit_reason = "Stopped out @ Breakout"
                # PnL is 0 because we exit at entry price
            
            # CHECK EXIT: Hit Target
            elif high >= target_price:
                return target_price, "Target hit @3%", 3, 0
        
        else:
            # NEW RE-ENTRY: Buy again if price crosses back ABOVE the breakout price
            if close > breakout_price:
                is_in_trade = True
                exit_reason = "Re-entered above Breakout"

        # FINAL EXIT: EOD 3:15 PM
        if row['date'].time() >= datetime.time(15, 15):
            if is_in_trade:
                last_exit_price = close
                pnl = ((last_exit_price - breakout_price) / breakout_price) * 100
                return last_exit_price, "EOD Exit @15:15", max(0, round(pnl, 2)), max(0, round(-pnl, 2))
            else:
                return last_exit_price, exit_reason, 0, 0

    return last_exit_price, exit_reason, 0, 0

# -------------------------- Main Scanner --------------------------

def process_symbol_for_date(sym, token, current_date, df_daily, df_15m, n_df_intra, nifty_50ma, rsi_series, df_hourly_full):
    results = []

    # Safety Check: Skip if data is invalid or missing columns
    if df_daily is None or df_daily.empty or 'date' not in df_daily.columns:
        return results
    if df_15m is None or df_15m.empty or 'date' not in df_15m.columns:
        return results

    df_today = df_15m[df_15m['date'].dt.date == current_date]
    if df_today.empty:
        return results

    day_open = float(df_today.iloc[0]['open'])
    prev_daily = df_daily[df_daily['date'].dt.date < current_date].tail(199)

    if len(prev_daily) < 199:
        return results

    df_hourly = df_hourly_full[df_hourly_full['date'].dt.date < current_date].tail(200)

    if len(df_hourly) < 200:
        return results

    d_prices = list(prev_daily['close'].values) + [day_open]
    h_prices = list(df_hourly['close'].values) + [day_open]

    # Using simple list slicing/summing is faster than creating 8 new pandas Series objects per symbol
    def get_ma(prices, period):
        return sum(prices[-period:]) / period if len(prices) >= period else 0

    daily_mas = [get_ma(d_prices, p) for p in [20, 50, 100, 200]]
    hourly_mas = [get_ma(h_prices, p) for p in [20, 50, 100, 200]]

    min_ma, max_ma = min(daily_mas + hourly_mas), max(daily_mas + hourly_mas)

    if not (day_open < min_ma and df_today['high'].max() > max_ma):
        return results

    breakout = df_today[
        ((df_today['close'] >= next_price_above(max_ma, 0.05)) |
        (df_today['open'] >= next_price_above(max_ma, 0.05))) &
        (df_today['date'].dt.time <= datetime.time(15, 10))
    ]

    if breakout.empty:
        return results

    first = breakout.iloc[0]
    b_time = first['date']
    b_price = max(first['open'], next_price_above(max_ma, 0.05))

    # Nifty
    nifty_entry = n_df_intra.iloc[0]['open']
    nifty_break = n_df_intra[n_df_intra.index <= b_time].iloc[-1]['close']

    # ---------- RSI ----------
    entry_time = df_today.iloc[0]['date']

    rsi_entry = rsi_series[df_15m['date'] <= entry_time].dropna().iloc[-1]
    rsi_break = rsi_series[df_15m['date'] <= b_time].dropna().iloc[-1]

    # ---------- VOLUME ----------
    avg_vol_20d = prev_daily['volume'].mean()
    vol_at_break = df_today[df_today['date'] <= b_time]['volume'].sum()
    vol_mult = round(vol_at_break / avg_vol_20d, 2) if avg_vol_20d > 0 else 0

    # ---------- ATR ----------
    atr_val = calculate_atr(prev_daily.tail(30))

    # ---------- BB SQUEEZE ----------
    is_squeeze = calculate_bb_squeeze(prev_daily.tail(50))

    # ---------- INTRADAY RANGE ----------
    df_hist_10 = prev_daily.tail(10).copy()
    df_hist_10['range_pct'] = ((df_hist_10['high'] - df_hist_10['low']) / df_hist_10['low']) * 100
    avg_range_10d = df_hist_10['range_pct'].mean()

    # ---------- PIVOTS ----------
    prev_day = prev_daily.iloc[-1]
    pivots = calculate_pivots(prev_day['high'], prev_day['low'], prev_day['close'])

    results.append({
        'Symbol': sym,
        'Breakout Date': current_date,
        'Time': b_time.time(),
        'Breakout Price': b_price,
        'df_today': df_today,
        'monitor_entry_time': df_today.iloc[0]['date'],
        'rsi_entry': rsi_entry,
        'rsi_break': rsi_break,
        'nifty_entry': nifty_entry,
        'nifty_break': nifty_break,
        'nifty_50ma': nifty_50ma,
        'atr_val': atr_val,
        'is_squeeze': is_squeeze,
        'vol_mult': vol_mult,
        'avg_range_10d': avg_range_10d,
        'avg_vol': avg_vol_20d,
        'vol_at_break': vol_at_break,
        'pivots': pivots,
        'prev_day': prev_day,
        'min_ma': min_ma,
        'max_ma': max_ma,
        'daily_mas': daily_mas,
        'hourly_mas': hourly_mas
    })
    return results

def run_scanner():
    daily_cache = {}
    intraday_cache = {}
    rsi_cache = {}
    hourly_cache = {}
    
    target_dates = pd.date_range(start="2026-02-16", end="2026-02-19").date               # FOR A WEEK

    # target_dates = pd.date_range(start="2026-02-02", end="2026-02-02").date             # FOR TODAY

    # Generates a list of dates for the last 30 days
    # target_dates = pd.date_range(end=datetime.date.today(), periods=30).date

    symbols = get_active_nse_symbols()
    
    inst = kite.instruments("NSE")
    inst_map = {i['tradingsymbol']: i['instrument_token'] for i in inst if i['tradingsymbol'] in symbols}

    import pickle
    cache_file = "market_data_cache.pkl"

    if os.path.exists(cache_file):
        logger.info(">>> SUCCESS: Found saved data on disk. Loading now...")
        with open(cache_file, "rb") as f:
            loaded_data = pickle.load(f)
            daily_cache = loaded_data['daily']
            intraday_cache = loaded_data['intraday']
            rsi_cache = loaded_data['rsi']
            hourly_cache = loaded_data['hourly']
        logger.info(f">>> LOADED: {len(daily_cache)} symbols ready for analysis.")
    else:
        logger.info(">>> STARTING PRELOAD: This will take ~15 mins. Logs will appear every 50 symbols.")
        
        def preload_worker(sym, token):
            d = fetch_history(token, min(target_dates) - datetime.timedelta(days=400), max(target_dates), "day")
            i = fetch_history(token, min(target_dates) - datetime.timedelta(days=100), max(target_dates), "15minute")
            rsi = calculate_rsi(i['close']) if not i.empty else pd.Series()
            h_full = i.set_index('date').resample('60min').agg({'close': 'last'}).dropna().reset_index() if not i.empty else pd.DataFrame()
            return token, d, i, rsi, h_full

        completed = 0
        total = len(inst_map)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_sym = {executor.submit(preload_worker, s, t): t for s, t in inst_map.items()}
            for future in concurrent.futures.as_completed(future_to_sym):
                try:
                    t, d, i, rsi, h_full = future.result()
                    daily_cache[t], intraday_cache[t] = d, i
                    rsi_cache[t], hourly_cache[t] = rsi, h_full
                    
                    completed += 1
                    if completed % 50 == 0: # Log every 50 so you see more updates
                        logger.info(f"Preload Progress: {completed}/{total} ({(completed/total)*100:.1f}%)")
                except Exception as e:
                    logger.error(f"Worker Error: {e}")

        # SAVE immediately after preloading so you never have to do this again
        cache_data = {'daily': daily_cache, 'intraday': intraday_cache, 'rsi': rsi_cache, 'hourly': hourly_cache}
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f">>> SAVED: Data stored to {cache_file}. Next run will be instant.")

    # --- REST OF THE SCRIPT (Nifty Fetching & Scanner) ---
    nifty_token = 256265 
    n_daily_full = fetch_history(nifty_token, min(target_dates) - datetime.timedelta(days=100), max(target_dates), "day")
    n_daily_full['ma50'] = n_daily_full['close'].rolling(50).mean().shift(1)
    n_intra_full = fetch_history(nifty_token, min(target_dates), max(target_dates), "15minute")

    final_results = []
    daily_total_budget = 20000

    for current_date in target_dates:
        logger.info(f"--- Processing Date: {current_date} ---")
        n_df_intra = n_intra_full[n_intra_full['date'].dt.date == current_date].set_index('date')
        nifty_row = n_daily_full[n_daily_full['date'].dt.date == current_date]
        
        if nifty_row.empty or n_df_intra.empty: continue
        nifty_50ma = nifty_row['ma50'].iloc[0]
        day_breakouts = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for sym, token in inst_map.items():
                if token not in intraday_cache or intraday_cache[token].empty or 'date' not in daily_cache[token].columns:
                    continue
                futures.append(executor.submit(process_symbol_for_date, sym, token, current_date, daily_cache[token], intraday_cache[token], n_df_intra, nifty_50ma, rsi_cache[token], hourly_cache[token]))

            for f in concurrent.futures.as_completed(futures):
                day_breakouts.extend(f.result())
            logger.info(f"Finished {current_date}: Found {len(day_breakouts)} breakout trades.")

        if day_breakouts:
            amt_per_trade = daily_total_budget / len(day_breakouts)
            for trade in day_breakouts:
                exit_p, reason, p_pct, l_pct = process_paper_trade(trade['df_today'], trade['Breakout Price'], pd.Timestamp.combine(trade['Breakout Date'], trade['Time']))
                df_t = trade.get('df_today')
                prev_d = trade.get('prev_day')

                final_results.append({
                    'Symbol': trade.get('Symbol'), 
                    'Breakout Date': pd.Timestamp.combine(trade.get('Breakout Date'), trade.get('Time')),
                    'Time': trade.get('Time'),
                    'monitor_entry_time': trade.get('monitor_entry_time'),
                    'Breakout Price': trade.get('Breakout Price'),
                    'High Price': df_t['high'].max() if (df_t is not None and not df_t.empty) else 0,
                    'Selling Price': exit_p, 
                    'Selling Price condition': reason,
                    'Trade Amount': amt_per_trade, 
                    'Profit %': p_pct,
                    'Loss %': l_pct,
                    'Day %Move': round(((df_t['high'].max() - trade.get('Breakout Price')) / trade.get('Breakout Price')) * 100, 2) if (df_t is not None and not df_t.empty) else 0,
                    'rsi_at_entry': trade.get('rsi_entry'),
                    'rsi_at_breakout': trade.get('rsi_break'),
                    'rsi_percent_move': ((trade.get('rsi_break') - trade.get('rsi_entry')) / trade.get('rsi_entry')) * 100 if trade.get('rsi_entry', 0) != 0 else 0,
                    'nifty_at_entry': trade.get('nifty_entry'),
                    'nifty_at_breakout': trade.get('nifty_break'),
                    'nifty_percent_move': ((trade.get('nifty_break') - trade.get('nifty_entry')) / trade.get('nifty_entry')) * 100 if trade.get('nifty_entry', 0) != 0 else 0,
                    'nifty_above_50ma': trade.get('nifty_break', 0) > trade.get('nifty_50ma', 0),
                    'volume_spike': (trade.get('vol_at_break', 0) / trade.get('avg_vol', 1)) > 1.5,
                    'vol_vs_avg_pct_at_breakout': (trade.get('vol_at_break', 0) / trade.get('avg_vol', 1)) * 100,
                    'volume_multiplier': trade.get('vol_mult'),
                    'avg_daily_vol_20d': trade.get('avg_vol'),
                    'day_low_price': df_t['low'].min() if (df_t is not None and not df_t.empty) else 0,
                    'prev_day_high': prev_d['high'] if prev_d is not None else 0,
                    'prev_day_low': prev_d['low'] if prev_d is not None else 0,
                    'atr_14': trade.get('atr_val'),
                    'avg_intraday_range_pct_10d': trade.get('avg_range_10d'),
                    'is_bb_squeeze': trade.get('is_squeeze'),
                    'pivot': trade.get('pivots', {}).get('pivot'),
                    's1': trade.get('pivots', {}).get('s1'),
                    'r1': trade.get('pivots', {}).get('r1'),
                    'min_ma': trade.get('min_ma'), 
                    'max_ma': trade.get('max_ma'),
                    'daily_ma20': trade.get('daily_mas')[0], 'daily_ma50': trade.get('daily_mas')[1],
                    'daily_ma100': trade.get('daily_mas')[2], 'daily_ma200': trade.get('daily_mas')[3],
                    'hourly_ma20': trade.get('hourly_mas')[0], 'hourly_ma50': trade.get('hourly_mas')[1],
                    'hourly_ma100': trade.get('hourly_mas')[2], 'hourly_ma200': trade.get('hourly_mas')[3]
                })

    # Final Output with Safety Catch
    try:
        if final_results:
            output_df = pd.DataFrame(final_results)
            output_filename = "Paper_trading_week_16th_feb.xlsx"
            
            # Save the file
            output_df.to_excel(output_filename, index=False)
            
            logger.info("--------------------------------------------------")
            logger.info(f"SUCCESS: {len(final_results)} trades saved to {output_filename}")
            logger.info("--------------------------------------------------")
        else:
            logger.warning("SCAN COMPLETE: No breakout trades met the criteria. No file created.")
            
    except Exception as e:
        logger.error(f"CRITICAL ERROR saving Excel file: {e}")
        # Emergency backup save as CSV if Excel fails
        if final_results:
            pd.DataFrame(final_results).to_csv("EMERGENCY_BACKUP_RESULTS.csv", index=False)
            logger.info("Emergency backup saved as CSV.")

if __name__ == "__main__":
    run_scanner()