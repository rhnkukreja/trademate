import datetime
import pandas as pd
import numpy as np
import time
import gc
from utils.common import supabase, kite, logger, batch_upsert_supabase, next_price_above
import io
import requests
from utils.news_agent import get_stock_news
from concurrent.futures import ThreadPoolExecutor

# -------------------------- Helper Functions -------------------------
def get_kite_token_map():
    """Builds symbol -> instrument_token map from Kite"""
    return {
        ins["tradingsymbol"]: ins["instrument_token"]
        for ins in kite.instruments("NSE")
        if ins.get("instrument_token")
    }

def get_derivative_symbols():
    """Fetches all symbols currently available in the F&O segment."""
    try:
        # Fetching NFO instruments to see which symbols have derivatives
        nfo_ins = kite.instruments("NFO")
        # Use a set for O(1) lookup speed
        return {ins["tradingsymbol"] for ins in nfo_ins}
    except Exception as e:
        logger.error(f"Failed to fetch NFO instruments: {e}")
        return set()
    
def get_all_nse_symbols():
    """Fetch all active NSE mainboard EQ symbols directly from official NSE CSV."""
    logger.info("Fetching NSE symbol list...")
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    
    # Browser-like headers to avoid blocking
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt+1}/{max_retries} failed to fetch symbol list: {e}")
                time.sleep(5)
            else:
                logger.error(f"All retries failed to fetch symbol list: {e}")
                return []  # Return empty on final failure
            
    # Get the list of F&O symbols        
    fno_symbols = get_derivative_symbols()
    # Clean and filter
    df.columns = df.columns.str.strip()
    df = df[df['SERIES'] == 'EQ']
    
    # Return in format expected by your code: list of dicts with 'symbol'
    symbols = [{"symbol": row['SYMBOL'], "tick_size": 0.05} for _, row in df.iterrows()]


    kite_token_map = get_kite_token_map()

    enriched_symbols = []
    missing_tokens = 0

    for s in symbols:
        token = kite_token_map.get(s["symbol"])
        if not token:
            missing_tokens += 1
            continue  # skip symbols not tradable on Kite

        enriched_symbols.append({
            "symbol": s["symbol"],
            "instrument_token": token,
            "tick_size": s.get("tick_size", 0.05),
            "has_derivative": s["symbol"] in fno_symbols
        })

    logger.info(f"Symbols dropped due to missing Kite token: {missing_tokens}")
    logger.info(f"Final tradable NSE symbols: {len(enriched_symbols)}")

    return enriched_symbols
    
def calculate_rsi(series, period=14):
    """Calculate RSI manually using pandas."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)."""
    if df.empty or not all(c in df.columns for c in ['high', 'low', 'close']):
        return np.nan
    
    # Ensure we are working with a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    df_copy['high_low'] = df_copy['high'] - df_copy['low']
    df_copy['high_prev_close'] = np.abs(df_copy['high'] - df_copy['close'].shift(1))
    df_copy['low_prev_close'] = np.abs(df_copy['low'] - df_copy['close'].shift(1))
    
    tr = df_copy[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    # Return the last value
    return atr.iloc[-1] if not atr.empty else np.nan

def calculate_pivots(prev_high, prev_low, prev_close):
    """Calculate standard pivot points."""
    if pd.isna(prev_high) or pd.isna(prev_low) or pd.isna(prev_close):
        return {}
    
    pivot = (prev_high + prev_low + prev_close) / 3
    s1 = (pivot * 2) - prev_high
    r1 = (pivot * 2) - prev_low
    s2 = pivot - (prev_high - prev_low)
    r2 = pivot + (prev_high - prev_low)
    
    return {
        'pivot': round(pivot, 2),
        's1': round(s1, 2),
        'r1': round(r1, 2),
        's2': round(s2, 2),
        'r2': round(r2, 2)
    }

def calculate_bb_squeeze(df, period=20, std_dev=2, kc_period=20, kc_atr_mult=1.5):
    """Detect Bollinger Band squeeze using BB vs Keltner Channel."""
    if df.empty or len(df) < period:
        return False

    df = df.copy()
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    bb_upper = sma + std_dev * std
    bb_lower = sma - std_dev * std
    bb_width = bb_upper - bb_lower

    atr = calculate_atr(df, kc_period)
    kc_middle = df["close"].rolling(kc_period).mean()
    kc_upper = kc_middle.iloc[-1] + kc_atr_mult * atr if pd.notna(atr) else np.nan
    kc_lower = kc_middle.iloc[-1] - kc_atr_mult * atr if pd.notna(atr) else np.nan
    kc_width = kc_upper - kc_lower

    if pd.notna(bb_width.iloc[-1]) and pd.notna(kc_width):
        return bb_width.iloc[-1] < kc_width
    return False

def preload_daily_data(stocks, from_date, to_date):
    """Fetches ONLY daily data."""
    data_dict = {}
    for stock in stocks:
        symbol = stock["symbol"]
        token = stock["instrument_token"]
        
        retries = 3
        for attempt in range(retries):
            try:
                data = kite.historical_data(instrument_token=token, from_date=from_date, to_date=to_date, interval="day")
                if data:
                    df = pd.DataFrame(data)
                    df = df.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    data_dict[symbol] = df[["date", "open", "high", "low", "close", "volume"]]
                break
            except Exception as e:
                wait_time = 1 if "429" not in str(e) else 2
                if attempt < retries - 1:
                    time.sleep(wait_time) 
                else:
                    logger.error(f"Failed to fetch daily for {symbol}: {e}")
        time.sleep(0.05) # Respect Kite rate limit
    return data_dict

def preload_hourly_data(stocks, from_date, to_date):
    """Fetches ONLY hourly data for stocks that survived the daily filter."""
    data_dict = {}
    hourly_start = max(from_date, to_date - datetime.timedelta(days=100))
    for stock in stocks:
        symbol = stock["symbol"]
        token = stock["instrument_token"]
        
        retries = 3
        for attempt in range(retries):
            try:
                data = kite.historical_data(instrument_token=token, from_date=hourly_start, to_date=to_date, interval="60minute")
                if data:
                    df = pd.DataFrame(data)
                    df = df.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    data_dict[symbol] = df.sort_values(by="date").drop_duplicates(subset="date").reset_index(drop=True)[["date", "open", "high", "low", "close", "volume"]]
                break
            except Exception as e:
                wait_time = 1 if "429" not in str(e) else 2
                if attempt < retries - 1:
                    time.sleep(wait_time) 
                else:
                    logger.error(f"Failed to fetch hourly for {symbol}: {e}")
        time.sleep(0.05)
    return data_dict

def precompute_mas(data_dict, stocks, today_open_prices):
    """Precomputes 8 MAs for each stock using history + today's open."""
    ma_dict = {}
    for stock in stocks:
        symbol = stock["symbol"]
        if symbol not in data_dict:
            continue
        ma_dict[symbol] = {}
        for interval in ["daily", "hourly"]:
            source_df = data_dict[symbol].get(interval)
            if source_df is None or source_df.empty or "close" not in source_df.columns:
                logger.warning(f"Skipping MA for {symbol} ({interval}): Invalid data.")
                continue
            dates = source_df["date"].tolist()
            prices = source_df["close"].tolist()
            if symbol in today_open_prices and today_open_prices[symbol] > 0:
                dates.append(pd.to_datetime(datetime.date.today()))
                prices.append(today_open_prices[symbol])
            ma_df = pd.DataFrame({"date": dates})
            price_series = pd.Series(prices)

            for period in [20, 50, 100, 200]:
                if len(price_series) >= period:
                    ma_df[f"ma{period}"] = price_series.rolling(window=period).mean()
                else:
                    ma_df[f"ma{period}"] = pd.NA

            ma_dict[symbol][interval] = ma_df.copy()
    return ma_dict

def process_batch(batch_info):
    batch_stocks, batch_idx, total_batches, from_date, to_date, auth_token = batch_info
    yesterday = to_date - datetime.timedelta(days=1)

    # 1. AUTHENTICATE KITE
    if hasattr(kite, "set_access_token"):
        kite.set_access_token(auth_token)
    elif hasattr(kite, "enctoken"):
        kite.enctoken = auth_token
        if hasattr(kite, "headers"):
            kite.headers["Authorization"] = f"enctoken {auth_token}"

    logger.info(f"Processing batch {batch_idx}/{total_batches}...")
    monitor_list = []
    
    # --- OPTIMIZATION 1: QUOTES FIRST ---
    # We only care about stocks that have valid live prices.
    symbols = [s["symbol"] for s in batch_stocks]
    try:
        quote_keys = [f"NSE:{s}" for s in symbols]
        raw_quotes = kite.quote(quote_keys)
        
        quotes = {}
        valid_stocks = []
        for stock in batch_stocks:
            sym = stock["symbol"]
            q_data = raw_quotes.get(f"NSE:{sym}", {})
            last_price = q_data.get("last_price") or q_data.get("ltp")
            open_price = q_data.get("open") or q_data.get("ohlc", {}).get("open")
            
            if last_price and open_price and open_price > 0:
                quotes[sym] = {"last_price": last_price, "open": open_price}
                valid_stocks.append(stock)
                
        if not valid_stocks:
            return monitor_list
    except Exception as e:
        logger.error(f"Batch {batch_idx}: Failed to fetch quotes: {e}")
        return monitor_list

    # --- OPTIMIZATION 2: DAILY FILTER ---
    # Fetch Daily data ONLY to check the Slingshot condition.
    daily_data_dict = preload_daily_data(valid_stocks, from_date, yesterday)
    
    surviving_stocks = []
    daily_mas_cache = {}
    
    for stock in valid_stocks:
        symbol = stock["symbol"]
        if symbol not in daily_data_dict:
            continue
            
        df_daily = daily_data_dict[symbol]
        current_open = quotes[symbol]["open"]
        
        # Calculate Daily MAs including today's open
        prices = df_daily["close"].tolist()
        prices.append(current_open)
        price_series = pd.Series(prices)
        
        daily_mas = []
        is_valid = True
        for p in [20, 50, 100, 200]:
            if len(price_series) >= p:
                ma_val = price_series.rolling(window=p).mean().iloc[-1]
                if pd.isna(ma_val) or ma_val <= 0:
                    is_valid = False
                    break
                daily_mas.append(float(ma_val))
            else:
                is_valid = False
                break
                
        if not is_valid:
            continue
            
        # MATH FILTER: If open >= min(daily MAs), it can't be a Slingshot.
        if current_open >= min(daily_mas):
            continue 
            
        surviving_stocks.append(stock)
        daily_mas_cache[symbol] = daily_mas

    surviving_set = {s["symbol"] for s in surviving_stocks}
    for sym in list(daily_data_dict.keys()):
        if sym not in surviving_set:
            del daily_data_dict[sym]
    gc.collect()

    # --- OPTIMIZATION 3: HOURLY FETCH ONLY FOR SURVIVORS ---
    # This is where the real time-saving happens.
    hourly_data_dict = preload_hourly_data(surviving_stocks, from_date, yesterday)

    surviving_symbols = [s["symbol"] for s in surviving_stocks]
    with ThreadPoolExecutor(max_workers=5) as executor:
        news_map = dict(zip(surviving_symbols, executor.map(get_stock_news, surviving_symbols)))

    for stock in surviving_stocks:
        symbol = stock["symbol"]
        if symbol not in hourly_data_dict:
            continue
            
        df_hourly = hourly_data_dict[symbol]
        current_open = quotes[symbol]["open"]
        current_ltp = quotes[symbol]["last_price"]
        
        # Calculate Hourly MAs including today's open
        h_prices = df_hourly["close"].tolist()
        h_prices.append(current_open)
        h_price_series = pd.Series(h_prices)
        
        hourly_mas = []
        is_valid = True
        for p in [20, 50, 100, 200]:
            if len(h_price_series) >= p:
                ma_val = h_price_series.rolling(window=p).mean().iloc[-1]
                if pd.isna(ma_val) or ma_val <= 0:
                    is_valid = False
                    break
                hourly_mas.append(float(ma_val))
            else:
                is_valid = False
                break
        
        if not is_valid:
            continue

        # Final MA Check
        all_mas = daily_mas_cache[symbol] + hourly_mas
        min_ma = min(all_mas)
        
        if current_open >= min_ma:
            continue

        # Stock is valid! Build final entry
        df_daily = daily_data_dict[symbol]
        prev_day = df_daily.iloc[-1]
        pivots = calculate_pivots(prev_day["high"], prev_day["low"], prev_day["close"])
        recent_daily = df_daily.tail(20)
        atr_14 = calculate_atr(recent_daily)
        
        is_bb_squeeze = calculate_bb_squeeze(df_daily.tail(30)) if len(df_daily) >= 30 else False
        df_hist_10 = df_daily.tail(10).copy()
        df_hist_10["range_pct"] = ((df_hist_10["high"] - df_hist_10["low"]) / df_hist_10["low"]) * 100
        avg_range = float(df_hist_10["range_pct"].mean())

        rsi_at_entry = calculate_rsi(df_daily["close"]).iloc[-1] if len(df_daily) >= 15 else None
        stock_news = news_map.get(symbol)

        monitor_entry = {
            "symbol": symbol,
            "date": datetime.date.today().strftime("%Y-%m-%d"),
            "latest_news": stock_news,
            "open_price": float(current_open),
            "current_price": float(current_ltp),
            "min_ma": float(min_ma),
            "max_ma": float(max(all_mas)),
            "monitoring_tier": "slow",
            "tick_size": float(stock["tick_size"]),
            "ma20_daily": daily_mas_cache[symbol][0],
            "ma50_daily": daily_mas_cache[symbol][1],
            "ma100_daily": daily_mas_cache[symbol][2],
            "ma200_daily": daily_mas_cache[symbol][3],
            "ma20_hourly": hourly_mas[0],
            "ma50_hourly": hourly_mas[1],
            "ma100_hourly": hourly_mas[2],
            "ma200_hourly": hourly_mas[3],
            "prev_day_high": float(prev_day["high"]),
            "prev_day_low": float(prev_day["low"]),
            "prev_day_close": float(prev_day["close"]),
            "pivot_points": pivots,
            "atr_14": float(atr_14) if pd.notna(atr_14) else None,
            "avg_daily_vol_20d": float(recent_daily["volume"].mean()),
            "rsi_at_entry": float(rsi_at_entry) if pd.notna(rsi_at_entry) else None,
            "is_bb_squeeze": bool(is_bb_squeeze),
            "avg_intraday_range_pct_10d": avg_range,
            "monitor_entry_time": datetime.datetime.combine(to_date, datetime.time(9, 15)).isoformat(),
            "has_derivative": bool(stock.get("has_derivative", False)),
        }
        print(f"✅ FOUND: {symbol} | Open: {current_open} | MinMA: {min_ma:.2f}")
        monitor_list.append(monitor_entry)

    return monitor_list

def create_monitor_list():
    """Builds monitor list sequentially to save memory on Render."""
    logger.info("Starting Memory-Optimized Monitor List Builder...")
    
    # 1. Get Auth Token
    try:
        auth_token = kite.access_token
    except AttributeError:
        try:
            auth_token = kite.enctoken
        except AttributeError:
            logger.error("Could not find access_token or enctoken on kite object.")
            return

    # 2. Set Dates (Target = Today)
    analysis_date = datetime.date.today()
    from_date = analysis_date - datetime.timedelta(days=365)
    
    # 3. Fetch Stocks (ALL Stocks)
    stocks = get_all_nse_symbols()
    if not stocks:
        logger.error("No stocks fetched. Exiting.")
        return
    

    # 4. Process Sequentially in Small Batches (Reduced to 50 for RAM safety)
    batch_size = 100
    total_stocks = len(stocks)
    total_batches = (total_stocks + batch_size - 1) // batch_size
    
    logger.info(f"Processing {total_stocks} stocks in {total_batches} sequential batches...")

    for i in range(0, total_stocks, batch_size):
        batch = stocks[i:i + batch_size]
        batch_idx = (i // batch_size) + 1
        
        # Prepare inputs for the existing process_batch function
        batch_input = (batch, batch_idx, total_batches, from_date, analysis_date, auth_token)
        
        try:
            # 🆕 CRITICAL: Run sequentially, NOT in a Pool
            batch_monitor_list = process_batch(batch_input)
            
            # Upsert this batch immediately to Supabase
            if batch_monitor_list:
                batch_upsert_supabase("monitor_list", batch_monitor_list)
                logger.info(f"✅ Batch {batch_idx}/{total_batches} saved and cleared from RAM.")
            
            # 🆕 CRITICAL: Clear batch data from memory immediately
            del batch_monitor_list
            gc.collect() 
            
        except Exception as e:
            logger.error(f"❌ Batch {batch_idx} failed: {e}")

    logger.info("🏁 Monitor List Builder completed.")
if __name__ == "__main__":
    create_monitor_list()