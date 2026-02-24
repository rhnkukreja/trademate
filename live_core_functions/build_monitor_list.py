import datetime
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import gc
from utils.common import supabase, kite, logger, batch_upsert_supabase, next_price_above
import io
import requests
from utils.news_agent import get_stock_news

# -------------------------- Helper Functions -------------------------
def get_kite_token_map():
    """Builds symbol -> instrument_token map from Kite"""
    return {
        ins["tradingsymbol"]: ins["instrument_token"]
        for ins in kite.instruments("NSE")
        if ins.get("instrument_token")
    }

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
            "tick_size": s.get("tick_size", 0.05)
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

def preload_stock_data(stocks, from_date, to_date):
    """Preloads historical daily and hourly data using direct API calls."""
    data_dict = {}
    nifty_token = None
    
    # 1. Identify Nifty Token
    for ins in kite.instruments("NSE"):
        if ins["tradingsymbol"] == "NIFTY 50":
            nifty_token = ins["instrument_token"]
            break
    
    def fetch_kite_data_simple(instrument_token, symbol, start_date, end_date, interval):
        """Simple retry wrapper for Kite API"""
        retries = 3
        for attempt in range(retries):
            try:
                data = kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=start_date,
                    to_date=end_date,
                    interval=interval
                )
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                df = df.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                return df[["date", "open", "high", "low", "close", "volume"]]
                
            except Exception as e:
                # 429 means too many requests - wait longer
                wait_time = 1 if "429" not in str(e) else 2
                if attempt < retries - 1:
                    time.sleep(wait_time) 
                else:
                    logger.error(f"Failed to fetch {interval} for {symbol}: {e}")
                    return pd.DataFrame()

    for stock in stocks:
        symbol = stock["symbol"]
        token = stock["instrument_token"]
        
        # --- OPTIMIZATION 1: Fetch Daily Data ---
        df_daily = fetch_kite_data_simple(token, symbol, from_date, to_date, "day")
        if df_daily.empty:
            continue

        # --- OPTIMIZATION 2: Fetch 60minute Data Directly ---
        # Kite allows fetching ~2000 candles in one go. 
        # 365 days * 7 trading hours = ~2500 candles. 
        # If your from_date is < 200 days ago, 1 call is enough.
        # If > 200 days, we might need a small chunk, but usually 1 call covers recent history.
        
        # We limit hourly fetch to last 200 days to be safe and fast
        # (You only need 200MA, so 200 hourly candles is roughly 30 trading days)
        hourly_start = max(from_date, to_date - datetime.timedelta(days=100)) 
        
        df_hourly = fetch_kite_data_simple(token, symbol, hourly_start, to_date, "60minute")
        
        # Standardize columns
        if not df_hourly.empty:
             df_hourly = df_hourly.sort_values(by="date").drop_duplicates(subset="date").reset_index(drop=True)

        data_dict[symbol] = {"daily": df_daily, "hourly": df_hourly}
        
        # IMPORTANT: Small sleep to respect Kite's 3 requests/second limit
        # Since we are running in Multiprocessing, this helps prevent collisions
        time.sleep(0.15)

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
    """
    Processes a batch of stocks.
    Includes logic to handle 'missing' daily rows by using the last available data.
    """
    batch_stocks, batch_idx, total_batches, from_date, to_date, auth_token = batch_info
    yesterday = to_date - datetime.timedelta(days=1)

    # 1. AUTHENTICATE KITE IN SUBPROCESS
    # This ensures every worker process has a valid login session
    if hasattr(kite, "set_access_token"):
        kite.set_access_token(auth_token)
    elif hasattr(kite, "enctoken"):
        kite.enctoken = auth_token
        if hasattr(kite, "headers"):
            kite.headers["Authorization"] = f"enctoken {auth_token}"

    logger.info(f"Processing batch {batch_idx}/{total_batches}...")
    symbols = [s["symbol"] for s in batch_stocks]
    monitor_list = []

    # Preload Data
    data_dict = preload_stock_data(batch_stocks, from_date, yesterday)
    if not data_dict:
        return []
    
    try:
        quote_keys = [f"NSE:{s}" for s in symbols]
        raw_quotes = kite.quote(quote_keys)

        # Clean up quote keys (remove 'NSE:')
        quotes = {}
        for k, v in raw_quotes.items():
            # Try different possible field names
            last_price = v.get("last_price") or v.get("ltp") or v.get("last_traded_price")
            open_price = v.get("open") or v.get("ohlc", {}).get("open")
            
            if last_price and open_price and open_price > 0:
                symbol = k.split(":")[1] if ":" in k else k
                quotes[symbol] = {
                    "last_price": last_price,
                    "open": open_price
                }
        today_open_prices = {symbol: quotes.get(symbol, {}).get("open", 0) for symbol in symbols}
        
        if not quotes:
            logger.warning(f"Batch {batch_idx}: No valid quotes received - skipping batch")
            return monitor_list
            
    except Exception as e:
        logger.error(f"Batch {batch_idx}: Failed to fetch quotes: {e}")
        return monitor_list

    ma_dict = precompute_mas(data_dict, batch_stocks, today_open_prices)

    token_map = {ins["tradingsymbol"]: ins["instrument_token"] for ins in kite.instruments("NSE")}

    for stock in batch_stocks:
        symbol = stock["symbol"]
        
        # 1. Basic Data Integrity Checks
        if (symbol not in data_dict or 
            symbol not in ma_dict or 
            symbol not in quotes or 
            symbol not in token_map):
            continue

        daily_mas_df = ma_dict[symbol]["daily"]
        
        # --- ROBUSTNESS FIX ---
        # Instead of failing if "Today's" row is missing, we take the LAST AVAILABLE row.
        # This represents the most recent valid Moving Average levels.
        if daily_mas_df.empty:
            continue
        
        # Ensure we're using TODAY's MA values (not yesterday's stale data)
        today_rows = daily_mas_df[daily_mas_df["date"].dt.date == to_date]
        if today_rows.empty:
            # Fallback to most recent if today's data isn't available yet
            daily_row = daily_mas_df.iloc[-1]
            if (to_date - daily_row["date"].date()).days > 1:  # Changed from 5 to 1
                continue
        else:
            daily_row = today_rows.iloc[-1]

        # 2. Calculate MAs (Ignore NaNs for young stocks)
        hourly_mas_df = ma_dict[symbol]["hourly"]
        last_hourly_row = hourly_mas_df.iloc[-1] if not hourly_mas_df.empty else None

        # STRICT MA VALIDATION
        mas = {}

        # Daily MAs
        for p in [20, 50, 100, 200]:
            val = daily_row.get(f"ma{p}")
            if pd.isna(val) or val <= 0:
                mas = {}
                break
            mas[f"ma{p}_daily"] = float(val)

        # Hourly MAs
        if mas and last_hourly_row is not None:
            for p in [20, 50, 100, 200]:
                val = last_hourly_row.get(f"ma{p}")
                if pd.isna(val) or val <= 0:
                    mas = {}
                    break
                mas[f"ma{p}_hourly"] = float(val)

        # ❌ Reject stock if ALL 8 MAs not present
        if len(mas) != 8:
            continue

        min_ma = min(mas.values())
        max_ma = max(mas.values())

        # 3. Get Previous Day Data (for Pivots & Dip Check)
        # We need historical context. If we only have 1 row, we can't do much.
        df_daily = data_dict[symbol]["daily"]
        if len(df_daily) < 2:
            continue
        
        prev_day_rows = df_daily[df_daily["date"].dt.date < datetime.date.today()]
        if prev_day_rows.empty:
            continue
        prev_day = prev_day_rows.iloc[-1]

        # 4. Live Data
        current_ltp = quotes[symbol]["last_price"]
        current_open = quotes[symbol]["open"]
        
        # --- SLINGSHOT LOGIC ---

        # B. DIP CONDITION - STRICTLY Open Price Only
        # 1. If Open Price is missing/invalid, SKIP the stock completely.
        if current_open is None or current_open <= 0:
            continue

        # 2. If Open Price is NOT below the Lowest MA, SKIP.
        if current_open >= min_ma:
            continue
        # -----------------------

        # 5. Build Entry
        pivots = calculate_pivots(prev_day["high"], prev_day["low"], prev_day["close"])
        recent_daily = df_daily.tail(20)
        atr_14 = calculate_atr(recent_daily)
        if pd.isna(atr_14):
            continue
        avg_daily_vol_20 = recent_daily["volume"].mean()

        # --- Extra context for AI / algo ---
        df_daily_history = df_daily[df_daily["date"].dt.date < to_date].copy()

        # BB squeeze flag (last ~30 days)
        if len(df_daily_history) >= 30:
            is_bb_squeeze = calculate_bb_squeeze(df_daily_history.tail(30))
        else:
            is_bb_squeeze = False

        if len(df_daily_history) < 10:
            continue

        df_hist_10 = df_daily_history.tail(10).copy()
        df_hist_10["range_pct"] = (
            (df_hist_10["high"] - df_hist_10["low"]) / df_hist_10["low"]
        ) * 100

        avg_intraday_range_pct_10d = float(df_hist_10["range_pct"].mean())


        # Calculate RSI when stock was added to monitor list (pre-market, using yesterday's close as latest)
        rsi_at_entry = None
        if len(df_daily) >= 15:  # Need at least 14 + current
            rsi_series = calculate_rsi(df_daily["close"])
            rsi_at_entry = rsi_series.iloc[-1]

        print(f"✅ FOUND: {symbol} | Price: {current_ltp} | Open: {current_open} | MinMA: {min_ma:.2f}")

        stock_news = get_stock_news(symbol)
        if stock_news is None:
            stock_news = None

        today = to_date

        monitor_entry_time = datetime.datetime.combine(today, datetime.time(9, 15))

        monitor_entry = {
            "symbol": symbol,
            "date": datetime.date.today().strftime("%Y-%m-%d"), # Always save as Today's list
            "latest_news": stock_news,
            "open_price": float(current_open),
            "current_price": float(current_ltp),
            "min_ma": float(min_ma),
            "max_ma": float(max_ma),
            "monitoring_tier": "slow",
            "tick_size": float(stock["tick_size"]),
            "ma20_daily": float(daily_row["ma20"]) if pd.notna(daily_row["ma20"]) else None,
            "ma50_daily": float(daily_row["ma50"]) if pd.notna(daily_row["ma50"]) else None,
            "ma100_daily": float(daily_row["ma100"]) if pd.notna(daily_row["ma100"]) else None,
            "ma200_daily": float(daily_row["ma200"]) if pd.notna(daily_row["ma200"]) else None,
            "ma20_hourly": float(last_hourly_row["ma20"]) if last_hourly_row is not None and pd.notna(last_hourly_row["ma20"]) else None,
            "ma50_hourly": float(last_hourly_row["ma50"]) if last_hourly_row is not None and pd.notna(last_hourly_row["ma50"]) else None,
            "ma100_hourly": float(last_hourly_row["ma100"]) if last_hourly_row is not None and pd.notna(last_hourly_row["ma100"]) else None,
            "ma200_hourly": float(last_hourly_row["ma200"]) if last_hourly_row is not None and pd.notna(last_hourly_row["ma200"]) else None,
            "prev_day_high": float(prev_day["high"]),
            "prev_day_low": float(prev_day["low"]),
            "prev_day_close": float(prev_day["close"]),
            "pivot_points": pivots,
            "atr_14": float(atr_14) if pd.notna(atr_14) else None,
            "avg_daily_vol_20d": float(avg_daily_vol_20),
            "rsi_at_entry": float(rsi_at_entry) if pd.notna(rsi_at_entry) else None,
            "is_bb_squeeze": bool(is_bb_squeeze),
            "avg_intraday_range_pct_10d": float(avg_intraday_range_pct_10d) if avg_intraday_range_pct_10d is not None and not pd.isna(avg_intraday_range_pct_10d) else None,
            "monitor_entry_time": monitor_entry_time.isoformat()
        }

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
    batch_size = 50 
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