import datetime
import pandas as pd
from utils.common import supabase, kite, logger, next_price_above


SCRIPT_START_TIME = datetime.datetime.now()

# -------------------------- Helper Functions -------------------------

def process_breakout(symbol, monitor_data, current_price, analysis_date, instrument_token):
    """Processes a potential breakout for a stock."""

    if not instrument_token:
        logger.warning(f"Skipping {symbol}: invalid or missing instrument_token")
        return "invalid_token", None
    
    max_ma = monitor_data.get("max_ma")
    if max_ma is None or pd.isna(max_ma):
        return "invalid_max_ma", None

    tick_size = monitor_data.get("tick_size") or 0.05
    breakout_price = next_price_above(max_ma, tick_size)

    if current_price < breakout_price:
        return "price_below_breakout", symbol

    existing = supabase.table("live_breakouts") \
        .select("symbol") \
        .eq("symbol", symbol) \
        .eq("breakout_date", analysis_date.strftime("%Y-%m-%d")) \
        .limit(1) \
        .execute()

    if existing.data:
        logger.debug(f"⏭️ Duplicate breakout ignored for {symbol}")
        return "duplicate", None
    
    # ============================================
    # BREAKOUT CHECK (8th MA = max_ma)
    # ============================================

    # Fetch minute data for the day
    try:
        minute_data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=analysis_date,
            to_date=analysis_date,
            interval="minute"
        )
        if not minute_data:
            return "no_intraday_data", symbol
        df_intraday = pd.DataFrame(minute_data)
        df_intraday = df_intraday.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        df_intraday["date"] = pd.to_datetime(df_intraday["date"]).dt.tz_localize(None)
    except Exception as e:
        logger.error(f"Failed to fetch minute data for {symbol}: {e}")
        return "no_intraday_data", symbol
    
    if df_intraday.empty:
        return "no_intraday_data", symbol
    
    df_intraday = df_intraday.set_index("date")

    # Check if price has crossed 8th MA (max_ma) = BREAKOUT!
    breakout_price = next_price_above(max_ma, tick_size)

    # Look back from the LAST CHECK TIME instead of current time
    entry_time_str = monitor_data.get("monitor_entry_time")
    if not entry_time_str:
        logger.error(f"{symbol}: monitor_entry_time missing — cannot backfill breakout safely")
        return "invalid_monitor_state", None

    db_entry_time = pd.to_datetime(entry_time_str)
    # This ensures we only look for candles AFTER the script started
    last_check_time = max(db_entry_time, SCRIPT_START_TIME)
    
    breakout_rows = df_intraday[(df_intraday["high"] >= breakout_price) & (df_intraday.index >= last_check_time)]
    # --- CRITICAL FIX: LTP FALLBACK ---
    # If historical candles don't show breakout yet, but live LTP does, FORCE IT.
    if breakout_rows.empty and current_price >= breakout_price:
        logger.info(f"⚠️ {symbol} Candle lag detected! LTP ({current_price}) > MaxMA ({max_ma}). Forcing breakout.")
        
        # Create a synthetic row for "now" to allow processing to continue
        now_ts = datetime.datetime.now().replace(second=0, microsecond=0)
        
        # Only add if it doesn't exist
        if now_ts not in df_intraday.index:
            new_row = df_intraday.iloc[-1].copy() # Copy last known candle structure
            new_row["high"] = max(new_row["high"], current_price)
            new_row["close"] = current_price
            new_row["low"] = min(new_row["low"], current_price) # Adjust low if needed
            
            # Append to DataFrame
            df_intraday.loc[now_ts] = new_row

            breakout_rows = df_intraday[
            (df_intraday["high"] >= breakout_price) &
            (df_intraday.index >= last_check_time)
        ]

    if breakout_rows.empty:
        logger.debug(f"{symbol}: No breakout | Current: {current_price:.2f} | Breakout: {breakout_price:.2f} | Max High: {df_intraday['high'].max():.2f}")
        return "no_breakout", symbol
    
    breakout_time = breakout_rows.index[0]

    monitor_entry_time = pd.to_datetime(entry_time_str)

    logger.info(f"🚀 BREAKOUT DETECTED: {symbol} crossed 8th MA ({max_ma:.2f}) at {breakout_time}")

    day_high = float(df_intraday["high"].max())
    early_breakout = {
        "symbol": symbol,
        "breakout_date": analysis_date.strftime("%Y-%m-%d"),
        "breakout_time": breakout_time.isoformat(),
        "breakout_price": float(breakout_price),
        "monitor_entry_time": monitor_entry_time.isoformat(),
        "high_price": day_high,
        "percent_move": round(((day_high - float(breakout_price)) / float(breakout_price)) * 100, 2)
    }

    supabase.table("live_breakouts").upsert(
        early_breakout,
        on_conflict="symbol,breakout_date"
    ).execute()

    return "success", early_breakout


def get_monitor_list(analysis_date, symbols=None):
    """Fetches today's monitor list from Supabase with pagination."""
    try:
        all_data = []
        page_size = 1000
        offset = 0
        
        while True:
            query = supabase.table("monitor_list").select(
                "symbol, min_ma, max_ma, open_price, tick_size, rsi_at_entry,monitor_entry_time, "
                "ma20_daily, ma50_daily, ma100_daily, ma200_daily, "
                "ma20_hourly, ma50_hourly, ma100_hourly, ma200_hourly, "
                "prev_day_high, prev_day_low, prev_day_close, "
                "atr_14, avg_daily_vol_20d, pivot_points, monitoring_tier, "
                "latest_news, is_bb_squeeze, avg_intraday_range_pct_10d"

            ).eq("date", analysis_date.strftime("%Y-%m-%d"))
            
            # Filter by symbols if provided
            if symbols:
                query = query.in_("symbol", symbols)
            
            # Add pagination
            query = query.range(offset, offset + page_size - 1)
            
            response = query.execute()
            
            if not response.data:
                break  # No more data
            
            all_data.extend(response.data)
            
            if len(response.data) < page_size:
                break  # Last page (fewer than 1000 rows returned)
            
            offset += page_size
        
        logger.info(f"Fetched {len(all_data)} stocks from monitor list for {analysis_date}")
        return all_data
        
    except Exception as e:
        logger.error(f"Failed to fetch monitor list: {e}")
        return []
