import numpy as np
import datetime
import time
import pandas as pd
import subprocess
import base64
import google.generativeai as genai
import os
import requests
import joblib
from sklearn.linear_model import LinearRegression
from utils.common import load_token_map, token_map
from utils.common import kite, supabase, logger

load_token_map()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
NIFTY_TOKEN = 256265

ML_BUNDLE = joblib.load("logistic_breakout_model_2.pkl")
ML_MODEL = ML_BUNDLE["model"]
ML_SCALER = ML_BUNDLE["scaler"]
ML_IMPUTER = ML_BUNDLE["imputer"]
ML_FEATURES = ML_BUNDLE["features"]
    
def run_ml_prediction(feature_dict):
    X = pd.DataFrame([feature_dict])[ML_FEATURES]
    X = ML_IMPUTER.transform(X)
    X = ML_SCALER.transform(X)
    prob = ML_MODEL.predict_proba(X)[0][1]
    return prob

def evaluate_breakout_with_ai(symbol, instrument_token, breakout_time, breakout_data):

    stock_df = fetch_30d_daily_ohlcv(
        symbol,
        instrument_token,
        breakout_time.date()
    )

    if stock_df is None:
        return None, "NO_STOCK_DATA"

    nifty_map = fetch_nifty_daily_map(breakout_time.date())

    rows = []
    for _, r in stock_df.iterrows():
        n = nifty_map.get(r["date"])
        rows.append({
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"],
            "nifty_open": n["open"] if n else 0,
            "nifty_close": n["close"] if n else 0
        })

    ml_df = pd.DataFrame(rows)

    features = get_ml_model_features(
        ml_df,
        ml_df.rename(columns={
            "nifty_open":"nifty_open",
            "nifty_close":"nifty_close"
        })
    )

    try:
        ml_prob = run_ml_prediction(features)
    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        ml_prob = None


    charts = capture_tv_charts(symbol, breakout_time)
    if not charts:
        return ml_prob, "NO_CHARTS"

    ai_decision, prompt = run_ai_evaluation(breakout_data, charts)

    return ml_prob, ai_decision

def capture_tv_charts(symbol, breakout_time):
    """Calls the screenshot scrapper and returns paths + base64 images."""

    dt_string = breakout_time.strftime("%Y-%m-%dT%H:%M:%S")

    day_num = breakout_time.day
    suffix = "th" if 11 <= day_num <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day_num % 10, "th")
    date_str = f"{day_num}{suffix}{breakout_time.strftime('%b%Y')}"
    safe_symbol = symbol.replace(":", "_")

    out_dir = f"./chart_screens/{safe_symbol}_{date_str}"
    os.makedirs(out_dir, exist_ok=True)

    # Call your screenshot script
    try:
        result = subprocess.run([
            "python", "screenshot_scrapper.py",
            "--symbol", symbol,
            "--dt", dt_string,
            "--out", out_dir,
            "--headless"
        ], timeout=180, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Screenshot failed for {symbol}: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Screenshot timeout for {symbol}")
        return None
    except Exception as e:
        logger.error(f"Screenshot error for {symbol}: {e}")
        return None

    # Build file paths
    img_15m = f"{out_dir}/{safe_symbol}_{date_str}_15m.png"
    img_1h  = f"{out_dir}/{safe_symbol}_{date_str}_1h.png"
    img_D   = f"{out_dir}/{safe_symbol}_{date_str}_D.png"

    # Check if files exist before encoding
    if not all(os.path.exists(p) for p in [img_15m, img_1h, img_D]):
        logger.error(f"Screenshot files missing for {symbol}")
        return None

    # Convert to base64
    def encode(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    return {
        "img_15m": encode(img_15m),
        "img_1h": encode(img_1h),
        "img_D": encode(img_D),
        "paths": [img_15m, img_1h, img_D]
    }
    
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


def get_historical_context(symbol, instrument_token, analysis_date):
    """
    Fetches past 30 trading days of OHLCV for the stock and Nifty.
    Adapted from build_ohlcv_dataset.py logic.
    """
    # 45 days buffer to ensure we get 30 trading days
    start_date = analysis_date - datetime.timedelta(days=45)
    end_date = analysis_date - datetime.timedelta(days=1)
    
    try:
        # 1. Fetch Stock Daily Data
        candles = kite.historical_data(instrument_token, start_date, end_date, "day")
        if not candles:
            return None, None
            
        df = pd.DataFrame(candles).tail(30)
        
        # 2. Fetch Nifty Daily Data
        nifty_token = token_map.get("NIFTY")
        nifty_candles = kite.historical_data(nifty_token, start_date, end_date, "day")
        n_df = pd.DataFrame(nifty_candles)
        
        # Create a mapping for easy Nifty lookup by date
        n_df["date"] = pd.to_datetime(n_df["date"]).dt.date
        nifty_map = n_df.set_index("date")[["open", "close"]].to_dict('index')

        # Format for AI Prompt context
        df["date_str"] = pd.to_datetime(df["date"]).dt.date
        history_rows = []
        for _, r in df.iterrows():
            d = r["date_str"]
            n_row = nifty_map.get(d, {"open": "N/A", "close": "N/A"})
            history_rows.append(
                f"{d}: O:{r['open']} H:{r['high']} L:{r['low']} C:{r['close']} V:{r['volume']} | Nifty_O:{n_row['open']} Nifty_C:{n_row['close']}"
            )
        
        return "\n".join(history_rows), len(df)
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {e}")
        return None, 0

def fetch_30d_daily_ohlcv(symbol, instrument_token, breakout_date):
    end_date = breakout_date - datetime.timedelta(days=1)
    start_date = breakout_date - datetime.timedelta(days=45)

    candles = kite.historical_data(
        instrument_token=instrument_token,
        from_date=start_date,
        to_date=end_date,
        interval="day"
    )

    if not candles:
        return None

    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").tail(30)

    return df if not df.empty else None

def fetch_nifty_daily_map(breakout_date):
    end_date = breakout_date - datetime.timedelta(days=1)
    start_date = breakout_date - datetime.timedelta(days=45)

    candles = kite.historical_data(
        instrument_token=NIFTY_TOKEN,
        from_date=start_date,
        to_date=end_date,
        interval="day"
    )

    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return {
        r["date"]: r
        for _, r in df.iterrows()
    }

def get_ml_model_features(stock_df, nifty_df):
    closes = stock_df["close"].values
    highs = stock_df["high"].values
    lows = stock_df["low"].values
    volumes = stock_df["volume"].values
    nifty_closes = nifty_df["nifty_close"].values
    if len(nifty_closes) < 2 or np.all(np.isnan(nifty_closes)):
        return_total_n = 0.0
        return_mean_n = 0.0
        return_std_n = 0.0
        volatility_n = 0.0
        corr = 0.0
        s_ret = np.diff(closes) / closes[:-1] if len(closes) >= 2 else np.array([0.0])
    else:
        n_ret = np.diff(nifty_closes) / nifty_closes[:-1]
        return_total_n = float((nifty_closes[-1] / nifty_closes[0]) - 1)
        return_mean_n = float(np.mean(n_ret))
        return_std_n = float(np.std(n_ret))
        volatility_n = float(np.std(n_ret))
        # For corr, similar check for s_ret
        s_ret = np.diff(closes) / closes[:-1] if len(closes) >= 2 else np.array([0.0])
        corr = np.corrcoef(s_ret, n_ret)[0, 1] if len(s_ret) == len(n_ret) and len(s_ret) > 0 else 0.0

    def compute_slope(series):
        x = np.arange(len(series)).reshape(-1, 1)
        return LinearRegression().fit(x, series).coef_[0]

    # Max Drawdown logic
    peak = closes[0]
    max_dd = 0.0
    for price in closes:
        if price > peak: peak = price
        max_dd = min(max_dd, (price - peak) / peak)

    return {
        "return_total": float((closes[-1] / closes[0]) - 1),
        "return_mean": float(np.mean(s_ret)),
        "return_std": float(np.std(s_ret)),
        "max_drawdown": float(max_dd),
        "price_slope": float(compute_slope(closes)),
        "volatility_mean": float(np.mean((highs - lows) / (closes + 1e-9))),
        "volume_mean": float(np.mean(volumes)),
        "volume_std": float(np.std(volumes)),
        "volume_trend_slope": float(compute_slope(volumes)),
        "close_position_mean": float(np.mean((closes - lows) / (highs - lows + 1e-9))),
        "nifty_return_total": return_total_n,
        "nifty_return_mean": return_mean_n,
        "nifty_return_std": return_std_n,
        "nifty_volatility": volatility_n,
        "stock_nifty_correlation": float(corr) if not np.isnan(corr) else 0.0
    }

# --- NEW: Sector Analysis Helper ---
def get_sector_performance(symbol, breakout_time, analysis_date):
    """
    Finds peers in the same sector and calculates their average % move 
    from market open till their HIGHEST price reached before breakout_time.
    Formula: ((high - open) / open) * 100
    """
    logger.info(f"🔍 Analyzing sector peak performance for {symbol}...")
    
    try:
        # 1. Identify the industry/sector from Supabase
        res = supabase.table("nse_equity_classification") \
            .select("industry") \
            .eq("symbol", symbol) \
            .limit(1) \
            .execute()
        
        if not res.data:
            logger.warning(f"Sector not found for {symbol}")
            return None
        
        target_industry = res.data[0]['industry']
        
        # 2. Fetch peer symbols in that industry
        peer_res = supabase.table("nse_equity_classification") \
            .select("symbol") \
            .eq("industry", target_industry) \
            .execute()
        
        peer_symbols = [row['symbol'] for row in peer_res.data if row['symbol'] != symbol]
        if not peer_symbols:
            return 0.0

        # 3. Calculate peak performance for peers
        moves = []
        start_dt = datetime.datetime.combine(analysis_date, datetime.time(9, 15))
        
        # Limit to 15 peers to maintain speed during live monitoring
        for peer in peer_symbols: 
            token = token_map.get(peer)
            if not token: continue
            
            try:
                # Fetch minute data from open to breakout time
                data = kite.historical_data(token, start_dt, breakout_time, "minute")
                if data:
                    open_p = None
                    for candle in data:
                        ts = candle.get("date")
                        if ts and ts.time() == datetime.time(9, 15):
                            open_p = candle["open"]
                            break

                    if open_p is None:
                        continue  # skip peer completely
                    
                    # Find the HIGHEST price across all candles until breakout time
                    highest_p = max(candle['high'] for candle in data)

                    if open_p <= 0:
                        continue
                    
                    # Calculate percentage move based on peak
                    move = ((highest_p - open_p) / open_p) * 100
                    moves.append(move)
                    time.sleep(0.05)
            except Exception as e:
                continue

        total_peers = len(peer_symbols)
        valid_peers = len(moves)

        if valid_peers == 0:
            logger.warning(f"Sector avg skipped for {symbol}: no valid peer data")
            return None

        coverage_pct = (valid_peers / total_peers) * 100

        if coverage_pct < 60:
            logger.warning(
                f"Sector avg weak for {symbol}: only {valid_peers}/{total_peers} peers ({coverage_pct:.1f}%)"
            )

        avg_peak_move = sum(moves) / valid_peers

        logger.info(
            f"📊 Sector Peak Avg: {avg_peak_move:.2f}% "
            f"(valid {valid_peers}/{total_peers}, coverage {coverage_pct:.1f}%)"
        )
        return round(avg_peak_move, 2)

    except Exception as e:
        logger.error(f"Sector peak check failed: {e}")
        return None

def run_ai_evaluation(breakout, chart_images):
    """Send breakout data to Gemini for YES/NO evaluation."""

    # --- unpack breakout fields ---
    symbol = breakout.get("symbol")
    breakout_price = breakout.get("breakout_price")
    rsi_at_entry = breakout.get("rsi_at_entry")
    percent_rsi_move = breakout.get("percent_rsi_move")
    rsi_at_breakout = breakout.get("rsi_at_breakout")
    volume_multiplier = breakout.get("volume_multiplier")
    volume_spike = breakout.get("volume_spike")
    atr_14 = breakout.get("atr_14")
    avg_daily_vol_20d = breakout.get("avg_daily_vol_20d")
    day_low_price = breakout.get("day_low_price")
    prev_day_high = breakout.get("prev_day_high")
    prev_day_low = breakout.get("prev_day_low")
    pivot_points = breakout.get("pivot_points")
    if not isinstance(pivot_points, dict):
        pivot_points = {"pivot": None, "s1": None, "s2": None, "r1": None, "r2": None}
    nifty_move = breakout.get("nifty_percent_move")
    nifty_above_50ma = breakout.get("nifty_above_50ma")
    breakout_time = breakout.get("breakout_time")
    nifty_value_at_entry = breakout.get("nifty_value_at_entry")
    nifty_value_at_breakout = breakout.get("nifty_value_at_breakout")
    is_bb_squeeze = breakout.get("is_bb_squeeze")
    avg_intraday_range_pct_10d = breakout.get("avg_intraday_range_pct_10d")
    vol_vs_avg_pct_at_breakout = breakout.get("vol_vs_avg_pct_at_breakout")

    # Moving averages (4 daily + 4 hourly)
    ma20_daily  = breakout.get("ma20_daily")
    ma50_daily  = breakout.get("ma50_daily")
    ma100_daily = breakout.get("ma100_daily")
    ma200_daily = breakout.get("ma200_daily")

    ma20_hourly  = breakout.get("ma20_hourly")
    ma50_hourly  = breakout.get("ma50_hourly")
    ma100_hourly = breakout.get("ma100_hourly")
    ma200_hourly = breakout.get("ma200_hourly")

    # --- decode images EXACTLY like your openai version expected ---
    import base64
    img_15m = base64.b64decode(chart_images["img_15m"])
    img_1h  = base64.b64decode(chart_images["img_1h"])
    img_D   = base64.b64decode(chart_images["img_D"])

    latest_news = breakout.get("latest_news")
    if latest_news is None:
        news_context = "No relevant news found for this stock in the last 24 hours."
    else:
        news_context = latest_news

    sector_avg_move = breakout.get("sector_avg_move", "N/A")

    # ---- BUILD PROMPT ----
    ai_prompt = f"""
You are an expert intraday breakout trader with 15+ years experience.

Evaluate the following breakout setup for {symbol} and determine if it is high-probability for a ≥3% intraday move.

RSI & MOMENTUM
-----------------------
• RSI Entry: {rsi_at_entry}
• RSI Breakout: {rsi_at_breakout}
• RSI Surge %: {percent_rsi_move}

VOLUME & VOLATILITY
-----------------------
• Volume Multiplier: {volume_multiplier}
• Volume Spike: {volume_spike}
• Vol vs Avg Vol till breakout (%): {vol_vs_avg_pct_at_breakout}
• ATR(14): {atr_14}
• Avg Daily Volume (20d): {avg_daily_vol_20d}
• Breakout Price: {breakout_price}
• Breakout Time: {breakout_time}

SECTOR & MARKET CONTEXT
-----------------------
• Sector Average Move (Open to Breakout): {sector_avg_move}%
• Nifty value at entry: {nifty_value_at_entry}
• Nifty value at breakout: {nifty_value_at_breakout}
• Nifty trend (% from entry to breakout): {nifty_move}
• Nifty above 50MA: {nifty_above_50ma}

PRICE ACTION & LEVELS
-----------------------
• Day Low: {day_low_price}br
• Previous Day High: {prev_day_high}
• Previous Day Low: {prev_day_low}

PIVOTS
-----------------------
• Pivot: {pivot_points['pivot']}
• S1: {pivot_points['s1']}
• S2: {pivot_points['s2']}
• R1: {pivot_points['r1']}
• R2: {pivot_points['r2']}

MOVING AVERAGES
-----------------------
• MA20 Daily: {ma20_daily}
• MA50 Daily: {ma50_daily}
• MA100 Daily: {ma100_daily}
• MA200 Daily: {ma200_daily}
• MA20 Hourly: {ma20_hourly}
• MA50 Hourly: {ma50_hourly}
• MA100 Hourly: {ma100_hourly}
• MA200 Hourly: {ma200_hourly}

HISTORICAL BEHAVIOUR
-----------------------
• BB Squeeze active (last 30 sessions): {is_bb_squeeze}
• Avg intraday range (last 10 days, %): {avg_intraday_range_pct_10d}

LATEST NEWS (Last 24 Hours)
-----------------------
• {news_context}

PAST 30 TRADING DAYS OHLCV & NIFTY CONTEXT
-----------------------
•{breakout.get('historical_context') or "Historical data not available."}

TASK
-----------------------
Based on all signals above (Technical + News + Chart Pattern + 30-day price action history), do you think the stock will move 2%+ after breakout by the end of the day?

Respond **only** with Yes or No.
"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            contents=[
                ai_prompt,
                {"mime_type": "image/png", "data": img_15m},
                {"mime_type": "image/png", "data": img_1h},
                {"mime_type": "image/png", "data": img_D},
            ]
        )
        ai_answer = response.text.strip()
    except Exception as e:
        logger.error(f"AI eval failed: {e}")
        ai_answer = "AI Unavailable - Proceed Manually"

    # Normalize YES/NO (optional but safe)
    if "yes" in ai_answer.lower():
        ai_answer = "Yes"
    elif "no" in ai_answer.lower():
        ai_answer = "No"

    return ai_answer, ai_prompt

def check_volume_spike(df_subset, lookback=20):
    """Calculate volume spike and return multiplier. Fixed to use proper lookback window."""
    if len(df_subset) <= lookback:
        return False, 1.0
    
    # Use volume data EXCLUDING the current candle for average
    historical_volume = df_subset['volume'].iloc[:-1]  # Exclude last candle
    
    # Calculate average volume over lookback period
    if len(historical_volume) < lookback:
        return False, 1.0  # CHANGED: Default to 1.0x instead of None
    
    avg_vol = historical_volume.tail(lookback).mean()
    current_vol = df_subset['volume'].iloc[-1]
    
    if pd.isna(avg_vol) or avg_vol == 0:
        return False, 1.0  # CHANGED: Default to 1.0x instead of None
    
    multiplier = current_vol / avg_vol
    is_spike = multiplier > 1.5
    
    return is_spike, round(multiplier, 2)
